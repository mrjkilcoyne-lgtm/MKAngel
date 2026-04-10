"""
Main WhatsApp bridge loop.

Spawns the Baileys Node subprocess, routes inbound WhatsApp messages to
:func:`agent_runner.run_agent`, and writes the agent's reply back to
Baileys for delivery.

Protocol on the Node <-> Python boundary (newline-delimited JSON):

    Node  -> Python (stdout):
        {"type": "ready"}
        {"type": "qr",    "qr": "<data url or ascii>"}
        {"type": "msg",   "from": "447700900123@s.whatsapp.net", "text": "..."}
        {"type": "error", "error": "..."}

    Python -> Node (stdin):
        {"type": "send", "to": "447700900123@s.whatsapp.net", "text": "..."}

Run with::

    python -m app.whatsapp.bridge
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import signal
import sys
from pathlib import Path

from app.whatsapp.agent_runner import repo_snapshot, run_agent
from app.whatsapp.config import BridgeConfig

logger = logging.getLogger("mkangel.whatsapp.bridge")


BAILEYS_DIR = Path(__file__).resolve().parent / "baileys"
BAILEYS_ENTRY = BAILEYS_DIR / "index.js"


class Bridge:
    def __init__(self, cfg: BridgeConfig) -> None:
        self.cfg = cfg
        self.proc: asyncio.subprocess.Process | None = None
        self.agent_lock = asyncio.Lock()

    # ── process management ──────────────────────────────────────────────

    async def start_baileys(self) -> None:
        if not BAILEYS_ENTRY.is_file():
            raise RuntimeError(
                f"Baileys entry script not found at {BAILEYS_ENTRY}. "
                "Run `npm install` in app/whatsapp/baileys first."
            )
        logger.info("spawning Baileys: node %s", BAILEYS_ENTRY)
        self.proc = await asyncio.create_subprocess_exec(
            "node",
            str(BAILEYS_ENTRY),
            cwd=str(BAILEYS_DIR),
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )

    async def stop(self) -> None:
        if self.proc and self.proc.returncode is None:
            logger.info("stopping Baileys subprocess")
            self.proc.terminate()
            try:
                await asyncio.wait_for(self.proc.wait(), timeout=5)
            except asyncio.TimeoutError:
                self.proc.kill()

    # ── send/receive ────────────────────────────────────────────────────

    async def send(self, to: str, text: str) -> None:
        assert self.proc and self.proc.stdin
        payload = json.dumps({"type": "send", "to": to, "text": text}) + "\n"
        self.proc.stdin.write(payload.encode())
        await self.proc.stdin.drain()
        self._log_traffic("tx", to, text)

    async def read_stderr(self) -> None:
        assert self.proc and self.proc.stderr
        async for raw in self.proc.stderr:
            line = raw.decode(errors="replace").rstrip()
            if line:
                logger.info("[baileys] %s", line)

    async def read_stdout(self) -> None:
        assert self.proc and self.proc.stdout
        async for raw in self.proc.stdout:
            line = raw.decode(errors="replace").rstrip()
            if not line:
                continue
            try:
                event = json.loads(line)
            except json.JSONDecodeError:
                logger.warning("non-JSON from baileys: %s", line)
                continue
            await self.handle_event(event)

    # ── events ──────────────────────────────────────────────────────────

    async def handle_event(self, event: dict) -> None:
        etype = event.get("type")
        if etype == "ready":
            logger.info("baileys ready; repo=%s", repo_snapshot(self.cfg))
        elif etype == "qr":
            logger.info("scan this QR from WhatsApp > Linked Devices:")
            logger.info("\n%s", event.get("qr", ""))
        elif etype == "msg":
            sender = event.get("from", "")
            text = (event.get("text") or "").strip()
            if not text:
                return
            if not self.cfg.is_allowed(sender):
                logger.warning("ignoring message from non-allowlisted %s", sender)
                return
            self._log_traffic("rx", sender, text)
            await self.dispatch(sender, text)
        elif etype == "error":
            logger.error("baileys error: %s", event.get("error"))
        else:
            logger.debug("unknown event: %s", event)

    async def dispatch(self, sender: str, text: str) -> None:
        if self.agent_lock.locked():
            await self.send(sender, "working on your previous one, queued this")
        async with self.agent_lock:
            await self.send(sender, "on it...")
            try:
                reply = await run_agent(text, self.cfg)
            except Exception as exc:  # noqa: BLE001
                logger.exception("agent crashed")
                reply = f"agent error: {exc}"
            await self.send(sender, _trim_for_whatsapp(reply))

    # ── logging ─────────────────────────────────────────────────────────

    def _log_traffic(self, direction: str, jid: str, text: str) -> None:
        try:
            self.cfg.log_file.parent.mkdir(parents=True, exist_ok=True)
            with self.cfg.log_file.open("a") as fh:
                fh.write(f"[{direction}] {jid}  {text!r}\n")
        except OSError as exc:
            logger.warning("could not write to log file: %s", exc)

    # ── main ────────────────────────────────────────────────────────────

    async def run(self) -> None:
        await self.start_baileys()
        assert self.proc is not None
        try:
            await asyncio.gather(
                self.read_stdout(),
                self.read_stderr(),
            )
        finally:
            await self.stop()


def _trim_for_whatsapp(text: str, limit: int = 3500) -> str:
    """WhatsApp's per-message limit is ~4096 chars; leave headroom."""
    if len(text) <= limit:
        return text
    return text[: limit - 30] + "\n\n...[truncated]"


def _install_signal_handlers(loop: asyncio.AbstractEventLoop, bridge: Bridge) -> None:
    def _handler() -> None:
        logger.info("shutdown signal received")
        asyncio.create_task(bridge.stop())
    for sig in (signal.SIGINT, signal.SIGTERM):
        try:
            loop.add_signal_handler(sig, _handler)
        except NotImplementedError:
            # Signal handlers don't work on Windows event loops.
            pass


def main() -> int:
    logging.basicConfig(
        level=os.environ.get("LOG_LEVEL", "INFO"),
        format="%(asctime)s %(name)s %(levelname)s: %(message)s",
    )
    try:
        cfg = BridgeConfig.from_env()
    except RuntimeError as exc:
        print(f"config error: {exc}", file=sys.stderr)
        return 2

    bridge = Bridge(cfg)
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    _install_signal_handlers(loop, bridge)
    try:
        loop.run_until_complete(bridge.run())
    except KeyboardInterrupt:
        pass
    finally:
        loop.run_until_complete(bridge.stop())
        loop.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
