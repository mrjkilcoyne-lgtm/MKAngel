"""
Slash-command dispatcher for the WhatsApp bridge.

Gives Matt a set of fast, agent-free shortcuts for things he'd otherwise
spin up a whole Claude Agent SDK session to ask. If the inbound WhatsApp
message starts with '/', the bridge routes it here instead of to the
agent. Replies are plain strings, capped at ~3500 chars.
"""

from __future__ import annotations

import shlex
import subprocess
from pathlib import Path

MAX_REPLY_CHARS = 3500
FREEBIES_LIMIT = 3000
GIT_TIMEOUT = 10

_COMMANDS = (
    ("/help", "list available commands"),
    ("/branch", "current git branch"),
    ("/status", "git status (porcelain, first 20 lines)"),
    ("/log [N]", "last N commits (default 5, max 20)"),
    ("/repo", "repo path, branch, head, remote"),
    ("/freebies", "CANZUK freebies triage doc"),
    ("/clear", "clear chat history (placeholder)"),
)


class SlashDispatcher:
    def __init__(self, repo_root: Path) -> None:
        self.repo_root = Path(repo_root)

    # ── public API ──────────────────────────────────────────────────────

    def is_slash(self, text: str) -> bool:
        """True iff text.lstrip() starts with '/'."""
        if not isinstance(text, str):
            return False
        stripped = text.lstrip()
        return stripped.startswith("/")

    def dispatch(self, text: str, jid: str) -> str:
        """Run the command. Only call when is_slash(text) is True.
        Never raises; errors are returned as a short text reply."""
        try:
            stripped = text.lstrip()
            try:
                parts = shlex.split(stripped)
            except ValueError:
                parts = stripped.split()
            if not parts:
                return "unknown command. try /help"

            cmd = parts[0].lower()
            args = parts[1:]

            handler = {
                "/help": self._cmd_help,
                "/branch": self._cmd_branch,
                "/status": self._cmd_status,
                "/log": self._cmd_log,
                "/repo": self._cmd_repo,
                "/freebies": self._cmd_freebies,
                "/clear": self._cmd_clear,
            }.get(cmd)

            if handler is None:
                return "unknown command. try /help"

            reply = handler(args, jid)
            return _cap(reply)
        except Exception as exc:  # noqa: BLE001
            return f"slash error: {exc.__class__.__name__}: {exc}"[:200]

    # ── commands ────────────────────────────────────────────────────────

    def _cmd_help(self, args: list[str], jid: str) -> str:
        lines = ["available commands:"]
        for name, desc in _COMMANDS:
            lines.append(f"  {name} -- {desc}")
        return "\n".join(lines)

    def _cmd_branch(self, args: list[str], jid: str) -> str:
        return self._git("branch", "--show-current")

    def _cmd_status(self, args: list[str], jid: str) -> str:
        out = self._git("status", "--porcelain")
        if out.startswith("git error:"):
            return out
        lines = out.splitlines()
        if not lines:
            return "status: clean working tree"
        header = f"status ({len(lines)} entries, showing first 20):"
        shown = lines[:20]
        return "\n".join([header, *shown])

    def _cmd_log(self, args: list[str], jid: str) -> str:
        n = 5
        if args:
            try:
                parsed = int(args[0])
                if 1 <= parsed <= 20:
                    n = parsed
            except (ValueError, TypeError):
                n = 5
        return self._git("log", f"-{n}", "--oneline", "--decorate")

    def _cmd_repo(self, args: list[str], jid: str) -> str:
        branch = self._git("branch", "--show-current").strip()
        head = self._git("rev-parse", "--short", "HEAD").strip()
        remote = self._git("config", "--get", "remote.origin.url").strip()
        return (
            f"repo={self.repo_root}\n"
            f"branch={branch}\n"
            f"head={head}\n"
            f"remote={remote}"
        )

    def _cmd_freebies(self, args: list[str], jid: str) -> str:
        path = self.repo_root / "docs" / "freebies_canzuk.md"
        try:
            text = path.read_text(encoding="utf-8")
        except FileNotFoundError:
            return f"freebies doc not found at {path}"
        except OSError as exc:
            return f"could not read freebies doc: {exc}"[:200]
        if len(text) > FREEBIES_LIMIT:
            return (
                text[:FREEBIES_LIMIT]
                + f"\n\n...[truncated, full doc is {len(text)} chars]"
            )
        return text

    def _cmd_clear(self, args: list[str], jid: str) -> str:
        return "history cleared (TODO: wire to SessionStore)"

    # ── helpers ─────────────────────────────────────────────────────────

    def _git(self, *git_args: str) -> str:
        try:
            result = subprocess.run(
                ["git", "-C", str(self.repo_root), *git_args],
                cwd=str(self.repo_root),
                capture_output=True,
                text=True,
                timeout=GIT_TIMEOUT,
            )
        except FileNotFoundError:
            return "git error: git executable not found"
        except subprocess.TimeoutExpired:
            return "git error: timeout"
        except OSError as exc:
            return f"git error: {exc}"[:200]

        if result.returncode != 0:
            err = (result.stderr or "").strip()
            return f"git error: {err[:200]}"
        return (result.stdout or "").rstrip("\n")


def _cap(text: str, limit: int = MAX_REPLY_CHARS) -> str:
    if text is None:
        return ""
    if len(text) <= limit:
        return text
    return text[: limit - 30] + "\n\n...[truncated]"
