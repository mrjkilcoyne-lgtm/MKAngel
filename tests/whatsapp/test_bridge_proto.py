"""Tests for app.whatsapp.bridge — _trim_for_whatsapp and handle_event."""

from __future__ import annotations

import asyncio
from pathlib import Path
from types import SimpleNamespace

import pytest

from app.whatsapp import bridge as bridge_mod
from app.whatsapp.bridge import Bridge, _trim_for_whatsapp


# ── _trim_for_whatsapp ─────────────────────────────────────────────────────


def test_trim_short_passthrough():
    text = "hello, world"
    assert _trim_for_whatsapp(text) == text


def test_trim_long_truncated():
    limit = 3500
    text = "x" * (limit + 500)
    result = _trim_for_whatsapp(text, limit=limit)
    assert result.endswith("...[truncated]")
    assert len(result) <= limit


def test_trim_exact_limit():
    limit = 100
    text = "y" * limit
    # len(text) == limit should be returned unchanged (the guard is `<=`).
    assert _trim_for_whatsapp(text, limit=limit) == text


# ── Bridge.handle_event routing ────────────────────────────────────────────


def _make_fake_cfg(tmp_path: Path, allowed: set[str]) -> SimpleNamespace:
    """Smallest possible stand-in for BridgeConfig.

    ``handle_event`` only reads ``is_allowed`` and ``log_file`` (via
    ``_log_traffic``), so we don't need the full dataclass here.
    """
    log_file = tmp_path / "whatsapp.log"
    return SimpleNamespace(
        is_allowed=lambda jid: jid.split("@", 1)[0].lstrip("+") in allowed,
        log_file=log_file,
    )


def _install_fake_dispatch(monkeypatch, bridge_instance):
    """Replace ``bridge_instance.dispatch`` with a recording stub."""
    calls: list[tuple[str, str]] = []

    async def fake_dispatch(sender: str, text: str) -> None:
        calls.append((sender, text))

    # Bind to the instance so ``self`` isn't passed.
    monkeypatch.setattr(bridge_instance, "dispatch", fake_dispatch)
    return calls


def test_handle_event_msg_allowlisted_dispatches(monkeypatch, tmp_path):
    cfg = _make_fake_cfg(tmp_path, allowed={"447700900123"})
    b = Bridge(cfg)  # type: ignore[arg-type]
    calls = _install_fake_dispatch(monkeypatch, b)

    event = {
        "type": "msg",
        "from": "447700900123@s.whatsapp.net",
        "text": "hello",
    }
    asyncio.run(b.handle_event(event))

    assert calls == [("447700900123@s.whatsapp.net", "hello")]


def test_handle_event_msg_not_allowlisted_skipped(monkeypatch, tmp_path):
    cfg = _make_fake_cfg(tmp_path, allowed={"447700900123"})
    b = Bridge(cfg)  # type: ignore[arg-type]
    calls = _install_fake_dispatch(monkeypatch, b)

    event = {
        "type": "msg",
        "from": "447700999999@s.whatsapp.net",
        "text": "hello",
    }
    asyncio.run(b.handle_event(event))

    assert calls == []


def test_handle_event_ready_no_dispatch(monkeypatch, tmp_path):
    cfg = _make_fake_cfg(tmp_path, allowed={"447700900123"})
    b = Bridge(cfg)  # type: ignore[arg-type]
    calls = _install_fake_dispatch(monkeypatch, b)

    # ``handle_event("ready")`` calls ``repo_snapshot(self.cfg)``, which
    # lives in ``agent_runner``. Stub it out to avoid touching the real
    # repo/SDK code.
    monkeypatch.setattr(bridge_mod, "repo_snapshot", lambda cfg: "stub")

    asyncio.run(b.handle_event({"type": "ready"}))

    assert calls == []


def test_handle_event_qr_no_dispatch(monkeypatch, tmp_path):
    cfg = _make_fake_cfg(tmp_path, allowed={"447700900123"})
    b = Bridge(cfg)  # type: ignore[arg-type]
    calls = _install_fake_dispatch(monkeypatch, b)

    asyncio.run(b.handle_event({"type": "qr", "qr": "abc"}))

    assert calls == []
