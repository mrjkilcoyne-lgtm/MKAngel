"""Tests for app.whatsapp.slash.SlashDispatcher."""

from __future__ import annotations

from pathlib import Path

import pytest

from app.whatsapp.slash import SlashDispatcher

REPO_ROOT = Path(__file__).resolve().parents[2]
JID = "447700900123@s.whatsapp.net"


@pytest.fixture
def dispatcher() -> SlashDispatcher:
    return SlashDispatcher(REPO_ROOT)


def test_is_slash_true(dispatcher: SlashDispatcher) -> None:
    assert dispatcher.is_slash("/help")
    assert dispatcher.is_slash("/branch")
    assert dispatcher.is_slash("  /help")  # leading whitespace OK
    assert dispatcher.is_slash("\t/log 3")


def test_is_slash_false(dispatcher: SlashDispatcher) -> None:
    assert not dispatcher.is_slash("hello")
    assert not dispatcher.is_slash("")
    assert not dispatcher.is_slash("   ")
    assert not dispatcher.is_slash("what is /help")


def test_help_lists_branch(dispatcher: SlashDispatcher) -> None:
    reply = dispatcher.dispatch("/help", JID)
    assert "/branch" in reply
    assert "/help" in reply
    assert "/log" in reply


def test_branch_returns_something(dispatcher: SlashDispatcher) -> None:
    reply = dispatcher.dispatch("/branch", JID)
    # Either a branch name or (rarely) empty if detached HEAD, but should
    # not be an error.
    assert not reply.startswith("git error:")
    assert not reply.startswith("slash error:")
    # Should be a short, single-line-ish response.
    assert len(reply) < 200


def test_unknown_command(dispatcher: SlashDispatcher) -> None:
    assert dispatcher.dispatch("/nope", JID) == "unknown command. try /help"
    assert dispatcher.dispatch("/banana split", JID) == "unknown command. try /help"


def test_freebies_contains_canzuk(dispatcher: SlashDispatcher) -> None:
    reply = dispatcher.dispatch("/freebies", JID)
    assert "CANZUK" in reply


def test_log_with_valid_arg(dispatcher: SlashDispatcher) -> None:
    reply = dispatcher.dispatch("/log 3", JID)
    assert not reply.startswith("git error:")
    assert not reply.startswith("slash error:")
    # Should contain at most 3 lines of commits (may be fewer in tiny repos).
    lines = [ln for ln in reply.splitlines() if ln.strip()]
    assert 1 <= len(lines) <= 3


def test_log_with_bad_arg_falls_back(dispatcher: SlashDispatcher) -> None:
    reply_banana = dispatcher.dispatch("/log banana", JID)
    reply_default = dispatcher.dispatch("/log", JID)
    assert not reply_banana.startswith("git error:")
    assert not reply_banana.startswith("slash error:")
    # Both should behave identically (default N=5).
    assert reply_banana == reply_default


def test_dispatch_never_raises(dispatcher: SlashDispatcher) -> None:
    # Unterminated quote would break shlex; must not raise.
    reply = dispatcher.dispatch('/log "unterminated', JID)
    assert isinstance(reply, str)
    assert reply  # non-empty


def test_repo_command_shape(dispatcher: SlashDispatcher) -> None:
    reply = dispatcher.dispatch("/repo", JID)
    assert "repo=" in reply
    assert "branch=" in reply
    assert "head=" in reply


def test_clear_placeholder(dispatcher: SlashDispatcher) -> None:
    reply = dispatcher.dispatch("/clear", JID)
    assert "history cleared" in reply
