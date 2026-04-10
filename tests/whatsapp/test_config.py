"""Tests for app.whatsapp.config — BridgeConfig and _load_dotenv."""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from app.whatsapp.config import BridgeConfig, _load_dotenv


# ── helpers ────────────────────────────────────────────────────────────────

_ENV_VARS = (
    "ANTHROPIC_API_KEY",
    "WHATSAPP_ALLOWLIST",
    "WHATSAPP_REPO_ROOT",
    "CLAUDE_MODEL",
    "WHATSAPP_MAX_TURNS",
    "WHATSAPP_LOG_FILE",
)


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch):
    """Strip all bridge-related env vars so tests start from a clean slate."""
    for name in _ENV_VARS:
        monkeypatch.delenv(name, raising=False)
    yield


def _init_git_repo(path: Path) -> None:
    subprocess.run(
        ["git", "init", "-q", str(path)],
        check=True,
        capture_output=True,
    )


# ── BridgeConfig.from_env ──────────────────────────────────────────────────


def test_from_env_requires_api_key(monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    with pytest.raises(RuntimeError, match="ANTHROPIC_API_KEY"):
        BridgeConfig.from_env()


def test_from_env_requires_allowlist(monkeypatch, tmp_path):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test-123")
    monkeypatch.delenv("WHATSAPP_ALLOWLIST", raising=False)
    # Point repo_root at a real git repo so the allowlist error is what fires.
    _init_git_repo(tmp_path)
    monkeypatch.setenv("WHATSAPP_REPO_ROOT", str(tmp_path))
    with pytest.raises(RuntimeError, match="WHATSAPP_ALLOWLIST"):
        BridgeConfig.from_env()


def test_from_env_rejects_non_git_repo(monkeypatch, tmp_path):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test-123")
    monkeypatch.setenv("WHATSAPP_ALLOWLIST", "447700900123")
    monkeypatch.setenv("WHATSAPP_REPO_ROOT", str(tmp_path))
    # tmp_path is a brand new dir with no .git -- should be rejected.
    with pytest.raises(RuntimeError, match="not a git checkout"):
        BridgeConfig.from_env()


def test_from_env_success(monkeypatch, tmp_path):
    _init_git_repo(tmp_path)
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test-123")
    monkeypatch.setenv(
        "WHATSAPP_ALLOWLIST", "447700900123, +447700900999"
    )
    monkeypatch.setenv("WHATSAPP_REPO_ROOT", str(tmp_path))

    cfg = BridgeConfig.from_env()

    assert cfg.anthropic_api_key == "sk-test-123"
    assert "447700900123" in cfg.allowlist
    assert "447700900999" in cfg.allowlist
    # The leading '+' must have been stripped.
    assert "+447700900999" not in cfg.allowlist
    assert cfg.repo_root == tmp_path.resolve()
    assert cfg.model == "claude-opus-4-6"
    assert cfg.max_turns == 20
    assert isinstance(cfg.log_file, Path)


# ── BridgeConfig.is_allowed ────────────────────────────────────────────────


def _make_cfg(tmp_path: Path) -> BridgeConfig:
    return BridgeConfig(
        anthropic_api_key="sk-test",
        allowlist=frozenset({"447700900123"}),
        repo_root=tmp_path,
    )


def test_is_allowed_bare_number(tmp_path):
    cfg = _make_cfg(tmp_path)
    assert cfg.is_allowed("447700900123") is True


def test_is_allowed_full_jid(tmp_path):
    cfg = _make_cfg(tmp_path)
    assert cfg.is_allowed("447700900123@s.whatsapp.net") is True


def test_is_allowed_with_plus(tmp_path):
    cfg = _make_cfg(tmp_path)
    assert cfg.is_allowed("+447700900123") is True


def test_is_allowed_rejects_other(tmp_path):
    cfg = _make_cfg(tmp_path)
    assert cfg.is_allowed("447700999999") is False


# ── _load_dotenv ───────────────────────────────────────────────────────────


def test_load_dotenv_parses_quoted_values(monkeypatch, tmp_path):
    # Make sure neither key is in the environment when we start.
    monkeypatch.delenv("MKANGEL_TEST_KEY", raising=False)
    monkeypatch.delenv("MKANGEL_TEST_HASH", raising=False)

    env_file = tmp_path / ".env"
    env_file.write_text(
        'MKANGEL_TEST_KEY="value with spaces"\n'
        "MKANGEL_TEST_HASH=abc123 # comment\n"
    )
    _load_dotenv(env_file)

    import os

    # Quoted value: the .strip('"') strips the surrounding quotes.
    assert os.environ.get("MKANGEL_TEST_KEY") == "value with spaces"
    # "abc123 # comment" — partition splits on '=' only, and there is no
    # in-line comment handling, so the whole tail (after .strip()) is the
    # value. We test what the implementation actually does.
    assert os.environ.get("MKANGEL_TEST_HASH") == "abc123 # comment"

    # Clean up process env to keep other tests isolated (autouse fixture
    # only clears the canonical bridge vars).
    monkeypatch.delenv("MKANGEL_TEST_KEY", raising=False)
    monkeypatch.delenv("MKANGEL_TEST_HASH", raising=False)


def test_load_dotenv_ignores_existing_env(monkeypatch, tmp_path):
    monkeypatch.setenv("MKANGEL_TEST_PRESET", "original")

    env_file = tmp_path / ".env"
    env_file.write_text("MKANGEL_TEST_PRESET=overridden\n")
    _load_dotenv(env_file)

    import os

    # setdefault -> existing env wins.
    assert os.environ.get("MKANGEL_TEST_PRESET") == "original"

    monkeypatch.delenv("MKANGEL_TEST_PRESET", raising=False)
