"""
Configuration for the WhatsApp bridge.

Reads from environment variables (typically populated via a `.env` file
in ``app/whatsapp/.env`` — see ``.env.example``).

Required:
    ANTHROPIC_API_KEY   -- for the Claude Agent SDK
    WHATSAPP_ALLOWLIST  -- comma-separated list of WhatsApp JIDs allowed
                           to trigger the agent (e.g. "447700900123")
    WHATSAPP_REPO_ROOT  -- absolute path to the MKAngel checkout the agent
                           should operate on

Optional:
    CLAUDE_MODEL        -- model ID (default: claude-opus-4-6)
    WHATSAPP_MAX_TURNS  -- cap on agent turns per message (default: 20)
    WHATSAPP_LOG_FILE   -- path to append all traffic (default: ./whatsapp.log)
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path


def _load_dotenv(path: Path) -> None:
    """Tiny .env loader — no python-dotenv dependency."""
    if not path.is_file():
        return
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        os.environ.setdefault(key, value)


@dataclass
class BridgeConfig:
    anthropic_api_key: str
    allowlist: frozenset[str]
    repo_root: Path
    model: str = "claude-opus-4-6"
    max_turns: int = 20
    log_file: Path = field(default_factory=lambda: Path("whatsapp.log"))

    @classmethod
    def from_env(cls) -> "BridgeConfig":
        here = Path(__file__).resolve().parent
        _load_dotenv(here / ".env")

        key = os.environ.get("ANTHROPIC_API_KEY", "").strip()
        if not key:
            raise RuntimeError(
                "ANTHROPIC_API_KEY is not set. "
                "Copy app/whatsapp/.env.example to .env and fill it in."
            )

        raw_allow = os.environ.get("WHATSAPP_ALLOWLIST", "").strip()
        if not raw_allow:
            raise RuntimeError(
                "WHATSAPP_ALLOWLIST is empty. Set it to your WhatsApp number "
                "(digits only, including country code, e.g. 447700900123)."
            )
        allowlist = frozenset(
            n.strip().lstrip("+").replace(" ", "")
            for n in raw_allow.split(",")
            if n.strip()
        )

        repo_root = Path(
            os.environ.get("WHATSAPP_REPO_ROOT", here.parent.parent)
        ).resolve()
        if not (repo_root / ".git").is_dir():
            raise RuntimeError(
                f"WHATSAPP_REPO_ROOT ({repo_root}) is not a git checkout."
            )

        return cls(
            anthropic_api_key=key,
            allowlist=allowlist,
            repo_root=repo_root,
            model=os.environ.get("CLAUDE_MODEL", "claude-opus-4-6"),
            max_turns=int(os.environ.get("WHATSAPP_MAX_TURNS", "20")),
            log_file=Path(
                os.environ.get("WHATSAPP_LOG_FILE", str(here / "whatsapp.log"))
            ).resolve(),
        )

    def is_allowed(self, jid: str) -> bool:
        """Return True if the given WhatsApp JID is on the allowlist."""
        # JIDs look like "447700900123@s.whatsapp.net" — strip the domain.
        number = jid.split("@", 1)[0].lstrip("+").replace(" ", "")
        return number in self.allowlist
