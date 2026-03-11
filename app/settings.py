"""
Settings and API key management for MKAngel.

Configuration is stored in ~/.mkangel/settings.json.
API keys are obfuscated with base64 — not encryption, but enough
to keep them out of casual plaintext view.
"""

from __future__ import annotations

import base64
import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


from app.paths import mkangel_dir

_CONFIG_DIR = mkangel_dir()
_SETTINGS_FILE = _CONFIG_DIR / "settings.json"

# Supported LLM providers
PROVIDERS = [
    "openai",
    "anthropic",
    "google",
    "mistral",
    "groq",
    "local",
]


def _obfuscate(plaintext: str) -> str:
    """Base64-encode a string (basic obfuscation, NOT encryption)."""
    return base64.b64encode(plaintext.encode("utf-8")).decode("ascii")


def _deobfuscate(encoded: str) -> str:
    """Decode a base64-obfuscated string."""
    return base64.b64decode(encoded.encode("ascii")).decode("utf-8")


@dataclass
class Settings:
    """Application settings with persistence.

    Manages model provider selection, API keys, theme, language,
    and offline mode.  Stored as JSON in ~/.mkangel/settings.json.
    """

    model_provider: str = "local"
    api_keys: dict[str, str] = field(default_factory=dict)
    theme: str = "dark"
    language: str = "en"
    offline_mode: bool = True

    # ------------------------------------------------------------------
    # API key management
    # ------------------------------------------------------------------

    def add_api_key(self, provider: str, key: str) -> None:
        """Store an API key for *provider* (base64-obfuscated)."""
        provider = provider.lower().strip()
        if provider not in PROVIDERS:
            raise ValueError(
                f"Unknown provider '{provider}'. "
                f"Supported: {', '.join(PROVIDERS)}"
            )
        self.api_keys[provider] = _obfuscate(key)
        self.save()

    def get_api_key(self, provider: str) -> str | None:
        """Retrieve the plaintext API key for *provider*, or None."""
        encoded = self.api_keys.get(provider.lower().strip())
        if encoded is None:
            return None
        try:
            return _deobfuscate(encoded)
        except Exception:
            return None

    def remove_api_key(self, provider: str) -> bool:
        """Remove the key for *provider*.  Returns True if one existed."""
        provider = provider.lower().strip()
        if provider in self.api_keys:
            del self.api_keys[provider]
            self.save()
            return True
        return False

    def list_providers(self) -> list[dict[str, Any]]:
        """Return a list of all providers with their configuration status."""
        result = []
        for p in PROVIDERS:
            has_key = p in self.api_keys
            is_active = self.model_provider == p
            result.append({
                "provider": p,
                "has_key": has_key,
                "active": is_active,
            })
        return result

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self) -> None:
        """Persist settings to disk."""
        _CONFIG_DIR.mkdir(parents=True, exist_ok=True)
        data = {
            "model_provider": self.model_provider,
            "api_keys": self.api_keys,       # already obfuscated
            "theme": self.theme,
            "language": self.language,
            "offline_mode": self.offline_mode,
        }
        with open(_SETTINGS_FILE, "w") as fh:
            json.dump(data, fh, indent=2)

    @classmethod
    def load(cls) -> "Settings":
        """Load settings from disk, or return defaults."""
        if not _SETTINGS_FILE.exists():
            return cls()
        try:
            with open(_SETTINGS_FILE) as fh:
                data = json.load(fh)
            return cls(
                model_provider=data.get("model_provider", "local"),
                api_keys=data.get("api_keys", {}),
                theme=data.get("theme", "dark"),
                language=data.get("language", "en"),
                offline_mode=data.get("offline_mode", True),
            )
        except (json.JSONDecodeError, OSError):
            return cls()

    # ------------------------------------------------------------------
    # Display
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        keys_summary = ", ".join(
            f"{p}={'*' * 8}" for p in self.api_keys
        )
        return (
            f"Settings(provider={self.model_provider}, "
            f"keys=[{keys_summary}], "
            f"theme={self.theme}, lang={self.language}, "
            f"offline={self.offline_mode})"
        )
