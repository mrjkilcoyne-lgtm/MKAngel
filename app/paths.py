"""
Shared filesystem paths for MKAngel.

On Android, Path.home() returns /data which is root-only.
All modules MUST use mkangel_dir() instead of Path.home() / ".mkangel".
"""

from __future__ import annotations

from pathlib import Path


def mkangel_dir() -> Path:
    """Return the base MKAngel data directory, Android-aware."""
    try:
        from android.storage import app_storage_path  # type: ignore[import]
        return Path(app_storage_path()) / ".mkangel"
    except ImportError:
        return Path.home() / ".mkangel"
