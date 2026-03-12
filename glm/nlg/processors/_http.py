"""Shared HTTP utilities for domain processors.

Thin wrapper around urllib with:
- JSON parsing
- Simple in-memory cache (5-minute TTL)
- Configurable timeout
- Graceful failure (returns None, never raises)
"""

from __future__ import annotations

import json
import time
import urllib.request
import urllib.error
from typing import Any, Dict, List, Optional, Union

_CACHE: Dict[str, tuple] = {}
_CACHE_TTL = 300  # seconds

JsonResult = Optional[Union[Dict[str, Any], List[Any]]]


def fetch_json(url: str, timeout: float = 5.0) -> JsonResult:
    """Fetch JSON from *url*, returning parsed data or ``None`` on failure."""
    now = time.time()

    # Check cache
    if url in _CACHE:
        ts, data = _CACHE[url]
        if now - ts < _CACHE_TTL:
            return data

    try:
        req = urllib.request.Request(
            url, headers={"User-Agent": "MKAngel-GLM/1.0", "Accept": "application/json"},
        )
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            raw = resp.read().decode("utf-8")
            data = json.loads(raw)
            _CACHE[url] = (now, data)
            return data
    except Exception:
        return None


def clear_cache() -> None:
    """Clear the HTTP cache (useful for testing)."""
    _CACHE.clear()
