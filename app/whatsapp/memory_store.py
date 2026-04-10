"""
SQLite-backed conversation history for the WhatsApp bridge.

One row per turn, keyed by WhatsApp JID. Thread-safe via a reentrant
lock. Uses WAL mode for low-contention read/write.
"""

from __future__ import annotations

import sqlite3
import threading
import time
from pathlib import Path


_SCHEMA = """
CREATE TABLE IF NOT EXISTS sessions (
    jid     TEXT NOT NULL,
    ts      REAL NOT NULL,
    role    TEXT NOT NULL,
    content TEXT NOT NULL
);
"""

_INDEX = """
CREATE INDEX IF NOT EXISTS idx_sessions_jid_ts
    ON sessions (jid, ts);
"""


class SessionStore:
    """
    Per-JID conversational memory backed by sqlite3.

    The store keeps one row per turn (``role`` is ``"user"`` or
    ``"assistant"``) and returns history in chronological order. It is
    safe to share a single :class:`SessionStore` across threads — all DB
    calls are guarded by a reentrant lock, and the underlying connection
    is opened with ``check_same_thread=False``.
    """

    def __init__(self, db_path: Path) -> None:
        self._db_path = Path(db_path)
        self._db_path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        self._conn = sqlite3.connect(
            str(self._db_path),
            check_same_thread=False,
            isolation_level=None,  # autocommit; we manage transactions explicitly
        )
        with self._lock:
            # WAL for low-contention concurrent reads/writes.
            self._conn.execute("PRAGMA journal_mode=WAL;")
            self._conn.execute("PRAGMA synchronous=NORMAL;")
            self._conn.execute(_SCHEMA)
            self._conn.execute(_INDEX)

    def get_history(self, jid: str, max_turns: int = 10) -> list[dict]:
        """
        Return the last ``2 * max_turns`` rows (user+assistant pairs)
        for ``jid`` in chronological order as
        ``[{"role": "user"|"assistant", "content": str}, ...]``.
        """
        limit = max(0, int(max_turns)) * 2
        if limit == 0:
            return []
        with self._lock:
            cur = self._conn.execute(
                """
                SELECT role, content FROM (
                    SELECT ts, role, content
                    FROM sessions
                    WHERE jid = ?
                    ORDER BY ts DESC
                    LIMIT ?
                )
                ORDER BY ts ASC
                """,
                (jid, limit),
            )
            rows = cur.fetchall()
        return [{"role": role, "content": content} for role, content in rows]

    def add_turn(self, jid: str, role: str, content: str) -> None:
        """Append a single turn to the history for ``jid``."""
        if role not in ("user", "assistant"):
            raise ValueError(f"role must be 'user' or 'assistant', got {role!r}")
        with self._lock:
            self._conn.execute(
                "INSERT INTO sessions (jid, ts, role, content) VALUES (?, ?, ?, ?)",
                (jid, time.time(), role, content),
            )

    def clear(self, jid: str) -> None:
        """Delete all history for ``jid``."""
        with self._lock:
            self._conn.execute("DELETE FROM sessions WHERE jid = ?", (jid,))

    def close(self) -> None:
        """Close the underlying sqlite connection."""
        with self._lock:
            try:
                self._conn.close()
            except sqlite3.Error:
                pass
