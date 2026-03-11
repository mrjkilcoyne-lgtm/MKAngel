"""
Persistent memory system for MKAngel.

Uses SQLite for local-first storage — no server needed, works
everywhere including Termux.  Stores conversation history, learned
patterns, and user preferences between sessions.

Database lives at ~/.mkangel/memory.db
"""

from __future__ import annotations

import json
import sqlite3
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any


from app.paths import mkangel_dir

_DB_PATH = mkangel_dir() / "memory.db"


@dataclass
class MemoryEntry:
    """A single memory record."""

    id: int
    category: str          # "session", "pattern", "preference"
    key: str
    value: str
    metadata: dict[str, Any]
    created_at: float
    accessed_at: float


class Memory:
    """Persistent memory backed by SQLite.

    Stores conversation sessions, learned patterns, and user
    preferences so the Angel can remember across restarts.
    """

    def __init__(self, db_path: Path | str | None = None):
        self._db_path = Path(db_path) if db_path else _DB_PATH
        self._db_path.parent.mkdir(parents=True, exist_ok=True)
        self._conn: sqlite3.Connection | None = None
        self._ensure_schema()

    # ------------------------------------------------------------------
    # Connection management
    # ------------------------------------------------------------------

    def _get_conn(self) -> sqlite3.Connection:
        if self._conn is None:
            self._conn = sqlite3.connect(str(self._db_path))
            self._conn.row_factory = sqlite3.Row
            self._conn.execute("PRAGMA journal_mode=WAL")
        return self._conn

    def _ensure_schema(self) -> None:
        conn = self._get_conn()
        conn.executescript("""
            CREATE TABLE IF NOT EXISTS memory (
                id          INTEGER PRIMARY KEY AUTOINCREMENT,
                category    TEXT    NOT NULL,
                key         TEXT    NOT NULL,
                value       TEXT    NOT NULL,
                metadata    TEXT    NOT NULL DEFAULT '{}',
                created_at  REAL   NOT NULL,
                accessed_at REAL   NOT NULL
            );
            CREATE INDEX IF NOT EXISTS idx_memory_category
                ON memory(category);
            CREATE INDEX IF NOT EXISTS idx_memory_key
                ON memory(key);
            CREATE INDEX IF NOT EXISTS idx_memory_created
                ON memory(created_at);

            CREATE TABLE IF NOT EXISTS sessions (
                id          INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id  TEXT    NOT NULL UNIQUE,
                messages    TEXT    NOT NULL DEFAULT '[]',
                summary     TEXT    NOT NULL DEFAULT '',
                created_at  REAL   NOT NULL,
                updated_at  REAL   NOT NULL
            );
            CREATE INDEX IF NOT EXISTS idx_sessions_sid
                ON sessions(session_id);
        """)
        conn.commit()

    def close(self) -> None:
        if self._conn is not None:
            self._conn.close()
            self._conn = None

    # ------------------------------------------------------------------
    # Session persistence
    # ------------------------------------------------------------------

    def save_session(
        self,
        session_id: str,
        messages: list[dict[str, str]],
        summary: str = "",
    ) -> None:
        """Save or update a conversation session."""
        conn = self._get_conn()
        now = time.time()
        conn.execute(
            """
            INSERT INTO sessions (session_id, messages, summary, created_at, updated_at)
            VALUES (?, ?, ?, ?, ?)
            ON CONFLICT(session_id) DO UPDATE SET
                messages   = excluded.messages,
                summary    = excluded.summary,
                updated_at = excluded.updated_at
            """,
            (session_id, json.dumps(messages), summary, now, now),
        )
        conn.commit()

    def load_session(
        self, session_id: str
    ) -> dict[str, Any] | None:
        """Load a conversation session by ID."""
        conn = self._get_conn()
        row = conn.execute(
            "SELECT * FROM sessions WHERE session_id = ?",
            (session_id,),
        ).fetchone()
        if row is None:
            return None
        return {
            "session_id": row["session_id"],
            "messages": json.loads(row["messages"]),
            "summary": row["summary"],
            "created_at": row["created_at"],
            "updated_at": row["updated_at"],
        }

    def list_sessions(self, limit: int = 20) -> list[dict[str, Any]]:
        """List recent sessions ordered by last update."""
        conn = self._get_conn()
        rows = conn.execute(
            """
            SELECT session_id, summary, created_at, updated_at
            FROM sessions ORDER BY updated_at DESC LIMIT ?
            """,
            (limit,),
        ).fetchall()
        return [
            {
                "session_id": r["session_id"],
                "summary": r["summary"],
                "created_at": r["created_at"],
                "updated_at": r["updated_at"],
            }
            for r in rows
        ]

    def delete_session(self, session_id: str) -> bool:
        """Delete a session.  Returns True if it existed."""
        conn = self._get_conn()
        cur = conn.execute(
            "DELETE FROM sessions WHERE session_id = ?", (session_id,)
        )
        conn.commit()
        return cur.rowcount > 0

    # ------------------------------------------------------------------
    # General memory (patterns, preferences, notes)
    # ------------------------------------------------------------------

    def store(
        self,
        category: str,
        key: str,
        value: str,
        metadata: dict[str, Any] | None = None,
    ) -> int:
        """Store a memory entry.  Returns the row id."""
        conn = self._get_conn()
        now = time.time()
        cur = conn.execute(
            """
            INSERT INTO memory (category, key, value, metadata, created_at, accessed_at)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (category, key, value, json.dumps(metadata or {}), now, now),
        )
        conn.commit()
        return cur.lastrowid  # type: ignore[return-value]

    def recall(self, category: str, key: str) -> list[MemoryEntry]:
        """Recall memory entries matching category and key."""
        conn = self._get_conn()
        now = time.time()
        rows = conn.execute(
            """
            SELECT * FROM memory WHERE category = ? AND key = ?
            ORDER BY created_at DESC
            """,
            (category, key),
        ).fetchall()
        # Touch accessed_at
        ids = [r["id"] for r in rows]
        if ids:
            placeholders = ",".join("?" for _ in ids)
            conn.execute(
                f"UPDATE memory SET accessed_at = ? WHERE id IN ({placeholders})",
                [now, *ids],
            )
            conn.commit()
        return [self._row_to_entry(r) for r in rows]

    def search_memory(self, query: str, limit: int = 20) -> list[MemoryEntry]:
        """Search across all memory entries (key and value) using LIKE."""
        conn = self._get_conn()
        pattern = f"%{query}%"
        rows = conn.execute(
            """
            SELECT * FROM memory
            WHERE key LIKE ? OR value LIKE ?
            ORDER BY accessed_at DESC
            LIMIT ?
            """,
            (pattern, pattern, limit),
        ).fetchall()
        return [self._row_to_entry(r) for r in rows]

    def learn_pattern(self, pattern: str, details: str = "") -> int:
        """Store a learned pattern for future reference."""
        return self.store(
            category="pattern",
            key=pattern,
            value=details or pattern,
            metadata={"type": "learned_pattern"},
        )

    def get_patterns(self, limit: int = 50) -> list[MemoryEntry]:
        """Retrieve all learned patterns."""
        conn = self._get_conn()
        rows = conn.execute(
            """
            SELECT * FROM memory WHERE category = 'pattern'
            ORDER BY accessed_at DESC LIMIT ?
            """,
            (limit,),
        ).fetchall()
        return [self._row_to_entry(r) for r in rows]

    def store_preference(self, key: str, value: str) -> int:
        """Store or update a user preference."""
        conn = self._get_conn()
        # Remove old preference with same key
        conn.execute(
            "DELETE FROM memory WHERE category = 'preference' AND key = ?",
            (key,),
        )
        conn.commit()
        return self.store("preference", key, value)

    def get_preference(self, key: str) -> str | None:
        """Get a stored preference value."""
        entries = self.recall("preference", key)
        return entries[0].value if entries else None

    # ------------------------------------------------------------------
    # Stats
    # ------------------------------------------------------------------

    def stats(self) -> dict[str, int]:
        """Return memory statistics."""
        conn = self._get_conn()
        total = conn.execute("SELECT COUNT(*) FROM memory").fetchone()[0]
        sessions = conn.execute("SELECT COUNT(*) FROM sessions").fetchone()[0]
        patterns = conn.execute(
            "SELECT COUNT(*) FROM memory WHERE category = 'pattern'"
        ).fetchone()[0]
        preferences = conn.execute(
            "SELECT COUNT(*) FROM memory WHERE category = 'preference'"
        ).fetchone()[0]
        return {
            "total_memories": total,
            "sessions": sessions,
            "patterns": patterns,
            "preferences": preferences,
        }

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _row_to_entry(row: sqlite3.Row) -> MemoryEntry:
        return MemoryEntry(
            id=row["id"],
            category=row["category"],
            key=row["key"],
            value=row["value"],
            metadata=json.loads(row["metadata"]),
            created_at=row["created_at"],
            accessed_at=row["accessed_at"],
        )
