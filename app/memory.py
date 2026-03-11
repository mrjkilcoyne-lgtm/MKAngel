"""
Persistent memory system for MKAngel.

Uses SQLite for local-first storage — no server needed, works
everywhere including Termux.  Stores conversation history, learned
patterns, and user preferences between sessions.

Database lives at ~/.mkangel/memory.db
"""

from __future__ import annotations

import datetime
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

            CREATE TABLE IF NOT EXISTS dreams (
                id              INTEGER PRIMARY KEY AUTOINCREMENT,
                type            TEXT    NOT NULL,
                content         TEXT    NOT NULL,
                source          TEXT    NOT NULL DEFAULT '[]',
                surprise_score  REAL    NOT NULL DEFAULT 0.0,
                position_x      REAL    NOT NULL DEFAULT 0.0,
                position_y      REAL    NOT NULL DEFAULT 0.0,
                vestment_hints  TEXT    NOT NULL DEFAULT '{}',
                created_at      TEXT    NOT NULL,
                status          TEXT    NOT NULL DEFAULT 'new'
            );
            CREATE INDEX IF NOT EXISTS idx_dreams_status
                ON dreams(status);
            CREATE INDEX IF NOT EXISTS idx_dreams_created
                ON dreams(created_at);

            CREATE TABLE IF NOT EXISTS dream_sessions (
                id                      INTEGER PRIMARY KEY AUTOINCREMENT,
                started_at              TEXT    NOT NULL,
                ended_at                TEXT,
                trigger_type            TEXT    NOT NULL,
                artifacts_count         INTEGER NOT NULL DEFAULT 0,
                pattern_buffer_snapshot  TEXT    NOT NULL DEFAULT '{}'
            );
            CREATE INDEX IF NOT EXISTS idx_dream_sessions_started
                ON dream_sessions(started_at);
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
        dreams = conn.execute("SELECT COUNT(*) FROM dreams").fetchone()[0]
        dream_sessions_count = conn.execute(
            "SELECT COUNT(*) FROM dream_sessions"
        ).fetchone()[0]
        return {
            "total_memories": total,
            "sessions": sessions,
            "patterns": patterns,
            "preferences": preferences,
            "dreams": dreams,
            "dream_sessions": dream_sessions_count,
        }

    # ------------------------------------------------------------------
    # Dream persistence
    # ------------------------------------------------------------------

    def save_dream(
        self,
        dream_type: str,
        content: str,
        source: list[str] | None = None,
        surprise_score: float = 0.0,
        position_x: float = 0.0,
        position_y: float = 0.0,
        vestment_hints: dict[str, Any] | None = None,
        created_at: str | None = None,
        status: str = "new",
    ) -> int:
        """Save a dream artifact. Returns the row id."""
        conn = self._get_conn()
        if created_at is None:
            created_at = datetime.datetime.now(
                datetime.timezone.utc
            ).isoformat()
        cur = conn.execute(
            """
            INSERT INTO dreams
                (type, content, source, surprise_score,
                 position_x, position_y, vestment_hints,
                 created_at, status)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                dream_type,
                content,
                json.dumps(source or []),
                surprise_score,
                position_x,
                position_y,
                json.dumps(vestment_hints or {}),
                created_at,
                status,
            ),
        )
        conn.commit()
        return cur.lastrowid  # type: ignore[return-value]

    def get_recent_dreams(self, limit: int = 20) -> list[dict[str, Any]]:
        """Return the most recent dreams ordered by created_at DESC."""
        conn = self._get_conn()
        rows = conn.execute(
            "SELECT * FROM dreams ORDER BY created_at DESC LIMIT ?",
            (limit,),
        ).fetchall()
        return [self._dream_row_to_dict(r) for r in rows]

    def get_unseen_dreams(self) -> list[dict[str, Any]]:
        """Return all dreams with status='new'."""
        conn = self._get_conn()
        rows = conn.execute(
            "SELECT * FROM dreams WHERE status = 'new' ORDER BY created_at DESC"
        ).fetchall()
        return [self._dream_row_to_dict(r) for r in rows]

    def archive_dream(self, dream_id: int) -> bool:
        """Set dream status to 'archived'. Returns True if row existed."""
        conn = self._get_conn()
        cur = conn.execute(
            "UPDATE dreams SET status = 'archived' WHERE id = ?",
            (dream_id,),
        )
        conn.commit()
        return cur.rowcount > 0

    def accept_patch(self, dream_id: int) -> dict[str, Any] | None:
        """Accept a self_patch dream: set status to 'accepted', return content.

        Only works on dreams with type='self_patch' and status in ('new', 'seen').
        Returns the dream dict on success, None on failure.
        """
        conn = self._get_conn()
        row = conn.execute(
            """
            SELECT * FROM dreams
            WHERE id = ? AND type = 'self_patch' AND status IN ('new', 'seen')
            """,
            (dream_id,),
        ).fetchone()
        if row is None:
            return None
        conn.execute(
            "UPDATE dreams SET status = 'accepted' WHERE id = ?",
            (dream_id,),
        )
        conn.commit()
        result = self._dream_row_to_dict(row)
        result["status"] = "accepted"
        return result

    def mark_dreams_seen(self, dream_ids: list[int]) -> int:
        """Set status='seen' for the given IDs. Returns count updated."""
        if not dream_ids:
            return 0
        conn = self._get_conn()
        placeholders = ",".join("?" for _ in dream_ids)
        cur = conn.execute(
            f"UPDATE dreams SET status = 'seen' WHERE id IN ({placeholders})",
            dream_ids,
        )
        conn.commit()
        return cur.rowcount

    def start_dream_session(
        self,
        trigger_type: str,
        pattern_buffer_snapshot: dict[str, Any] | None = None,
    ) -> int:
        """Create a dream_sessions row. Returns the row id."""
        conn = self._get_conn()
        now = datetime.datetime.now(datetime.timezone.utc).isoformat()
        cur = conn.execute(
            """
            INSERT INTO dream_sessions
                (started_at, trigger_type, pattern_buffer_snapshot)
            VALUES (?, ?, ?)
            """,
            (now, trigger_type, json.dumps(pattern_buffer_snapshot or {})),
        )
        conn.commit()
        return cur.lastrowid  # type: ignore[return-value]

    def end_dream_session(
        self,
        session_id: int,
        artifacts_count: int,
    ) -> None:
        """Set ended_at and artifacts_count for a dream session."""
        conn = self._get_conn()
        now = datetime.datetime.now(datetime.timezone.utc).isoformat()
        conn.execute(
            """
            UPDATE dream_sessions
            SET ended_at = ?, artifacts_count = ?
            WHERE id = ?
            """,
            (now, artifacts_count, session_id),
        )
        conn.commit()

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

    @staticmethod
    def _dream_row_to_dict(row: sqlite3.Row) -> dict[str, Any]:
        return {
            "id": row["id"],
            "type": row["type"],
            "content": row["content"],
            "source": json.loads(row["source"]),
            "surprise_score": row["surprise_score"],
            "position_x": row["position_x"],
            "position_y": row["position_y"],
            "vestment_hints": json.loads(row["vestment_hints"]),
            "created_at": row["created_at"],
            "status": row["status"],
        }
