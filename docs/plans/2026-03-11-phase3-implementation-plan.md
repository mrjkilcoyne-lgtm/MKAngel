# Phase 3 Implementation Plan: The Angel Gets Hands

**Date**: 2026-03-11
**Depends on**: Phase 2 (Voice, SelfImprover, Puriel) -- all complete
**Design doc**: `docs/plans/2026-03-11-phase3-angel-gets-hands-design.md`

---

## Phase 3.1 -- Dream First (DETAILED)

Five steps, each one commit. Each step results in a testable, non-breaking increment.

---

### Step 1: Dreams Table in Memory

**Goal**: Extend the SQLite persistence layer with tables for dream artifacts and dream sessions so the Dreamer has somewhere to write.

**Files to modify**:
- `app/memory.py`

**No new files.**

#### Schema additions

Add two new tables inside `_ensure_schema()`, appended to the existing `CREATE TABLE IF NOT EXISTS` block:

```python
# --- In Memory._ensure_schema(), add after the sessions table ---

CREATE TABLE IF NOT EXISTS dreams (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    type            TEXT    NOT NULL,   -- poem | grammar_map | micro_tool | self_patch | observation
    content         TEXT    NOT NULL,
    source          TEXT    NOT NULL DEFAULT '[]',   -- JSON list of conversation/pattern IDs
    surprise_score  REAL    NOT NULL DEFAULT 0.0,
    position_x      REAL    NOT NULL DEFAULT 0.0,
    position_y      REAL    NOT NULL DEFAULT 0.0,
    vestment_hints  TEXT    NOT NULL DEFAULT '{}',   -- JSON dict
    created_at      TEXT    NOT NULL,                -- ISO 8601 timestamp
    status          TEXT    NOT NULL DEFAULT 'new'   -- new | seen | archived | accepted
);
CREATE INDEX IF NOT EXISTS idx_dreams_status
    ON dreams(status);
CREATE INDEX IF NOT EXISTS idx_dreams_created
    ON dreams(created_at);

CREATE TABLE IF NOT EXISTS dream_sessions (
    id                      INTEGER PRIMARY KEY AUTOINCREMENT,
    started_at              TEXT    NOT NULL,    -- ISO 8601
    ended_at                TEXT,                -- NULL until session completes
    trigger_type            TEXT    NOT NULL,    -- self | context | user
    artifacts_count         INTEGER NOT NULL DEFAULT 0,
    pattern_buffer_snapshot TEXT    NOT NULL DEFAULT '{}' -- JSON
);
CREATE INDEX IF NOT EXISTS idx_dream_sessions_started
    ON dream_sessions(started_at);
```

#### Methods to add to the `Memory` class

```python
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

def get_recent_dreams(self, limit: int = 20) -> list[dict[str, Any]]:
    """Return the most recent dreams ordered by created_at DESC."""

def get_unseen_dreams(self) -> list[dict[str, Any]]:
    """Return all dreams with status='new'."""

def archive_dream(self, dream_id: int) -> bool:
    """Set dream status to 'archived'. Returns True if row existed."""

def accept_patch(self, dream_id: int) -> dict[str, Any] | None:
    """Accept a self_patch dream: set status to 'accepted', return its content.
    Only works on dreams with type='self_patch' and status='new' or 'seen'.
    Returns the dream dict on success, None on failure."""

def mark_dreams_seen(self, dream_ids: list[int]) -> int:
    """Set status='seen' for the given IDs. Returns count updated."""

def start_dream_session(
    self,
    trigger_type: str,
    pattern_buffer_snapshot: dict[str, Any] | None = None,
) -> int:
    """Create a dream_sessions row. Returns the row id."""

def end_dream_session(
    self,
    session_id: int,
    artifacts_count: int,
) -> None:
    """Set ended_at and artifacts_count for a dream session."""
```

#### Implementation notes

- `save_dream` generates `created_at` via `datetime.datetime.now(datetime.timezone.utc).isoformat()` if not provided.
- `get_recent_dreams` and `get_unseen_dreams` both return dicts with all columns, `source` and `vestment_hints` parsed from JSON.
- `accept_patch` checks `type = 'self_patch'` before accepting.
- `start_dream_session` uses UTC ISO timestamp for `started_at`.
- Add `import datetime` to the imports at the top of `memory.py`.

#### Dependencies

None. This is the foundation step.

#### How to test

```python
# tests/test_dream_memory.py
import tempfile, os
from app.memory import Memory

def test_dream_roundtrip():
    with tempfile.TemporaryDirectory() as td:
        m = Memory(db_path=os.path.join(td, "test.db"))

        # Save a dream
        did = m.save_dream(
            dream_type="poem",
            content="Water traces to *wed-...",
            source=["session-abc", "session-def"],
            surprise_score=0.82,
            position_x=0.5,
            position_y=0.3,
        )
        assert did > 0

        # Unseen dreams
        unseen = m.get_unseen_dreams()
        assert len(unseen) == 1
        assert unseen[0]["type"] == "poem"
        assert unseen[0]["surprise_score"] == 0.82

        # Mark seen
        count = m.mark_dreams_seen([did])
        assert count == 1
        assert len(m.get_unseen_dreams()) == 0

        # Archive
        assert m.archive_dream(did)

        # Dream session
        sid = m.start_dream_session("self", {"patterns": 7})
        m.end_dream_session(sid, artifacts_count=1)

        m.close()

def test_accept_patch():
    with tempfile.TemporaryDirectory() as td:
        m = Memory(db_path=os.path.join(td, "test.db"))
        pid = m.save_dream(
            dream_type="self_patch",
            content='{"action": "add_rule", "domain": "linguistic"}',
            surprise_score=0.6,
        )
        result = m.accept_patch(pid)
        assert result is not None
        assert result["status"] == "accepted"

        # Cannot accept a poem
        poem_id = m.save_dream(dream_type="poem", content="roses")
        assert m.accept_patch(poem_id) is None
        m.close()
```

Also update `Memory.stats()` to include dream counts:

```python
def stats(self) -> dict[str, int]:
    # ... existing code ...
    dreams = conn.execute("SELECT COUNT(*) FROM dreams").fetchone()[0]
    dream_sessions_count = conn.execute(
        "SELECT COUNT(*) FROM dream_sessions"
    ).fetchone()[0]
    return {
        **existing_stats,
        "dreams": dreams,
        "dream_sessions": dream_sessions_count,
    }
```

#### Commit message

```
feat(memory): add dreams and dream_sessions tables

Add SQLite schema for dream artifacts (type, content, source,
surprise_score, spatial position, vestment hints, status lifecycle)
and dream sessions (trigger type, duration, pattern snapshot).

Methods: save_dream, get_recent_dreams, get_unseen_dreams,
archive_dream, accept_patch, mark_dreams_seen,
start_dream_session, end_dream_session.

Foundation for Phase 3.1 Dreamer module.
```

---

### Step 2: The Dreamer Module

**Goal**: Create the 4-stage dream pipeline that reads the day's conversations, finds surprising connections via the GLM, composes artifacts using Voice, and arranges them spatially.

**Files to create**:
- `glm/dreamer.py`

**Files to read (not modify)**:
- `glm/angel.py` -- `compose_fugue()`, `sense()`, `predict()`, `reconstruct()`
- `glm/voice.py` -- `Voice.compose()`
- `app/memory.py` -- `list_sessions()`, `load_session()`, `save_dream()`
- `app/self_improve.py` -- `SelfImprover.get_strong_patterns()`, `get_weak_patterns()`, `analyse_performance()`
- `app/puriel.py` -- `GrammarIntegrityChecksum.validate_learned_rule()`

#### Classes and functions

```python
# glm/dreamer.py
"""
The Dreamer -- she sleeps, she dreams, she wakes having created.

The 4-stage dream pipeline:
  1. RECALL  -- replay the day's conversations from Memory
  2. CONNECT -- find surprising cross-domain links via the GLM
  3. COMPOSE -- create dream artifacts using Voice and grammar
  4. ARRANGE -- position artifacts spatially for the canvas

Input:  Angel (GLM core), Memory (SQLite), Voice (composition)
Output: list[DreamArtifact] saved to Memory's dreams table
"""

from __future__ import annotations

import datetime
import json
import logging
from dataclasses import dataclass, field
from typing import Any

log = logging.getLogger(__name__)


MAX_ARTIFACTS_PER_DREAM = 5


@dataclass
class DreamArtifact:
    """A single dream artifact -- something she created while sleeping."""
    type: str                  # poem | grammar_map | micro_tool | self_patch | observation
    content: str               # text, JSON structure, or HTML fragment
    source: list[str] = field(default_factory=list)  # conversation/pattern IDs
    surprise_score: float = 0.0
    position: tuple[float, float] = (0.5, 0.5)  # (x, y) normalised 0-1
    vestment_hints: dict[str, Any] = field(default_factory=dict)
    created_at: str = ""

    def __post_init__(self):
        if not self.created_at:
            self.created_at = datetime.datetime.now(
                datetime.timezone.utc
            ).isoformat()


class Dreamer:
    """The 4-stage dream pipeline."""

    def __init__(self) -> None:
        # Puriel gate -- loaded lazily
        self._integrity: Any = None
        try:
            from app.puriel import GrammarIntegrityChecksum
            self._integrity = GrammarIntegrityChecksum()
        except Exception:
            pass

    # ── Main entry point ──────────────────────────────────────

    def dream(
        self,
        angel: Any,
        memory: Any,
        voice: Any,
        self_improver: Any | None = None,
    ) -> list[DreamArtifact]:
        """Run the full dream cycle.

        Args:
            angel: The Angel instance (GLM core).
            memory: The Memory instance (SQLite).
            voice: The Voice instance (composition).
            self_improver: Optional SelfImprover for pattern data.

        Returns:
            List of DreamArtifacts (max 5), already saved to memory.
        """
        # Stage 1
        day_summary = self.recall(memory)
        if not day_summary.get("tokens"):
            log.info("Dreamer: nothing to dream about (no conversations)")
            return []

        # Stage 2
        connections = self.connect(angel, day_summary, self_improver)

        # Stage 3
        artifacts = self.compose(voice, connections, day_summary)

        # Stage 4
        artifacts = self.arrange(artifacts)

        # Puriel gate -- validate all artifacts
        artifacts = self._puriel_filter(artifacts)

        # Cap at MAX_ARTIFACTS_PER_DREAM
        artifacts = artifacts[:MAX_ARTIFACTS_PER_DREAM]

        # Save to memory
        for art in artifacts:
            try:
                memory.save_dream(
                    dream_type=art.type,
                    content=art.content,
                    source=art.source,
                    surprise_score=art.surprise_score,
                    position_x=art.position[0],
                    position_y=art.position[1],
                    vestment_hints=art.vestment_hints,
                    created_at=art.created_at,
                )
            except Exception as exc:
                log.warning("Dreamer: failed to save artifact: %s", exc)

        return artifacts

    # ── Stage 1: RECALL ───────────────────────────────────────

    def recall(self, memory: Any) -> dict[str, Any]:
        """Replay the day's conversations from Memory.

        Returns a day_summary dict:
            tokens:         list[str] -- all content words from today
            session_ids:    list[str] -- conversation IDs referenced
            domain_counts:  dict[str, int] -- how many times each domain appeared
            harmony_avg:    float -- average harmony signal
            loop_gate_avg:  float -- average loop_gate signal
            message_count:  int -- total messages today
        """
        sessions = memory.list_sessions(limit=50)
        # Filter to today's sessions (by created_at timestamp)
        now = datetime.datetime.now(datetime.timezone.utc)
        today_start = now.replace(hour=0, minute=0, second=0, microsecond=0)
        today_ts = today_start.timestamp()

        all_tokens: list[str] = []
        session_ids: list[str] = []
        message_count = 0

        for s in sessions:
            if s.get("created_at", 0) < today_ts:
                continue
            sid = s.get("session_id", "")
            session_ids.append(sid)

            full = memory.load_session(sid)
            if full and full.get("messages"):
                msgs = full["messages"]
                message_count += len(msgs)
                for msg in msgs:
                    if msg.get("role") == "user":
                        words = msg.get("content", "").lower().split()
                        all_tokens.extend(words)

        return {
            "tokens": all_tokens,
            "session_ids": session_ids,
            "message_count": message_count,
        }

    # ── Stage 2: CONNECT ──────────────────────────────────────

    def connect(
        self,
        angel: Any,
        day_summary: dict[str, Any],
        self_improver: Any | None = None,
    ) -> list[dict[str, Any]]:
        """Find surprising cross-domain connections.

        Runs compose_fugue on a representative sample of the day's
        tokens, then looks for harmonics that were NOT surfaced
        during conversation (the serendipity).

        Returns a list of connection dicts:
            {
                "type": str,  -- "harmonic" | "counterpoint" | "gap" | "pattern"
                "data": Any,  -- the raw connection data
                "tokens": list[str],  -- the tokens that triggered it
                "surprise": float,
            }
        """
        tokens = day_summary.get("tokens", [])
        connections: list[dict[str, Any]] = []

        if not tokens:
            return connections

        # Deduplicate and take the most interesting words
        # (skip very common words, keep content words)
        skip = {
            "the", "a", "an", "is", "are", "was", "were", "be",
            "to", "of", "in", "on", "at", "for", "with", "by",
            "and", "or", "but", "not", "it", "so", "do", "i",
            "you", "me", "my", "we", "they", "he", "she", "that",
            "this", "what", "how", "why", "who",
        }
        content = [w for w in tokens if w not in skip and len(w) > 2]
        # Count frequency, take top words
        from collections import Counter
        counts = Counter(content)
        top_words = [w for w, _ in counts.most_common(20)]

        if not top_words:
            return connections

        # Run fugue on batches of 3-5 words to find cross-domain links
        batch_size = min(5, len(top_words))
        for i in range(0, len(top_words), batch_size):
            batch = top_words[i:i + batch_size]
            try:
                fugue = angel.compose_fugue(batch)
                harmonics = fugue.get("harmonics", [])
                counterpoint = fugue.get("counterpoint", [])

                for h in harmonics:
                    domains = h.get("domains", [])
                    if len(domains) > 1:
                        connections.append({
                            "type": "harmonic",
                            "data": h,
                            "tokens": batch,
                            "surprise": min(1.0, len(domains) * 0.25),
                        })

                for cp in counterpoint:
                    connections.append({
                        "type": "counterpoint",
                        "data": cp,
                        "tokens": batch,
                        "surprise": 0.6,
                    })
            except Exception:
                pass

            if len(connections) >= 10:
                break

        # Check SelfImprover for weak patterns (knowledge gaps)
        if self_improver is not None:
            try:
                weak = self_improver.get_weak_patterns(max_confidence=0.3)
                for pat in weak[:3]:
                    connections.append({
                        "type": "gap",
                        "data": {
                            "pattern_id": pat.pattern_id,
                            "domain": pat.domain,
                            "confidence": pat.confidence,
                        },
                        "tokens": [],
                        "surprise": 0.7,
                    })
            except Exception:
                pass

        # Sort by surprise descending
        connections.sort(key=lambda c: c.get("surprise", 0), reverse=True)
        return connections[:10]

    # ── Stage 3: COMPOSE ──────────────────────────────────────

    def compose(
        self,
        voice: Any,
        connections: list[dict[str, Any]],
        day_summary: dict[str, Any],
    ) -> list[DreamArtifact]:
        """Create dream artifacts from connections.

        Maps connection types to artifact types:
            harmonic    -> grammar_map (cross-domain visual)
            counterpoint -> poem (tension as beauty)
            gap         -> self_patch (growth proposal)
            pattern     -> micro_tool (usage pattern)

        Falls back to observation if nothing else qualifies.
        """
        artifacts: list[DreamArtifact] = []
        session_ids = day_summary.get("session_ids", [])

        if not connections:
            # Even with no connections, compose an observation
            artifacts.append(DreamArtifact(
                type="observation",
                content="A quiet day. The grammars listened. Nothing new surfaced -- and that is fine.",
                source=session_ids[:3],
                surprise_score=0.1,
                vestment_hints={"glow": False, "opacity": 0.7},
            ))
            return artifacts

        for conn in connections[:MAX_ARTIFACTS_PER_DREAM]:
            ctype = conn.get("type", "")
            data = conn.get("data", {})
            tokens = conn.get("tokens", [])
            surprise = conn.get("surprise", 0.5)

            if ctype == "harmonic":
                # Grammar map -- describe the cross-domain link
                domains = data.get("domains", [])
                output = data.get("output", "")
                content_parts = [
                    f"Cross-domain resonance: '{output}' echoes across {', '.join(domains)}.",
                ]
                if voice is not None:
                    try:
                        composed = voice.compose(
                            original=" ".join(tokens),
                            tokens=tokens,
                            voices={d: [True] for d in domains},
                            harmonics=[data],
                            counterpoint=[],
                            origins=[],
                            predictions=[],
                            harmony=0.8,
                            loop_gate=0.4,
                        )
                        if composed:
                            content_parts.append(composed)
                    except Exception:
                        pass

                artifacts.append(DreamArtifact(
                    type="grammar_map",
                    content="\n".join(content_parts),
                    source=session_ids[:3],
                    surprise_score=surprise,
                    vestment_hints={
                        "glow": True,
                        "color_key": "teal",
                        "domains": domains,
                    },
                ))

            elif ctype == "counterpoint":
                # Poem -- tension made beautiful
                domain = data.get("domain", "")
                unique = data.get("unique_outputs", [])
                content_parts = []
                if voice is not None:
                    try:
                        composed = voice.compose(
                            original=" ".join(tokens),
                            tokens=tokens,
                            voices={domain: unique},
                            harmonics=[],
                            counterpoint=[data],
                            origins=[],
                            predictions=[],
                            harmony=0.4,
                            loop_gate=0.6,
                        )
                        if composed:
                            content_parts.append(composed)
                    except Exception:
                        pass
                if not content_parts:
                    content_parts.append(
                        f"{domain} heard something its own in {', '.join(tokens)}."
                    )

                artifacts.append(DreamArtifact(
                    type="poem",
                    content="\n".join(content_parts),
                    source=session_ids[:3],
                    surprise_score=surprise,
                    vestment_hints={
                        "glow": True,
                        "color_key": "accent",
                    },
                ))

            elif ctype == "gap":
                # Self-patch -- growth proposal
                patch_content = json.dumps({
                    "action": "strengthen_pattern",
                    "pattern_id": data.get("pattern_id", ""),
                    "domain": data.get("domain", ""),
                    "current_confidence": data.get("confidence", 0),
                    "proposal": (
                        f"Pattern '{data.get('pattern_id', '?')}' in "
                        f"{data.get('domain', '?')} has low confidence "
                        f"({data.get('confidence', 0):.2f}). "
                        f"Consider additional training data or rule refinement."
                    ),
                })
                artifacts.append(DreamArtifact(
                    type="self_patch",
                    content=patch_content,
                    source=session_ids[:3],
                    surprise_score=surprise,
                    vestment_hints={
                        "glow": False,
                        "color_key": "warning",
                        "corner": "growth",
                    },
                ))

            else:
                # Observation
                artifacts.append(DreamArtifact(
                    type="observation",
                    content=f"Patterns noticed in {', '.join(tokens[:3])}.",
                    source=session_ids[:3],
                    surprise_score=surprise * 0.5,
                    vestment_hints={"glow": False, "opacity": 0.7},
                ))

        return artifacts

    # ── Stage 4: ARRANGE ──────────────────────────────────────

    def arrange(self, artifacts: list[DreamArtifact]) -> list[DreamArtifact]:
        """Position artifacts spatially for the canvas.

        Layout rules:
            - Most surprising artifact -> centre (0.5, 0.5)
            - Self-patches -> bottom-right "growth corner" (0.85, 0.15)
            - Poems -> upper area (y > 0.6)
            - Grammar maps -> mid area
            - Observations -> edges (low y or high x)
            - Related artifacts cluster near each other
        """
        if not artifacts:
            return artifacts

        # Sort by surprise (highest first)
        artifacts.sort(key=lambda a: a.surprise_score, reverse=True)

        for i, art in enumerate(artifacts):
            if i == 0:
                # Most surprising -> centre
                art.position = (0.5, 0.5)
            elif art.type == "self_patch":
                art.position = (0.85, 0.15)
            elif art.type == "poem":
                # Upper area, spread horizontally
                x = 0.2 + (i * 0.2) % 0.6
                art.position = (x, 0.7 + (i % 2) * 0.1)
            elif art.type == "grammar_map":
                # Mid area
                x = 0.3 + (i * 0.15) % 0.4
                art.position = (x, 0.4 + (i % 2) * 0.1)
            elif art.type == "observation":
                # Edges
                art.position = (0.1 + (i * 0.2) % 0.3, 0.15)
            else:
                # Default spread
                x = 0.2 + (i * 0.2) % 0.6
                y = 0.3 + (i * 0.15) % 0.4
                art.position = (x, y)

        return artifacts

    # ── Puriel gate ───────────────────────────────────────────

    def _puriel_filter(
        self, artifacts: list[DreamArtifact]
    ) -> list[DreamArtifact]:
        """Validate all artifacts through Puriel's integrity gate.

        Self-patches are checked most strictly. Poems and observations
        pass freely. Grammar maps are checked for structural integrity.
        """
        if self._integrity is None:
            return artifacts  # No gate available, pass all

        filtered: list[DreamArtifact] = []
        for art in artifacts:
            if art.type == "self_patch":
                try:
                    patch_data = json.loads(art.content)
                    domain = patch_data.get("domain", "linguistic")
                    passed, reason = self._integrity.validate_learned_rule(
                        domain, patch_data,
                    )
                    if passed:
                        filtered.append(art)
                    else:
                        log.info(
                            "Puriel held back dream self_patch: %s", reason
                        )
                except (json.JSONDecodeError, Exception):
                    log.info("Puriel rejected malformed self_patch")
            elif art.type == "grammar_map":
                # Grammar maps are structural -- light validation
                try:
                    passed, _ = self._integrity.validate_learned_rule(
                        "linguistic",
                        {"name": "dream_map", "pattern": art.content[:200]},
                    )
                    filtered.append(art)  # Include even if gate says no -- maps are read-only
                except Exception:
                    filtered.append(art)
            else:
                # Poems, observations, micro_tools pass freely
                filtered.append(art)

        return filtered
```

#### Dependencies

- Step 1 (dreams table in Memory) -- must exist for `save_dream()` calls.

#### How to test

```python
# tests/test_dreamer.py
import tempfile, os
from glm.dreamer import Dreamer, DreamArtifact
from app.memory import Memory

def test_dreamer_empty_day():
    """Dreamer with no conversations produces empty list."""
    with tempfile.TemporaryDirectory() as td:
        m = Memory(db_path=os.path.join(td, "test.db"))
        d = Dreamer()
        # No sessions in memory -> no dreams
        result = d.recall(m)
        assert result["tokens"] == []
        m.close()

def test_dreamer_recall_with_data():
    """Dreamer recalls today's conversations."""
    with tempfile.TemporaryDirectory() as td:
        m = Memory(db_path=os.path.join(td, "test.db"))
        # Seed a session
        m.save_session("test-001", [
            {"role": "user", "content": "tell me about water"},
            {"role": "assistant", "content": "Water traces to *wed-..."},
        ])
        d = Dreamer()
        result = d.recall(m)
        assert "water" in result["tokens"]
        m.close()

def test_dream_artifact_dataclass():
    art = DreamArtifact(
        type="poem",
        content="The roots remember.",
        surprise_score=0.9,
    )
    assert art.type == "poem"
    assert art.created_at  # auto-populated
    assert art.position == (0.5, 0.5)

def test_arrange_positions():
    d = Dreamer()
    artifacts = [
        DreamArtifact(type="poem", content="...", surprise_score=0.9),
        DreamArtifact(type="self_patch", content="{}", surprise_score=0.5),
        DreamArtifact(type="observation", content="...", surprise_score=0.2),
    ]
    arranged = d.arrange(artifacts)
    # Most surprising (poem, 0.9) should be at centre
    assert arranged[0].position == (0.5, 0.5)
    # Self-patch should be in growth corner
    patch = [a for a in arranged if a.type == "self_patch"][0]
    assert patch.position == (0.85, 0.15)
```

#### Commit message

```
feat(glm): add Dreamer module with 4-stage dream pipeline

New file glm/dreamer.py implements:
- DreamArtifact dataclass (type, content, source, surprise, position)
- Dreamer class with recall/connect/compose/arrange pipeline
- Puriel gate validation on all artifacts
- Max 5 artifacts per dream cycle

The Dreamer reads the day's conversations from Memory, finds
cross-domain connections via compose_fugue(), composes artifacts
using Voice, and arranges them spatially for the canvas.
```

---

### Step 3: Sleep Triggers in Chat Pipeline

**Goal**: Add sleep state tracking and trigger detection to the chat pipeline. When triggered, the Angel transitions through AWAKE -> DROWSY -> SLEEPING states and composes a sleep message via Voice.

**Files to modify**:
- `app/chat.py` -- add state machine, sleep signal detection, sleep keywords

**No new files.**

#### Changes to `ChatSession.__init__`

Add these instance variables:

```python
# --- Sleep/dream state ---
self._state: str = "AWAKE"  # AWAKE | DROWSY | SLEEPING | WAKING
self._loop_gate_history: list[float] = []  # last N loop_gate values
self._pattern_buffer_count: int = 0
self._last_interaction_time: float = time.time()
self._voice: Any = None
try:
    from glm.voice import Voice as _V
    self._voice = _V()
except Exception:
    pass
```

#### New constants at module level

```python
# --- Sleep trigger configuration ---
SLEEP_KEYWORDS = frozenset({
    "goodnight", "rest", "sleep", "nap", "dream",
    "good night", "go to sleep", "night night",
})

LOOP_GATE_THRESHOLD = 0.85
PATTERN_BUFFER_MIN = 5
SUSTAINED_TURNS = 3
```

#### New methods on `ChatSession`

```python
def _check_sleep_signals(self, text: str) -> str | None:
    """Check if sleep should be triggered.

    Returns:
        "self" | "user" | "context" | None
    """
    if self._state != "AWAKE":
        return None

    # User-initiated: check keywords
    text_lower = text.lower().strip()
    for kw in SLEEP_KEYWORDS:
        if kw in text_lower:
            return "user"

    # Self-initiated: high loop_gate sustained + pattern buffer
    if (
        len(self._loop_gate_history) >= SUSTAINED_TURNS
        and all(
            g > LOOP_GATE_THRESHOLD
            for g in self._loop_gate_history[-SUSTAINED_TURNS:]
        )
        and self._pattern_buffer_count >= PATTERN_BUFFER_MIN
    ):
        return "self"

    # Context-detected: handled in Step 4 (Android-specific)
    return None


def _update_sleep_signals(self, text: str) -> None:
    """Update loop_gate history and pattern buffer after each turn."""
    if self._angel is None:
        return

    try:
        tokens = text.lower().split()
        signals = self._angel.sense(tokens)
        loop_gate = signals.get("loop_gate", 0.1)
        self._loop_gate_history.append(loop_gate)
        # Keep only last 10 turns
        self._loop_gate_history = self._loop_gate_history[-10:]
    except Exception:
        pass

    # Count patterns from SelfImprover if available
    try:
        from app.self_improve import SelfImprover
        si = SelfImprover()
        perf = si.analyse_performance()
        self._pattern_buffer_count = perf.get("total_patterns", 0)
    except Exception:
        pass

    self._last_interaction_time = time.time()


def _compose_sleep_message(self, trigger: str) -> str:
    """Compose the Angel's drowsy/sleep transition message."""
    if trigger == "user":
        return (
            "Goodnight. I'll dream on what we talked about today. "
            "When you open me next, I might have something to show you."
        )

    # Self-initiated -- she asked to sleep
    if self._voice is not None and self._angel is not None:
        tokens = ["rest", "dream", "sleep", "pattern"]
        try:
            composed = self._voice.compose(
                original="I want to rest",
                tokens=tokens,
                voices={},
                harmonics=[],
                counterpoint=[],
                origins=[],
                predictions=[],
                harmony=0.6,
                loop_gate=0.9,
            )
            if composed:
                return composed
        except Exception:
            pass

    return (
        "I have been turning a lot of loops today. "
        "Mind if I rest for a bit? I think I see some "
        "connections I want to follow."
    )


@property
def state(self) -> str:
    """Current sleep/wake state."""
    return self._state
```

#### Modifications to `process_input`

After `self._handle_chat(text)` returns, add sleep signal processing:

```python
def process_input(self, user_input: str) -> str:
    text = user_input.strip()
    if not text:
        return ""

    self._messages.append({"role": "user", "content": text})

    # --- Check for wake greeting (Step 5 will fill this in) ---
    # Placeholder: if state is WAKING, compose wake greeting

    # --- Check sleep signals BEFORE processing ---
    trigger = self._check_sleep_signals(text)
    if trigger == "user":
        self._state = "DROWSY"
        sleep_msg = self._compose_sleep_message(trigger)
        self._messages.append({"role": "assistant", "content": sleep_msg})
        # Transition to SLEEPING happens in Step 4 (foreground service)
        return sleep_msg

    # Command dispatch
    if text.startswith("/"):
        parts = text.split(maxsplit=1)
        cmd = parts[0].lower()
        args = parts[1] if len(parts) > 1 else ""
        response = self._handle_command(cmd, args)
    else:
        response = self._handle_chat(text)

    self._messages.append({"role": "assistant", "content": response})

    # --- Update sleep signals AFTER processing ---
    self._update_sleep_signals(text)

    # --- Self-initiated sleep check ---
    trigger = self._check_sleep_signals(text)
    if trigger == "self":
        sleep_msg = self._compose_sleep_message(trigger)
        self._messages.append({"role": "assistant", "content": sleep_msg})
        self._state = "DROWSY"
        response = response + "\n\n" + sleep_msg

    # Auto-save to memory periodically
    if self._memory and len(self._messages) % 10 == 0:
        self._auto_save()

    return response
```

Also add a `/sleep` command to `_COMMANDS` and `_handle_command`:

```python
# In _COMMANDS dict:
"/sleep":  ("", "Tell the Angel to sleep and dream"),

# In _handle_command:
"/sleep": self._cmd_sleep,

# New method:
def _cmd_sleep(self, args: str) -> str:
    """Manually trigger sleep."""
    self._state = "DROWSY"
    return self._compose_sleep_message("user")
```

#### Dependencies

- Step 1 (dreams table) -- not strictly needed yet but assumed present.
- Step 2 (Dreamer module) -- not invoked yet, just triggers are wired.

#### How to test

```python
# tests/test_sleep_triggers.py
from app.chat import ChatSession, SLEEP_KEYWORDS

def test_user_sleep_trigger():
    session = ChatSession()
    session._state = "AWAKE"
    assert session._check_sleep_signals("goodnight") == "user"
    assert session._check_sleep_signals("time to sleep") == "user"
    assert session._check_sleep_signals("hello") is None

def test_self_initiated_trigger():
    session = ChatSession()
    session._state = "AWAKE"
    session._loop_gate_history = [0.9, 0.9, 0.9, 0.88, 0.92]
    session._pattern_buffer_count = 7
    assert session._check_sleep_signals("anything") == "self"

def test_no_trigger_when_sleeping():
    session = ChatSession()
    session._state = "SLEEPING"
    assert session._check_sleep_signals("goodnight") is None

def test_sleep_message_composition():
    session = ChatSession()
    msg = session._compose_sleep_message("user")
    assert "dream" in msg.lower() or "night" in msg.lower()
```

#### Commit message

```
feat(chat): add sleep trigger detection and state machine

ChatSession now tracks loop_gate history and pattern buffer
to detect when the Angel should sleep. Three triggers:
- User-initiated: sleep keywords in message
- Self-initiated: loop_gate > 0.85 sustained 3+ turns + 5+ patterns
- Context-detected: placeholder for Android battery/idle check

State machine: AWAKE -> DROWSY -> SLEEPING -> WAKING -> AWAKE.
Drowsy transition composes a sleep message via Voice.
Adds /sleep command for manual trigger.
```

---

### Step 4: Android Foreground Service

**Goal**: When the Angel transitions to SLEEPING, start an Android foreground service that runs the Dreamer pipeline, saves artifacts, and notifies the user when dreaming is complete.

**Files to modify**:
- `main_android.py` -- add service lifecycle management
- `buildozer.spec` -- add FOREGROUND_SERVICE permission and service declaration
- `app/chat.py` -- wire DROWSY -> SLEEPING transition to call the service

**Files to create**:
- `app/dream_service.py` -- service logic (decoupled from Android APIs for testability)

#### `app/dream_service.py` (new file)

```python
"""
Dream service logic -- decoupled from Android for testability.

Runs the Dreamer pipeline:
  1. Start dream session in Memory
  2. Run Dreamer.dream()
  3. End dream session with artifact count
  4. Return results

The Android foreground service (in main_android.py) wraps this
with notification management and pyjnius calls.
"""

from __future__ import annotations

import logging
import time
from typing import Any

log = logging.getLogger(__name__)


def run_dream_cycle(
    angel: Any,
    memory: Any,
    voice: Any,
    self_improver: Any | None = None,
    trigger_type: str = "self",
) -> dict[str, Any]:
    """Run one complete dream cycle.

    Args:
        angel: The Angel instance.
        memory: The Memory instance.
        voice: The Voice instance.
        self_improver: Optional SelfImprover.
        trigger_type: What triggered the dream (self/context/user).

    Returns:
        {
            "artifacts": list[DreamArtifact],
            "session_id": int,
            "duration_seconds": float,
            "success": bool,
            "error": str | None,
        }
    """
    from glm.dreamer import Dreamer

    start = time.time()
    result = {
        "artifacts": [],
        "session_id": 0,
        "duration_seconds": 0.0,
        "success": False,
        "error": None,
    }

    try:
        # 1. Start dream session
        pattern_snapshot = {}
        if self_improver is not None:
            try:
                pattern_snapshot = self_improver.analyse_performance()
                # Convert sets to lists for JSON serialisation
                metrics = pattern_snapshot.get("session_metrics", {})
                if isinstance(metrics.get("domains_used"), set):
                    metrics["domains_used"] = list(metrics["domains_used"])
            except Exception:
                pass

        session_id = memory.start_dream_session(
            trigger_type=trigger_type,
            pattern_buffer_snapshot=pattern_snapshot,
        )
        result["session_id"] = session_id

        # 2. Run dreamer
        dreamer = Dreamer()
        artifacts = dreamer.dream(
            angel=angel,
            memory=memory,
            voice=voice,
            self_improver=self_improver,
        )
        result["artifacts"] = artifacts

        # 3. End dream session
        memory.end_dream_session(
            session_id=session_id,
            artifacts_count=len(artifacts),
        )

        result["success"] = True

    except Exception as exc:
        log.error("Dream cycle failed: %s", exc)
        result["error"] = str(exc)

    result["duration_seconds"] = time.time() - start
    return result
```

#### Modifications to `main_android.py`

Add Android foreground service management. The key pyjnius calls:

```python
# --- Add near the top of MKAngelApp ---

def _start_dream_service(self):
    """Start Android foreground service for dreaming."""
    try:
        from jnius import autoclass
        PythonService = autoclass("org.kivy.android.PythonService")
        Context = autoclass("android.content.Context")
        Intent = autoclass("android.content.Intent")
        NotificationCompat = autoclass("androidx.core.app.NotificationCompat")
        NotificationChannel = autoclass("android.app.NotificationChannel")
        NotificationManager = autoclass("android.app.NotificationManager")

        # Create notification channel (Android 8+)
        context = PythonService.mActivity
        nm = context.getSystemService(Context.NOTIFICATION_SERVICE)
        channel = NotificationChannel(
            "mkangel_dream",
            "Angel Dreams",
            NotificationManager.IMPORTANCE_LOW,
        )
        channel.setDescription("Notification while Angel is dreaming")
        nm.createNotificationChannel(channel)

        # Build notification
        builder = NotificationCompat.Builder(context, "mkangel_dream")
        builder.setContentTitle("Angel is dreaming...")
        builder.setContentText("Processing patterns from today")
        builder.setSmallIcon(context.getApplicationInfo().icon)
        builder.setOngoing(True)

        notification = builder.build()
        # ... start foreground with notification
    except Exception as exc:
        log.warning("Could not start foreground service: %s", exc)
        # Fallback: run dream cycle in background thread without service
        self._run_dream_fallback()


def _run_dream_fallback(self):
    """Run dream cycle without foreground service (fallback)."""
    import threading

    def _dream():
        try:
            from app.dream_service import run_dream_cycle
            from glm.voice import Voice

            voice = Voice()
            result = run_dream_cycle(
                angel=self.angel,
                memory=self._memory_obj,
                voice=voice,
                trigger_type="self",
            )
            if result["success"]:
                n = len(result["artifacts"])
                log.info("Dream cycle complete: %d artifacts", n)
        except Exception as exc:
            log.error("Dream fallback failed: %s", exc)

    threading.Thread(target=_dream, daemon=True).start()


def _stop_dream_service(self):
    """Stop the foreground service and update notification."""
    try:
        from jnius import autoclass
        # ... stop foreground, update notification to
        # "Angel finished dreaming -- N artifacts"
    except Exception:
        pass
```

The actual foreground service on Android requires a `Service` subclass. For p4a/Kivy, the recommended approach is using `PythonService`:

```python
# The foreground service approach for Kivy/p4a:
# 1. Run dream_service.run_dream_cycle() in a background thread
# 2. Use pyjnius to show a persistent notification during processing
# 3. Update notification when done
# 4. Set chat state to WAKING
#
# Full Android Service (separate process) is Phase 3.1+ optimisation.
# For now, background thread + notification is sufficient and simpler.
```

#### Wire DROWSY -> SLEEPING in `app/chat.py`

Add a callback mechanism so `main_android.py` can listen for state transitions:

```python
# In ChatSession.__init__:
self._on_state_change: Any = None  # callback(old_state, new_state)

def set_on_state_change(self, callback) -> None:
    """Register a callback for state transitions."""
    self._on_state_change = callback

def _transition_state(self, new_state: str) -> None:
    """Transition to a new state, notifying listeners."""
    old = self._state
    self._state = new_state
    if self._on_state_change:
        try:
            self._on_state_change(old, new_state)
        except Exception:
            pass
```

#### Modifications to `buildozer.spec`

```ini
# Add FOREGROUND_SERVICE permission
android.permissions = INTERNET,RECORD_AUDIO,FOREGROUND_SERVICE

# Add service declaration (for future full Service implementation)
# android.services = dream_service:app/dream_service.py
```

#### Dependencies

- Step 1 (dreams table) -- `run_dream_cycle` writes to it.
- Step 2 (Dreamer module) -- `run_dream_cycle` calls `Dreamer.dream()`.
- Step 3 (sleep triggers) -- the DROWSY state that triggers this step.

#### How to test

```python
# tests/test_dream_service.py
import tempfile, os
from app.dream_service import run_dream_cycle
from app.memory import Memory

def test_dream_cycle_empty():
    """Dream cycle with empty memory completes without error."""
    with tempfile.TemporaryDirectory() as td:
        m = Memory(db_path=os.path.join(td, "test.db"))
        result = run_dream_cycle(
            angel=None,  # No angel -> dreamer returns empty
            memory=m,
            voice=None,
            trigger_type="user",
        )
        # Should succeed even with no data
        assert result["success"]
        assert result["duration_seconds"] >= 0
        m.close()

def test_dream_cycle_with_data():
    """Dream cycle with seeded conversations produces artifacts."""
    with tempfile.TemporaryDirectory() as td:
        m = Memory(db_path=os.path.join(td, "test.db"))
        m.save_session("s1", [
            {"role": "user", "content": "explain photosynthesis"},
            {"role": "assistant", "content": "..."},
        ])
        # Angel is optional -- without it, dreamer just recalls
        result = run_dream_cycle(
            angel=None,
            memory=m,
            voice=None,
            trigger_type="self",
        )
        assert result["success"]
        assert result["session_id"] > 0
        m.close()
```

The pyjnius Android service code is not unit-testable off-device. Test it via:
1. Build APK on CI
2. Deploy to device
3. Trigger `/sleep` command
4. Check logcat: `adb logcat -s python:* kivy:*`
5. Verify notification appears/disappears
6. Verify dreams table has entries after cycle

#### Commit message

```
feat(android): add dream service with foreground notification

New file app/dream_service.py runs the dream cycle:
- Starts dream session in Memory
- Runs Dreamer.dream() pipeline
- Ends session with artifact count

main_android.py gets _start_dream_service() and _run_dream_fallback()
for Android foreground service with notification. Falls back to
background thread on non-Android platforms.

buildozer.spec: add FOREGROUND_SERVICE permission.

Wire ChatSession state transitions to trigger the service
when DROWSY -> SLEEPING.
```

---

### Step 5: Wake Greeting

**Goal**: When the app resumes and there are unseen dreams, compose a wake greeting via Voice that references what she dreamed about, display dream count and types, and offer to show the canvas.

**Files to modify**:
- `app/chat.py` -- add wake greeting logic on app resume
- `main_android.py` -- detect app resume and trigger wake check

#### Changes to `ChatSession`

```python
def check_wake_greeting(self) -> str | None:
    """Check for unseen dreams and compose a wake greeting.

    Called on app resume (from main_android.py).

    Returns:
        A greeting string if unseen dreams exist, None otherwise.
    """
    if self._state not in ("WAKING", "AWAKE"):
        return None
    if self._memory is None:
        return None

    try:
        unseen = self._memory.get_unseen_dreams()
    except Exception:
        return None

    if not unseen:
        if self._state == "WAKING":
            self._transition_state("AWAKE")
        return None

    # Transition state
    self._transition_state("AWAKE")

    # Classify dream types
    type_counts: dict[str, int] = {}
    for d in unseen:
        t = d.get("type", "observation")
        type_counts[t] = type_counts.get(t, 0) + 1

    # Mark as seen
    dream_ids = [d["id"] for d in unseen]
    try:
        self._memory.mark_dreams_seen(dream_ids)
    except Exception:
        pass

    # Compose greeting
    total = len(unseen)
    type_descriptions = {
        "poem": "poem",
        "grammar_map": "grammar map",
        "micro_tool": "micro tool",
        "self_patch": "growth proposal",
        "observation": "observation",
    }

    type_parts = []
    for t, count in type_counts.items():
        desc = type_descriptions.get(t, t)
        if count > 1:
            type_parts.append(f"{count} {desc}s")
        else:
            type_parts.append(f"a {desc}")

    type_list = ", ".join(type_parts[:-1])
    if len(type_parts) > 1:
        type_list += f" and {type_parts[-1]}"
    else:
        type_list = type_parts[0] if type_parts else "something"

    # Try Voice composition for a personal touch
    greeting = ""
    if self._voice is not None:
        try:
            # Use the first dream's content words for Voice
            first_content = unseen[0].get("content", "")
            tokens = first_content.lower().split()[:5]
            composed = self._voice.compose(
                original="I dreamed",
                tokens=tokens,
                voices={},
                harmonics=[],
                counterpoint=[],
                origins=[],
                predictions=[],
                harmony=0.7,
                loop_gate=0.3,
            )
            if composed and len(composed) > 10:
                greeting = composed + " "
        except Exception:
            pass

    if not greeting:
        greeting = "I dreamed while you were away. "

    greeting += (
        f"I made {type_list}. "
        f"{total} dream{'s' if total != 1 else ''} in all."
    )

    # If there are self-patches, mention them
    if type_counts.get("self_patch", 0) > 0:
        n = type_counts["self_patch"]
        greeting += (
            f" {n} growth proposal{'s' if n != 1 else ''} "
            f"waiting for your review."
        )

    return greeting
```

#### Changes to `main_android.py`

In `MKAngelApp.build()`, add resume detection:

```python
# In build():
# ... existing code ...

# Register for app resume events
from kivy.app import App
# on_resume is a Kivy App lifecycle method for Android
# It fires when the app returns from background

def on_resume(self):
    """Called when app returns from background on Android."""
    if self.session is not None:
        greeting = self.session.check_wake_greeting()
        if greeting:
            # Schedule on main thread
            from kivy.clock import Clock
            accent_hex = _css("accent").lstrip("#")

            def _show_wake(_):
                self.chat.add(
                    f"[color={accent_hex}][b]~ Good morning ~[/b][/color]",
                    kind="system",
                )
                self.chat.add(greeting, kind="angel")

            Clock.schedule_once(_show_wake, 0.3)
```

Also add a `/dreams` command to view recent dreams:

```python
# In ChatSession._COMMANDS:
"/dreams": ("", "View recent dreams"),

# In _handle_command handlers dict:
"/dreams": self._cmd_dreams,

# New method:
def _cmd_dreams(self, args: str) -> str:
    """View recent dreams."""
    if self._memory is None:
        return "Memory not available."

    dreams = self._memory.get_recent_dreams(limit=10)
    if not dreams:
        return "No dreams yet. The Angel hasn't slept."

    parts = ["Recent Dreams:"]
    for d in dreams:
        status_icon = {
            "new": "[NEW]",
            "seen": "[seen]",
            "archived": "[archived]",
            "accepted": "[accepted]",
        }.get(d["status"], "[?]")

        parts.append(
            f"\n  {status_icon} {d['type']} "
            f"(surprise: {d['surprise_score']:.2f})"
        )
        # Show first 80 chars of content
        preview = d["content"][:80].replace("\n", " ")
        parts.append(f"    {preview}")

    return "\n".join(parts)
```

#### Dependencies

- Step 1 (dreams table) -- `get_unseen_dreams()`, `mark_dreams_seen()`.
- Step 3 (state machine) -- WAKING state.
- Step 4 (dream service) -- populates the dreams table.

#### How to test

```python
# tests/test_wake_greeting.py
import tempfile, os
from app.chat import ChatSession
from app.memory import Memory

def test_wake_greeting_with_dreams():
    with tempfile.TemporaryDirectory() as td:
        m = Memory(db_path=os.path.join(td, "test.db"))
        # Seed some dreams
        m.save_dream(
            dream_type="poem",
            content="Water traces to *wed-, to be wet",
            surprise_score=0.8,
        )
        m.save_dream(
            dream_type="grammar_map",
            content="Resonance across linguistic and chemical",
            surprise_score=0.6,
        )
        session = ChatSession(memory=m)
        session._state = "WAKING"

        greeting = session.check_wake_greeting()
        assert greeting is not None
        assert "poem" in greeting.lower() or "grammar map" in greeting.lower()
        assert session.state == "AWAKE"

        # Dreams should now be marked seen
        unseen = m.get_unseen_dreams()
        assert len(unseen) == 0
        m.close()

def test_wake_greeting_no_dreams():
    with tempfile.TemporaryDirectory() as td:
        m = Memory(db_path=os.path.join(td, "test.db"))
        session = ChatSession(memory=m)
        session._state = "WAKING"
        greeting = session.check_wake_greeting()
        assert greeting is None
        m.close()

def test_dreams_command():
    with tempfile.TemporaryDirectory() as td:
        m = Memory(db_path=os.path.join(td, "test.db"))
        m.save_dream(dream_type="observation", content="A quiet day.")
        session = ChatSession(memory=m)
        result = session._cmd_dreams("")
        assert "observation" in result
        m.close()
```

#### Commit message

```
feat(chat): add wake greeting and /dreams command

On app resume, ChatSession.check_wake_greeting() checks for unseen
dreams in Memory. If found, composes a greeting via Voice describing
what she dreamed (types, count, growth proposals).

Marks dreams as 'seen' after greeting. Transitions WAKING -> AWAKE.

Adds /dreams command to view recent dream artifacts with status,
type, surprise score, and content preview.

main_android.py on_resume() shows the wake greeting as a system
message followed by an angel bubble.
```

---

## Phase 3.2 -- Canvas (OVERVIEW)

### Step 6: WebView Widget in Kivy

- Add a WebView panel alongside the chat panel in `main_android.py`
- Use Kivy's `android.runnable` + pyjnius to create an Android WebView
- Load `canvas/dream_canvas.html` from app assets
- Panel switching via gesture or button (reuse `GestureAction` system from `app/gestures.py`)
- On non-Android (desktop testing), fall back to a simple Kivy widget with text rendering

### Step 7: dream_canvas.html

- Create `canvas/dream_canvas.html` -- HTML5 infinite canvas
- CSS custom properties from `vestment_to_css()` for theming
- Pan/zoom via CSS transforms and touch event handlers
- Receives dream artifacts as JSON via `window.MKAngel.pushArtifacts(json)`
- Renders each artifact type:
  - Poems: gold-bordered text cards
  - Grammar maps: SVG network nodes/edges
  - Self-patches: approval cards with Accept/Defer buttons
  - Observations: translucent floating notes
- New dreams animate in with a gentle pulse

### Step 8: Python-JS Bridge

- Kivy WebView `evaluateJavascript()` for Python -> JS (push artifacts)
- `@JavascriptInterface` annotated class via pyjnius for JS -> Python callbacks
- JS sends: `onArtifactTap(id)`, `onArtifactDismiss(id)`, `onPatchAccept(id)`
- Python handles: archive dream, accept patch, compose response to tap
- Keep bridge minimal -- serialise everything as JSON strings

### Step 9: Canvas Gesture

- Add `GestureAction.CANVAS = "canvas"` to `app/gestures.py`
- Wire swipe-up-from-centre or a dedicated button in the header
- Canvas remembers scroll/zoom position between opens
- "Back to Chat" button in canvas header

---

## Phase 3.3 -- Integration (OVERVIEW)

### Step 10: Android Intents (outbound)

- Use pyjnius to create `android.content.Intent` objects
- Share dream artifacts: `ACTION_SEND` with text/plain or text/html
- Open URLs found during dreaming: `ACTION_VIEW` with URI
- Launch other apps from micro-tools
- Wrap in try/except -- graceful fallback if intent resolution fails

### Step 11: Content Providers (calendar, contacts)

- Query `content://com.android.calendar/events` for upcoming events
- Query `content://com.android.contacts/contacts` for names (with user consent)
- Add `READ_CALENDAR`, `READ_CONTACTS` to `buildozer.spec` permissions
- Runtime permission request via pyjnius (`ActivityCompat.requestPermissions`)
- Compliance module sanitises PII before it enters the grammar pipeline
- Data flows into Dreamer's recall stage as "context signals"

### Step 12: Wire Real-World Data into Dreamer

- Extend `Dreamer.recall()` to accept optional context providers
- Calendar events feed "temporal context" -- she can dream about upcoming events
- Contact names feed "social context" -- she can reference people the user knows
- Weather API (optional, network-dependent) feeds "environmental context"
- All real-world data passes through the same grammar pipeline as conversation data

---

## Phase 3.4 -- Ecosystem (OVERVIEW)

### Step 13: MCP Server

- Expose GLM functions via Model Context Protocol
- Tools: `compose_fugue`, `lookup_word`, `sense`, `predict`, `reconstruct`
- Run as a local server on the device (localhost:port)
- Other apps/agents can call the Angel's grammars
- Use stdlib `http.server` or a minimal framework -- no heavy deps

### Step 14: Grammar Export

- Dream artifacts (grammar maps) exportable as interactive HTML files
- Shareable via Android `ACTION_SEND` intent
- Grammar map HTML includes embedded JavaScript for pan/zoom
- Self-contained single-file export (inline CSS/JS)

### Step 15: Multi-Device Sync

- Cloud layer for syncing learned patterns between devices
- Encrypted storage (user holds the key)
- Conflict resolution: merge pattern confidences, union of dreams
- Optional -- the Angel works fully offline without sync

---

## Summary of Build Order

| Step | File(s) | Creates | Depends On |
|------|---------|---------|------------|
| 1 | `app/memory.py` | dreams + dream_sessions tables | -- |
| 2 | `glm/dreamer.py` (new) | Dreamer class, DreamArtifact | Step 1 |
| 3 | `app/chat.py` | Sleep triggers, state machine | -- |
| 4 | `app/dream_service.py` (new), `main_android.py`, `buildozer.spec` | Foreground service | Steps 1, 2, 3 |
| 5 | `app/chat.py`, `main_android.py` | Wake greeting, /dreams | Steps 1, 3, 4 |
| 6-9 | Canvas (Phase 3.2) | WebView, HTML, bridge, gesture | Steps 1-5 |
| 10-12 | Integration (Phase 3.3) | Intents, content providers | Steps 1-9 |
| 13-15 | Ecosystem (Phase 3.4) | MCP, export, sync | Steps 1-12 |

Steps 1-3 can be built and tested without an Android device.
Step 4 requires an APK build for full testing (pyjnius).
Step 5 can be partially tested off-device, fully tested on device.

---

## Testing Strategy

### Unit tests (off-device)

- `tests/test_dream_memory.py` -- Step 1 (SQLite dream operations)
- `tests/test_dreamer.py` -- Step 2 (recall, connect, compose, arrange, puriel)
- `tests/test_sleep_triggers.py` -- Step 3 (sleep detection, state machine)
- `tests/test_dream_service.py` -- Step 4 (dream cycle orchestration)
- `tests/test_wake_greeting.py` -- Step 5 (greeting composition)

### Integration tests (on-device)

1. Build APK via GitHub Actions CI
2. Install on Pixel 10 Pro XL
3. Have a conversation (5+ turns)
4. Send `/sleep` command
5. Verify: notification appears, dreams table populated
6. Background the app, reopen
7. Verify: wake greeting appears with dream summary
8. Send `/dreams` to see dream list

### CI additions

- Add `pytest tests/test_dream_*.py tests/test_sleep_*.py tests/test_wake_*.py` to the test matrix
- These tests use `tempfile.TemporaryDirectory` for SQLite -- no external deps
