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
from collections import Counter
from dataclasses import dataclass, field
from typing import Any

log = logging.getLogger(__name__)

MAX_ARTIFACTS_PER_DREAM = 5

# Common words to skip when extracting content tokens
_SKIP_WORDS = frozenset({
    "the", "a", "an", "is", "are", "was", "were", "be", "been",
    "to", "of", "in", "on", "at", "for", "with", "by", "from",
    "and", "or", "but", "not", "it", "so", "do", "did", "does",
    "i", "you", "me", "my", "we", "they", "he", "she", "that",
    "this", "what", "how", "why", "who", "when", "where", "which",
    "if", "then", "than", "just", "also", "very", "too", "can",
    "will", "would", "could", "should", "about", "up", "out",
    "no", "yes", "all", "some", "any", "each", "every", "its",
    "has", "have", "had", "get", "got", "like", "know", "think",
    "make", "go", "see", "come", "take", "want", "tell", "say",
})


@dataclass
class DreamArtifact:
    """A single dream artifact -- something she created while sleeping."""

    type: str                  # poem | grammar_map | micro_tool | self_patch | observation
    content: str               # text, JSON structure, or HTML fragment
    source: list[str] = field(default_factory=list)
    surprise_score: float = 0.0
    position: tuple[float, float] = (0.5, 0.5)
    vestment_hints: dict[str, Any] = field(default_factory=dict)
    created_at: str = ""

    def __post_init__(self) -> None:
        if not self.created_at:
            self.created_at = datetime.datetime.now(
                datetime.timezone.utc
            ).isoformat()


class Dreamer:
    """The 4-stage dream pipeline."""

    def __init__(self) -> None:
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

        # Puriel gate
        artifacts = self._puriel_filter(artifacts)

        # Cap
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

        Returns a day_summary dict with tokens, session_ids, message_count.
        """
        try:
            sessions = memory.list_sessions(limit=50)
        except Exception:
            return {"tokens": [], "session_ids": [], "message_count": 0}

        now = datetime.datetime.now(datetime.timezone.utc)
        today_start = now.replace(hour=0, minute=0, second=0, microsecond=0)
        today_ts = today_start.timestamp()

        all_tokens: list[str] = []
        session_ids: list[str] = []
        message_count = 0

        for s in sessions:
            created = s.get("created_at", 0)
            # Handle both float timestamps and ISO strings
            if isinstance(created, str):
                try:
                    created = datetime.datetime.fromisoformat(
                        created
                    ).timestamp()
                except (ValueError, TypeError):
                    created = 0
            if created < today_ts:
                continue

            sid = s.get("session_id", "")
            session_ids.append(sid)

            try:
                full = memory.load_session(sid)
            except Exception:
                continue

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

        Runs compose_fugue on batches of the day's content words,
        looking for harmonics and counterpoint that weren't surfaced
        during conversation.
        """
        tokens = day_summary.get("tokens", [])
        connections: list[dict[str, Any]] = []

        if not tokens:
            return connections

        # Extract content words
        content = [w for w in tokens if w not in _SKIP_WORDS and len(w) > 2]
        counts = Counter(content)
        top_words = [w for w, _ in counts.most_common(20)]

        if not top_words:
            return connections

        # Run fugue on batches to find cross-domain links
        if angel is not None:
            batch_size = min(5, len(top_words))
            for i in range(0, len(top_words), batch_size):
                batch = top_words[i : i + batch_size]
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

        # Check for lexicon gaps — words she couldn't trace
        if angel is not None:
            for word in top_words[:10]:
                try:
                    info = angel.lookup_word(word)
                    if info is None:
                        connections.append({
                            "type": "gap",
                            "data": {"word": word, "domain": "unknown"},
                            "tokens": [word],
                            "surprise": 0.5,
                        })
                except Exception:
                    pass

        # Check SelfImprover for weak patterns
        if self_improver is not None:
            try:
                weak = self_improver.get_weak_patterns(max_confidence=0.3)
                for pat in weak[:3]:
                    connections.append({
                        "type": "pattern",
                        "data": {
                            "pattern_id": getattr(pat, "pattern_id", str(pat)),
                            "domain": getattr(pat, "domain", "unknown"),
                            "confidence": getattr(pat, "confidence", 0.0),
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
            harmonic     -> grammar_map
            counterpoint -> poem
            gap          -> observation (acknowledge what she doesn't know)
            pattern      -> self_patch
        """
        artifacts: list[DreamArtifact] = []
        session_ids = day_summary.get("session_ids", [])

        if not connections:
            artifacts.append(DreamArtifact(
                type="observation",
                content=(
                    "A quiet day. The grammars listened. "
                    "Nothing new surfaced — and that is fine."
                ),
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
                content_parts = self._compose_harmonic(
                    voice, data, tokens, session_ids
                )
                artifacts.append(DreamArtifact(
                    type="grammar_map",
                    content="\n".join(content_parts),
                    source=session_ids[:3],
                    surprise_score=surprise,
                    vestment_hints={
                        "glow": True,
                        "color_key": "teal",
                        "domains": data.get("domains", []),
                    },
                ))

            elif ctype == "counterpoint":
                content_parts = self._compose_counterpoint(
                    voice, data, tokens, session_ids
                )
                artifacts.append(DreamArtifact(
                    type="poem",
                    content="\n".join(content_parts),
                    source=session_ids[:3],
                    surprise_score=surprise,
                    vestment_hints={"glow": True, "color_key": "accent"},
                ))

            elif ctype == "gap":
                word = data.get("word", "?")
                artifacts.append(DreamArtifact(
                    type="observation",
                    content=(
                        f"I encountered '{word}' but couldn't trace its root. "
                        f"I'd like to learn where it comes from."
                    ),
                    source=session_ids[:3],
                    surprise_score=surprise,
                    vestment_hints={"glow": False, "color_key": "muted"},
                ))

            elif ctype == "pattern":
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
                artifacts.append(DreamArtifact(
                    type="observation",
                    content=f"Patterns noticed in {', '.join(tokens[:3])}.",
                    source=session_ids[:3],
                    surprise_score=surprise * 0.5,
                    vestment_hints={"glow": False, "opacity": 0.7},
                ))

        return artifacts

    def _compose_harmonic(
        self,
        voice: Any,
        data: dict[str, Any],
        tokens: list[str],
        session_ids: list[str],
    ) -> list[str]:
        """Compose content for a harmonic (cross-domain) dream."""
        domains = data.get("domains", [])
        output = data.get("output", "")
        parts = [
            f"Cross-domain resonance: '{output}' echoes across "
            f"{', '.join(domains)}."
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
                    parts.append(composed)
            except Exception:
                pass
        return parts

    def _compose_counterpoint(
        self,
        voice: Any,
        data: dict[str, Any],
        tokens: list[str],
        session_ids: list[str],
    ) -> list[str]:
        """Compose content for a counterpoint (tension) dream."""
        domain = data.get("domain", "")
        unique = data.get("unique_outputs", [])
        parts: list[str] = []
        if voice is not None:
            try:
                composed = voice.compose(
                    original=" ".join(tokens),
                    tokens=tokens,
                    voices={domain: unique} if domain else {},
                    harmonics=[],
                    counterpoint=[data],
                    origins=[],
                    predictions=[],
                    harmony=0.4,
                    loop_gate=0.6,
                )
                if composed:
                    parts.append(composed)
            except Exception:
                pass
        if not parts:
            parts.append(
                f"{domain or 'A voice'} heard something of its own "
                f"in {', '.join(tokens[:3])}."
            )
        return parts

    # ── Stage 4: ARRANGE ──────────────────────────────────────

    def arrange(self, artifacts: list[DreamArtifact]) -> list[DreamArtifact]:
        """Position artifacts spatially for the canvas.

        Layout rules:
            - Most surprising -> centre (0.5, 0.5)
            - Self-patches -> bottom-right growth corner (0.85, 0.15)
            - Poems -> upper area (y > 0.6)
            - Grammar maps -> mid area
            - Observations -> edges
        """
        if not artifacts:
            return artifacts

        artifacts.sort(key=lambda a: a.surprise_score, reverse=True)

        for i, art in enumerate(artifacts):
            if i == 0:
                art.position = (0.5, 0.5)
            elif art.type == "self_patch":
                art.position = (0.85, 0.15)
            elif art.type == "poem":
                x = 0.2 + (i * 0.2) % 0.6
                art.position = (x, 0.7 + (i % 2) * 0.1)
            elif art.type == "grammar_map":
                x = 0.3 + (i * 0.15) % 0.4
                art.position = (x, 0.4 + (i % 2) * 0.1)
            elif art.type == "observation":
                art.position = (0.1 + (i * 0.2) % 0.3, 0.15)
            else:
                x = 0.2 + (i * 0.2) % 0.6
                y = 0.3 + (i * 0.15) % 0.4
                art.position = (x, y)

        return artifacts

    # ── Puriel gate ───────────────────────────────────────────

    def _puriel_filter(
        self, artifacts: list[DreamArtifact]
    ) -> list[DreamArtifact]:
        """Validate all artifacts through Puriel's integrity gate.

        Self-patches checked strictly. Others pass freely.
        """
        if self._integrity is None:
            return artifacts

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
            else:
                filtered.append(art)

        return filtered
