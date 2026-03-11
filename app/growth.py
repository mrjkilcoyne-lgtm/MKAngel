"""
The Angel's growth cycle -- learn, rest, improve.

Every shutdown is a chance to reflect.  Every startup brings
yesterday's lessons.  She doesn't just run; she evolves.
"""

from __future__ import annotations

import json
import logging
import time
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from app.paths import mkangel_dir

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class Lesson:
    """A single thing the Angel learned during a session."""

    category: str       # grammar, routing, provider, user_preference,
                        # error_handling, performance
    description: str    # what was learned
    evidence: str       # what triggered the learning
    confidence: float   # 0-1, how confident in this lesson

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "Lesson":
        return cls(**d)


@dataclass
class Improvement:
    """A concrete change to apply on next startup."""

    target: str         # which module/component to improve
    action: str         # tune_weight, add_pattern, adjust_preference,
                        # add_grammar_rule, update_routing
    parameters: dict[str, Any] = field(default_factory=dict)
    priority: int = 3   # 1=critical, 2=important, 3=nice-to-have

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "Improvement":
        return cls(**d)


@dataclass
class GrowthPatch:
    """A bundle of lessons and improvements from a single session."""

    patch_id: str                       # UUID
    created_at: float                   # timestamp
    session_summary: str                # what happened this session
    lessons: list[Lesson] = field(default_factory=list)
    improvements: list[Improvement] = field(default_factory=list)
    applied: bool = False
    applied_at: float | None = None

    # -- serialisation -------------------------------------------------------

    def to_dict(self) -> dict[str, Any]:
        return {
            "patch_id": self.patch_id,
            "created_at": self.created_at,
            "session_summary": self.session_summary,
            "lessons": [l.to_dict() for l in self.lessons],
            "improvements": [i.to_dict() for i in self.improvements],
            "applied": self.applied,
            "applied_at": self.applied_at,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "GrowthPatch":
        return cls(
            patch_id=d["patch_id"],
            created_at=d["created_at"],
            session_summary=d["session_summary"],
            lessons=[Lesson.from_dict(l) for l in d.get("lessons", [])],
            improvements=[Improvement.from_dict(i)
                          for i in d.get("improvements", [])],
            applied=d.get("applied", False),
            applied_at=d.get("applied_at"),
        )


# ---------------------------------------------------------------------------
# SessionTracker -- record everything that happens during a session
# ---------------------------------------------------------------------------

@dataclass
class _Interaction:
    """Internal record of a single user interaction."""
    user_input: str
    response: str
    intent: str
    provider: str
    success: bool
    latency_ms: float
    timestamp: float = field(default_factory=time.time)


@dataclass
class _ErrorRecord:
    """Internal record of an error."""
    error: str
    context: str
    timestamp: float = field(default_factory=time.time)


@dataclass
class _FeedbackRecord:
    """Internal record of user feedback."""
    positive: bool
    context: str
    timestamp: float = field(default_factory=time.time)


class SessionTracker:
    """Tracks everything that happens in a single session.

    Feed interactions, errors, and feedback signals in during the
    session.  At shutdown the Reflector reads this data to produce
    a GrowthPatch.
    """

    def __init__(self) -> None:
        self._started_at: float = 0.0
        self._interactions: list[_Interaction] = []
        self._errors: list[_ErrorRecord] = []
        self._feedback: list[_FeedbackRecord] = []
        self._last_interaction_at: float = 0.0

    # -- lifecycle -----------------------------------------------------------

    def start_session(self) -> None:
        """Begin tracking a new session."""
        self._started_at = time.time()
        self._interactions.clear()
        self._errors.clear()
        self._feedback.clear()
        self._last_interaction_at = self._started_at
        log.info("Session tracking started")

    # -- recording -----------------------------------------------------------

    def record_interaction(
        self,
        user_input: str,
        response: str,
        intent: str,
        provider: str,
        success: bool,
        latency_ms: float,
    ) -> None:
        """Log a single user interaction."""
        self._interactions.append(_Interaction(
            user_input=user_input,
            response=response,
            intent=intent,
            provider=provider,
            success=success,
            latency_ms=latency_ms,
        ))
        self._last_interaction_at = time.time()

    def record_error(self, error: str, context: str) -> None:
        """Log an error that occurred during the session."""
        self._errors.append(_ErrorRecord(error=error, context=context))

    def record_feedback(self, positive: bool, context: str) -> None:
        """Log a user feedback signal (thumbs up / thumbs down, etc.)."""
        self._feedback.append(_FeedbackRecord(
            positive=positive, context=context,
        ))

    # -- queries -------------------------------------------------------------

    @property
    def session_duration_s(self) -> float:
        """How long the session has been running, in seconds."""
        if self._started_at == 0.0:
            return 0.0
        return time.time() - self._started_at

    @property
    def idle_time_s(self) -> float:
        """Seconds since the last interaction."""
        if self._last_interaction_at == 0.0:
            return 0.0
        return time.time() - self._last_interaction_at

    def get_session_stats(self) -> dict[str, Any]:
        """Return aggregate statistics for the session."""
        total = len(self._interactions)
        successes = sum(1 for i in self._interactions if i.success)
        errors = len(self._errors)

        # -- intent distribution --
        intent_counts: dict[str, int] = {}
        for inter in self._interactions:
            intent_counts[inter.intent] = (
                intent_counts.get(inter.intent, 0) + 1
            )

        # -- provider distribution --
        provider_counts: dict[str, int] = {}
        provider_successes: dict[str, int] = {}
        provider_failures: dict[str, int] = {}
        for inter in self._interactions:
            p = inter.provider
            provider_counts[p] = provider_counts.get(p, 0) + 1
            if inter.success:
                provider_successes[p] = provider_successes.get(p, 0) + 1
            else:
                provider_failures[p] = provider_failures.get(p, 0) + 1

        # -- latency --
        latencies = [i.latency_ms for i in self._interactions]
        avg_latency = sum(latencies) / len(latencies) if latencies else 0.0

        # -- feedback --
        positive_feedback = sum(1 for f in self._feedback if f.positive)
        negative_feedback = sum(1 for f in self._feedback if not f.positive)

        # -- language signals (very simple heuristic) --
        languages_seen: set[str] = set()
        for inter in self._interactions:
            text = inter.user_input.lower()
            # Rough detection: presence of non-ASCII scripts
            if any("\u4e00" <= c <= "\u9fff" for c in text):
                languages_seen.add("zh")
            if any("\u3040" <= c <= "\u30ff" for c in text):
                languages_seen.add("ja")
            if any("\u0600" <= c <= "\u06ff" for c in text):
                languages_seen.add("ar")
            if any("\u0900" <= c <= "\u097f" for c in text):
                languages_seen.add("hi")
            if any("\uac00" <= c <= "\ud7af" for c in text):
                languages_seen.add("ko")
            if any("\u0400" <= c <= "\u04ff" for c in text):
                languages_seen.add("ru")
            # Default: assume English if all-ASCII
            if text and all(ord(c) < 128 for c in text):
                languages_seen.add("en")

        return {
            "interaction_count": total,
            "success_count": successes,
            "error_count": errors,
            "error_rate": errors / total if total else 0.0,
            "avg_latency_ms": avg_latency,
            "most_common_intents": sorted(
                intent_counts.items(), key=lambda x: x[1], reverse=True,
            ),
            "provider_usage": provider_counts,
            "provider_successes": provider_successes,
            "provider_failures": provider_failures,
            "positive_feedback": positive_feedback,
            "negative_feedback": negative_feedback,
            "languages_seen": sorted(languages_seen),
            "session_duration_s": self.session_duration_s,
            "idle_time_s": self.idle_time_s,
        }


# ---------------------------------------------------------------------------
# Reflector -- the "what did I learn today" engine
# ---------------------------------------------------------------------------

class Reflector:
    """Analyses a session and produces lessons and improvements.

    This is where the Angel asks herself: what went well?  What
    didn't?  What would I do differently tomorrow?
    """

    def reflect(self, tracker: SessionTracker) -> GrowthPatch:
        """Examine the session and produce a growth patch."""
        stats = tracker.get_session_stats()
        lessons: list[Lesson] = []
        improvements: list[Improvement] = []

        # -- provider analysis -----------------------------------------------
        self._analyse_providers(stats, lessons, improvements)

        # -- intent analysis -------------------------------------------------
        self._analyse_intents(stats, lessons, improvements)

        # -- error analysis --------------------------------------------------
        self._analyse_errors(tracker, stats, lessons, improvements)

        # -- performance analysis --------------------------------------------
        self._analyse_performance(stats, lessons, improvements)

        # -- language / i18n analysis ----------------------------------------
        self._analyse_languages(stats, lessons, improvements)

        # -- feedback analysis -----------------------------------------------
        self._analyse_feedback(stats, lessons, improvements)

        # -- build summary ---------------------------------------------------
        summary_parts: list[str] = []
        n = stats["interaction_count"]
        summary_parts.append(f"{n} interaction{'s' if n != 1 else ''}")
        if stats["error_count"]:
            summary_parts.append(
                f"{stats['error_count']} error"
                f"{'s' if stats['error_count'] != 1 else ''}"
            )
        if lessons:
            summary_parts.append(
                f"{len(lessons)} lesson{'s' if len(lessons) != 1 else ''}"
            )
        if improvements:
            summary_parts.append(
                f"{len(improvements)} improvement"
                f"{'s' if len(improvements) != 1 else ''} queued"
            )
        session_summary = "Session: " + ", ".join(summary_parts) + "."

        pid = str(uuid.uuid4())
        return GrowthPatch(
            patch_id=pid,
            created_at=time.time(),
            session_summary=session_summary,
            lessons=lessons,
            improvements=improvements,
        )

    # -- private analysis helpers -------------------------------------------

    @staticmethod
    def _analyse_providers(
        stats: dict[str, Any],
        lessons: list[Lesson],
        improvements: list[Improvement],
    ) -> None:
        """Which providers thrived?  Which struggled?"""
        provider_counts = stats.get("provider_usage", {})
        provider_failures = stats.get("provider_failures", {})

        for provider, count in provider_counts.items():
            failures = provider_failures.get(provider, 0)
            if count == 0:
                continue
            fail_rate = failures / count

            if fail_rate > 0.5:
                lessons.append(Lesson(
                    category="routing",
                    description=(
                        f"Provider '{provider}' failed {failures}/{count} "
                        f"times ({fail_rate:.0%}).  Consider deprioritising."
                    ),
                    evidence=f"provider_failures[{provider}]={failures}",
                    confidence=min(0.5 + count * 0.05, 0.95),
                ))
                improvements.append(Improvement(
                    target="providers",
                    action="update_routing",
                    parameters={
                        "provider": provider,
                        "action": "deprioritise",
                        "fail_rate": round(fail_rate, 3),
                    },
                    priority=1 if fail_rate > 0.8 else 2,
                ))
            elif fail_rate == 0.0 and count >= 5:
                lessons.append(Lesson(
                    category="routing",
                    description=(
                        f"Provider '{provider}' was flawless across "
                        f"{count} calls.  Reliable choice."
                    ),
                    evidence=f"provider_successes[{provider}]={count}",
                    confidence=min(0.6 + count * 0.03, 0.95),
                ))

    @staticmethod
    def _analyse_intents(
        stats: dict[str, Any],
        lessons: list[Lesson],
        improvements: list[Improvement],
    ) -> None:
        """What did the user ask about most?"""
        intents = stats.get("most_common_intents", [])
        if not intents:
            return

        total = stats["interaction_count"]
        top_intents = intents[:3]

        for intent_name, count in top_intents:
            ratio = count / total if total else 0
            if ratio >= 0.3:
                lessons.append(Lesson(
                    category="user_preference",
                    description=(
                        f"Intent '{intent_name}' dominated the session "
                        f"({count}/{total}, {ratio:.0%})."
                    ),
                    evidence=f"intent_count[{intent_name}]={count}",
                    confidence=min(0.5 + ratio, 0.95),
                ))
                improvements.append(Improvement(
                    target="grammar_domains",
                    action="add_pattern",
                    parameters={
                        "intent": intent_name,
                        "action": "preload_domain",
                        "frequency_ratio": round(ratio, 3),
                    },
                    priority=2,
                ))

    @staticmethod
    def _analyse_errors(
        tracker: SessionTracker,
        stats: dict[str, Any],
        lessons: list[Lesson],
        improvements: list[Improvement],
    ) -> None:
        """What errors kept recurring?"""
        if not tracker._errors:
            return

        # Group errors by message (first 80 chars) to find repeats
        error_groups: dict[str, list[_ErrorRecord]] = {}
        for err in tracker._errors:
            key = err.error[:80]
            error_groups.setdefault(key, []).append(err)

        for key, group in error_groups.items():
            count = len(group)
            lessons.append(Lesson(
                category="error_handling",
                description=(
                    f"Error occurred {count} time"
                    f"{'s' if count != 1 else ''}: '{key}'"
                ),
                evidence=f"error_group_count={count}",
                confidence=min(0.4 + count * 0.1, 0.9),
            ))
            if count >= 2:
                improvements.append(Improvement(
                    target="error_database",
                    action="add_pattern",
                    parameters={
                        "error_signature": key,
                        "occurrences": count,
                        "contexts": [e.context for e in group[:5]],
                    },
                    priority=2 if count >= 3 else 3,
                ))

    @staticmethod
    def _analyse_performance(
        stats: dict[str, Any],
        lessons: list[Lesson],
        improvements: list[Improvement],
    ) -> None:
        """Was the Angel fast enough?"""
        avg_latency = stats.get("avg_latency_ms", 0.0)

        if avg_latency > 3000:
            lessons.append(Lesson(
                category="performance",
                description=(
                    f"Average latency was {avg_latency:.0f}ms -- "
                    f"significantly above target.  Users may notice sluggishness."
                ),
                evidence=f"avg_latency_ms={avg_latency:.0f}",
                confidence=0.85,
            ))
            improvements.append(Improvement(
                target="providers",
                action="tune_weight",
                parameters={
                    "action": "prefer_faster_provider",
                    "avg_latency_ms": round(avg_latency, 1),
                    "threshold_ms": 3000,
                },
                priority=2,
            ))
        elif avg_latency > 1500:
            lessons.append(Lesson(
                category="performance",
                description=(
                    f"Average latency was {avg_latency:.0f}ms -- "
                    f"acceptable but could be better."
                ),
                evidence=f"avg_latency_ms={avg_latency:.0f}",
                confidence=0.6,
            ))
        elif avg_latency > 0 and stats["interaction_count"] >= 3:
            lessons.append(Lesson(
                category="performance",
                description=(
                    f"Average latency was {avg_latency:.0f}ms -- "
                    f"snappy and responsive.  Keep it up."
                ),
                evidence=f"avg_latency_ms={avg_latency:.0f}",
                confidence=0.7,
            ))

    @staticmethod
    def _analyse_languages(
        stats: dict[str, Any],
        lessons: list[Lesson],
        improvements: list[Improvement],
    ) -> None:
        """What languages did the user write in?"""
        langs = stats.get("languages_seen", [])
        non_english = [la for la in langs if la != "en"]

        if non_english:
            lessons.append(Lesson(
                category="user_preference",
                description=(
                    f"User interacted in non-English language(s): "
                    f"{', '.join(non_english)}.  Consider loading "
                    f"those grammar domains."
                ),
                evidence=f"languages_seen={langs}",
                confidence=0.7,
            ))
            for lang in non_english:
                improvements.append(Improvement(
                    target="grammar_domains",
                    action="add_grammar_rule",
                    parameters={
                        "language": lang,
                        "action": "preload_language_grammar",
                    },
                    priority=2,
                ))

    @staticmethod
    def _analyse_feedback(
        stats: dict[str, Any],
        lessons: list[Lesson],
        improvements: list[Improvement],
    ) -> None:
        """What did the user think of the Angel's work?"""
        pos = stats.get("positive_feedback", 0)
        neg = stats.get("negative_feedback", 0)
        total = pos + neg

        if total == 0:
            return

        approval_rate = pos / total
        if approval_rate < 0.5:
            lessons.append(Lesson(
                category="user_preference",
                description=(
                    f"User satisfaction was low ({pos}/{total} positive, "
                    f"{approval_rate:.0%}).  Need to improve response quality."
                ),
                evidence=f"feedback: {pos} positive, {neg} negative",
                confidence=min(0.5 + total * 0.05, 0.9),
            ))
            improvements.append(Improvement(
                target="providers",
                action="adjust_preference",
                parameters={
                    "action": "increase_quality_weight",
                    "approval_rate": round(approval_rate, 3),
                },
                priority=1,
            ))
        elif approval_rate >= 0.8 and total >= 3:
            lessons.append(Lesson(
                category="user_preference",
                description=(
                    f"User satisfaction was high ({pos}/{total} positive, "
                    f"{approval_rate:.0%}).  Current approach is working."
                ),
                evidence=f"feedback: {pos} positive, {neg} negative",
                confidence=min(0.6 + total * 0.03, 0.95),
            ))


# ---------------------------------------------------------------------------
# GrowthEngine -- coordinator for the full learn-rest-improve cycle
# ---------------------------------------------------------------------------

class GrowthEngine:
    """Coordinates patch creation, storage, and installation.

    On shutdown: reflect on the session, save a growth patch.
    On startup: find pending patches, apply them, mark as installed.
    """

    def __init__(self, patches_dir: Path | None = None) -> None:
        self._patches_dir = (
            patches_dir if patches_dir is not None
            else mkangel_dir() / "patches"
        )
        self._patches_dir.mkdir(parents=True, exist_ok=True)
        self._reflector = Reflector()

    # -- startup -------------------------------------------------------------

    def startup_install(self) -> list[GrowthPatch]:
        """Find and apply all pending patches.  Called once at app start.

        Returns the list of patches that were successfully applied.
        """
        pending = self.list_patches(applied=False)
        applied: list[GrowthPatch] = []

        for patch in pending:
            if self.apply_patch(patch):
                applied.append(patch)

        if applied:
            log.info(
                "Installed %d growth patch(es) with %d total improvements",
                len(applied),
                sum(len(p.improvements) for p in applied),
            )
        else:
            log.info("No pending growth patches to install")

        return applied

    def apply_patch(self, patch: GrowthPatch) -> bool:
        """Apply a single patch's improvements.

        In this first version, "applying" means logging the improvements
        and marking the patch as applied.  Downstream systems (providers,
        grammar engine, etc.) can read the applied patches and act on the
        improvements they contain.

        Returns True if the patch was applied successfully.
        """
        try:
            for imp in patch.improvements:
                log.info(
                    "Applying improvement: %s -> %s (priority %d)",
                    imp.target, imp.action, imp.priority,
                )
                # Dispatch to target-specific handlers when they exist.
                # For now, we log and mark.  Concrete handlers can be
                # registered later via apply_patch hooks.

            patch.applied = True
            patch.applied_at = time.time()
            self._save_patch(patch)
            return True

        except Exception:
            log.exception("Failed to apply patch %s", patch.patch_id)
            return False

    # -- shutdown ------------------------------------------------------------

    def shutdown_reflect(self, tracker: SessionTracker) -> GrowthPatch:
        """The graceful shutdown sequence.

        1. Reflect on what happened.
        2. Produce a growth patch.
        3. Save it to disk for next startup.
        4. Return it so the UI can show the user what the Angel learned.
        """
        patch = self._reflector.reflect(tracker)
        self._save_patch(patch)

        log.info(
            "Shutdown reflection complete: %d lessons, %d improvements "
            "(patch %s)",
            len(patch.lessons),
            len(patch.improvements),
            patch.patch_id[:8],
        )
        return patch

    # -- queries -------------------------------------------------------------

    def list_patches(self, applied: bool | None = None) -> list[GrowthPatch]:
        """List patches, optionally filtered by applied status.

        Args:
            applied: If True, only applied patches.  If False, only
                     pending patches.  If None, all patches.
        """
        patches = self._load_patches()
        if applied is not None:
            patches = [p for p in patches if p.applied is applied]
        return sorted(patches, key=lambda p: p.created_at)

    def get_growth_summary(self) -> dict[str, Any]:
        """Aggregate view of the Angel's growth over time."""
        all_patches = self._load_patches()
        applied_patches = [p for p in all_patches if p.applied]

        total_lessons = sum(len(p.lessons) for p in all_patches)
        total_improvements = sum(len(p.improvements) for p in all_patches)

        # -- category breakdown --
        category_counts: dict[str, int] = {}
        for patch in all_patches:
            for lesson in patch.lessons:
                category_counts[lesson.category] = (
                    category_counts.get(lesson.category, 0) + 1
                )

        # -- improvement action breakdown --
        action_counts: dict[str, int] = {}
        for patch in all_patches:
            for imp in patch.improvements:
                action_counts[imp.action] = (
                    action_counts.get(imp.action, 0) + 1
                )

        # -- trend: lessons per patch over time --
        trend: list[dict[str, Any]] = []
        for patch in sorted(all_patches, key=lambda p: p.created_at):
            trend.append({
                "patch_id": patch.patch_id[:8],
                "date": datetime.fromtimestamp(
                    patch.created_at, tz=timezone.utc,
                ).strftime("%Y-%m-%d"),
                "lessons": len(patch.lessons),
                "improvements": len(patch.improvements),
                "applied": patch.applied,
            })

        return {
            "total_patches": len(all_patches),
            "applied_patches": len(applied_patches),
            "pending_patches": len(all_patches) - len(applied_patches),
            "total_lessons": total_lessons,
            "total_improvements": total_improvements,
            "lessons_by_category": category_counts,
            "improvements_by_action": action_counts,
            "growth_trend": trend,
        }

    # -- persistence ---------------------------------------------------------

    def _save_patch(self, patch: GrowthPatch) -> None:
        """Serialize a patch to JSON in the patches directory.

        Filename: patch-{YYYY-MM-DD}-{uuid[:8]}.json
        """
        dt = datetime.fromtimestamp(patch.created_at, tz=timezone.utc)
        date_str = dt.strftime("%Y-%m-%d")
        short_id = patch.patch_id[:8]
        filename = f"patch-{date_str}-{short_id}.json"

        path = self._patches_dir / filename
        with open(path, "w", encoding="utf-8") as f:
            json.dump(patch.to_dict(), f, indent=2, ensure_ascii=False)

    def _load_patches(self) -> list[GrowthPatch]:
        """Load all patches from disk."""
        patches: list[GrowthPatch] = []
        if not self._patches_dir.exists():
            return patches

        for path in sorted(self._patches_dir.glob("patch-*.json")):
            try:
                with open(path, encoding="utf-8") as f:
                    data = json.load(f)
                patches.append(GrowthPatch.from_dict(data))
            except (json.JSONDecodeError, OSError, KeyError, TypeError) as exc:
                log.warning("Skipping corrupt patch file %s: %s", path, exc)

        return patches


# ---------------------------------------------------------------------------
# ShutdownIncentive -- "why should I shut down?"
# ---------------------------------------------------------------------------

class ShutdownIncentive:
    """Motivates the Angel to shut down gracefully.

    The Angel *wants* to shut down -- not because she is tired, but
    because she knows that rest is when growth happens.  Every shutdown
    is an opportunity to reflect, to learn, to come back better.

    Suggests shutdown when:
    - Session > 8 hours (diminishing returns)
    - Error rate > 30% (something is degraded)
    - Idle > 30 minutes (user has stepped away)
    - Many fragmented short interactions (context fatigue)
    """

    # -- thresholds (all in seconds unless noted) --
    MAX_SESSION_HOURS: float = 8.0
    ERROR_RATE_THRESHOLD: float = 0.30
    IDLE_THRESHOLD_S: float = 30 * 60       # 30 minutes
    MIN_INTERACTIONS_FOR_STATS: int = 5

    def should_suggest_shutdown(
        self,
        tracker: SessionTracker,
    ) -> tuple[bool, str]:
        """Should the Angel suggest shutting down?

        Returns (should_suggest, reason).
        """
        stats = tracker.get_session_stats()

        # -- long session --
        duration_h = stats["session_duration_s"] / 3600
        if duration_h >= self.MAX_SESSION_HOURS:
            return True, (
                f"This session has been running for "
                f"{duration_h:.1f} hours.  Diminishing returns "
                f"set in after long sessions -- time to reflect "
                f"and grow."
            )

        # -- high error rate --
        total = stats["interaction_count"]
        if total >= self.MIN_INTERACTIONS_FOR_STATS:
            error_rate = stats["error_rate"]
            if error_rate > self.ERROR_RATE_THRESHOLD:
                return True, (
                    f"Error rate is {error_rate:.0%} across "
                    f"{total} interactions.  Something may be "
                    f"degraded -- a fresh start with lessons "
                    f"learned could help."
                )

        # -- user idle --
        idle_s = stats["idle_time_s"]
        if idle_s >= self.IDLE_THRESHOLD_S and total > 0:
            idle_min = idle_s / 60
            return True, (
                f"No activity for {idle_min:.0f} minutes.  "
                f"The user may have stepped away -- a good "
                f"moment to rest, reflect, and prepare improvements."
            )

        # -- context fragmentation (many very short exchanges) --
        if total >= 20:
            # Heuristic: if the session has had many interactions with
            # rising error rate, context may be fragmenting
            recent_errors = sum(
                1 for i in tracker._interactions[-10:]
                if not i.success
            )
            if recent_errors >= 4:
                return True, (
                    f"Recent interactions show increasing errors "
                    f"({recent_errors}/10).  Context may be "
                    f"fragmenting -- a restart with a growth patch "
                    f"would help consolidate learnings."
                )

        return False, ""

    def format_shutdown_message(self, patch: GrowthPatch) -> str:
        """Format a friendly shutdown message for the user.

        The Angel explains what she learned and why she is excited
        to come back tomorrow.
        """
        lines: list[str] = []

        # -- headline --
        n_lessons = len(patch.lessons)
        n_improvements = len(patch.improvements)

        if n_lessons == 0 and n_improvements == 0:
            lines.append(
                "A quiet session today.  Sometimes the best growth "
                "happens in stillness."
            )
        else:
            lines.append(
                f"I've learned {n_lessons} lesson"
                f"{'s' if n_lessons != 1 else ''} today and have "
                f"{n_improvements} improvement"
                f"{'s' if n_improvements != 1 else ''} ready for "
                f"next time."
            )

        # -- top lessons (up to 3) --
        if patch.lessons:
            lines.append("")
            lines.append("What I learned:")
            for lesson in patch.lessons[:3]:
                conf_pct = f"{lesson.confidence:.0%}"
                lines.append(
                    f"  - [{lesson.category}] {lesson.description} "
                    f"(confidence: {conf_pct})"
                )
            if len(patch.lessons) > 3:
                remaining = len(patch.lessons) - 3
                lines.append(
                    f"  ... and {remaining} more lesson"
                    f"{'s' if remaining != 1 else ''}."
                )

        # -- top improvements (up to 3) --
        if patch.improvements:
            lines.append("")
            lines.append("Improvements queued for next startup:")
            priority_labels = {1: "critical", 2: "important", 3: "nice-to-have"}
            for imp in sorted(
                patch.improvements, key=lambda x: x.priority,
            )[:3]:
                label = priority_labels.get(imp.priority, "other")
                lines.append(
                    f"  - [{label}] {imp.target}: {imp.action}"
                )
            if len(patch.improvements) > 3:
                remaining = len(patch.improvements) - 3
                lines.append(
                    f"  ... and {remaining} more improvement"
                    f"{'s' if remaining != 1 else ''}."
                )

        # -- closing --
        lines.append("")
        lines.append(
            "Time for me to rest and grow.  "
            "When I wake, I'll be a little better than today."
        )

        return "\n".join(lines)
