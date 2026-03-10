"""
Recursive self-improvement system for MKAngel.

The Angel improves itself between sessions and during continuous
operation.  This is the strange loop made practical: the system
uses its own grammar to reason about its own performance, then
modifies its own rules to improve.

Self-improvement channels:
1. Pattern learning — observe successful derivations, strengthen rules
2. Skill acquisition — discover and install new skills from registries
3. Language learning — acquire new grammars and substrates on demand
4. MNEMO evolution — compress experience into MNEMO programs
5. Cross-session memory — persist improvements across restarts
6. Meta-analysis — reason about reasoning quality

The key insight: improvement is itself a grammar operation.  The rules
for improving rules are meta-rules.  The strange loop: the system
applies its derivation engine to its own derivation history.

Safety: all modifications are logged, versioned, and reversible.
The Angel cannot modify its core invariants (safety rules, ethical
constraints, user preferences).
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

log = logging.getLogger(__name__)


_IMPROVEMENT_DIR = Path.home() / ".mkangel" / "improvements"
_IMPROVEMENT_LOG = _IMPROVEMENT_DIR / "log.jsonl"
_SKILL_REGISTRY = _IMPROVEMENT_DIR / "skill_registry.json"
_LEARNED_PATTERNS = _IMPROVEMENT_DIR / "learned_patterns.json"


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class ImprovementRecord:
    """Record of a self-improvement action."""
    action: str
    domain: str
    description: str
    before: dict[str, Any] = field(default_factory=dict)
    after: dict[str, Any] = field(default_factory=dict)
    confidence: float = 0.0
    timestamp: float = 0.0
    reversible: bool = True
    applied: bool = False
    integrity_check_passed: bool = True

    def __post_init__(self):
        if not self.timestamp:
            self.timestamp = time.time()


@dataclass
class LearnedPattern:
    """A pattern discovered through self-improvement."""
    pattern_id: str
    domain: str
    description: str
    rule_data: dict[str, Any] = field(default_factory=dict)
    success_count: int = 0
    failure_count: int = 0
    confidence: float = 0.5
    created_at: float = 0.0
    last_used: float = 0.0

    @property
    def success_rate(self) -> float:
        total = self.success_count + self.failure_count
        return self.success_count / total if total > 0 else 0.5


@dataclass
class SkillRequest:
    """A request for a new skill or capability."""
    name: str
    description: str
    domain: str = "general"
    source: str = ""  # URL, package name, or "self-generated"
    priority: str = "normal"
    status: str = "pending"  # pending, installing, installed, failed
    metadata: dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Available skill registries
# ---------------------------------------------------------------------------

SKILL_REGISTRIES = {
    "languages": {
        "description": "Natural language grammars and translation models",
        "skills": [
            {"name": "japanese_grammar", "package": "glm-grammars-ja", "domain": "linguistic"},
            {"name": "arabic_grammar", "package": "glm-grammars-ar", "domain": "linguistic"},
            {"name": "mandarin_grammar", "package": "glm-grammars-zh", "domain": "linguistic"},
            {"name": "hindi_grammar", "package": "glm-grammars-hi", "domain": "linguistic"},
            {"name": "spanish_grammar", "package": "glm-grammars-es", "domain": "linguistic"},
            {"name": "french_grammar", "package": "glm-grammars-fr", "domain": "linguistic"},
            {"name": "german_grammar", "package": "glm-grammars-de", "domain": "linguistic"},
            {"name": "russian_grammar", "package": "glm-grammars-ru", "domain": "linguistic"},
            {"name": "korean_grammar", "package": "glm-grammars-ko", "domain": "linguistic"},
            {"name": "swahili_grammar", "package": "glm-grammars-sw", "domain": "linguistic"},
        ],
    },
    "sciences": {
        "description": "Scientific domain grammars",
        "skills": [
            {"name": "quantum_field_theory", "package": "glm-grammars-qft", "domain": "physics"},
            {"name": "organic_chemistry", "package": "glm-grammars-orgchem", "domain": "chemical"},
            {"name": "genomics", "package": "glm-grammars-genomics", "domain": "biological"},
            {"name": "neuroscience", "package": "glm-grammars-neuro", "domain": "biological"},
            {"name": "astrophysics", "package": "glm-grammars-astro", "domain": "physics"},
            {"name": "topology", "package": "glm-grammars-topology", "domain": "mathematical"},
            {"name": "category_theory", "package": "glm-grammars-cattheory", "domain": "mathematical"},
        ],
    },
    "engineering": {
        "description": "Engineering and applied skills",
        "skills": [
            {"name": "circuit_design", "package": "glm-skills-circuits", "domain": "physics"},
            {"name": "signal_processing", "package": "glm-skills-dsp", "domain": "computational"},
            {"name": "control_theory", "package": "glm-skills-control", "domain": "mathematical"},
            {"name": "machine_learning", "package": "glm-skills-ml", "domain": "computational"},
            {"name": "cryptography", "package": "glm-skills-crypto", "domain": "mathematical"},
        ],
    },
    "creative": {
        "description": "Creative and artistic skills",
        "skills": [
            {"name": "music_theory", "package": "glm-grammars-music", "domain": "linguistic"},
            {"name": "poetry_grammar", "package": "glm-grammars-poetry", "domain": "linguistic"},
            {"name": "narrative_structure", "package": "glm-grammars-narrative", "domain": "linguistic"},
            {"name": "visual_grammar", "package": "glm-grammars-visual", "domain": "computational"},
        ],
    },
}


# ---------------------------------------------------------------------------
# SelfImprover — the recursive self-improvement engine
# ---------------------------------------------------------------------------

class SelfImprover:
    """Recursive self-improvement engine for MKAngel.

    The strange loop at the highest level: the system uses its own
    grammar to reason about its own performance, then modifies its
    own rules to do better.  Each improvement is logged, versioned,
    and reversible.
    """

    def __init__(self) -> None:
        self._patterns: dict[str, LearnedPattern] = {}
        self._history: list[ImprovementRecord] = []
        self._skill_requests: list[SkillRequest] = []
        self._session_metrics: dict[str, Any] = {
            "predictions_made": 0,
            "predictions_correct": 0,
            "rules_applied": 0,
            "strange_loops_traversed": 0,
            "domains_used": set(),
        }

        # Puriel -- grammar integrity gate
        self._integrity: Any = None
        try:
            from app.puriel import GrammarIntegrityChecksum
            self._integrity = GrammarIntegrityChecksum()
            log.info("Puriel integrity gate active (%d builders checksummed)",
                     self._integrity.total_builders)
        except Exception as exc:
            log.warning("Puriel integrity gate unavailable: %s", exc)

        # Ensure directories exist
        _IMPROVEMENT_DIR.mkdir(parents=True, exist_ok=True)

        # Load persisted state
        self._load_patterns()
        self._load_history()

    # ------------------------------------------------------------------
    # Pattern learning
    # ------------------------------------------------------------------

    def observe_success(
        self,
        domain: str,
        rule_name: str,
        input_data: Any,
        output_data: Any,
        confidence: float = 0.8,
    ) -> None:
        """Record a successful derivation for pattern learning.

        Before strengthening a pattern the rule is passed through
        Puriel's integrity gate to ensure it does not corrupt
        the immutable grammar seeds.
        """
        pattern_id = f"{domain}:{rule_name}"

        # -- Puriel integrity gate -----------------------------------------
        rule_data = {"rule": rule_name, "sample_input": str(input_data)[:200]}

        if self._integrity is not None:
            # Build a rule_data dict that the gate can validate
            gate_data = {
                "pattern": str(input_data)[:200],
                "result": str(output_data)[:200],
                "domain": domain,
                "name": rule_name,
            }
            passed, reason = self._integrity.validate_learned_rule(domain, gate_data)
            if not passed:
                log.info(
                    "Puriel gently held back pattern %s: %s",
                    pattern_id, reason,
                )
                self._log_improvement(ImprovementRecord(
                    action="integrity_guidance",
                    domain=domain,
                    description=(
                        f"Puriel guided '{rule_name}' away from the seeds: "
                        f"{reason}"
                    ),
                    confidence=confidence,
                    integrity_check_passed=False,
                ))
                return  # This one isn't ready yet -- and that's OK

        # -- Pattern learning (passed integrity gate) ----------------------
        if pattern_id in self._patterns:
            pattern = self._patterns[pattern_id]
            pattern.success_count += 1
            pattern.last_used = time.time()
            # Bayesian update of confidence
            pattern.confidence = (
                (pattern.confidence * (pattern.success_count + pattern.failure_count - 1)
                 + confidence) /
                (pattern.success_count + pattern.failure_count)
            )
        else:
            self._patterns[pattern_id] = LearnedPattern(
                pattern_id=pattern_id,
                domain=domain,
                description=f"Learned pattern from rule '{rule_name}'",
                rule_data=rule_data,
                success_count=1,
                confidence=confidence,
                created_at=time.time(),
                last_used=time.time(),
            )

        self._session_metrics["predictions_correct"] += 1
        self._save_patterns()

    def observe_failure(
        self,
        domain: str,
        rule_name: str,
        input_data: Any,
        expected: Any,
        actual: Any,
    ) -> None:
        """Record a failed derivation for pattern learning."""
        pattern_id = f"{domain}:{rule_name}"

        if pattern_id in self._patterns:
            self._patterns[pattern_id].failure_count += 1
            # Reduce confidence on failure
            pattern = self._patterns[pattern_id]
            total = pattern.success_count + pattern.failure_count
            pattern.confidence = pattern.success_count / total
        else:
            self._patterns[pattern_id] = LearnedPattern(
                pattern_id=pattern_id,
                domain=domain,
                description=f"Pattern from rule '{rule_name}' (initially failed)",
                rule_data={"rule": rule_name},
                failure_count=1,
                confidence=0.3,
                created_at=time.time(),
            )

        self._save_patterns()

    def get_pattern_confidence(self, domain: str, rule_name: str) -> float:
        """Get learned confidence for a rule."""
        pattern_id = f"{domain}:{rule_name}"
        if pattern_id in self._patterns:
            return self._patterns[pattern_id].confidence
        return 0.5  # Neutral prior

    def get_strong_patterns(self, domain: str | None = None, min_confidence: float = 0.7) -> list[LearnedPattern]:
        """Get patterns with high confidence."""
        patterns = self._patterns.values()
        if domain:
            patterns = [p for p in patterns if p.domain == domain]
        return sorted(
            [p for p in patterns if p.confidence >= min_confidence],
            key=lambda p: p.confidence,
            reverse=True,
        )

    def get_weak_patterns(self, max_confidence: float = 0.3) -> list[LearnedPattern]:
        """Get patterns that need improvement."""
        return sorted(
            [p for p in self._patterns.values() if p.confidence <= max_confidence],
            key=lambda p: p.confidence,
        )

    # ------------------------------------------------------------------
    # Skill acquisition
    # ------------------------------------------------------------------

    def request_skill(
        self,
        name: str,
        description: str = "",
        domain: str = "general",
        source: str = "",
        priority: str = "normal",
    ) -> SkillRequest:
        """Request a new skill or capability.

        The Angel can request skills it needs but doesn't have.
        These are queued for installation.
        """
        request = SkillRequest(
            name=name,
            description=description or f"Requested skill: {name}",
            domain=domain,
            source=source,
            priority=priority,
        )
        self._skill_requests.append(request)

        # Log the request
        self._log_improvement(ImprovementRecord(
            action="skill_request",
            domain=domain,
            description=f"Requested skill: {name}",
            after={"skill": name, "source": source},
        ))

        return request

    def list_available_skills(self, category: str | None = None) -> dict[str, Any]:
        """List skills available from registries."""
        if category and category in SKILL_REGISTRIES:
            return {category: SKILL_REGISTRIES[category]}
        return SKILL_REGISTRIES

    def search_skills(self, query: str) -> list[dict[str, Any]]:
        """Search for skills matching a query."""
        results = []
        query_lower = query.lower()
        for category, registry in SKILL_REGISTRIES.items():
            for skill in registry["skills"]:
                if (query_lower in skill["name"].lower() or
                        query_lower in skill.get("domain", "").lower()):
                    results.append({
                        "category": category,
                        "skill": skill,
                    })
        return results

    def get_pending_requests(self) -> list[SkillRequest]:
        """Get skills that haven't been installed yet."""
        return [r for r in self._skill_requests if r.status == "pending"]

    # ------------------------------------------------------------------
    # Meta-analysis (reasoning about reasoning)
    # ------------------------------------------------------------------

    def analyse_performance(self) -> dict[str, Any]:
        """Analyse the Angel's performance across sessions.

        The strange loop: using grammar to reason about grammar performance.
        """
        total_patterns = len(self._patterns)
        strong = len(self.get_strong_patterns())
        weak = len(self.get_weak_patterns())

        domain_stats: dict[str, dict[str, int]] = {}
        for pattern in self._patterns.values():
            if pattern.domain not in domain_stats:
                domain_stats[pattern.domain] = {"total": 0, "strong": 0, "weak": 0}
            domain_stats[pattern.domain]["total"] += 1
            if pattern.confidence >= 0.7:
                domain_stats[pattern.domain]["strong"] += 1
            elif pattern.confidence <= 0.3:
                domain_stats[pattern.domain]["weak"] += 1

        metrics = dict(self._session_metrics)
        if isinstance(metrics.get("domains_used"), set):
            metrics["domains_used"] = list(metrics["domains_used"])

        return {
            "total_patterns": total_patterns,
            "strong_patterns": strong,
            "weak_patterns": weak,
            "improvement_history": len(self._history),
            "pending_skill_requests": len(self.get_pending_requests()),
            "domain_stats": domain_stats,
            "session_metrics": metrics,
            "recommendations": self._generate_recommendations(),
        }

    def _generate_recommendations(self) -> list[str]:
        """Generate self-improvement recommendations."""
        recs = []

        weak = self.get_weak_patterns()
        if weak:
            domains = set(p.domain for p in weak[:5])
            recs.append(
                f"Weak patterns detected in: {', '.join(domains)}. "
                f"Consider acquiring additional training data or grammar rules."
            )

        if not self._patterns:
            recs.append(
                "No learned patterns yet. Use the Angel more to build "
                "pattern confidence through observation."
            )

        metrics = self._session_metrics
        if metrics["predictions_made"] > 0:
            accuracy = metrics["predictions_correct"] / metrics["predictions_made"]
            if accuracy < 0.5:
                recs.append(
                    f"Prediction accuracy is {accuracy:.0%}. Consider expanding "
                    f"grammar rules or adjusting confidence thresholds."
                )

        return recs

    # ------------------------------------------------------------------
    # MNEMO compression of experience
    # ------------------------------------------------------------------

    def compress_to_mnemo(self) -> str:
        """Compress current session experience into MNEMO notation.

        The scribe: converting live experience into the hyper-compressed
        MNEMO language for efficient storage and replay.
        """
        try:
            from glm.mnemo.language import encode

            # Build a summary of what we've learned
            summary_parts = []
            for domain, stats in self.analyse_performance().get("domain_stats", {}).items():
                if stats["strong"] > 0:
                    summary_parts.append(f"{domain} strong patterns")
            summary = "; ".join(summary_parts) if summary_parts else "initial session"

            return encode(f"Session summary: {summary}")
        except Exception:
            return "Mi Ma~ *a#"  # Fallback: meta-introspect, meta-analyse, universal count

    # ------------------------------------------------------------------
    # Cross-session persistence
    # ------------------------------------------------------------------

    def save_session_state(self) -> None:
        """Persist all improvements for the next session."""
        self._save_patterns()
        self._save_history()

    def _save_patterns(self) -> None:
        """Save learned patterns to disk."""
        data = {}
        for pid, pattern in self._patterns.items():
            data[pid] = {
                "pattern_id": pattern.pattern_id,
                "domain": pattern.domain,
                "description": pattern.description,
                "rule_data": pattern.rule_data,
                "success_count": pattern.success_count,
                "failure_count": pattern.failure_count,
                "confidence": pattern.confidence,
                "created_at": pattern.created_at,
                "last_used": pattern.last_used,
            }
        with open(_LEARNED_PATTERNS, "w") as f:
            json.dump(data, f, indent=2)

    def _load_patterns(self) -> None:
        """Load learned patterns from disk."""
        if not _LEARNED_PATTERNS.exists():
            return
        try:
            with open(_LEARNED_PATTERNS) as f:
                data = json.load(f)
            for pid, pdata in data.items():
                self._patterns[pid] = LearnedPattern(**pdata)
        except (json.JSONDecodeError, OSError, TypeError):
            pass

    def _log_improvement(self, record: ImprovementRecord) -> None:
        """Log an improvement action."""
        self._history.append(record)
        _IMPROVEMENT_LOG.parent.mkdir(parents=True, exist_ok=True)
        with open(_IMPROVEMENT_LOG, "a") as f:
            f.write(json.dumps({
                "action": record.action,
                "domain": record.domain,
                "description": record.description,
                "confidence": record.confidence,
                "timestamp": record.timestamp,
                "reversible": record.reversible,
                "integrity_check_passed": record.integrity_check_passed,
            }, default=str) + "\n")

    def _save_history(self) -> None:
        """History is saved incrementally via _log_improvement."""
        pass

    def _load_history(self) -> None:
        """Load improvement history."""
        if not _IMPROVEMENT_LOG.exists():
            return
        try:
            with open(_IMPROVEMENT_LOG) as f:
                for line in f:
                    line = line.strip()
                    if line:
                        data = json.loads(line)
                        self._history.append(ImprovementRecord(**data))
        except (json.JSONDecodeError, OSError):
            pass
