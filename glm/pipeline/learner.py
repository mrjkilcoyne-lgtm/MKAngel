"""
learner.py -- Self-improvement engine for the GSM reasoning pipeline.

The GSM's LEARN and SLEEP cycles analyse pipeline output and generate
patches to the grammar rules themselves -- not prompt text, but actual
structural changes to the grammar engine.

    LEARN (per-session):
        Analyses a single pipeline run's output.  Identifies which grammar
        rules fired, which failed, and which were missing.  Generates
        patches: weight adjustments, new rules, pattern refinements.

    SLEEP (cross-session):
        Consolidates patterns across multiple LEARN sessions.  Identifies
        persistent weaknesses and produces system-level grammar upgrades:
        new productions, refined loop structures, cross-domain connections.

The key insight: because MKAngel's reasoning IS grammar operations, self-
improvement means improving the grammar.  A weight adjustment on a rule
directly changes how the pipeline reasons.  A new production directly
adds a new reasoning capability.  This is not prompt engineering -- it
is structural evolution of the reasoning engine itself.
"""

from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple

from glm.core.grammar import Direction, Grammar, Production, Rule, StrangeLoop
from glm.core.engine import DerivationEngine

from .result import (
    DisconfirmResult,
    EvidenceLevel,
    FallacyType,
    PipelineResult,
    SkeletonResult,
    SynthesisResult,
    Triple,
)


# ---------------------------------------------------------------------------
# Patch -- a proposed change to a grammar
# ---------------------------------------------------------------------------

@dataclass
class GrammarPatch:
    """A proposed modification to a grammar rule or production.

    Patches are the output of the LEARN and SLEEP cycles.  They describe
    a specific change to apply to the grammar engine.

    Patch types:
        weight_adjust:   Change a rule's confidence weight.
        new_rule:        Add a new Rule to a grammar.
        new_production:  Add a new Production to a grammar.
        pattern_refine:  Modify a rule's pattern or result.
        deprecate:       Mark a rule for removal (set weight to 0).
        loop_modify:     Add or modify a StrangeLoop.

    Attributes:
        id:            Unique patch identifier.
        patch_type:    Type of modification.
        target_grammar: Name of the grammar to patch.
        target_rule_id: ID of the rule to modify (for existing rules).
        description:   Human-readable explanation of why this patch helps.
        payload:       The actual change data (type-dependent).
        confidence:    How confident the learner is this patch helps [0, 1].
        source_session: ID of the pipeline run that motivated this patch.
    """
    patch_type: str
    target_grammar: str
    description: str
    payload: Dict[str, Any] = field(default_factory=dict)
    target_rule_id: str = ""
    confidence: float = 0.5
    source_session: str = ""
    id: str = field(default_factory=lambda: uuid.uuid4().hex[:10])

    def __repr__(self) -> str:
        return (
            f"GrammarPatch({self.patch_type}, "
            f"grammar={self.target_grammar!r}, "
            f"confidence={self.confidence:.2f})"
        )


# ---------------------------------------------------------------------------
# LearnCycle -- per-session analysis and patching
# ---------------------------------------------------------------------------

class LearnCycle:
    """Analyse a single pipeline run and generate grammar patches.

    The LEARN cycle inspects the full PipelineResult to identify:

    1. **Underperforming rules** -- rules that fired but contributed to
       weak or disconfirmed claims.  Patch: reduce their weight.

    2. **Missing rules** -- claims that fell through to heuristic
       extraction (low grammar coverage).  Patch: propose new rules
       that would have caught them.

    3. **Overconfident rules** -- rules with high weight that produced
       claims subsequently disconfirmed.  Patch: reduce weight.

    4. **Structural gaps** -- cycles in the DAG that correspond to
       missing strange loops in the grammar.  Patch: add loops.

    5. **Cross-domain misses** -- isomorphisms that should exist but
       were not found.  Patch: add bridging productions.

    The learner does NOT apply patches automatically.  It produces a
    list of GrammarPatch objects that can be reviewed and applied.

    Attributes:
        engine:     Shared DerivationEngine for analysis.
        grammars:   The grammars being analysed.
        patches:    Generated patches from the most recent analysis.
        history:    All patches generated across calls.
    """

    def __init__(
        self,
        engine: Optional[DerivationEngine] = None,
        grammars: Optional[Dict[str, Grammar]] = None,
    ) -> None:
        self.engine = engine or DerivationEngine()
        self.grammars = grammars or {}
        self.patches: List[GrammarPatch] = []
        self.history: List[GrammarPatch] = []

    def analyse(self, result: PipelineResult) -> List[GrammarPatch]:
        """Analyse a pipeline result and generate grammar patches.

        Parameters:
            result: A complete PipelineResult from a pipeline run.

        Returns:
            List of GrammarPatch objects.  These are proposed changes,
            not yet applied.
        """
        self.patches = []
        session_id = result.run_id

        # Analysis 1: Underperforming rules (low-confidence claims).
        if result.skeleton:
            self._analyse_skeleton(result.skeleton, session_id)

        # Analysis 2: Structural issues from DAG.
        if result.dag:
            self._analyse_dag_structure(result.dag, session_id)

        # Analysis 3: Disconfirmation feedback (rules that led to weakness).
        if result.disconfirm:
            self._analyse_disconfirmation(result.disconfirm, session_id)

        # Analysis 4: Synthesis gaps (unproven claims suggest missing rules).
        if result.synthesis:
            self._analyse_synthesis_gaps(result.synthesis, session_id)

        # Analysis 5: Coverage gaps (low grammar_coverage in skeleton).
        if result.skeleton and result.skeleton.grammar_coverage < 0.5:
            self._propose_coverage_rules(result.skeleton, session_id)

        self.history.extend(self.patches)
        return self.patches

    # -- internal: skeleton analysis ----------------------------------------

    def _analyse_skeleton(
        self,
        skeleton: SkeletonResult,
        session_id: str,
    ) -> None:
        """Identify rules that should fire better during extraction."""
        for triple in skeleton.triples:
            if triple.confidence < 0.4 and triple.source_rule:
                # The rule that extracted this triple has low confidence.
                # Suggest refining it.
                self.patches.append(GrammarPatch(
                    patch_type="pattern_refine",
                    target_grammar="syntactic",
                    target_rule_id=triple.source_rule,
                    description=(
                        f"Rule '{triple.source_rule}' produced a low-confidence "
                        f"triple ({triple.confidence:.2f}). Consider refining its "
                        f"pattern to better match '{triple.raw_text}'."
                    ),
                    payload={
                        "current_confidence": triple.confidence,
                        "example_text": triple.raw_text,
                        "triple": triple.as_tuple,
                    },
                    confidence=0.4,
                    source_session=session_id,
                ))

        # Check noise -- if many clauses became noise, the grammar is missing rules.
        if len(skeleton.noise) > len(skeleton.triples):
            self.patches.append(GrammarPatch(
                patch_type="new_rule",
                target_grammar="syntactic",
                description=(
                    f"High noise ratio ({len(skeleton.noise)} noise vs "
                    f"{len(skeleton.triples)} triples). The syntactic grammar "
                    f"is missing rules for these clause patterns."
                ),
                payload={
                    "noise_samples": skeleton.noise[:5],
                    "noise_count": len(skeleton.noise),
                    "triple_count": len(skeleton.triples),
                },
                confidence=0.6,
                source_session=session_id,
            ))

    # -- internal: DAG structure analysis -----------------------------------

    def _analyse_dag_structure(
        self,
        dag: Any,
        session_id: str,
    ) -> None:
        """Identify structural issues in the dependency graph."""
        # Cycles suggest grammar should model these as strange loops.
        for cycle in dag.cycles:
            self.patches.append(GrammarPatch(
                patch_type="loop_modify",
                target_grammar="syntactic",
                description=(
                    f"Dependency cycle of length {len(cycle)} detected. "
                    f"Consider adding a StrangeLoop to the grammar to "
                    f"model this cyclic reasoning pattern explicitly."
                ),
                payload={
                    "cycle_node_ids": cycle,
                    "cycle_length": len(cycle),
                },
                confidence=0.5,
                source_session=session_id,
            ))

        # Orphans suggest missing productions that would connect them.
        if dag.orphans:
            self.patches.append(GrammarPatch(
                patch_type="new_production",
                target_grammar="syntactic",
                description=(
                    f"{len(dag.orphans)} orphan nodes found -- claims that "
                    f"are disconnected from the argument. Consider adding "
                    f"productions that connect these to the main graph."
                ),
                payload={
                    "orphan_count": len(dag.orphans),
                    "orphan_ids": dag.orphans,
                },
                confidence=0.4,
                source_session=session_id,
            ))

    # -- internal: disconfirmation feedback ---------------------------------

    def _analyse_disconfirmation(
        self,
        disconfirm: DisconfirmResult,
        session_id: str,
    ) -> None:
        """Generate patches based on disconfirmation results."""
        for weakness in disconfirm.weaknesses:
            if weakness.fallacy == FallacyType.CIRCULAR:
                # Circular reasoning detected -- the grammar's loop detection
                # should model this.
                self.patches.append(GrammarPatch(
                    patch_type="weight_adjust",
                    target_grammar="syntactic",
                    description=(
                        f"Circular reasoning detected involving claim "
                        f"'{weakness.claim.subject if weakness.claim else '?'} "
                        f"{weakness.claim.relation if weakness.claim else '?'}'. "
                        f"Reduce weight of rules that enabled this cycle."
                    ),
                    payload={
                        "adjustment": -0.1,
                        "reason": "circular_reasoning",
                        "backward_trace": weakness.backward_trace,
                    },
                    confidence=0.6,
                    source_session=session_id,
                ))

            elif weakness.weakness_type == "weak_premise":
                # A premise was too weak -- the rule that produced it
                # needs its weight lowered.
                if weakness.claim and weakness.claim.source_rule:
                    self.patches.append(GrammarPatch(
                        patch_type="weight_adjust",
                        target_grammar="syntactic",
                        target_rule_id=weakness.claim.source_rule,
                        description=(
                            f"Weak premise '{weakness.claim.subject} "
                            f"{weakness.claim.relation}' -- reduce rule weight."
                        ),
                        payload={
                            "adjustment": -0.05,
                            "current_confidence": weakness.claim.confidence,
                            "reason": "weak_premise",
                        },
                        confidence=0.5,
                        source_session=session_id,
                    ))

        # Overconfident rules: claims that were survived but had weaknesses.
        for triple in disconfirm.weakened_claims:
            if triple.confidence > 0.6 and triple.source_rule:
                self.patches.append(GrammarPatch(
                    patch_type="weight_adjust",
                    target_grammar="syntactic",
                    target_rule_id=triple.source_rule,
                    description=(
                        f"Rule '{triple.source_rule}' was overconfident: "
                        f"produced high-confidence claim that was later "
                        f"weakened. Reduce weight for calibration."
                    ),
                    payload={
                        "adjustment": -0.08,
                        "original_confidence": triple.confidence,
                        "reason": "overconfident",
                    },
                    confidence=0.55,
                    source_session=session_id,
                ))

    # -- internal: synthesis gap analysis -----------------------------------

    def _analyse_synthesis_gaps(
        self,
        synthesis: SynthesisResult,
        session_id: str,
    ) -> None:
        """Generate patches for claims that couldn't be proven."""
        for node in synthesis.unproven:
            if node.claim is None:
                continue
            if node.level == EvidenceLevel.UNSUBSTANTIATED:
                # No derivation support -- suggest a new rule.
                self.patches.append(GrammarPatch(
                    patch_type="new_rule",
                    target_grammar="syntactic",
                    description=(
                        f"Claim '{node.claim.subject} {node.claim.relation} "
                        f"{node.claim.object}' could not be derived. "
                        f"Consider adding a rule to support this pattern."
                    ),
                    payload={
                        "claim": node.claim.as_tuple,
                        "evidence_level": node.level.value,
                        "suggested_pattern": node.claim.relation,
                    },
                    confidence=0.35,
                    source_session=session_id,
                ))

    # -- internal: coverage improvement rules --------------------------------

    def _propose_coverage_rules(
        self,
        skeleton: SkeletonResult,
        session_id: str,
    ) -> None:
        """Propose rules to improve grammar coverage of input text."""
        # Analyse noise segments to find common patterns.
        if not skeleton.noise:
            return

        # Look for repeated words or structures in noise.
        word_freq: Dict[str, int] = {}
        for noise_seg in skeleton.noise:
            for word in noise_seg.lower().split():
                if len(word) > 3:  # Skip short words.
                    word_freq[word] = word_freq.get(word, 0) + 1

        # Most frequent uncaptured words suggest missing vocabulary.
        frequent = sorted(word_freq.items(), key=lambda x: -x[1])[:5]

        if frequent:
            self.patches.append(GrammarPatch(
                patch_type="new_rule",
                target_grammar="morphological",
                description=(
                    f"Low grammar coverage ({skeleton.grammar_coverage:.1%}). "
                    f"Frequent uncaptured tokens suggest missing morphological "
                    f"rules for: {', '.join(w for w, _ in frequent)}."
                ),
                payload={
                    "frequent_uncaptured": frequent,
                    "coverage": skeleton.grammar_coverage,
                },
                confidence=0.45,
                source_session=session_id,
            ))

    # -- patch application --------------------------------------------------

    def apply_patches(
        self,
        patches: List[GrammarPatch],
        grammars: Dict[str, Grammar],
        min_confidence: float = 0.4,
    ) -> List[GrammarPatch]:
        """Apply a list of patches to grammars.

        Only patches with confidence >= min_confidence are applied.

        Parameters:
            patches:        Patches to apply.
            grammars:       The grammar dict to modify in place.
            min_confidence: Minimum patch confidence to apply.

        Returns:
            List of patches that were actually applied.
        """
        applied: List[GrammarPatch] = []

        for patch in patches:
            if patch.confidence < min_confidence:
                continue

            grammar = grammars.get(patch.target_grammar)
            if grammar is None:
                continue

            success = self._apply_single_patch(patch, grammar)
            if success:
                applied.append(patch)

        return applied

    def _apply_single_patch(
        self,
        patch: GrammarPatch,
        grammar: Grammar,
    ) -> bool:
        """Apply a single patch to a grammar. Returns True if applied."""
        try:
            if patch.patch_type == "weight_adjust":
                return self._apply_weight_adjust(patch, grammar)
            elif patch.patch_type == "new_rule":
                return self._apply_new_rule(patch, grammar)
            elif patch.patch_type == "new_production":
                return self._apply_new_production(patch, grammar)
            elif patch.patch_type == "deprecate":
                return self._apply_deprecate(patch, grammar)
            elif patch.patch_type == "loop_modify":
                return self._apply_loop_modify(patch, grammar)
            elif patch.patch_type == "pattern_refine":
                # Pattern refinement requires manual review -- log it but
                # don't auto-apply.
                return False
            else:
                return False
        except Exception:
            return False

    @staticmethod
    def _apply_weight_adjust(patch: GrammarPatch, grammar: Grammar) -> bool:
        """Adjust the weight of an existing rule."""
        adjustment = patch.payload.get("adjustment", 0.0)
        if not adjustment:
            return False

        if patch.target_rule_id:
            # Target specific rule.
            rule_idx = grammar.rule_index()
            rule = rule_idx.get(patch.target_rule_id)
            if rule:
                rule.weight = max(0.01, min(1.0, rule.weight + adjustment))
                return True
        else:
            # Adjust all rules matching the trace.
            trace = patch.payload.get("backward_trace", [])
            if trace:
                rule_idx = grammar.rule_index()
                for rid in trace:
                    rule = rule_idx.get(rid)
                    if rule:
                        rule.weight = max(0.01, min(1.0, rule.weight + adjustment))
                return True

        return False

    @staticmethod
    def _apply_new_rule(patch: GrammarPatch, grammar: Grammar) -> bool:
        """Add a new rule to the grammar."""
        pattern = patch.payload.get("suggested_pattern", "")
        if not pattern:
            return False

        new_rule = Rule(
            name=f"learned_{patch.id}",
            pattern=pattern,
            result={"learned": True, "source": patch.source_session},
            weight=0.3,  # Start conservative.
            metadata={"patch_id": patch.id, "auto_generated": True},
        )
        grammar.add_rule(new_rule)
        return True

    @staticmethod
    def _apply_new_production(patch: GrammarPatch, grammar: Grammar) -> bool:
        """Add a new production to the grammar."""
        # For now, new productions require more context than patches provide.
        # This is a placeholder for the SLEEP cycle's deeper analysis.
        return False

    @staticmethod
    def _apply_deprecate(patch: GrammarPatch, grammar: Grammar) -> bool:
        """Deprecate a rule by setting its weight to near-zero."""
        if not patch.target_rule_id:
            return False

        rule_idx = grammar.rule_index()
        rule = rule_idx.get(patch.target_rule_id)
        if rule:
            rule.weight = 0.01
            rule.metadata["deprecated"] = True
            rule.metadata["deprecation_reason"] = patch.description
            return True
        return False

    @staticmethod
    def _apply_loop_modify(patch: GrammarPatch, grammar: Grammar) -> bool:
        """Add or modify a StrangeLoop in the grammar."""
        cycle = patch.payload.get("cycle_node_ids", [])
        if not cycle:
            return False

        new_loop = StrangeLoop(
            cycle=cycle,
            entry=cycle[0] if cycle else None,
            level_delta=0,
            grammar_id=grammar.id,
        )
        grammar.strange_loops.append(new_loop)
        return True


# ---------------------------------------------------------------------------
# SleepCycle -- cross-session consolidation and system upgrades
# ---------------------------------------------------------------------------

class SleepCycle:
    """Consolidate patterns across sessions and produce system-level upgrades.

    The SLEEP cycle operates on the accumulated history of LEARN patches
    and pipeline results.  It identifies persistent patterns and generates
    deeper structural changes to the grammar:

    1. **Weight convergence** -- if the same rule has been adjusted in the
       same direction across multiple sessions, commit the aggregate change.

    2. **New production discovery** -- if multiple sessions suggest new
       rules for the same pattern, distill them into a proper Production.

    3. **Cross-domain bridge** -- if isomorphisms appear consistently,
       create explicit cross-domain productions.

    4. **Loop consolidation** -- if the same cyclic pattern appears in
       multiple sessions, promote it to a named StrangeLoop with
       accurate level_delta.

    5. **Rule pruning** -- rules that never fire across multiple sessions
       are candidates for removal.

    The SLEEP cycle is designed to run periodically (e.g., daily, or
    after N pipeline runs), not after every run.

    Attributes:
        learner:         The LearnCycle whose history to consolidate.
        session_results: Accumulated PipelineResults for analysis.
        upgrades:        Generated system-level upgrade patches.
    """

    def __init__(
        self,
        learner: Optional[LearnCycle] = None,
    ) -> None:
        self.learner = learner or LearnCycle()
        self.session_results: List[PipelineResult] = []
        self.upgrades: List[GrammarPatch] = []
        self._consolidation_count = 0

    def record_session(self, result: PipelineResult) -> None:
        """Record a pipeline result for future consolidation."""
        self.session_results.append(result)

    def consolidate(
        self,
        min_sessions: int = 3,
    ) -> List[GrammarPatch]:
        """Run the SLEEP consolidation cycle.

        Analyses all accumulated LEARN patches and session results to
        produce system-level grammar upgrades.

        Parameters:
            min_sessions: Minimum number of sessions before consolidation
                          makes sense.  Fewer sessions lack the statistical
                          power to distinguish persistent patterns from noise.

        Returns:
            List of system-level GrammarPatch objects.
        """
        self.upgrades = []
        self._consolidation_count += 1

        if len(self.session_results) < min_sessions:
            return self.upgrades

        # Strategy 1: Weight convergence.
        self._consolidate_weight_adjustments()

        # Strategy 2: New production discovery.
        self._discover_new_productions()

        # Strategy 3: Loop consolidation.
        self._consolidate_loops()

        # Strategy 4: Rule pruning candidates.
        self._identify_prune_candidates()

        # Strategy 5: Cross-domain bridges.
        self._identify_cross_domain_bridges()

        return self.upgrades

    # -- internal: weight convergence ---------------------------------------

    def _consolidate_weight_adjustments(self) -> None:
        """Aggregate weight adjustments for rules adjusted multiple times."""
        # Collect all weight_adjust patches grouped by target rule.
        rule_adjustments: Dict[str, List[float]] = {}

        for patch in self.learner.history:
            if patch.patch_type == "weight_adjust" and patch.target_rule_id:
                adj = patch.payload.get("adjustment", 0.0)
                rule_adjustments.setdefault(
                    patch.target_rule_id, []
                ).append(adj)

        # For rules adjusted consistently in the same direction, commit.
        for rule_id, adjustments in rule_adjustments.items():
            if len(adjustments) < 2:
                continue

            avg_adj = sum(adjustments) / len(adjustments)
            # Check direction consistency.
            all_negative = all(a < 0 for a in adjustments)
            all_positive = all(a > 0 for a in adjustments)

            if all_negative or all_positive:
                self.upgrades.append(GrammarPatch(
                    patch_type="weight_adjust",
                    target_grammar="syntactic",
                    target_rule_id=rule_id,
                    description=(
                        f"Consistent weight adjustment across "
                        f"{len(adjustments)} sessions. "
                        f"Average adjustment: {avg_adj:+.3f}."
                    ),
                    payload={
                        "adjustment": avg_adj,
                        "session_count": len(adjustments),
                        "reason": "sleep_consolidation",
                    },
                    confidence=min(0.9, 0.5 + 0.1 * len(adjustments)),
                    source_session=f"sleep_{self._consolidation_count}",
                ))

    # -- internal: new production discovery ---------------------------------

    def _discover_new_productions(self) -> None:
        """Look for patterns in new_rule patches that suggest productions."""
        # Group new_rule patches by their suggested pattern.
        pattern_groups: Dict[str, List[GrammarPatch]] = {}

        for patch in self.learner.history:
            if patch.patch_type == "new_rule":
                pattern = str(patch.payload.get("suggested_pattern", ""))
                if pattern:
                    pattern_groups.setdefault(pattern, []).append(patch)

        # Patterns suggested across multiple sessions deserve a production.
        for pattern, patches in pattern_groups.items():
            if len(patches) < 2:
                continue

            # Collect example claims.
            examples = []
            for p in patches:
                claim = p.payload.get("claim", ())
                if claim:
                    examples.append(claim)

            self.upgrades.append(GrammarPatch(
                patch_type="new_production",
                target_grammar="syntactic",
                description=(
                    f"Pattern '{pattern}' suggested as a new rule in "
                    f"{len(patches)} sessions. Promote to a Production "
                    f"for systematic handling."
                ),
                payload={
                    "pattern": pattern,
                    "session_count": len(patches),
                    "examples": examples[:5],
                    "suggested_lhs": pattern,
                    "suggested_rhs": [ex[2] if len(ex) > 2 else "" for ex in examples[:3]],
                },
                confidence=min(0.85, 0.4 + 0.15 * len(patches)),
                source_session=f"sleep_{self._consolidation_count}",
            ))

    # -- internal: loop consolidation ---------------------------------------

    def _consolidate_loops(self) -> None:
        """Promote recurring cyclic patterns to named StrangeLoops."""
        # Collect all loop_modify patches.
        loop_patches: List[GrammarPatch] = [
            p for p in self.learner.history
            if p.patch_type == "loop_modify"
        ]

        if len(loop_patches) < 2:
            return

        # Group by cycle length (rough similarity metric).
        by_length: Dict[int, List[GrammarPatch]] = {}
        for p in loop_patches:
            length = p.payload.get("cycle_length", 0)
            by_length.setdefault(length, []).append(p)

        for length, patches in by_length.items():
            if len(patches) >= 2:
                self.upgrades.append(GrammarPatch(
                    patch_type="loop_modify",
                    target_grammar="syntactic",
                    description=(
                        f"Cyclic pattern of length {length} detected in "
                        f"{len(patches)} sessions. Promote to a named "
                        f"StrangeLoop for explicit modelling."
                    ),
                    payload={
                        "cycle_length": length,
                        "session_count": len(patches),
                        "representative_cycle": patches[0].payload.get(
                            "cycle_node_ids", []
                        ),
                    },
                    confidence=min(0.8, 0.45 + 0.1 * len(patches)),
                    source_session=f"sleep_{self._consolidation_count}",
                ))

    # -- internal: prune candidates -----------------------------------------

    def _identify_prune_candidates(self) -> None:
        """Identify rules that never fire and are candidates for pruning.

        Strategy: look at all sessions' derivation counts.  If a grammar
        has rules that never contributed to any extraction, they may be
        dead weight.
        """
        # Collect all source_rule IDs that actually fired in skeletons.
        fired_rules: Set[str] = set()
        for result in self.session_results:
            if result.skeleton:
                for triple in result.skeleton.all_claims:
                    if triple.source_rule:
                        fired_rules.add(triple.source_rule)

        # Count sessions with skeleton results.
        sessions_with_skeleton = sum(
            1 for r in self.session_results if r.skeleton
        )

        if sessions_with_skeleton < 3:
            return

        # For each grammar, find rules that never fired.
        for gname, grammar in self.learner.grammars.items():
            for rule in grammar.all_rules():
                if rule.id not in fired_rules:
                    self.upgrades.append(GrammarPatch(
                        patch_type="deprecate",
                        target_grammar=gname,
                        target_rule_id=rule.id,
                        description=(
                            f"Rule '{rule.name or rule.id}' has not fired in "
                            f"{sessions_with_skeleton} sessions. Consider "
                            f"deprecating to reduce grammar complexity."
                        ),
                        payload={
                            "rule_name": rule.name,
                            "current_weight": rule.weight,
                            "sessions_checked": sessions_with_skeleton,
                        },
                        confidence=min(
                            0.6,
                            0.2 + 0.1 * sessions_with_skeleton,
                        ),
                        source_session=f"sleep_{self._consolidation_count}",
                    ))

    # -- internal: cross-domain bridges -------------------------------------

    def _identify_cross_domain_bridges(self) -> None:
        """Identify cross-domain patterns that should be linked.

        If multiple synthesis results show the same grammar domains
        validating claims, suggest explicit bridging productions.
        """
        # Count (domain_a, domain_b) validation co-occurrences.
        domain_pairs: Dict[Tuple[str, str], int] = {}

        for result in self.session_results:
            if result.synthesis:
                for validation in result.synthesis.cross_domain_validation:
                    ga = validation.get("grammar_a", "")
                    gb = validation.get("grammar_b", "")
                    if ga and gb:
                        pair = (min(ga, gb), max(ga, gb))
                        domain_pairs[pair] = domain_pairs.get(pair, 0) + 1

        for (ga, gb), count in domain_pairs.items():
            if count >= 2:
                self.upgrades.append(GrammarPatch(
                    patch_type="new_production",
                    target_grammar=ga,
                    description=(
                        f"Grammars '{ga}' and '{gb}' consistently produce "
                        f"isomorphisms ({count} sessions). Add a bridging "
                        f"production to formalise the cross-domain link."
                    ),
                    payload={
                        "grammar_a": ga,
                        "grammar_b": gb,
                        "co_occurrence_count": count,
                    },
                    confidence=min(0.75, 0.3 + 0.15 * count),
                    source_session=f"sleep_{self._consolidation_count}",
                ))

    # -- state management ---------------------------------------------------

    def clear_history(self) -> None:
        """Clear accumulated session results and learner history."""
        self.session_results.clear()
        self.learner.history.clear()
        self.learner.patches.clear()
        self.upgrades.clear()
