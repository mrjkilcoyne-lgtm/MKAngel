"""
result.py -- Data classes for the GSM reasoning pipeline results.

Each stage of the pipeline (Skeleton, DAG, Disconfirm, Synthesis) produces
a typed result object.  These results flow forward through the pipeline:
each stage consumes the previous stage's output and produces its own.

The final ``PipelineResult`` aggregates all stage outputs into a single
auditable record of the entire reasoning process -- from raw input to
minimum-viable-logic conclusion.

Design principle: every intermediate product is preserved.  The pipeline
never discards evidence; it only refines it.  This makes the reasoning
fully inspectable and replayable.
"""

from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Set, Tuple


# ---------------------------------------------------------------------------
# Enumerations
# ---------------------------------------------------------------------------

class ClaimStrength(Enum):
    """How strongly a claim is supported after skeleton extraction."""
    STRONG = "strong"          # High rule-confidence, clean S-R-O
    MODERATE = "moderate"      # Partial match or hedged language
    WEAK = "weak"              # Low confidence, implicit or inferred
    NOISE = "noise"            # Identified as decoration / filler


class EvidenceLevel(Enum):
    """Hierarchy of evidence quality in the synthesis stage."""
    PROVEN = "proven"              # Derivable from axioms / strong rules
    SUPPORTED = "supported"        # Backed by multiple derivation paths
    PLAUSIBLE = "plausible"        # Single derivation path, moderate weight
    UNSUBSTANTIATED = "unsubstantiated"  # No derivation support found
    CONTRADICTED = "contradicted"  # Actively disconfirmed


class FallacyType(Enum):
    """Structural fallacies detectable via grammar analysis."""
    CIRCULAR = "circular"          # Strange loop with no level delta
    BEGGING_QUESTION = "begging_question"  # Conclusion in premises
    NON_SEQUITUR = "non_sequitur"  # No derivation path between premises
    EQUIVOCATION = "equivocation"  # Same form, different grammar rules
    FALSE_CAUSE = "false_cause"    # Spurious dependency in DAG
    STRAW_MAN = "straw_man"        # Steel-man counter diverges from original


# ---------------------------------------------------------------------------
# Triple -- the atomic unit of structured claims
# ---------------------------------------------------------------------------

@dataclass
class Triple:
    """A Subject-Relation-Object triple extracted from input.

    This is the GSM's atomic claim unit.  The skeleton stage decomposes
    natural language into these triples using syntactic grammar rules
    (S -> NP VP maps to Subject-Relation-Object).

    Attributes:
        subject:     The entity or concept (from NP extraction).
        relation:    The predicate / verb / link (from VP head).
        object:      The target / complement (from VP complement).
        confidence:  Rule-weight-based confidence in extraction [0, 1].
        source_rule: ID of the grammar rule that produced this triple.
        raw_text:    The original text segment this triple was extracted from.
        implicit:    True if this triple was inferred (not stated directly).
    """
    subject: str
    relation: str
    object: str
    confidence: float = 1.0
    source_rule: str = ""
    raw_text: str = ""
    implicit: bool = False

    @property
    def as_tuple(self) -> Tuple[str, str, str]:
        return (self.subject, self.relation, self.object)


# ---------------------------------------------------------------------------
# SkeletonResult -- output of the SKELETON stage
# ---------------------------------------------------------------------------

@dataclass
class SkeletonResult:
    """Result of the Skeleton stage: structural decomposition of input.

    The skeleton strips the input to its logical bones -- S-R-O triples,
    with implicit premises surfaced and noise removed.

    Grammar mapping:
        - Syntactic grammar productions (S -> NP VP) drive triple extraction.
        - Morphological grammar rules identify hedging/decoration for removal.
        - Rule confidence scores weight skeleton strength.

    Attributes:
        triples:          Extracted Subject-Relation-Object claims.
        implicit_premises: Premises not stated but required by grammar rules.
        noise:            Segments identified as decoration (hedging, filler).
        raw_input:        The original input before decomposition.
        derivation_count: How many grammar derivations were explored.
        grammar_coverage: Fraction of input covered by grammar rules [0, 1].
    """
    triples: List[Triple] = field(default_factory=list)
    implicit_premises: List[Triple] = field(default_factory=list)
    noise: List[str] = field(default_factory=list)
    raw_input: Any = None
    derivation_count: int = 0
    grammar_coverage: float = 0.0

    @property
    def all_claims(self) -> List[Triple]:
        """All claims: explicit triples + implicit premises."""
        return self.triples + self.implicit_premises

    @property
    def strong_claims(self) -> List[Triple]:
        """Claims with confidence above 0.7."""
        return [t for t in self.all_claims if t.confidence > 0.7]


# ---------------------------------------------------------------------------
# DAGNode / DAGEdge / DAGResult -- output of the DAG stage
# ---------------------------------------------------------------------------

@dataclass
class DAGNode:
    """A node in the dependency graph.

    Each node represents a claim (triple) from the skeleton.

    Attributes:
        id:          Unique node identifier.
        triple:      The claim this node represents.
        is_root:     True if no other claim depends on this one being true.
        is_leaf:     True if this claim has no premises (terminal/axiom).
        is_orphan:   True if disconnected from the main argument graph.
        depth:       Distance from nearest root node.
    """
    id: str = field(default_factory=lambda: uuid.uuid4().hex[:8])
    triple: Optional[Triple] = None
    is_root: bool = False
    is_leaf: bool = False
    is_orphan: bool = False
    depth: int = 0


@dataclass
class DAGEdge:
    """A directed edge in the dependency graph: source depends on target.

    Grammar mapping: an edge exists when a grammar production can derive
    the source triple's relation from the target triple.

    Attributes:
        source_id:   ID of the dependent node.
        target_id:   ID of the node depended upon.
        edge_type:   Kind of dependency (causal, evidential, definitional).
        weight:      Strength of the dependency (from production weights).
        rule_id:     The grammar production that established this link.
    """
    source_id: str
    target_id: str
    edge_type: str = "dependency"
    weight: float = 1.0
    rule_id: str = ""


@dataclass
class DAGResult:
    """Result of the DAG stage: dependency graph over skeleton claims.

    Grammar mapping:
        - Production references map claim-to-claim dependencies.
        - Grammar.find_loops() detects circular dependencies.
        - Root/leaf detection via graph traversal on the production graph.

    Attributes:
        nodes:          All nodes in the dependency graph.
        edges:          All directed edges (dependencies).
        roots:          IDs of root nodes (conclusions / top-level claims).
        leaves:         IDs of leaf nodes (axioms / base evidence).
        cycles:         Detected circular dependency chains.
        orphans:        IDs of disconnected nodes.
        critical_path:  The longest dependency chain (bottleneck path).
        depth:          Maximum depth of the dependency graph.
    """
    nodes: List[DAGNode] = field(default_factory=list)
    edges: List[DAGEdge] = field(default_factory=list)
    roots: List[str] = field(default_factory=list)
    leaves: List[str] = field(default_factory=list)
    cycles: List[List[str]] = field(default_factory=list)
    orphans: List[str] = field(default_factory=list)
    critical_path: List[str] = field(default_factory=list)
    depth: int = 0

    @property
    def node_index(self) -> Dict[str, DAGNode]:
        """Map node-id -> DAGNode for fast lookup."""
        return {n.id: n for n in self.nodes}

    def adjacency(self) -> Dict[str, List[str]]:
        """Build adjacency list: source -> [targets]."""
        adj: Dict[str, List[str]] = {n.id: [] for n in self.nodes}
        for e in self.edges:
            adj.setdefault(e.source_id, []).append(e.target_id)
        return adj


# ---------------------------------------------------------------------------
# DisconfirmResult -- output of the DISCONFIRM stage
# ---------------------------------------------------------------------------

@dataclass
class WeaknessReport:
    """A structural weakness found during disconfirmation.

    Attributes:
        claim:          The claim under scrutiny (as Triple).
        weakness_type:  The kind of weakness (fallacy, weak_premise, etc.).
        fallacy:        Specific fallacy detected, if any.
        explanation:    Human-readable description of the weakness.
        counter_claim:  A steel-man counter-argument (strongest objection).
        severity:       How damaging this weakness is [0, 1].
        loop:           The strange loop involved, if circular reasoning.
        backward_trace: Derivation trace from conclusion back to weak point.
    """
    claim: Optional[Triple] = None
    weakness_type: str = "unknown"
    fallacy: Optional[FallacyType] = None
    explanation: str = ""
    counter_claim: Optional[Triple] = None
    severity: float = 0.5
    loop: Optional[Any] = None
    backward_trace: List[str] = field(default_factory=list)


@dataclass
class DisconfirmResult:
    """Result of the Disconfirm stage: structural weakness hunting.

    Grammar mapping:
        - Backward derivation from conclusions finds weak premises.
        - Strange loop analysis detects circular reasoning.
        - Fugue composition finds structural contradictions across domains.

    Attributes:
        weaknesses:       All structural weaknesses found.
        fallacies:        Specific logical fallacies detected.
        steel_man:        The strongest counter-argument constructed.
        survived_claims:  Claims that passed disconfirmation intact.
        weakened_claims:  Claims whose confidence was reduced.
        contradictions:   Pairs of claims that contradict each other.
        cross_domain:     Cross-domain structural issues (from fugue analysis).
    """
    weaknesses: List[WeaknessReport] = field(default_factory=list)
    fallacies: List[FallacyType] = field(default_factory=list)
    steel_man: Optional[str] = None
    survived_claims: List[Triple] = field(default_factory=list)
    weakened_claims: List[Triple] = field(default_factory=list)
    contradictions: List[Tuple[Triple, Triple]] = field(default_factory=list)
    cross_domain: List[Dict[str, Any]] = field(default_factory=list)

    @property
    def severity_score(self) -> float:
        """Average severity of all weaknesses [0, 1]. 0 = no issues."""
        if not self.weaknesses:
            return 0.0
        return sum(w.severity for w in self.weaknesses) / len(self.weaknesses)


# ---------------------------------------------------------------------------
# SynthesisResult -- output of the SYNTHESIS stage
# ---------------------------------------------------------------------------

@dataclass
class EvidenceNode:
    """A node in the final evidence hierarchy.

    Attributes:
        claim:     The claim.
        level:     Evidence quality level.
        support:   Number of independent derivation paths supporting it.
        domains:   Grammar domains that validated this claim (cross-domain).
        mnemo:     MNEMO-compressed representation of this node.
    """
    claim: Optional[Triple] = None
    level: EvidenceLevel = EvidenceLevel.UNSUBSTANTIATED
    support: int = 0
    domains: List[str] = field(default_factory=list)
    mnemo: str = ""


@dataclass
class SynthesisResult:
    """Result of the Synthesis stage: minimum viable logic.

    Grammar mapping:
        - Forward derivation from proven roots only.
        - Isomorphism discovery for cross-domain validation.
        - MNEMO compression of the clean argument.

    Attributes:
        proven:           Claims at PROVEN or SUPPORTED evidence level.
        unproven:         Claims that could not be derived from axioms.
        evidence_hierarchy: Full ordered hierarchy of evidence nodes.
        clean_argument:   The minimum viable argument (proven chain only).
        cross_domain_validation: Isomorphisms that strengthen conclusions.
        mnemo_compressed: The entire clean argument in MNEMO notation.
        confidence:       Overall argument confidence [0, 1].
    """
    proven: List[EvidenceNode] = field(default_factory=list)
    unproven: List[EvidenceNode] = field(default_factory=list)
    evidence_hierarchy: List[EvidenceNode] = field(default_factory=list)
    clean_argument: List[Triple] = field(default_factory=list)
    cross_domain_validation: List[Dict[str, Any]] = field(default_factory=list)
    mnemo_compressed: str = ""
    confidence: float = 0.0


# ---------------------------------------------------------------------------
# PipelineResult -- the aggregate result of the full 4-stage pipeline
# ---------------------------------------------------------------------------

@dataclass
class PipelineResult:
    """Aggregate result of the full GSM reasoning pipeline.

    Contains the output of every stage plus metadata about the pipeline
    run itself.  This is the complete audit trail: from raw input through
    skeleton extraction, dependency mapping, disconfirmation, and synthesis
    to the final minimum-viable-logic conclusion.

    Attributes:
        skeleton:       Output of stage 1 (SKELETON).
        dag:            Output of stage 2 (DAG).
        disconfirm:     Output of stage 3 (DISCONFIRM).
        synthesis:      Output of stage 4 (SYNTHESIS).
        raw_input:      The original input to the pipeline.
        run_id:         Unique identifier for this pipeline run.
        timestamp:      When the pipeline ran.
        stage_timings:  Wall-clock time for each stage (seconds).
        metadata:       Arbitrary extra data.
    """
    skeleton: Optional[SkeletonResult] = None
    dag: Optional[DAGResult] = None
    disconfirm: Optional[DisconfirmResult] = None
    synthesis: Optional[SynthesisResult] = None
    raw_input: Any = None
    run_id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])
    timestamp: float = field(default_factory=time.time)
    stage_timings: Dict[str, float] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def is_complete(self) -> bool:
        """True if all four stages have produced output."""
        return all([
            self.skeleton is not None,
            self.dag is not None,
            self.disconfirm is not None,
            self.synthesis is not None,
        ])

    @property
    def overall_confidence(self) -> float:
        """Overall pipeline confidence: synthesis confidence weighted by
        disconfirmation severity (weaknesses reduce confidence)."""
        if self.synthesis is None:
            return 0.0
        base = self.synthesis.confidence
        if self.disconfirm is not None:
            penalty = self.disconfirm.severity_score * 0.5
            return max(0.0, base - penalty)
        return base

    @property
    def summary(self) -> Dict[str, Any]:
        """Quick summary of the pipeline run."""
        return {
            "run_id": self.run_id,
            "complete": self.is_complete,
            "confidence": round(self.overall_confidence, 3),
            "claims_extracted": len(self.skeleton.all_claims) if self.skeleton else 0,
            "dependencies_mapped": len(self.dag.edges) if self.dag else 0,
            "weaknesses_found": len(self.disconfirm.weaknesses) if self.disconfirm else 0,
            "proven_claims": len(self.synthesis.proven) if self.synthesis else 0,
            "mnemo": self.synthesis.mnemo_compressed if self.synthesis else "",
        }
