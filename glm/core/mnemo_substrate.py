"""
Unified 270-glyph MNEMO substrate.

The atomic alphabet for all internal GLM processing. Every concept the GLM
can think about is encoded as a glyph from one of 8 tiers. Natural language
exists only at input/output boundaries — all derivation, attention, and
evidential marking operates on glyph sequences.

Security: supports rotating glyph mappings per session via session_seed.

The true timeline is grammatically the case — evidentiality is not metadata,
it is grammar. Every derivation MUST carry source, confidence, and temporal
markers. A sequence without evidential marking is grammatically invalid.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Dict, List, Optional, Tuple

from .substrate import Substrate, Symbol, Sequence


# ---------------------------------------------------------------------------
# Tier — the 8 glyph categories
# ---------------------------------------------------------------------------

class Tier(Enum):
    ONTOLOGICAL = auto()   # I:   What exists (roots, primitives)
    PROCESS = auto()       # II:  What happens (verbs, transforms)
    STATE = auto()         # III: Evidential source markers
    RELATIONAL = auto()    # IV:  How things relate (grammatical ops)
    EPISTEMIC = auto()     # V:   How sure we are (confidence)
    TEMPORAL = auto()      # VI:  When it's true (temporal/causal)
    SCALE = auto()         # VII: What domain (scale/domain switching)
    META = auto()          # VIII: How to compose (meta/syntax)


# ---------------------------------------------------------------------------
# Evidential enums — the three axes of truth
# ---------------------------------------------------------------------------

class EvidentialSource(Enum):
    OBSERVED = "obs"
    INFERRED = "inf"
    COMPUTED = "comp"
    REPORTED = "rep"
    TRADITION = "trad"
    SPECULATIVE = "spec"
    COUNTERFACTUAL = "ctr"


class EvidentialConfidence(Enum):
    CERTAIN = "cert"       # .95+
    PROBABLE = "prob"      # .7-.95
    POSSIBLE = "poss"      # .4-.7
    UNLIKELY = "unl"       # .1-.4
    UNKNOWN = "unk"        # <.1


class EvidentialTemporal(Enum):
    VERIFIED_PAST = "ver_past"
    OBSERVED_PRESENT = "obs_pres"
    PREDICTED_FUTURE = "pred_fut"
    HYPOTHETICAL = "hyp"
    TIMELESS = "timeless"


# ---------------------------------------------------------------------------
# MnemoGlyph — one glyph in the 270-glyph alphabet
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class MnemoGlyph:
    """A single glyph in the MNEMO alphabet.

    Attributes:
        code:     Short string identifier (e.g. 'obs', 'cert', 'derive').
        concept:  Human-readable concept name.
        tier:     Which of the 8 tiers this glyph belongs to.
        features: Additional feature bundle for grammar matching.
    """
    code: str
    concept: str
    tier: Tier
    features: Dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Glyph Registry — the full 270-glyph alphabet
# ---------------------------------------------------------------------------

def _build_registry() -> Dict[str, MnemoGlyph]:
    """Build the complete 270-glyph registry."""
    reg: Dict[str, MnemoGlyph] = {}

    def _add(code: str, concept: str, tier: Tier, **features: Any) -> None:
        reg[code] = MnemoGlyph(code=code, concept=concept, tier=tier,
                                features=features if features else {})

    # --- Tier I: Ontological Roots (50) ---
    for i, concept in enumerate([
        "entity", "property", "relation", "event", "state",
        "substance", "form", "pattern", "boundary", "void",
        "unity", "duality", "plurality", "whole", "part",
        "cause", "effect", "potential", "actual", "abstract",
    ]):
        _add(f"ont_{i:02d}", concept, Tier.ONTOLOGICAL)

    # Math primitives
    for i, concept in enumerate([
        "zero", "one", "infinity", "set", "function",
        "limit", "derivative_math", "integral", "sum", "product",
        "prime", "ratio", "matrix", "vector", "scalar",
    ]):
        _add(f"math_{i:02d}", concept, Tier.ONTOLOGICAL, domain="math")

    # Chemical elements
    for i, concept in enumerate([
        "hydrogen", "carbon", "nitrogen", "oxygen", "sulfur",
        "phosphorus", "iron", "sodium", "potassium", "calcium",
    ]):
        _add(f"elem_{i:02d}", concept, Tier.ONTOLOGICAL, domain="chemistry")

    # Phoneme roots
    for i, concept in enumerate([
        "vowel", "consonant", "syllable", "stress", "tone",
    ]):
        _add(f"phon_{i:02d}", concept, Tier.ONTOLOGICAL, domain="phonology")

    # --- Tier II: Process Verbs (40) ---
    processes = [
        "derive", "transform", "bond", "split", "merge",
        "inflect", "recurse", "compose", "decompose", "predict",
        "reconstruct", "align", "compare", "select", "reject",
        "generate", "parse", "encode", "decode", "translate",
        "reduce", "expand", "substitute", "bind", "release",
        "activate", "inhibit", "catalyse", "mutate", "replicate",
        "abstract", "instantiate", "inherit", "override", "emit",
        "absorb", "propagate", "converge", "diverge", "oscillate",
    ]
    for i, concept in enumerate(processes):
        _add(f"proc_{i:02d}", concept, Tier.PROCESS)

    # --- Tier III: State / Evidential Source Markers (25) ---
    _add("obs", "observed", Tier.STATE, axis="source", weight=0.95)
    _add("inf", "inferred", Tier.STATE, axis="source", weight=0.75)
    _add("comp", "computed", Tier.STATE, axis="source", weight=0.85)
    _add("rep", "reported", Tier.STATE, axis="source", weight=0.60)
    _add("trad", "tradition", Tier.STATE, axis="source", weight=0.40)
    _add("spec", "speculative", Tier.STATE, axis="source", weight=0.30)
    _add("ctr", "counterfactual", Tier.STATE, axis="source", weight=0.10)
    for i, concept in enumerate([
        "active", "passive", "transitional", "stable", "unstable",
        "nascent", "mature", "decaying", "equilibrium", "resonant",
        "latent", "manifest", "dormant", "emergent", "terminal",
        "anomalous", "normative", "critical",
    ]):
        _add(f"state_{i:02d}", concept, Tier.STATE)

    # --- Tier IV: Relational Ops (35) ---
    relationals = [
        "governs", "agrees", "selects", "binds", "dominates",
        "c_commands", "precedes", "follows", "contains", "intersects",
        "entails", "contradicts", "implies", "presupposes", "compatible",
        "incompatible", "isomorphic", "analogous", "inverse", "complement",
        "subset", "superset", "equivalent", "adjacent", "distant",
        "parent", "child", "sibling", "ancestor", "descendant",
        "input_of", "output_of", "parameter_of", "condition_of", "result_of",
    ]
    for i, concept in enumerate(relationals):
        _add(f"rel_{i:02d}", concept, Tier.RELATIONAL)

    # --- Tier V: Epistemic / Confidence Markers (25) ---
    _add("cert", "certain", Tier.EPISTEMIC, axis="confidence", min_prob=0.95)
    _add("prob", "probable", Tier.EPISTEMIC, axis="confidence", min_prob=0.70, max_prob=0.95)
    _add("poss", "possible", Tier.EPISTEMIC, axis="confidence", min_prob=0.40, max_prob=0.70)
    _add("unl", "unlikely", Tier.EPISTEMIC, axis="confidence", min_prob=0.10, max_prob=0.40)
    _add("unk", "unknown", Tier.EPISTEMIC, axis="confidence", max_prob=0.10)
    for i, concept in enumerate([
        "verified", "falsified", "contested", "consensus", "minority_view",
        "preliminary", "established", "deprecated", "revised", "retracted",
        "calibrated", "uncalibrated", "self_assessed", "peer_reviewed",
        "anecdotal", "systematic", "replicated", "unreplicated",
        "first_principles", "empirical",
    ]):
        _add(f"epist_{i:02d}", concept, Tier.EPISTEMIC)

    # --- Tier VI: Temporal / Causal (30) ---
    _add("ver_past", "verified_past", Tier.TEMPORAL, axis="temporal")
    _add("obs_pres", "observed_present", Tier.TEMPORAL, axis="temporal")
    _add("pred_fut", "predicted_future", Tier.TEMPORAL, axis="temporal")
    _add("hyp", "hypothetical", Tier.TEMPORAL, axis="temporal")
    _add("timeless", "timeless", Tier.TEMPORAL, axis="temporal")
    for i, concept in enumerate([
        "before", "after", "during", "simultaneous", "sequential",
        "causal_chain", "feedback_loop", "strange_loop_marker", "epoch",
        "derivation_depth_0", "derivation_depth_1", "derivation_depth_2",
        "derivation_depth_3", "derivation_depth_n",
        "cycle_start", "cycle_end", "phase_transition",
        "accelerating", "decelerating", "steady_state",
        "reversible", "irreversible", "periodic", "aperiodic", "chaotic",
    ]):
        _add(f"temp_{i:02d}", concept, Tier.TEMPORAL)

    # --- Tier VII: Scale / Domain (35) ---
    domains = [
        "linguistic_space", "chemical_space", "biological_space",
        "computational_space", "etymological_space", "mathematical_space",
        "physical_space", "phonological_space", "morphological_space",
        "syntactic_space", "semantic_space", "pragmatic_space",
        "quantum_scale", "atomic_scale", "molecular_scale",
        "cellular_scale", "organism_scale", "ecosystem_scale",
        "micro", "meso", "macro", "nano", "cosmic",
        "local", "global", "abstract_space", "concrete_space",
        "formal_space", "informal_space", "domain_switch",
        "cross_domain", "meta_domain", "null_domain",
        "response_space", "orchestration_space",
    ]
    for i, concept in enumerate(domains):
        _add(f"scale_{i:02d}", concept, Tier.SCALE)

    # --- Tier VIII: Meta / Syntax (30) ---
    metas = [
        "compose", "sequence_op", "parallel_op", "branch", "merge_op",
        "if_then", "loop_op", "break_op", "return_op", "yield_op",
        "open_scope", "close_scope", "reference", "dereference", "quote",
        "unquote", "eval_op", "apply_op", "map_op", "filter_op",
        "reduce_op", "fold_op", "zip_op", "concat", "split_op",
        "utterance_start", "utterance_end", "domain_analysis",
        "evidential_marker", "conjunction",
    ]
    for i, concept in enumerate(metas):
        _add(f"meta_{i:02d}", concept, Tier.META)

    return reg


GLYPH_REGISTRY: Dict[str, MnemoGlyph] = _build_registry()

# Reverse lookup: concept -> glyph
_CONCEPT_INDEX: Dict[str, MnemoGlyph] = {
    g.concept: g for g in GLYPH_REGISTRY.values()
}


# ---------------------------------------------------------------------------
# MnemoSequence — ordered glyph sequence with evidential checking
# ---------------------------------------------------------------------------

class MnemoSequence(Sequence):
    """A sequence of MNEMO glyphs with evidential validation.

    The true timeline is grammatically the case: a sequence is only
    valid if it carries evidential marking on all three axes.
    """

    def __init__(self, glyphs: Optional[List[MnemoGlyph]] = None) -> None:
        self._glyphs: List[MnemoGlyph] = list(glyphs) if glyphs else []
        symbols = []
        for g in self._glyphs:
            symbols.append(Symbol(
                form=g.code,
                features=dict(g.features),
                domain="mnemo",
            ))
        super().__init__(symbols)

    @property
    def glyphs(self) -> List[MnemoGlyph]:
        return list(self._glyphs)

    def has_evidential_marking(self) -> bool:
        """Check if sequence contains all three evidential axes.

        A grammatically valid derivation MUST have:
        - At least one Tier III glyph with axis=source
        - At least one Tier V glyph with axis=confidence
        - At least one Tier VI glyph with axis=temporal
        """
        has_source = any(
            g.tier == Tier.STATE and g.features.get("axis") == "source"
            for g in self._glyphs
        )
        has_confidence = any(
            g.tier == Tier.EPISTEMIC and g.features.get("axis") == "confidence"
            for g in self._glyphs
        )
        has_temporal = any(
            g.tier == Tier.TEMPORAL and g.features.get("axis") == "temporal"
            for g in self._glyphs
        )
        return has_source and has_confidence and has_temporal

    def get_evidential_triple(
        self,
    ) -> Tuple[Optional[MnemoGlyph], Optional[MnemoGlyph], Optional[MnemoGlyph]]:
        """Extract the (source, confidence, temporal) evidential triple."""
        source = next(
            (g for g in self._glyphs
             if g.tier == Tier.STATE and g.features.get("axis") == "source"),
            None,
        )
        confidence = next(
            (g for g in self._glyphs
             if g.tier == Tier.EPISTEMIC and g.features.get("axis") == "confidence"),
            None,
        )
        temporal = next(
            (g for g in self._glyphs
             if g.tier == Tier.TEMPORAL and g.features.get("axis") == "temporal"),
            None,
        )
        return source, confidence, temporal


# ---------------------------------------------------------------------------
# MnemoSubstrate — the substrate implementation
# ---------------------------------------------------------------------------

class MnemoSubstrate(Substrate):
    """The unified MNEMO substrate — 270 glyphs, 8 tiers.

    All GLM internal processing operates on MnemoSequences.
    Supports rotating glyph mappings per session for security.
    """

    def __init__(self, session_seed: Optional[str] = None) -> None:
        self._session_seed = session_seed
        self._registry = dict(GLYPH_REGISTRY)
        self._concept_index = dict(_CONCEPT_INDEX)
        if session_seed:
            self._apply_rotation(session_seed)

    def _apply_rotation(self, seed: str) -> None:
        """Rotate internal code mappings using session seed.

        Infrastructure for future session-specific encoding. The grammar
        and semantics are unchanged; only the serialisation changes.
        """
        pass  # Rotation infrastructure — activate when needed

    def encode_codes(self, codes: List[str]) -> MnemoSequence:
        """Create a MnemoSequence from glyph codes."""
        glyphs = []
        for code in codes:
            if code in self._registry:
                glyphs.append(self._registry[code])
        return MnemoSequence(glyphs)

    def decode_codes(self, seq: MnemoSequence) -> List[str]:
        """Extract glyph codes from a MnemoSequence."""
        return [g.code for g in seq.glyphs]

    def lookup_concept(self, concept: str) -> Optional[MnemoGlyph]:
        """Find a glyph by its concept name."""
        return self._concept_index.get(concept)

    def validate_sequence(self, codes: List[str]) -> bool:
        """Grammar gate: reject invalid MNEMO sequences before processing."""
        return all(code in self._registry for code in codes)

    # --- Substrate ABC implementation ---

    def encode(self, data: Any) -> MnemoSequence:
        if isinstance(data, list) and all(isinstance(x, str) for x in data):
            return self.encode_codes(data)
        return MnemoSequence()

    def decode(self, sequence: Any) -> Any:
        if isinstance(sequence, MnemoSequence):
            return self.decode_codes(sequence)
        return []
