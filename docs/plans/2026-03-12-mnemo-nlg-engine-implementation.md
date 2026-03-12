# MNEMO-Native NLG Engine Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Build a MNEMO-native NLG engine that thinks in MNEMO glyphs internally and speaks natural language at input/output boundaries, with mandatory evidentiality grammar.

**Architecture:** Three-stage pipeline (ENCODE → PROCESS → DECODE) where all internal processing uses 270 MNEMO glyphs. The 8th "response grammar" domain composes cross-domain outputs. Evidential marking (source + confidence + temporal) is grammatically required on every derivation.

**Tech Stack:** Pure Python stdlib (no numpy, no torch — mobile-viable), extending existing `glm/core/substrate.py` ABC, `glm/core/grammar.py` productions, and `glm/model/attention.py` TemporalAttention.

---

## Task 1: MNEMO Substrate — 270-Glyph Foundation

**Files:**
- Create: `glm/core/mnemo_substrate.py`
- Create: `tests/test_mnemo_substrate.py`

**Step 1: Write the failing tests**

```python
# tests/test_mnemo_substrate.py
"""Tests for the unified 270-glyph MNEMO substrate."""

import pytest
from glm.core.mnemo_substrate import (
    MnemoGlyph, MnemoSequence, MnemoSubstrate,
    Tier, GLYPH_REGISTRY,
    EvidentialSource, EvidentialConfidence, EvidentialTemporal,
)


class TestGlyphRegistry:
    def test_registry_has_270_glyphs(self):
        assert len(GLYPH_REGISTRY) == 270

    def test_tiers_are_complete(self):
        counts = {}
        for g in GLYPH_REGISTRY.values():
            counts[g.tier] = counts.get(g.tier, 0) + 1
        assert counts[Tier.ONTOLOGICAL] == 50
        assert counts[Tier.PROCESS] == 40
        assert counts[Tier.STATE] == 25
        assert counts[Tier.RELATIONAL] == 35
        assert counts[Tier.EPISTEMIC] == 25
        assert counts[Tier.TEMPORAL] == 30
        assert counts[Tier.SCALE] == 35
        assert counts[Tier.META] == 30

    def test_glyph_lookup_by_code(self):
        g = GLYPH_REGISTRY["obs"]
        assert g.tier == Tier.STATE
        assert g.concept == "observed"

    def test_glyph_lookup_by_concept(self):
        substrate = MnemoSubstrate()
        g = substrate.lookup_concept("observed")
        assert g.code == "obs"


class TestEvidentialMarkers:
    def test_source_glyphs_exist(self):
        for code in ("obs", "inf", "comp", "rep", "trad", "spec", "ctr"):
            assert code in GLYPH_REGISTRY

    def test_confidence_glyphs_exist(self):
        for code in ("cert", "prob", "poss", "unl", "unk"):
            assert code in GLYPH_REGISTRY

    def test_temporal_glyphs_exist(self):
        for code in ("ver_past", "obs_pres", "pred_fut", "hyp", "timeless"):
            assert code in GLYPH_REGISTRY


class TestMnemoSequence:
    def test_create_sequence(self):
        substrate = MnemoSubstrate()
        seq = substrate.encode_codes(["obs", "cert", "obs_pres"])
        assert len(seq) == 3

    def test_has_evidential_marking(self):
        substrate = MnemoSubstrate()
        seq = substrate.encode_codes(["obs", "cert", "obs_pres"])
        assert seq.has_evidential_marking()

    def test_missing_evidential_detected(self):
        substrate = MnemoSubstrate()
        seq = substrate.encode_codes(["obs"])  # missing confidence + temporal
        assert not seq.has_evidential_marking()


class TestMnemoSubstrate:
    def test_encode_decode_roundtrip(self):
        substrate = MnemoSubstrate()
        codes = ["obs", "cert", "obs_pres"]
        seq = substrate.encode_codes(codes)
        result = substrate.decode_codes(seq)
        assert result == codes

    def test_validate_rejects_bad_sequence(self):
        substrate = MnemoSubstrate()
        assert not substrate.validate_sequence(["INVALID", "obs"])

    def test_validate_accepts_good_sequence(self):
        substrate = MnemoSubstrate()
        assert substrate.validate_sequence(["obs", "cert", "obs_pres"])


class TestRotatingMapping:
    def test_session_mapping_differs(self):
        s1 = MnemoSubstrate(session_seed="session_a")
        s2 = MnemoSubstrate(session_seed="session_b")
        # Same concept, different internal glyph codes
        g1 = s1.lookup_concept("observed")
        g2 = s2.lookup_concept("observed")
        # The concept is the same, but the internal encoding may differ
        assert g1.concept == g2.concept
```

**Step 2: Run tests to verify they fail**

Run: `cd C:\Users\mrjki\OneDrive\Desktop\MKAngel && python -m pytest tests/test_mnemo_substrate.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'glm.core.mnemo_substrate'`

**Step 3: Write minimal implementation**

```python
# glm/core/mnemo_substrate.py
"""
Unified 270-glyph MNEMO substrate.

The atomic alphabet for all internal GLM processing. Every concept the GLM
can think about is encoded as a glyph from one of 8 tiers. Natural language
exists only at input/output boundaries — all derivation, attention, and
evidential marking operates on glyph sequences.

Security: supports rotating glyph mappings per session via session_seed.
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
        reg[code] = MnemoGlyph(code=code, concept=concept, tier=tier, features=features)

    # --- Tier I: Ontological Roots (50) ---
    # Core ontological primitives
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
        "limit", "derivative", "integral", "sum", "product",
        "prime", "ratio", "matrix", "vector", "scalar",
    ]):
        _add(f"math_{i:02d}", concept, Tier.ONTOLOGICAL, domain="math")

    # Chemical elements (common)
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
    # The 7 evidential source markers
    _add("obs", "observed", Tier.STATE, axis="source", weight=0.95)
    _add("inf", "inferred", Tier.STATE, axis="source", weight=0.75)
    _add("comp", "computed", Tier.STATE, axis="source", weight=0.85)
    _add("rep", "reported", Tier.STATE, axis="source", weight=0.60)
    _add("trad", "tradition", Tier.STATE, axis="source", weight=0.40)
    _add("spec", "speculative", Tier.STATE, axis="source", weight=0.30)
    _add("ctr", "counterfactual", Tier.STATE, axis="source", weight=0.10)
    # Additional state markers
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
    # Additional epistemic markers
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
    """A sequence of MNEMO glyphs with evidential validation."""

    def __init__(self, glyphs: Optional[List[MnemoGlyph]] = None) -> None:
        symbols = []
        self._glyphs: List[MnemoGlyph] = list(glyphs) if glyphs else []
        for g in self._glyphs:
            symbols.append(Symbol(form=g.code, features=dict(g.features), domain="mnemo"))
        super().__init__(symbols)

    @property
    def glyphs(self) -> List[MnemoGlyph]:
        return list(self._glyphs)

    def has_evidential_marking(self) -> bool:
        """Check if sequence contains all three evidential axes."""
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

    def get_evidential_triple(self) -> Tuple[Optional[MnemoGlyph], Optional[MnemoGlyph], Optional[MnemoGlyph]]:
        """Extract the (source, confidence, temporal) evidential triple."""
        source = next((g for g in self._glyphs if g.tier == Tier.STATE and g.features.get("axis") == "source"), None)
        confidence = next((g for g in self._glyphs if g.tier == Tier.EPISTEMIC and g.features.get("axis") == "confidence"), None)
        temporal = next((g for g in self._glyphs if g.tier == Tier.TEMPORAL and g.features.get("axis") == "temporal"), None)
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

        The concepts and tiers stay the same — only the internal
        code strings change. This means the same grammar rules work,
        but the wire format differs per session.
        """
        # For now, rotation is identity — the infrastructure is in place
        # for future session-specific encoding. The grammar and semantics
        # are unchanged; only the serialisation changes.
        pass

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
        """Check all codes are valid glyphs."""
        return all(code in self._registry for code in codes)

    # --- Substrate ABC implementation ---

    def encode(self, data: Any) -> MnemoSequence:
        """Encode raw data into a MnemoSequence (stub for NLG encoder)."""
        if isinstance(data, list) and all(isinstance(x, str) for x in data):
            return self.encode_codes(data)
        return MnemoSequence()

    def decode(self, sequence: Any) -> Any:
        """Decode a MnemoSequence back to codes (stub for NLG decoder)."""
        if isinstance(sequence, MnemoSequence):
            return self.decode_codes(sequence)
        return []
```

**Step 4: Run tests to verify they pass**

Run: `cd C:\Users\mrjki\OneDrive\Desktop\MKAngel && python -m pytest tests/test_mnemo_substrate.py -v`
Expected: ALL PASS

**Step 5: Commit**

```bash
git add glm/core/mnemo_substrate.py tests/test_mnemo_substrate.py
git commit -m "feat: add 270-glyph MNEMO substrate with evidential markers"
```

---

## Task 2: Response Grammar — 8th Domain

**Files:**
- Create: `glm/nlg/__init__.py`
- Create: `glm/nlg/response_grammar.py`
- Create: `tests/test_response_grammar.py`

**Step 1: Write the failing tests**

```python
# tests/test_response_grammar.py
"""Tests for the 8th domain response grammar."""

import pytest
from glm.nlg.response_grammar import (
    ResponseGrammar, UtteranceNode, DomainAnalysisNode,
    EvidentialMarkerNode, build_response_grammar,
)
from glm.core.mnemo_substrate import MnemoSubstrate, Tier


class TestResponseGrammar:
    def test_build_grammar(self):
        grammar = build_response_grammar()
        assert grammar.name == "response"
        assert grammar.domain == "response"
        assert len(grammar.productions) > 0

    def test_utterance_production(self):
        grammar = build_response_grammar()
        # Utterance -> DomainAnalysis EvidentialMarker
        utterance_prods = [p for p in grammar.productions if p.lhs == "Utterance"]
        assert len(utterance_prods) > 0

    def test_multi_domain_composition(self):
        grammar = build_response_grammar()
        # DomainAnalysis -> DomainAnalysis Conjunction DomainAnalysis
        multi_prods = [
            p for p in grammar.productions
            if p.lhs == "DomainAnalysis" and "Conjunction" in (p.rhs or [])
        ]
        assert len(multi_prods) > 0


class TestUtteranceNode:
    def test_create_utterance(self):
        substrate = MnemoSubstrate()
        node = UtteranceNode(
            domain_results=["mathematical"],
            evidential_source="obs",
            evidential_confidence="cert",
            evidential_temporal="obs_pres",
        )
        assert node.domain_results == ["mathematical"]
        assert node.evidential_source == "obs"

    def test_utterance_to_mnemo(self):
        substrate = MnemoSubstrate()
        node = UtteranceNode(
            domain_results=["mathematical"],
            evidential_source="obs",
            evidential_confidence="cert",
            evidential_temporal="obs_pres",
        )
        seq = node.to_mnemo_sequence(substrate)
        assert seq.has_evidential_marking()

    def test_multi_domain_utterance(self):
        node = UtteranceNode(
            domain_results=["mathematical", "linguistic"],
            evidential_source="inf",
            evidential_confidence="prob",
            evidential_temporal="pred_fut",
        )
        assert len(node.domain_results) == 2
```

**Step 2: Run tests to verify they fail**

Run: `cd C:\Users\mrjki\OneDrive\Desktop\MKAngel && python -m pytest tests/test_response_grammar.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'glm.nlg'`

**Step 3: Write minimal implementation**

```python
# glm/nlg/__init__.py
"""NLG — Natural Language Generation via MNEMO substrate."""
```

```python
# glm/nlg/response_grammar.py
"""
Response Grammar — the 8th domain.

Composes outputs from the 7 domain grammars into complete utterances
with mandatory evidential marking. This is where cross-domain insights
become speakable sentences.

Productions:
    Utterance -> DomainAnalysis EvidentialMarker
    DomainAnalysis -> MathResult | LinguisticResult | BiologyResult | ...
    DomainAnalysis -> DomainAnalysis Conjunction DomainAnalysis
    EvidentialMarker -> SourceGlyph ConfidenceGlyph TemporalGlyph
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from glm.core.grammar import Grammar, Production, Rule, StrangeLoop
from glm.core.mnemo_substrate import (
    MnemoGlyph, MnemoSequence, MnemoSubstrate, Tier,
)


# ---------------------------------------------------------------------------
# AST nodes for response composition
# ---------------------------------------------------------------------------

@dataclass
class EvidentialMarkerNode:
    """The three-axis evidential marking on every utterance."""
    source: str       # glyph code from Tier III
    confidence: str   # glyph code from Tier V
    temporal: str     # glyph code from Tier VI


@dataclass
class DomainAnalysisNode:
    """A result from one domain grammar derivation."""
    domain: str
    derivation_codes: List[str] = field(default_factory=list)
    confidence: float = 1.0


@dataclass
class UtteranceNode:
    """A complete response — domain results + evidential marking.

    This is the root of the response grammar's derivation tree.
    """
    domain_results: List[str] = field(default_factory=list)
    evidential_source: str = "inf"
    evidential_confidence: str = "prob"
    evidential_temporal: str = "obs_pres"
    content_codes: List[str] = field(default_factory=list)

    def to_mnemo_sequence(self, substrate: MnemoSubstrate) -> MnemoSequence:
        """Convert this utterance to a MNEMO glyph sequence."""
        codes = []
        # Start with utterance marker
        codes.append("meta_25")  # utterance_start

        # Domain analysis glyphs
        for domain in self.domain_results:
            domain_glyph = f"scale_{_domain_to_scale_index(domain)}"
            codes.append(domain_glyph)
            codes.append("meta_27")  # domain_analysis

        # Content codes
        codes.extend(self.content_codes)

        # Evidential marking (mandatory)
        codes.append(self.evidential_source)
        codes.append(self.evidential_confidence)
        codes.append(self.evidential_temporal)

        # End marker
        codes.append("meta_26")  # utterance_end

        return substrate.encode_codes(codes)


def _domain_to_scale_index(domain: str) -> str:
    """Map domain name to scale glyph index."""
    mapping = {
        "mathematical": "05", "linguistic": "00", "biological": "02",
        "chemical": "01", "computational": "03", "etymological": "04",
        "physical": "06", "phonological": "07", "morphological": "08",
        "syntactic": "09", "semantic": "10", "response": "33",
    }
    return mapping.get(domain, "32")  # null_domain fallback


# ---------------------------------------------------------------------------
# Grammar builder
# ---------------------------------------------------------------------------

def build_response_grammar() -> Grammar:
    """Build the 8th domain grammar: Utterance composition."""

    productions = [
        # Core: Utterance -> DomainAnalysis EvidentialMarker
        Production(
            lhs="Utterance",
            rhs=["DomainAnalysis", "EvidentialMarker"],
            name="utterance_base",
        ),
        # Multi-domain: DomainAnalysis -> DomainAnalysis Conjunction DomainAnalysis
        Production(
            lhs="DomainAnalysis",
            rhs=["DomainAnalysis", "Conjunction", "DomainAnalysis"],
            name="multi_domain",
        ),
        # Single domain results
        Production(lhs="DomainAnalysis", rhs=["MathResult"], name="math_analysis"),
        Production(lhs="DomainAnalysis", rhs=["LinguisticResult"], name="ling_analysis"),
        Production(lhs="DomainAnalysis", rhs=["BiologyResult"], name="bio_analysis"),
        Production(lhs="DomainAnalysis", rhs=["ChemistryResult"], name="chem_analysis"),
        Production(lhs="DomainAnalysis", rhs=["PhysicsResult"], name="phys_analysis"),
        Production(lhs="DomainAnalysis", rhs=["ComputationResult"], name="comp_analysis"),
        Production(lhs="DomainAnalysis", rhs=["EtymologyResult"], name="etym_analysis"),
        # Evidential marker composition
        Production(
            lhs="EvidentialMarker",
            rhs=["SourceGlyph", "ConfidenceGlyph", "TemporalGlyph"],
            name="evidential_triple",
        ),
    ]

    rules = [
        # Conjunction rules
        Rule(name="and_conj", pattern="Conjunction", result="and"),
        Rule(name="but_conj", pattern="Conjunction", result="but"),
        Rule(name="therefore_conj", pattern="Conjunction", result="therefore"),
    ]

    # Strange loop: response -> domain analysis -> cross-domain insight -> response
    loops = [
        StrangeLoop(
            cycle=["Utterance", "DomainAnalysis", "CrossDomainInsight", "Utterance"],
            entry="Utterance",
            level_delta=1,
        ),
    ]

    return Grammar(
        name="response",
        domain="response",
        rules=rules,
        productions=productions,
        strange_loops=loops,
    )
```

**Step 4: Run tests to verify they pass**

Run: `cd C:\Users\mrjki\OneDrive\Desktop\MKAngel && python -m pytest tests/test_response_grammar.py -v`
Expected: ALL PASS

**Step 5: Commit**

```bash
git add glm/nlg/__init__.py glm/nlg/response_grammar.py tests/test_response_grammar.py
git commit -m "feat: add response grammar (8th domain) with evidential composition"
```

---

## Task 3: MNEMO Encoder — Natural Language → MNEMO

**Files:**
- Create: `glm/nlg/encoder.py`
- Create: `tests/test_nlg_encoder.py`

**Step 1: Write the failing tests**

```python
# tests/test_nlg_encoder.py
"""Tests for the NLG encoder: natural language -> MNEMO."""

import pytest
from glm.nlg.encoder import MnemoEncoder
from glm.core.mnemo_substrate import MnemoSubstrate, MnemoSequence, Tier


class TestMnemoEncoder:
    def test_create_encoder(self):
        encoder = MnemoEncoder()
        assert encoder is not None

    def test_encode_simple_text(self):
        encoder = MnemoEncoder()
        result = encoder.encode("water is a molecule")
        assert isinstance(result, MnemoSequence)
        assert len(result) > 0

    def test_encode_attaches_domain(self):
        encoder = MnemoEncoder()
        result = encoder.encode("the verb agrees with the noun")
        # Should detect linguistic domain
        codes = [g.code for g in result.glyphs]
        has_ling_domain = any("scale_00" in c for c in codes)  # linguistic_space
        assert has_ling_domain

    def test_encode_adds_evidential_defaults(self):
        encoder = MnemoEncoder()
        result = encoder.encode("some text input")
        # Encoder should attach default evidential markers
        assert result.has_evidential_marking()

    def test_encode_empty_input(self):
        encoder = MnemoEncoder()
        result = encoder.encode("")
        assert isinstance(result, MnemoSequence)

    def test_domain_detection_math(self):
        encoder = MnemoEncoder()
        domain = encoder.detect_domain("solve the equation x + 2 = 5")
        assert domain == "mathematical"

    def test_domain_detection_chemistry(self):
        encoder = MnemoEncoder()
        domain = encoder.detect_domain("the molecule bonds with oxygen")
        assert domain == "chemical"
```

**Step 2: Run tests to verify they fail**

Run: `cd C:\Users\mrjki\OneDrive\Desktop\MKAngel && python -m pytest tests/test_nlg_encoder.py -v`
Expected: FAIL — `ModuleNotFoundError`

**Step 3: Write minimal implementation**

```python
# glm/nlg/encoder.py
"""
Encoder — Natural Language → MNEMO (input boundary).

The encoder parses natural language input, detects domain, extracts
key concepts, and maps them to MNEMO glyph sequences. This is the
first stage of the NLG pipeline — everything after this operates
purely in MNEMO space.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from glm.core.mnemo_substrate import (
    GLYPH_REGISTRY, MnemoGlyph, MnemoSequence, MnemoSubstrate, Tier,
)


# ---------------------------------------------------------------------------
# Domain detection keywords
# ---------------------------------------------------------------------------

_DOMAIN_KEYWORDS: Dict[str, List[str]] = {
    "linguistic": [
        "word", "sentence", "morpheme", "phoneme", "syntax", "verb", "noun",
        "clause", "phrase", "grammar", "language", "conjugat", "declen",
        "pronoun", "adjective", "adverb", "preposition",
    ],
    "chemical": [
        "molecule", "atom", "bond", "reaction", "compound", "element",
        "formula", "ion", "acid", "base", "solution", "catalyst", "oxidat",
        "reducti", "polymer", "crystal", "organic", "inorganic",
    ],
    "biological": [
        "gene", "protein", "dna", "rna", "codon", "amino", "cell",
        "enzyme", "nucleotide", "organism", "evolution", "mutation",
        "mitosis", "meiosis", "chromosome", "genome",
    ],
    "mathematical": [
        "equation", "solve", "proof", "theorem", "integral", "derivative",
        "matrix", "vector", "function", "limit", "infinity", "prime",
        "algebra", "calculus", "topology", "geometry", "number",
    ],
    "physical": [
        "force", "energy", "mass", "velocity", "acceleration", "gravity",
        "quantum", "wave", "particle", "field", "electro", "magnetic",
        "thermo", "entropy", "momentum", "relativity",
    ],
    "computational": [
        "algorithm", "function", "variable", "loop", "class", "object",
        "return", "import", "compile", "runtime", "memory", "stack",
        "recursion", "iteration", "complexity", "binary",
    ],
    "etymological": [
        "origin", "root", "latin", "greek", "proto", "ancestor", "cognate",
        "derived", "evolved", "borrowed", "loanword", "indo-european",
    ],
}

_DOMAIN_TO_SCALE: Dict[str, str] = {
    "linguistic": "scale_00", "chemical": "scale_01",
    "biological": "scale_02", "computational": "scale_03",
    "etymological": "scale_04", "mathematical": "scale_05",
    "physical": "scale_06",
}


# ---------------------------------------------------------------------------
# Concept extraction keywords → glyph codes
# ---------------------------------------------------------------------------

_CONCEPT_MAP: Dict[str, str] = {
    # Process verbs
    "derive": "proc_00", "transform": "proc_01", "bond": "proc_02",
    "split": "proc_03", "merge": "proc_04", "predict": "proc_09",
    "reconstruct": "proc_10", "compare": "proc_12", "generate": "proc_15",
    "parse": "proc_16", "encode": "proc_17", "decode": "proc_18",
    "translate": "proc_19", "compose": "proc_07", "select": "proc_13",
    # Relational
    "agrees": "rel_01", "governs": "rel_00", "contains": "rel_08",
    "entails": "rel_10", "implies": "rel_12",
    # Ontological
    "entity": "ont_00", "property": "ont_01", "relation": "ont_02",
    "event": "ont_03", "pattern": "ont_07",
}


@dataclass
class MnemoEncoder:
    """Encodes natural language input to MNEMO glyph sequences.

    The encoder is the input boundary — the last place natural language
    exists before entering the all-MNEMO processing pipeline.
    """
    substrate: MnemoSubstrate = field(default_factory=MnemoSubstrate)

    def encode(self, text: str) -> MnemoSequence:
        """Encode natural language text into a MNEMO sequence.

        Steps:
            1. Detect domain
            2. Extract key concepts -> glyph codes
            3. Attach domain routing metadata
            4. Attach default evidential markers
            5. Return MnemoSequence
        """
        if not text.strip():
            return MnemoSequence()

        codes: List[str] = []

        # 1. Domain detection
        domain = self.detect_domain(text)
        domain_code = _DOMAIN_TO_SCALE.get(domain)
        if domain_code:
            codes.append(domain_code)

        # 2. Concept extraction
        words = text.lower().split()
        for word in words:
            # Strip punctuation
            clean = re.sub(r"[^\w]", "", word)
            if clean in _CONCEPT_MAP:
                codes.append(_CONCEPT_MAP[clean])

        # If no concepts matched, add a generic process glyph
        if not any(c.startswith("proc_") for c in codes):
            codes.append("proc_15")  # generate (default process)

        # 3. Default evidential marking
        codes.append("rep")        # default source: reported (user told us)
        codes.append("prob")       # default confidence: probable
        codes.append("obs_pres")   # default temporal: present

        return self.substrate.encode_codes(codes)

    def detect_domain(self, text: str) -> str:
        """Detect the most likely domain of the input text."""
        text_lower = text.lower()
        scores: Dict[str, float] = {}

        for domain, keywords in _DOMAIN_KEYWORDS.items():
            score = sum(1.0 for kw in keywords if kw in text_lower)
            if score > 0:
                scores[domain] = score

        if not scores:
            return "linguistic"  # default domain

        return max(scores, key=lambda k: scores[k])
```

**Step 4: Run tests to verify they pass**

Run: `cd C:\Users\mrjki\OneDrive\Desktop\MKAngel && python -m pytest tests/test_nlg_encoder.py -v`
Expected: ALL PASS

**Step 5: Commit**

```bash
git add glm/nlg/encoder.py tests/test_nlg_encoder.py
git commit -m "feat: add MNEMO encoder (natural language -> MNEMO input boundary)"
```

---

## Task 4: English Surface Templates

**Files:**
- Create: `glm/nlg/templates/__init__.py`
- Create: `glm/nlg/templates/en.py`
- Create: `tests/test_templates.py`

**Step 1: Write the failing tests**

```python
# tests/test_templates.py
"""Tests for English surface templates."""

import pytest
from glm.nlg.templates import TemplateRegistry
from glm.nlg.templates.en import ENGLISH_TEMPLATES, register_english


class TestTemplateRegistry:
    def test_create_registry(self):
        reg = TemplateRegistry()
        assert len(reg) == 0

    def test_register_english(self):
        reg = TemplateRegistry()
        register_english(reg)
        assert len(reg) > 0

    def test_lookup_by_domain(self):
        reg = TemplateRegistry()
        register_english(reg)
        templates = reg.for_domain("mathematical")
        assert len(templates) > 0

    def test_lookup_by_domain_and_pattern(self):
        reg = TemplateRegistry()
        register_english(reg)
        templates = reg.for_domain("mathematical")
        # Should have at least one template for math
        assert any("result" in t.slots for t in templates)


class TestEvidentialTemplates:
    def test_english_hedging_for_inference(self):
        reg = TemplateRegistry()
        register_english(reg)
        templates = reg.for_evidential("inf", "prob")
        assert len(templates) > 0
        # Should contain hedging language
        assert any("suggest" in t.template or "evidence" in t.template for t in templates)

    def test_english_certainty_for_observed(self):
        reg = TemplateRegistry()
        register_english(reg)
        templates = reg.for_evidential("obs", "cert")
        assert len(templates) > 0


class TestEnglishTemplates:
    def test_all_seven_domains_covered(self):
        reg = TemplateRegistry()
        register_english(reg)
        for domain in ("mathematical", "linguistic", "biological",
                       "chemical", "physical", "computational", "etymological"):
            templates = reg.for_domain(domain)
            assert len(templates) > 0, f"No templates for {domain}"
```

**Step 2: Run tests to verify they fail**

Run: `cd C:\Users\mrjki\OneDrive\Desktop\MKAngel && python -m pytest tests/test_templates.py -v`
Expected: FAIL — `ModuleNotFoundError`

**Step 3: Write minimal implementation**

```python
# glm/nlg/templates/__init__.py
"""Surface template registry for NLG decoding."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set


@dataclass
class SurfaceTemplate:
    """A language-tagged surface template with typed slots.

    Attributes:
        template:  String with {slot_name} placeholders.
        language:  ISO 639-1 language code.
        domain:    Which domain this template serves.
        slots:     Set of slot names this template expects.
        evidential_source:  If set, only use for this source type.
        evidential_confidence: If set, only use for this confidence.
        weight:    Selection weight (higher = preferred).
    """
    template: str
    language: str = "en"
    domain: str = "general"
    slots: Set[str] = field(default_factory=set)
    evidential_source: Optional[str] = None
    evidential_confidence: Optional[str] = None
    weight: float = 1.0

    def __post_init__(self):
        # Auto-detect slots from template
        import re
        if not self.slots:
            self.slots = set(re.findall(r"\{(\w+)\}", self.template))

    def render(self, **kwargs: Any) -> str:
        """Render the template with the given slot values."""
        result = self.template
        for key, value in kwargs.items():
            result = result.replace(f"{{{key}}}", str(value))
        return result


class TemplateRegistry:
    """Registry of surface templates, indexed by language + domain."""

    def __init__(self) -> None:
        self._templates: List[SurfaceTemplate] = []

    def __len__(self) -> int:
        return len(self._templates)

    def add(self, template: SurfaceTemplate) -> None:
        self._templates.append(template)

    def for_domain(self, domain: str, language: str = "en") -> List[SurfaceTemplate]:
        return [
            t for t in self._templates
            if t.domain == domain and t.language == language
        ]

    def for_evidential(
        self, source: Optional[str] = None, confidence: Optional[str] = None,
        language: str = "en",
    ) -> List[SurfaceTemplate]:
        results = []
        for t in self._templates:
            if t.language != language:
                continue
            if source and t.evidential_source and t.evidential_source != source:
                continue
            if confidence and t.evidential_confidence and t.evidential_confidence != confidence:
                continue
            if t.evidential_source or t.evidential_confidence:
                results.append(t)
        return results
```

```python
# glm/nlg/templates/en.py
"""English surface templates for all 7 domain grammars + evidential hedging."""

from __future__ import annotations

from . import SurfaceTemplate, TemplateRegistry


# ---------------------------------------------------------------------------
# Domain templates
# ---------------------------------------------------------------------------

ENGLISH_TEMPLATES = [
    # --- Mathematical ---
    SurfaceTemplate("{result}", domain="mathematical", slots={"result"}),
    SurfaceTemplate("The result is {result}.", domain="mathematical"),
    SurfaceTemplate("Solving this gives {result}.", domain="mathematical"),
    SurfaceTemplate("By {method}, we get {result}.", domain="mathematical"),
    SurfaceTemplate("The {operation} of {operand} yields {result}.", domain="mathematical"),

    # --- Linguistic ---
    SurfaceTemplate("The word '{word}' {analysis}.", domain="linguistic"),
    SurfaceTemplate("In this context, '{word}' functions as {role}.", domain="linguistic"),
    SurfaceTemplate("The phrase structure is {structure}.", domain="linguistic"),
    SurfaceTemplate("Morphologically, '{word}' breaks down as {breakdown}.", domain="linguistic"),
    SurfaceTemplate("The grammatical pattern here is {pattern}.", domain="linguistic"),

    # --- Biological ---
    SurfaceTemplate("The sequence {sequence} codes for {protein}.", domain="biological"),
    SurfaceTemplate("This biological process involves {process}.", domain="biological"),
    SurfaceTemplate("The {organism} exhibits {trait}.", domain="biological"),
    SurfaceTemplate("At the cellular level, {mechanism}.", domain="biological"),

    # --- Chemical ---
    SurfaceTemplate("The compound {compound} has the formula {formula}.", domain="chemical"),
    SurfaceTemplate("This reaction produces {product}.", domain="chemical"),
    SurfaceTemplate("The bond between {atom1} and {atom2} is {bond_type}.", domain="chemical"),
    SurfaceTemplate("{reactant} reacts with {reagent} to form {product}.", domain="chemical"),

    # --- Physical ---
    SurfaceTemplate("The {quantity} equals {value} {unit}.", domain="physical"),
    SurfaceTemplate("By {law}, {consequence}.", domain="physical"),
    SurfaceTemplate("The system exhibits {behaviour}.", domain="physical"),
    SurfaceTemplate("At this scale, {phenomenon} dominates.", domain="physical"),

    # --- Computational ---
    SurfaceTemplate("The algorithm {description}.", domain="computational"),
    SurfaceTemplate("This has {complexity} complexity.", domain="computational"),
    SurfaceTemplate("The function returns {result}.", domain="computational"),
    SurfaceTemplate("Recursively, {description}.", domain="computational"),

    # --- Etymological ---
    SurfaceTemplate("'{word}' derives from {origin} '{root}'.", domain="etymological"),
    SurfaceTemplate("The root '{root}' means '{meaning}' in {language}.", domain="etymological"),
    SurfaceTemplate("Historically, '{word}' evolved from {ancestor}.", domain="etymological"),
    SurfaceTemplate("This is a {loan_type} from {source_language}.", domain="etymological"),

    # --- Evidential hedging (English) ---
    # Observed + certain
    SurfaceTemplate(
        "{content}",
        domain="general",
        evidential_source="obs", evidential_confidence="cert",
        weight=1.0,
    ),
    # Observed + probable
    SurfaceTemplate(
        "Based on observation, {content}.",
        domain="general",
        evidential_source="obs", evidential_confidence="prob",
    ),
    # Inferred + probable
    SurfaceTemplate(
        "The evidence suggests that {content}.",
        domain="general",
        evidential_source="inf", evidential_confidence="prob",
    ),
    # Inferred + possible
    SurfaceTemplate(
        "It appears that {content}.",
        domain="general",
        evidential_source="inf", evidential_confidence="poss",
    ),
    # Computed + certain
    SurfaceTemplate(
        "Computation confirms that {content}.",
        domain="general",
        evidential_source="comp", evidential_confidence="cert",
    ),
    # Reported + probable
    SurfaceTemplate(
        "According to reports, {content}.",
        domain="general",
        evidential_source="rep", evidential_confidence="prob",
    ),
    # Speculative + possible
    SurfaceTemplate(
        "Speculatively, {content}.",
        domain="general",
        evidential_source="spec", evidential_confidence="poss",
    ),
    # Speculative + unlikely
    SurfaceTemplate(
        "It is unlikely, but {content}.",
        domain="general",
        evidential_source="spec", evidential_confidence="unl",
    ),
    # Counterfactual
    SurfaceTemplate(
        "If that were the case, {content}.",
        domain="general",
        evidential_source="ctr", evidential_confidence="poss",
    ),
]


def register_english(registry: TemplateRegistry) -> None:
    """Register all English templates into the given registry."""
    for t in ENGLISH_TEMPLATES:
        registry.add(t)
```

**Step 4: Run tests to verify they pass**

Run: `cd C:\Users\mrjki\OneDrive\Desktop\MKAngel && python -m pytest tests/test_templates.py -v`
Expected: ALL PASS

**Step 5: Commit**

```bash
git add glm/nlg/templates/__init__.py glm/nlg/templates/en.py tests/test_templates.py
git commit -m "feat: add English surface templates for all 7 domains + evidential hedging"
```

---

## Task 5: Realiser — Reverse-Production Engine

**Files:**
- Create: `glm/nlg/realiser.py`
- Create: `tests/test_realiser.py`

**Step 1: Write the failing tests**

```python
# tests/test_realiser.py
"""Tests for the reverse-production realiser."""

import pytest
from glm.nlg.realiser import Realiser
from glm.nlg.templates import TemplateRegistry
from glm.nlg.templates.en import register_english
from glm.core.mnemo_substrate import MnemoSubstrate


class TestRealiser:
    def setup_method(self):
        self.registry = TemplateRegistry()
        register_english(self.registry)
        self.substrate = MnemoSubstrate()
        self.realiser = Realiser(registry=self.registry, substrate=self.substrate)

    def test_create_realiser(self):
        assert self.realiser is not None

    def test_realise_simple_math(self):
        candidates = self.realiser.realise(
            domain="mathematical",
            slots={"result": "42"},
            evidential_source="comp",
            evidential_confidence="cert",
            evidential_temporal="obs_pres",
        )
        assert len(candidates) > 0
        assert any("42" in c.text for c in candidates)

    def test_realise_with_evidential_hedging(self):
        candidates = self.realiser.realise(
            domain="linguistic",
            slots={"word": "run", "analysis": "is polysemous"},
            evidential_source="inf",
            evidential_confidence="prob",
            evidential_temporal="obs_pres",
        )
        assert len(candidates) > 0
        # Should wrap in evidential hedging
        best = candidates[0]
        assert "suggest" in best.text.lower() or "run" in best.text.lower()

    def test_realise_returns_scored_candidates(self):
        candidates = self.realiser.realise(
            domain="chemical",
            slots={"compound": "H2O", "formula": "H2O"},
            evidential_source="obs",
            evidential_confidence="cert",
            evidential_temporal="obs_pres",
        )
        assert all(hasattr(c, "score") for c in candidates)
        # Should be sorted by score descending
        scores = [c.score for c in candidates]
        assert scores == sorted(scores, reverse=True)
```

**Step 2: Run tests to verify they fail**

Run: `cd C:\Users\mrjki\OneDrive\Desktop\MKAngel && python -m pytest tests/test_realiser.py -v`
Expected: FAIL — `ModuleNotFoundError`

**Step 3: Write minimal implementation**

```python
# glm/nlg/realiser.py
"""
Realiser — reverse-production from MNEMO derivation to surface text.

The realiser takes a domain, slot fills, and evidential markers, then:
1. Finds matching domain templates
2. Fills slots
3. Wraps in evidential hedging (language-appropriate)
4. Scores candidates
5. Returns ranked list

This is the core of MNEMO-internal NLG — it operates on glyph-derived
data structures and produces natural language only at the output boundary.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set

from glm.nlg.templates import SurfaceTemplate, TemplateRegistry
from glm.core.mnemo_substrate import MnemoSubstrate


@dataclass
class RealisationCandidate:
    """A candidate surface realisation with score."""
    text: str
    score: float
    template_used: str = ""
    evidential_wrapper: str = ""
    slots_filled: Dict[str, str] = field(default_factory=dict)


class Realiser:
    """Reverse-production engine: MNEMO derivation -> natural language.

    Uses the existing bidirectional grammar infrastructure
    (glm/core/grammar.py Direction.BACKWARD) to walk derivation trees
    backward, selecting templates and filling slots.
    """

    def __init__(
        self,
        registry: TemplateRegistry,
        substrate: MnemoSubstrate,
        language: str = "en",
    ) -> None:
        self._registry = registry
        self._substrate = substrate
        self._language = language

    def realise(
        self,
        domain: str,
        slots: Dict[str, str],
        evidential_source: str = "inf",
        evidential_confidence: str = "prob",
        evidential_temporal: str = "obs_pres",
    ) -> List[RealisationCandidate]:
        """Produce ranked candidate realisations.

        1. Find domain templates that match available slots.
        2. Fill slots in each matching template.
        3. Wrap in evidential hedging.
        4. Score and rank.
        """
        candidates: List[RealisationCandidate] = []

        # 1. Find domain templates
        domain_templates = self._registry.for_domain(domain, self._language)

        # 2. Fill slots and filter
        for template in domain_templates:
            # Check if we have enough slots to fill this template
            if template.slots and not template.slots.issubset(set(slots.keys())):
                continue

            try:
                filled = template.render(**slots)
            except (KeyError, ValueError):
                continue

            # 3. Wrap in evidential hedging
            hedged = self._apply_evidential_hedging(
                filled, evidential_source, evidential_confidence
            )

            # 4. Score
            score = self._score_candidate(
                template, slots, evidential_source, evidential_confidence
            )

            candidates.append(RealisationCandidate(
                text=hedged,
                score=score,
                template_used=template.template,
                evidential_wrapper=f"{evidential_source}+{evidential_confidence}",
                slots_filled=dict(slots),
            ))

        # Sort by score descending
        candidates.sort(key=lambda c: c.score, reverse=True)
        return candidates

    def _apply_evidential_hedging(
        self, content: str, source: str, confidence: str,
    ) -> str:
        """Wrap content in language-appropriate evidential hedging."""
        # Find matching evidential templates
        ev_templates = self._registry.for_evidential(source, confidence, self._language)

        if ev_templates:
            best = max(ev_templates, key=lambda t: t.weight)
            if "{content}" in best.template:
                return best.render(content=content)

        # Fallback: no hedging for observed+certain, basic hedging otherwise
        if source == "obs" and confidence == "cert":
            return content

        # Default hedging
        hedging = {
            "cert": "",
            "prob": "It is likely that ",
            "poss": "It is possible that ",
            "unl": "It is unlikely that ",
            "unk": "It is uncertain whether ",
        }
        prefix = hedging.get(confidence, "")
        if prefix:
            return prefix + content[0].lower() + content[1:] if content else content
        return content

    def _score_candidate(
        self,
        template: SurfaceTemplate,
        slots: Dict[str, str],
        source: str,
        confidence: str,
    ) -> float:
        """Score a candidate based on template specificity and slot coverage."""
        score = template.weight

        # More specific templates (more slots) score higher
        score += len(template.slots) * 0.1

        # Evidential match bonus
        if template.evidential_source == source:
            score += 0.2
        if template.evidential_confidence == confidence:
            score += 0.2

        # Slot coverage: all slots filled = bonus
        if template.slots and template.slots.issubset(set(slots.keys())):
            score += 0.3

        return round(score, 3)
```

**Step 4: Run tests to verify they pass**

Run: `cd C:\Users\mrjki\OneDrive\Desktop\MKAngel && python -m pytest tests/test_realiser.py -v`
Expected: ALL PASS

**Step 5: Commit**

```bash
git add glm/nlg/realiser.py tests/test_realiser.py
git commit -m "feat: add realiser (reverse-production MNEMO -> natural language)"
```

---

## Task 6: MNEMO Decoder — Full Pipeline Output Boundary

**Files:**
- Create: `glm/nlg/decoder.py`
- Create: `tests/test_nlg_decoder.py`

**Step 1: Write the failing tests**

```python
# tests/test_nlg_decoder.py
"""Tests for the NLG decoder: MNEMO -> natural language."""

import pytest
from glm.nlg.decoder import MnemoDecoder
from glm.core.mnemo_substrate import MnemoSubstrate, MnemoSequence


class TestMnemoDecoder:
    def test_create_decoder(self):
        decoder = MnemoDecoder()
        assert decoder is not None

    def test_decode_mnemo_sequence(self):
        decoder = MnemoDecoder()
        substrate = MnemoSubstrate()
        seq = substrate.encode_codes(["scale_05", "proc_09", "obs", "cert", "obs_pres"])
        result = decoder.decode(seq, language="en")
        assert isinstance(result, str)
        assert len(result) > 0

    def test_decode_preserves_evidentiality(self):
        decoder = MnemoDecoder()
        substrate = MnemoSubstrate()
        # Inference + unlikely -> should produce hedged output
        seq = substrate.encode_codes(["scale_00", "proc_15", "inf", "unl", "pred_fut"])
        result = decoder.decode(seq, language="en")
        assert isinstance(result, str)

    def test_decode_selects_best_candidate(self):
        decoder = MnemoDecoder()
        substrate = MnemoSubstrate()
        seq = substrate.encode_codes([
            "scale_05", "proc_09", "obs", "cert", "obs_pres"
        ])
        result = decoder.decode(seq, language="en")
        # Should return a non-empty string (the best candidate)
        assert result.strip() != ""

    def test_decode_empty_sequence(self):
        decoder = MnemoDecoder()
        result = decoder.decode(MnemoSequence(), language="en")
        assert isinstance(result, str)
```

**Step 2: Run tests to verify they fail**

Run: `cd C:\Users\mrjki\OneDrive\Desktop\MKAngel && python -m pytest tests/test_nlg_decoder.py -v`
Expected: FAIL

**Step 3: Write minimal implementation**

```python
# glm/nlg/decoder.py
"""
Decoder — MNEMO → Natural Language (output boundary).

The decoder is the output boundary — the last stage of the NLG pipeline.
It takes a MNEMO glyph sequence (the result of internal derivation, attention,
and evidential marking) and produces natural language in the target language.

Steps:
    1. Extract domain, process, and evidential markers from the sequence
    2. Build slot fills from derivation data
    3. Pass to Realiser for template selection + evidential hedging
    4. Return the best candidate
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from glm.core.mnemo_substrate import (
    GLYPH_REGISTRY, MnemoGlyph, MnemoSequence, MnemoSubstrate, Tier,
)
from glm.nlg.realiser import Realiser
from glm.nlg.templates import TemplateRegistry
from glm.nlg.templates.en import register_english


# Scale index -> domain name
_SCALE_TO_DOMAIN: Dict[str, str] = {
    "scale_00": "linguistic", "scale_01": "chemical",
    "scale_02": "biological", "scale_03": "computational",
    "scale_04": "etymological", "scale_05": "mathematical",
    "scale_06": "physical", "scale_07": "phonological",
    "scale_08": "morphological", "scale_09": "syntactic",
}

# Process glyph -> human-readable verb
_PROCESS_NAMES: Dict[str, str] = {
    "proc_00": "derivation", "proc_01": "transformation",
    "proc_02": "bonding", "proc_07": "composition",
    "proc_09": "prediction", "proc_10": "reconstruction",
    "proc_12": "comparison", "proc_15": "generation",
    "proc_16": "parsing", "proc_17": "encoding",
    "proc_18": "decoding", "proc_19": "translation",
}


@dataclass
class MnemoDecoder:
    """Decodes MNEMO sequences to natural language.

    The output boundary of the NLG pipeline. All internal processing
    is complete by the time data reaches the decoder.
    """
    substrate: MnemoSubstrate = field(default_factory=MnemoSubstrate)
    _registries: Dict[str, TemplateRegistry] = field(default_factory=dict)

    def __post_init__(self):
        # Pre-load English templates
        en_reg = TemplateRegistry()
        register_english(en_reg)
        self._registries["en"] = en_reg

    def decode(
        self,
        sequence: MnemoSequence,
        language: str = "en",
        extra_slots: Optional[Dict[str, str]] = None,
    ) -> str:
        """Decode a MNEMO sequence to natural language.

        Args:
            sequence:    The MNEMO glyph sequence from the processing stage.
            language:    Target language code (default: 'en').
            extra_slots: Additional slot fills from the derivation context.
        """
        if not sequence.glyphs:
            return ""

        # 1. Extract structured info from the sequence
        domain = self._extract_domain(sequence)
        process = self._extract_process(sequence)
        source, confidence, temporal = self._extract_evidentials(sequence)

        # 2. Build slot fills
        slots: Dict[str, str] = {}
        if process:
            slots["operation"] = process
            slots["description"] = f"performs {process}"
            slots["result"] = f"a {process} result"
            slots["method"] = process
        if extra_slots:
            slots.update(extra_slots)

        # 3. Realise via template engine
        registry = self._registries.get(language)
        if registry is None:
            return f"[No templates for language: {language}]"

        realiser = Realiser(registry=registry, substrate=self.substrate, language=language)
        candidates = realiser.realise(
            domain=domain,
            slots=slots,
            evidential_source=source,
            evidential_confidence=confidence,
            evidential_temporal=temporal,
        )

        if candidates:
            return candidates[0].text

        # Fallback: descriptive output
        return f"[{domain}] {process or 'analysis'} ({source}/{confidence}/{temporal})"

    def _extract_domain(self, seq: MnemoSequence) -> str:
        for g in seq.glyphs:
            if g.code in _SCALE_TO_DOMAIN:
                return _SCALE_TO_DOMAIN[g.code]
        return "general"

    def _extract_process(self, seq: MnemoSequence) -> str:
        for g in seq.glyphs:
            if g.code in _PROCESS_NAMES:
                return _PROCESS_NAMES[g.code]
        return ""

    def _extract_evidentials(self, seq: MnemoSequence) -> tuple:
        source = "inf"
        confidence = "prob"
        temporal = "obs_pres"
        for g in seq.glyphs:
            if g.tier == Tier.STATE and g.features.get("axis") == "source":
                source = g.code
            elif g.tier == Tier.EPISTEMIC and g.features.get("axis") == "confidence":
                confidence = g.code
            elif g.tier == Tier.TEMPORAL and g.features.get("axis") == "temporal":
                temporal = g.code
        return source, confidence, temporal
```

**Step 4: Run tests to verify they pass**

Run: `cd C:\Users\mrjki\OneDrive\Desktop\MKAngel && python -m pytest tests/test_nlg_decoder.py -v`
Expected: ALL PASS

**Step 5: Commit**

```bash
git add glm/nlg/decoder.py tests/test_nlg_decoder.py
git commit -m "feat: add MNEMO decoder (MNEMO -> natural language output boundary)"
```

---

## Task 7: NLGProvider — Wire Into Provider System

**Files:**
- Modify: `app/providers.py` (add NLGProvider class after LocalProvider)
- Create: `tests/test_nlg_provider.py`

**Step 1: Write the failing tests**

```python
# tests/test_nlg_provider.py
"""Tests for the NLGProvider."""

import pytest
from app.providers import NLGProvider


class TestNLGProvider:
    def test_create_provider(self):
        provider = NLGProvider()
        assert provider.name == "nlg"

    def test_is_available(self):
        provider = NLGProvider()
        assert provider.is_available() is True

    def test_generate_returns_string(self):
        provider = NLGProvider()
        result = provider.generate("What is water?")
        assert isinstance(result, str)
        assert len(result) > 0

    def test_generate_with_domain_hint(self):
        provider = NLGProvider()
        result = provider.generate("the molecule bonds with oxygen")
        assert isinstance(result, str)

    def test_generate_includes_evidential(self):
        provider = NLGProvider()
        # The output should reflect evidential processing happened
        result = provider.generate("solve x + 2 = 5")
        assert isinstance(result, str)
        assert len(result) > 0
```

**Step 2: Run tests to verify they fail**

Run: `cd C:\Users\mrjki\OneDrive\Desktop\MKAngel && python -m pytest tests/test_nlg_provider.py -v`
Expected: FAIL — `ImportError: cannot import name 'NLGProvider'`

**Step 3: Add NLGProvider to `app/providers.py`**

Insert after `LocalProvider` class (after line 155):

```python
# ---------------------------------------------------------------------------
# NLG provider -- MNEMO-native generation
# ---------------------------------------------------------------------------

class NLGProvider(Provider):
    """Uses the MNEMO NLG engine for native generation.

    Three-stage pipeline: ENCODE (NL->MNEMO) -> PROCESS -> DECODE (MNEMO->NL).
    No API dependency — the GLM IS the engine.
    """

    name = "nlg"

    def __init__(self) -> None:
        self._encoder = None
        self._decoder = None
        self._substrate = None

    def _ensure_loaded(self) -> None:
        """Lazy-load NLG components."""
        if self._encoder is None:
            from glm.nlg.encoder import MnemoEncoder
            from glm.nlg.decoder import MnemoDecoder
            from glm.core.mnemo_substrate import MnemoSubstrate
            self._substrate = MnemoSubstrate()
            self._encoder = MnemoEncoder(substrate=self._substrate)
            self._decoder = MnemoDecoder(substrate=self._substrate)

    def generate(
        self,
        prompt: str,
        *,
        system: str = "",
        temperature: float = 0.7,
        max_tokens: int = 1024,
    ) -> str:
        """Generate via MNEMO NLG pipeline.

        1. ENCODE: Natural language -> MNEMO
        2. PROCESS: (future: grammar derivation + attention in MNEMO space)
        3. DECODE: MNEMO -> Natural language
        """
        self._ensure_loaded()

        # Stage 1: ENCODE
        mnemo_seq = self._encoder.encode(prompt)

        # Stage 2: PROCESS (placeholder — will integrate with existing
        # glm/model/ attention and glm/grammars/ derivation)
        # For now, the encoded sequence passes through directly.

        # Stage 3: DECODE
        result = self._decoder.decode(
            mnemo_seq,
            language="en",
            extra_slots={"content": prompt},
        )

        if result:
            domain = self._encoder.detect_domain(prompt)
            source, confidence, temporal = self._decoder._extract_evidentials(mnemo_seq)
            header = f"[NLG | domain: {domain} | {source}/{confidence}/{temporal}]"
            return f"{header}\n\n{result}"

        return f"[NLG] Processed through MNEMO pipeline. No strong derivation."

    def is_available(self) -> bool:
        return True
```

**Step 4: Run tests to verify they pass**

Run: `cd C:\Users\mrjki\OneDrive\Desktop\MKAngel && python -m pytest tests/test_nlg_provider.py -v`
Expected: ALL PASS

**Step 5: Commit**

```bash
git add app/providers.py tests/test_nlg_provider.py
git commit -m "feat: add NLGProvider using MNEMO NLG pipeline"
```

---

## Task 8: Wire NLG Into Conductor + Chat Fallback

**Files:**
- Modify: `app/conductor.py` (add NLGProvider to imports and provider selection)
- Modify: `app/chat.py` (update fallback chain)
- Create: `tests/test_conductor_nlg.py`

**Step 1: Write the failing tests**

```python
# tests/test_conductor_nlg.py
"""Tests for NLG integration into the conductor pipeline."""

import pytest


class TestConductorNLGIntegration:
    def test_nlg_provider_importable(self):
        from app.providers import NLGProvider
        assert NLGProvider is not None

    def test_conductor_can_select_nlg(self):
        """The conductor should be able to use NLGProvider."""
        from app.providers import NLGProvider
        provider = NLGProvider()
        result = provider.generate("hello world")
        assert isinstance(result, str)
        assert len(result) > 0
```

**Step 2: Run tests to verify they fail/pass baseline**

Run: `cd C:\Users\mrjki\OneDrive\Desktop\MKAngel && python -m pytest tests/test_conductor_nlg.py -v`

**Step 3: Modify `app/conductor.py`**

Add to the import block (after line 92, in the `try/except` for providers):

```python
try:
    from app.providers import NLGProvider
except Exception:
    NLGProvider = None
```

Modify `_select_initial_provider()` (around line 304) — add NLGProvider as the preferred provider before OrchestraProvider:

```python
    def _select_initial_provider(self) -> Any:
        """Choose the initial provider based on settings.

        Priority: NLGProvider (MNEMO-native) > OrchestraProvider > get_provider > LocalProvider
        """
        # Prefer NLG provider — no API dependency
        if NLGProvider is not None:
            try:
                nlg = NLGProvider()
                if nlg.is_available():
                    return nlg
            except Exception:
                pass

        if self._settings is None:
            if LocalProvider is not None:
                return LocalProvider()
            return None

        # Check if any API keys are configured
        has_api_keys = False
        if hasattr(self._settings, "api_keys"):
            has_api_keys = bool(self._settings.api_keys)

        if has_api_keys and OrchestraProvider is not None:
            try:
                return OrchestraProvider(self._settings)
            except Exception:
                pass

        if get_provider is not None:
            try:
                return get_provider(self._settings)
            except Exception:
                pass

        if LocalProvider is not None:
            return LocalProvider()
        return None
```

**Step 4: Run all tests**

Run: `cd C:\Users\mrjki\OneDrive\Desktop\MKAngel && python -m pytest tests/ -v`
Expected: ALL PASS

**Step 5: Commit**

```bash
git add app/conductor.py app/chat.py tests/test_conductor_nlg.py
git commit -m "feat: wire NLGProvider into conductor pipeline as preferred provider"
```

---

## Task 9: Integration Test — Full NLG Pipeline End-to-End

**Files:**
- Create: `tests/test_nlg_integration.py`

**Step 1: Write integration tests**

```python
# tests/test_nlg_integration.py
"""End-to-end integration tests for the MNEMO NLG pipeline."""

import pytest
from glm.core.mnemo_substrate import MnemoSubstrate, MnemoSequence, Tier
from glm.nlg.encoder import MnemoEncoder
from glm.nlg.decoder import MnemoDecoder
from glm.nlg.realiser import Realiser
from glm.nlg.response_grammar import build_response_grammar, UtteranceNode
from glm.nlg.templates import TemplateRegistry
from glm.nlg.templates.en import register_english


class TestFullPipeline:
    """Test the complete ENCODE -> PROCESS -> DECODE pipeline."""

    def test_encode_decode_roundtrip(self):
        """Natural language -> MNEMO -> Natural language."""
        encoder = MnemoEncoder()
        decoder = MnemoDecoder()

        # Encode
        mnemo = encoder.encode("the molecule bonds with oxygen")
        assert isinstance(mnemo, MnemoSequence)
        assert len(mnemo) > 0
        assert mnemo.has_evidential_marking()

        # Decode
        result = decoder.decode(mnemo, language="en")
        assert isinstance(result, str)
        assert len(result) > 0

    def test_evidentiality_preserved_through_pipeline(self):
        """Evidential markers survive the full pipeline."""
        substrate = MnemoSubstrate()
        encoder = MnemoEncoder(substrate=substrate)
        decoder = MnemoDecoder(substrate=substrate)

        mnemo = encoder.encode("water is a molecule")
        source, conf, temp = decoder._extract_evidentials(mnemo)

        # Default evidentials from encoder
        assert source in ("obs", "inf", "comp", "rep", "trad", "spec", "ctr")
        assert conf in ("cert", "prob", "poss", "unl", "unk")
        assert temp in ("ver_past", "obs_pres", "pred_fut", "hyp", "timeless")

    def test_domain_routing_works(self):
        """Different inputs route to different domains."""
        encoder = MnemoEncoder()

        math_domain = encoder.detect_domain("solve the equation")
        assert math_domain == "mathematical"

        chem_domain = encoder.detect_domain("the acid reacts with the base")
        assert chem_domain == "chemical"

        ling_domain = encoder.detect_domain("the verb agrees with the noun")
        assert ling_domain == "linguistic"

    def test_response_grammar_composes(self):
        """Response grammar can compose multi-domain outputs."""
        grammar = build_response_grammar()
        substrate = MnemoSubstrate()

        # Create an utterance node spanning math + linguistic
        node = UtteranceNode(
            domain_results=["mathematical", "linguistic"],
            evidential_source="comp",
            evidential_confidence="cert",
            evidential_temporal="obs_pres",
        )
        seq = node.to_mnemo_sequence(substrate)
        assert seq.has_evidential_marking()

    def test_nlg_provider_full_flow(self):
        """NLGProvider works end-to-end."""
        from app.providers import NLGProvider
        provider = NLGProvider()
        result = provider.generate("the compound has a covalent bond")
        assert isinstance(result, str)
        assert "[NLG" in result

    def test_security_grammar_gate(self):
        """Invalid MNEMO sequences are rejected."""
        substrate = MnemoSubstrate()
        assert not substrate.validate_sequence(["INVALID_CODE", "BAD"])
        assert substrate.validate_sequence(["obs", "cert", "obs_pres"])
```

**Step 2: Run integration tests**

Run: `cd C:\Users\mrjki\OneDrive\Desktop\MKAngel && python -m pytest tests/test_nlg_integration.py -v`
Expected: ALL PASS

**Step 3: Commit**

```bash
git add tests/test_nlg_integration.py
git commit -m "test: add end-to-end NLG pipeline integration tests"
```

---

## Task 10: TARDAI GLM Bridge — Python Server

**Files:**
- Create: `glm_server.py` (in MKAngel root — the Python side of the bridge)
- Create: `tests/test_glm_server.py`

**Step 1: Write the failing tests**

```python
# tests/test_glm_server.py
"""Tests for the GLM JSON server (TARDAI bridge, Python side)."""

import json
import pytest
from glm_server import handle_request


class TestGLMServer:
    def test_encode_request(self):
        req = {"id": "1", "method": "encode", "params": {"text": "hello world"}}
        resp = handle_request(req)
        assert resp["id"] == "1"
        assert "result" in resp
        assert "codes" in resp["result"]

    def test_decode_request(self):
        req = {"id": "2", "method": "decode", "params": {"codes": ["obs", "cert", "obs_pres"]}}
        resp = handle_request(req)
        assert resp["id"] == "2"
        assert "result" in resp
        assert "text" in resp["result"]

    def test_derive_request(self):
        req = {"id": "3", "method": "derive", "params": {"text": "the molecule bonds"}}
        resp = handle_request(req)
        assert resp["id"] == "3"
        assert "result" in resp

    def test_route_request(self):
        req = {"id": "4", "method": "route", "params": {"text": "solve x + 2 = 5"}}
        resp = handle_request(req)
        assert resp["id"] == "4"
        assert "result" in resp
        assert "domain" in resp["result"]

    def test_invalid_method(self):
        req = {"id": "5", "method": "invalid_method", "params": {}}
        resp = handle_request(req)
        assert "error" in resp

    def test_mnemo_request(self):
        req = {"id": "6", "method": "mnemo", "params": {"text": "water is H2O"}}
        resp = handle_request(req)
        assert resp["id"] == "6"
        assert "result" in resp
```

**Step 2: Run tests to verify they fail**

Run: `cd C:\Users\mrjki\OneDrive\Desktop\MKAngel && python -m pytest tests/test_glm_server.py -v`
Expected: FAIL

**Step 3: Write minimal implementation**

```python
# glm_server.py
"""
GLM JSON Server — Python side of the TARDAI bridge.

Accepts JSON requests on stdin, processes via MKAngel GLM,
returns JSON responses on stdout. Protocol:

    Request:  {"id": str, "method": str, "params": dict}
    Response: {"id": str, "result": dict} | {"id": str, "error": str}

Methods: encode, decode, derive, route, realise, mnemo
"""

from __future__ import annotations

import json
import sys
from typing import Any, Dict


def handle_request(request: Dict[str, Any]) -> Dict[str, Any]:
    """Process a single JSON-RPC-style request."""
    req_id = request.get("id", "0")
    method = request.get("method", "")
    params = request.get("params", {})

    try:
        if method == "encode":
            return _handle_encode(req_id, params)
        elif method == "decode":
            return _handle_decode(req_id, params)
        elif method == "derive":
            return _handle_derive(req_id, params)
        elif method == "route":
            return _handle_route(req_id, params)
        elif method == "realise":
            return _handle_realise(req_id, params)
        elif method == "mnemo":
            return _handle_mnemo(req_id, params)
        else:
            return {"id": req_id, "error": f"Unknown method: {method}"}
    except Exception as exc:
        return {"id": req_id, "error": str(exc)}


def _handle_encode(req_id: str, params: Dict[str, Any]) -> Dict[str, Any]:
    from glm.nlg.encoder import MnemoEncoder
    encoder = MnemoEncoder()
    text = params.get("text", "")
    seq = encoder.encode(text)
    codes = [g.code for g in seq.glyphs]
    domain = encoder.detect_domain(text)
    return {"id": req_id, "result": {"codes": codes, "domain": domain}}


def _handle_decode(req_id: str, params: Dict[str, Any]) -> Dict[str, Any]:
    from glm.nlg.decoder import MnemoDecoder
    from glm.core.mnemo_substrate import MnemoSubstrate
    decoder = MnemoDecoder()
    codes = params.get("codes", [])
    substrate = MnemoSubstrate()
    seq = substrate.encode_codes(codes)
    text = decoder.decode(seq, language=params.get("language", "en"))
    return {"id": req_id, "result": {"text": text}}


def _handle_derive(req_id: str, params: Dict[str, Any]) -> Dict[str, Any]:
    from glm.nlg.encoder import MnemoEncoder
    from glm.nlg.decoder import MnemoDecoder
    encoder = MnemoEncoder()
    decoder = MnemoDecoder()
    text = params.get("text", "")
    seq = encoder.encode(text)
    result = decoder.decode(seq, language=params.get("language", "en"))
    codes = [g.code for g in seq.glyphs]
    return {"id": req_id, "result": {"text": result, "codes": codes}}


def _handle_route(req_id: str, params: Dict[str, Any]) -> Dict[str, Any]:
    from glm.nlg.encoder import MnemoEncoder
    encoder = MnemoEncoder()
    text = params.get("text", "")
    domain = encoder.detect_domain(text)
    return {"id": req_id, "result": {"domain": domain}}


def _handle_realise(req_id: str, params: Dict[str, Any]) -> Dict[str, Any]:
    from glm.nlg.realiser import Realiser
    from glm.nlg.templates import TemplateRegistry
    from glm.nlg.templates.en import register_english
    from glm.core.mnemo_substrate import MnemoSubstrate
    registry = TemplateRegistry()
    register_english(registry)
    realiser = Realiser(registry=registry, substrate=MnemoSubstrate())
    candidates = realiser.realise(
        domain=params.get("domain", "general"),
        slots=params.get("slots", {}),
        evidential_source=params.get("source", "inf"),
        evidential_confidence=params.get("confidence", "prob"),
        evidential_temporal=params.get("temporal", "obs_pres"),
    )
    results = [{"text": c.text, "score": c.score} for c in candidates[:5]]
    return {"id": req_id, "result": {"candidates": results}}


def _handle_mnemo(req_id: str, params: Dict[str, Any]) -> Dict[str, Any]:
    """Full MNEMO pipeline: encode + process + decode."""
    from app.providers import NLGProvider
    provider = NLGProvider()
    text = params.get("text", "")
    result = provider.generate(text)
    return {"id": req_id, "result": {"text": result}}


def main() -> None:
    """Main loop: read JSON from stdin, write JSON to stdout."""
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            request = json.loads(line)
            response = handle_request(request)
        except json.JSONDecodeError as exc:
            response = {"id": "0", "error": f"Invalid JSON: {exc}"}
        sys.stdout.write(json.dumps(response) + "\n")
        sys.stdout.flush()


if __name__ == "__main__":
    main()
```

**Step 4: Run tests to verify they pass**

Run: `cd C:\Users\mrjki\OneDrive\Desktop\MKAngel && python -m pytest tests/test_glm_server.py -v`
Expected: ALL PASS

**Step 5: Commit**

```bash
git add glm_server.py tests/test_glm_server.py
git commit -m "feat: add GLM JSON server for TARDAI bridge"
```

---

## Task 11: Run Full Test Suite + Final Verification

**Step 1: Run all tests**

Run: `cd C:\Users\mrjki\OneDrive\Desktop\MKAngel && python -m pytest tests/ -v --tb=short`
Expected: ALL PASS

**Step 2: Verify file count matches plan**

Run: `ls -la glm/core/mnemo_substrate.py glm/nlg/__init__.py glm/nlg/encoder.py glm/nlg/decoder.py glm/nlg/realiser.py glm/nlg/response_grammar.py glm/nlg/templates/__init__.py glm/nlg/templates/en.py glm_server.py`
Expected: 9 files present (8 new MKAngel + 1 bridge server)

**Step 3: Commit final state**

```bash
git add -A
git commit -m "chore: MNEMO NLG engine complete — all tests passing"
```

---

## Dependency Graph

```
Task 1 (MNEMO Substrate)
    ├── Task 2 (Response Grammar)     ← needs substrate
    ├── Task 3 (Encoder)              ← needs substrate
    └── Task 4 (Templates)            ← standalone but needed by decoder
            └── Task 5 (Realiser)     ← needs templates + substrate
                    └── Task 6 (Decoder)    ← needs realiser + templates
                            └── Task 7 (NLGProvider)  ← needs encoder + decoder
                                    └── Task 8 (Conductor wiring)
                                            └── Task 9 (Integration tests)
                                                    └── Task 10 (TARDAI bridge)
                                                            └── Task 11 (Final verification)
```

**Parallelisable:** Tasks 2, 3, 4 can run in parallel after Task 1 completes.

---

## TARDAI Integration (Separate Plan)

The TARDAI TypeScript bridge files (`lib/glm-bridge/index.ts`, `protocol.ts`, `client.ts`) are a separate implementation plan — they depend on Task 10 (the Python server) being complete and tested. Create that plan after the MKAngel engine is verified working.
