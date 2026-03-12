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

StrangeLoop: response -> domain analysis -> cross-domain insight -> response
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
    source: str
    confidence: str
    temporal: str


@dataclass
class DomainAnalysisNode:
    """A result from one domain grammar derivation."""
    domain: str
    derivation_codes: List[str] = field(default_factory=list)
    confidence: float = 1.0


@dataclass
class UtteranceNode:
    """A complete response — domain results + evidential marking.

    The root of the response grammar's derivation tree.
    """
    domain_results: List[str] = field(default_factory=list)
    evidential_source: str = "inf"
    evidential_confidence: str = "prob"
    evidential_temporal: str = "obs_pres"
    content_codes: List[str] = field(default_factory=list)

    def to_mnemo_sequence(self, substrate: MnemoSubstrate) -> MnemoSequence:
        """Convert this utterance to a MNEMO glyph sequence."""
        codes = []
        codes.append("meta_25")  # utterance_start

        for domain in self.domain_results:
            domain_glyph = f"scale_{_domain_to_scale_index(domain)}"
            codes.append(domain_glyph)
            codes.append("meta_27")  # domain_analysis

        codes.extend(self.content_codes)

        # Evidential marking (mandatory — the true timeline is grammatical)
        codes.append(self.evidential_source)
        codes.append(self.evidential_confidence)
        codes.append(self.evidential_temporal)

        codes.append("meta_26")  # utterance_end
        return substrate.encode_codes(codes)


def _domain_to_scale_index(domain: str) -> str:
    mapping = {
        "mathematical": "05", "linguistic": "00", "biological": "02",
        "chemical": "01", "computational": "03", "etymological": "04",
        "physical": "06", "phonological": "07", "morphological": "08",
        "syntactic": "09", "semantic": "10", "response": "33",
    }
    return mapping.get(domain, "32")


# ---------------------------------------------------------------------------
# Grammar builder
# ---------------------------------------------------------------------------

def build_response_grammar() -> Grammar:
    """Build the 8th domain grammar: Utterance composition."""
    productions = [
        Production(lhs="Utterance", rhs=["DomainAnalysis", "EvidentialMarker"],
                   name="utterance_base"),
        Production(lhs="DomainAnalysis",
                   rhs=["DomainAnalysis", "Conjunction", "DomainAnalysis"],
                   name="multi_domain"),
        Production(lhs="DomainAnalysis", rhs=["MathResult"], name="math_analysis"),
        Production(lhs="DomainAnalysis", rhs=["LinguisticResult"], name="ling_analysis"),
        Production(lhs="DomainAnalysis", rhs=["BiologyResult"], name="bio_analysis"),
        Production(lhs="DomainAnalysis", rhs=["ChemistryResult"], name="chem_analysis"),
        Production(lhs="DomainAnalysis", rhs=["PhysicsResult"], name="phys_analysis"),
        Production(lhs="DomainAnalysis", rhs=["ComputationResult"], name="comp_analysis"),
        Production(lhs="DomainAnalysis", rhs=["EtymologyResult"], name="etym_analysis"),
        Production(lhs="EvidentialMarker",
                   rhs=["SourceGlyph", "ConfidenceGlyph", "TemporalGlyph"],
                   name="evidential_triple"),
    ]

    rules = [
        Rule(name="and_conj", pattern="Conjunction", result="and"),
        Rule(name="but_conj", pattern="Conjunction", result="but"),
        Rule(name="therefore_conj", pattern="Conjunction", result="therefore"),
    ]

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
