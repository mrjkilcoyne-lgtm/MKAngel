"""
Encoder — Natural Language -> MNEMO (input boundary).

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
        "simplify", "factor", "zeroes", "tangent", "area", "integrate",
        "derive", "cos", "sin", "log", "sqrt", "sum", "product",
        "x^", "x+", "x-", "x=", "x*", "^2", "^3",
    ],
    "physical": [
        "force", "energy", "mass", "velocity", "acceleration", "gravity",
        "quantum", "wave", "particle", "field", "electro", "magnetic",
        "thermo", "entropy", "momentum", "relativity", "speed of light",
        "planck", "boltzmann", "constant", "newton", "joule", "watt",
    ],
    "computational": [
        "algorithm", "variable", "loop", "class", "object",
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
# Concept extraction: keyword -> glyph code
# ---------------------------------------------------------------------------

_CONCEPT_MAP: Dict[str, str] = {
    "derive": "proc_00", "transform": "proc_01", "bond": "proc_02",
    "split": "proc_03", "merge": "proc_04", "predict": "proc_09",
    "reconstruct": "proc_10", "compare": "proc_12", "generate": "proc_15",
    "parse": "proc_16", "encode": "proc_17", "decode": "proc_18",
    "translate": "proc_19", "compose": "proc_07", "select": "proc_13",
    "agrees": "rel_01", "governs": "rel_00", "contains": "rel_08",
    "entails": "rel_10", "implies": "rel_12",
    "entity": "ont_00", "property": "ont_01", "relation": "ont_02",
    "event": "ont_03", "pattern": "ont_07",
}


@dataclass
class MnemoEncoder:
    """Encodes natural language input to MNEMO glyph sequences.

    The input boundary — the last place natural language exists before
    entering the all-MNEMO processing pipeline.
    """
    substrate: MnemoSubstrate = field(default_factory=MnemoSubstrate)

    def encode(self, text: str) -> MnemoSequence:
        """Encode natural language text into a MNEMO sequence."""
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
            clean = re.sub(r"[^\w]", "", word)
            if clean in _CONCEPT_MAP:
                codes.append(_CONCEPT_MAP[clean])

        # Default process glyph if none matched
        if not any(c.startswith("proc_") for c in codes):
            codes.append("proc_15")  # generate

        # 3. Default evidential marking (the true timeline is grammatical)
        codes.append("rep")        # source: reported (user told us)
        codes.append("prob")       # confidence: probable
        codes.append("obs_pres")   # temporal: present

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
            return "linguistic"

        return max(scores, key=lambda k: scores[k])
