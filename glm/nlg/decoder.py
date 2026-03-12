"""
Decoder — MNEMO -> Natural Language (output boundary).

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
from glm.nlg.templates.fr import register_french
from glm.nlg.templates.es import register_spanish
from glm.nlg.templates.de import register_german
from glm.nlg.templates.tr import register_turkish
from glm.nlg.templates.cy import register_welsh


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
        # Build a single registry with all languages
        registry = TemplateRegistry()
        register_english(registry)
        register_french(registry)
        register_spanish(registry)
        register_german(registry)
        register_turkish(registry)
        register_welsh(registry)
        # Index by language code for fast lookup
        for lang in ("en", "fr", "es", "de", "tr", "cy"):
            self._registries[lang] = registry

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
