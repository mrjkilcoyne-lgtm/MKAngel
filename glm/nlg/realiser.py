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
        ev_templates = self._registry.for_evidential(source, confidence, self._language)

        if ev_templates:
            best = max(ev_templates, key=lambda t: t.weight)
            if "{content}" in best.template:
                return best.render(content=content)

        # Fallback: no hedging for observed+certain
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
