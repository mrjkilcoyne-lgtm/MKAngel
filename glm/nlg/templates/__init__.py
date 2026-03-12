"""Surface template registry for NLG decoding."""

from __future__ import annotations

import re
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

    def add(self, template: SurfaceTemplate, language: Optional[str] = None) -> None:
        if language is not None:
            template.language = language
        self._templates.append(template)

    def for_domain(self, domain: str, language: str = "en") -> List[SurfaceTemplate]:
        return [
            t for t in self._templates
            if t.domain == domain and t.language == language
        ]

    def for_evidential(
        self,
        source: Optional[str] = None,
        confidence: Optional[str] = None,
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
