"""Domain-specific processors for Stage 2 of the NLG pipeline.

Each processor:
1. Receives the input text + MNEMO sequence from Stage 1 (ENCODE)
2. Queries a domain-specific public API (or built-in knowledge base)
3. Returns slot fills that the decoder's Realiser uses for template rendering

The ProcessorDispatcher routes to the correct processor by detected domain.
All API failures are graceful — an empty dict means the decoder falls back
to its existing behaviour.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Dict, Optional

from glm.core.mnemo_substrate import MnemoSequence


class DomainProcessor(ABC):
    """Base class for domain-specific processors."""

    domain: str = "general"

    @abstractmethod
    def process(self, text: str, mnemo_seq: MnemoSequence) -> Dict[str, str]:
        """Process *text* and return slot fills for the Realiser.

        Must never raise — return ``{}`` on failure.
        """
        ...


class ProcessorDispatcher:
    """Routes processing to domain-specific processors."""

    def __init__(self) -> None:
        self._processors: Dict[str, DomainProcessor] = {}

    def register(self, processor: DomainProcessor) -> None:
        self._processors[processor.domain] = processor

    @property
    def domains(self) -> list:
        return list(self._processors.keys())

    def process(
        self, domain: str, text: str, mnemo_seq: MnemoSequence,
    ) -> Dict[str, str]:
        """Dispatch to the right processor, returning slot fills."""
        processor = self._processors.get(domain)
        if processor is None:
            return {}
        try:
            return processor.process(text, mnemo_seq)
        except Exception:
            return {}


def create_default_dispatcher() -> ProcessorDispatcher:
    """Build a dispatcher with all 7 domain processors registered."""
    from .mathematical import MathProcessor
    from .linguistic import LinguisticProcessor
    from .chemical import ChemicalProcessor
    from .biological import BiologicalProcessor
    from .physical import PhysicalProcessor
    from .computational import ComputationalProcessor
    from .etymological import EtymologicalProcessor

    dispatcher = ProcessorDispatcher()
    for cls in [
        MathProcessor, LinguisticProcessor, ChemicalProcessor,
        BiologicalProcessor, PhysicalProcessor, ComputationalProcessor,
        EtymologicalProcessor,
    ]:
        dispatcher.register(cls())
    return dispatcher
