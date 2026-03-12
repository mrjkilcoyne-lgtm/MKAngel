"""Core grammar primitives — the foundational scales."""

from .grammar import Rule, Production, Grammar, StrangeLoop, Direction
from .lexicon import Lexicon, LexicalEntry
from .engine import DerivationEngine, Derivation, DerivationTree
from .reasoning import ReasoningEngine, ReasoningChain, ReasoningStep

# Substrate imports are deferred — the module may not exist yet.
try:
    from .substrate import Substrate, Symbol, Sequence
except ImportError:
    Substrate = Symbol = Sequence = None  # type: ignore[assignment,misc]

__all__ = [
    "Rule",
    "Production",
    "Grammar",
    "StrangeLoop",
    "Direction",
    "Substrate",
    "Symbol",
    "Sequence",
    "Lexicon",
    "LexicalEntry",
    "DerivationEngine",
    "Derivation",
    "DerivationTree",
    "ReasoningEngine",
    "ReasoningChain",
    "ReasoningStep",
]
