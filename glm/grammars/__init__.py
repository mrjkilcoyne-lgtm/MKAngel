"""Domain-specific grammar sets for the Grammar Language Model.

Each module provides builder functions that return Grammar objects populated
with real, meaningful rules for their domain.  These grammars are the compact
"scales" from which infinite variety is generated — they encode the deep
structural regularities of natural language, chemistry, biology, code, and
the historical evolution of words.

Cross-domain isomorphisms (fugues) emerge naturally: phrase-structure rules
mirror molecular-structure rules; evolutionary grammars echo sound-change
grammars; type systems parallel chemical valence.
"""

from .linguistic import (
    build_syntactic_grammar,
    build_phonological_grammar,
    build_morphological_grammar,
)
from .etymological import (
    build_etymology_grammar,
    build_substrate_transfer_grammar,
    build_cognate_detection_grammar,
)
from .chemical import (
    build_bonding_grammar,
    build_reaction_grammar,
    build_molecular_grammar,
)
from .biological import (
    build_genetic_grammar,
    build_protein_grammar,
    build_evolutionary_grammar,
)
from .computational import (
    build_syntax_grammar,
    build_type_grammar,
    build_pattern_grammar,
)

__all__ = [
    # Linguistic
    "build_syntactic_grammar",
    "build_phonological_grammar",
    "build_morphological_grammar",
    # Etymological
    "build_etymology_grammar",
    "build_substrate_transfer_grammar",
    "build_cognate_detection_grammar",
    # Chemical
    "build_bonding_grammar",
    "build_reaction_grammar",
    "build_molecular_grammar",
    # Biological
    "build_genetic_grammar",
    "build_protein_grammar",
    "build_evolutionary_grammar",
    # Computational
    "build_syntax_grammar",
    "build_type_grammar",
    "build_pattern_grammar",
]
