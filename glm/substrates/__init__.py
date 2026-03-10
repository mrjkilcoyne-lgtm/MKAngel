"""
Substrates — the media that grammars operate on.

Each substrate is a complete symbolic system with its own alphabet,
combination rules, feature system, and transformation rules.  Despite
their surface differences, all substrates share deep structural
isomorphisms: valence, constituency, recursion, agreement, and the
possibility of strange loops (self-reference).

Available substrates
────────────────────
phonological   Sound patterns — phonemes, syllables, sound change
morphological  Word formation — morphemes, affixation, compounding
molecular      Chemistry — atoms, bonds, molecular grammar
symbolic       Code / formal logic — tokens, syntax trees, types
"""

from .phonological import Phoneme, PhonologicalSubstrate, Syllable
from .morphological import (
    GrammaticalFunction,
    Morpheme,
    MorphemeType,
    MorphologicalSubstrate,
)
from .molecular import Atom, Bond, MolecularSubstrate
from .symbolic import (
    ASTNode,
    SymbolicSubstrate,
    Token,
    TokenCategory,
)

__all__ = [
    # Phonological
    "Phoneme",
    "PhonologicalSubstrate",
    "Syllable",
    # Morphological
    "GrammaticalFunction",
    "Morpheme",
    "MorphemeType",
    "MorphologicalSubstrate",
    # Molecular
    "Atom",
    "Bond",
    "MolecularSubstrate",
    # Symbolic
    "ASTNode",
    "SymbolicSubstrate",
    "Token",
    "TokenCategory",
]
