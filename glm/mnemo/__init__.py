"""
mnemo — Hyper-compressed encoding language for the Grammar Language Model.

MNEMO is a maximally dense grammar where 150 characters can represent
75 billion parameter possibilities.  The key insight: if you have a grammar
model that understands deep structural rules, you don't need to spell
everything out.  Short codes reference grammar rules that the engine
expands into full meaning — like how musical notation compresses an
orchestra performance into marks on paper.

    Position 1 — Domain       (L C B X E M *)
    Position 2 — Operation    (p r f t i d c s a)
    Position 3 — Modifier     (+ - ~ ! ? @ # &)  optional

Modules
-------
    language    MNEMO language specification: tokens, grammar, encode/decode
    codec       Compression/decompression between full representations and MNEMO
    interpreter Execute MNEMO programs against the grammar engine
    rules       Production rules, vocabulary, and validation
"""

from .rules import (
    DOMAIN_CODES,
    OPERATION_CODES,
    MODIFIER_CODES,
    MNEMO_VOCABULARY,
    COMPOUND_OPERATIONS,
    META_OPERATIONS,
    validate,
    validate_token,
    lookup,
    describe,
)
from .language import MnemoToken, MnemoGrammar, encode, decode, expand
from .codec import MnemoCodec
from .interpreter import MnemoInterpreter

__all__ = [
    # rules
    "DOMAIN_CODES",
    "OPERATION_CODES",
    "MODIFIER_CODES",
    "MNEMO_VOCABULARY",
    "COMPOUND_OPERATIONS",
    "META_OPERATIONS",
    "validate",
    "validate_token",
    "lookup",
    "describe",
    # language
    "MnemoToken",
    "MnemoGrammar",
    "encode",
    "decode",
    "expand",
    # codec
    "MnemoCodec",
    # interpreter
    "MnemoInterpreter",
]
