"""
GLM — Grammar Language Model

A small, rule-driven model that learns the deep grammars underlying
natural languages, chemical notation, biological encoding, and programming
languages.  Instead of memorising surfaces it internalises the *scales* —
the structural rules, substrates, and derivation patterns shared across
every symbolic domain — so that it can compose masterpieces from first
principles.

Architecture
────────────
    core/       Grammar primitives: rules, productions, derivation engine
    grammars/   Domain-specific rule sets (linguistic, chemical, biological …)
    substrates/ The media grammars operate on (phonemes, morphemes, molecules …)
    model/      The neural GLM: grammar-aware embeddings, attention, inference
    angel.py    The Angel — the beating heart that unifies every layer
"""

__version__ = "0.1.0"
