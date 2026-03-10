"""
Mathematical Substrate — the grammar of mathematical notation.

Mathematics is grammar distilled: every symbol has precise meaning,
every expression is a derivation tree, and the rules of manipulation
are the production rules of a formal grammar.

This substrate treats mathematical notation as a symbolic system:
- Symbols are the atoms (numbers, variables, operators, functions)
- Expressions are the molecules (well-formed formulae)
- Equations are reactions (transformations that preserve equality)

Strange loops: Gödel numbering (encoding proofs as numbers about which
you can then prove things), recursive functions that compute their own
Gödel numbers, the Y combinator (a function that takes a function and
finds its fixed point).

Isomorphisms:
- Mathematical variables ↔ linguistic pronouns (both refer to unbound entities)
- Operator precedence ↔ syntactic constituency (both impose structure on linear sequence)
- Proof by induction ↔ recursive derivation ↔ recursive function definition
- Group isomorphism ↔ translation between grammars
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Dict, List, Optional, Set, Tuple

from ..core.substrate import Sequence, Substrate, Symbol, TransformationRule


# ---------------------------------------------------------------------------
# Mathematical symbol categories
# ---------------------------------------------------------------------------

class MathCategory(Enum):
    """Categories of mathematical symbols."""
    NUMBER = auto()
    VARIABLE = auto()
    OPERATOR = auto()
    FUNCTION = auto()
    RELATION = auto()
    GROUPING = auto()
    QUANTIFIER = auto()
    CONSTANT = auto()
    SET_OP = auto()
    LOGIC_OP = auto()
    CALCULUS_OP = auto()
    GREEK = auto()


# ---------------------------------------------------------------------------
# MathSymbol — Symbol subclass for mathematical entities
# ---------------------------------------------------------------------------

@dataclass
class MathSymbol(Symbol):
    """A mathematical symbol — the atomic unit of mathematical notation.

    Like tokens in code or phonemes in language, mathematical symbols
    have categories and features that determine how they combine.
    """

    category: MathCategory = MathCategory.VARIABLE
    precedence: int = 0
    associativity: str = "left"
    arity: int = 0

    def __post_init__(self) -> None:
        if self.domain == "generic":
            self.domain = "mathematical"
        self.features.setdefault("category", self.category.name.lower())
        if self.precedence:
            self.features["precedence"] = str(self.precedence)
        if self.associativity:
            self.features["associativity"] = self.associativity

    def __hash__(self) -> int:
        return hash((self.form, self.domain, self.category.name))

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, MathSymbol):
            return Symbol.__eq__(self, other)
        return (self.form == other.form and self.domain == other.domain
                and self.category == other.category)

    @property
    def is_operator(self) -> bool:
        return self.category in (MathCategory.OPERATOR, MathCategory.LOGIC_OP,
                                  MathCategory.SET_OP, MathCategory.CALCULUS_OP)

    @property
    def is_value(self) -> bool:
        return self.category in (MathCategory.NUMBER, MathCategory.VARIABLE,
                                  MathCategory.CONSTANT)


# ---------------------------------------------------------------------------
# Built-in mathematical symbols
# ---------------------------------------------------------------------------

MATH_OPERATORS: Dict[str, Dict[str, Any]] = {
    "+":  {"precedence": 1, "arity": 2, "associativity": "left", "category": "OPERATOR"},
    "-":  {"precedence": 1, "arity": 2, "associativity": "left", "category": "OPERATOR"},
    "×":  {"precedence": 2, "arity": 2, "associativity": "left", "category": "OPERATOR"},
    "÷":  {"precedence": 2, "arity": 2, "associativity": "left", "category": "OPERATOR"},
    "^":  {"precedence": 3, "arity": 2, "associativity": "right", "category": "OPERATOR"},
    "√":  {"precedence": 3, "arity": 1, "associativity": "right", "category": "OPERATOR"},
    "!":  {"precedence": 4, "arity": 1, "associativity": "left", "category": "OPERATOR"},
}

MATH_RELATIONS: Dict[str, Dict[str, Any]] = {
    "=":  {"arity": 2, "category": "RELATION"},
    "≠":  {"arity": 2, "category": "RELATION"},
    "<":  {"arity": 2, "category": "RELATION"},
    ">":  {"arity": 2, "category": "RELATION"},
    "≤":  {"arity": 2, "category": "RELATION"},
    "≥":  {"arity": 2, "category": "RELATION"},
    "≈":  {"arity": 2, "category": "RELATION"},
    "∈":  {"arity": 2, "category": "RELATION"},
    "⊂":  {"arity": 2, "category": "RELATION"},
    "⊆":  {"arity": 2, "category": "RELATION"},
    "≡":  {"arity": 2, "category": "RELATION"},
    "∝":  {"arity": 2, "category": "RELATION"},
}

MATH_FUNCTIONS: Dict[str, Dict[str, Any]] = {
    "sin": {"arity": 1}, "cos": {"arity": 1}, "tan": {"arity": 1},
    "log": {"arity": 1}, "ln":  {"arity": 1}, "exp": {"arity": 1},
    "lim": {"arity": 1}, "max": {"arity": 2}, "min": {"arity": 2},
    "gcd": {"arity": 2}, "lcm": {"arity": 2},
    "det": {"arity": 1}, "tr":  {"arity": 1},
}

MATH_CONSTANTS: Dict[str, str] = {
    "π": "pi", "e": "euler", "i": "imaginary_unit",
    "∞": "infinity", "ℵ₀": "aleph_null", "φ": "golden_ratio",
    "ℏ": "reduced_planck", "c": "speed_of_light",
}

CALCULUS_OPS: Dict[str, Dict[str, Any]] = {
    "∫":  {"name": "integral", "arity": 1},
    "∂":  {"name": "partial", "arity": 1},
    "∇":  {"name": "nabla/gradient", "arity": 1},
    "Σ":  {"name": "summation", "arity": 1},
    "Π":  {"name": "product", "arity": 1},
    "d":  {"name": "differential", "arity": 1},
}

LOGIC_OPS: Dict[str, Dict[str, Any]] = {
    "∧": {"name": "and", "arity": 2},
    "∨": {"name": "or", "arity": 2},
    "¬": {"name": "not", "arity": 1},
    "→": {"name": "implies", "arity": 2},
    "↔": {"name": "iff", "arity": 2},
    "∀": {"name": "forall", "arity": 1},
    "∃": {"name": "exists", "arity": 1},
}

SET_OPS: Dict[str, Dict[str, Any]] = {
    "∪": {"name": "union", "arity": 2},
    "∩": {"name": "intersection", "arity": 2},
    "∖": {"name": "set_minus", "arity": 2},
    "×": {"name": "cartesian_product", "arity": 2},
}

GROUPING: Dict[str, str] = {
    "(": ")", "[": "]", "{": "}", "⟨": "⟩", "|": "|",
}

GREEK_LETTERS = [
    "α", "β", "γ", "δ", "ε", "ζ", "η", "θ",
    "ι", "κ", "λ", "μ", "ν", "ξ", "ο", "π",
    "ρ", "σ", "τ", "υ", "φ", "χ", "ψ", "ω",
    "Γ", "Δ", "Θ", "Λ", "Ξ", "Π", "Σ", "Φ", "Ψ", "Ω",
]


# ---------------------------------------------------------------------------
# MathSubstrate
# ---------------------------------------------------------------------------

class MathSubstrate(Substrate):
    """Substrate for mathematical notation.

    Treats mathematical expressions as symbolic systems with their own
    grammar: operator precedence defines constituency, parentheses
    define scope, and well-formedness rules parallel syntax.
    """

    def __init__(self, name: str = "mathematical") -> None:
        super().__init__(name, domain="mathematical")
        self._build_inventory()

    def _build_inventory(self) -> None:
        """Register mathematical symbols."""
        for op, data in MATH_OPERATORS.items():
            self.add_symbol(MathSymbol(
                form=op, domain="mathematical",
                valence=data.get("arity", 2),
                category=MathCategory.OPERATOR,
                precedence=data.get("precedence", 0),
                associativity=data.get("associativity", "left"),
                arity=data.get("arity", 2),
            ))

        for rel, data in MATH_RELATIONS.items():
            self.add_symbol(MathSymbol(
                form=rel, domain="mathematical",
                valence=data.get("arity", 2),
                category=MathCategory.RELATION,
                arity=data.get("arity", 2),
            ))

        for func, data in MATH_FUNCTIONS.items():
            self.add_symbol(MathSymbol(
                form=func, domain="mathematical",
                valence=data.get("arity", 1),
                category=MathCategory.FUNCTION,
                arity=data.get("arity", 1),
            ))

        for const, meaning in MATH_CONSTANTS.items():
            self.add_symbol(MathSymbol(
                form=const, domain="mathematical",
                valence=0,
                category=MathCategory.CONSTANT,
                features={"meaning": meaning},
            ))

        for cop, data in CALCULUS_OPS.items():
            self.add_symbol(MathSymbol(
                form=cop, domain="mathematical",
                valence=data.get("arity", 1),
                category=MathCategory.CALCULUS_OP,
                arity=data.get("arity", 1),
            ))

        for lop, data in LOGIC_OPS.items():
            self.add_symbol(MathSymbol(
                form=lop, domain="mathematical",
                valence=data.get("arity", 1),
                category=MathCategory.LOGIC_OP,
                arity=data.get("arity", 1),
            ))

        for letter in GREEK_LETTERS:
            self.add_symbol(MathSymbol(
                form=letter, domain="mathematical",
                valence=0,
                category=MathCategory.GREEK,
            ))

        for opener, closer in GROUPING.items():
            self.add_symbol(MathSymbol(
                form=opener, domain="mathematical", valence=0,
                category=MathCategory.GROUPING,
                features={"role": "open", "match": closer},
            ))
            if closer != opener:
                self.add_symbol(MathSymbol(
                    form=closer, domain="mathematical", valence=0,
                    category=MathCategory.GROUPING,
                    features={"role": "close", "match": opener},
                ))

    # -- encode / decode ---------------------------------------------------

    def encode(self, raw_input: str) -> Sequence:
        """Tokenise mathematical notation into a Sequence of MathSymbols."""
        symbols: List[Symbol] = []
        i = 0
        src = raw_input.strip()
        all_known = set()
        for sym in self._inventory.values():
            all_known.add(sym.form)

        while i < len(src):
            if src[i] in (" ", "\t", "\n"):
                i += 1
                continue

            # Try multi-char tokens first (function names, etc.)
            matched = False
            for length in range(6, 0, -1):
                candidate = src[i:i + length]
                if candidate in all_known:
                    sym = self._inventory[candidate]
                    symbols.append(MathSymbol(
                        form=candidate, domain="mathematical",
                        valence=sym.valence,
                        category=sym.category if isinstance(sym, MathSymbol) else MathCategory.VARIABLE,
                    ))
                    i += length
                    matched = True
                    break

            if matched:
                continue

            # Numbers (including decimals)
            if src[i].isdigit() or (src[i] == '.' and i + 1 < len(src) and src[i + 1].isdigit()):
                j = i
                has_dot = False
                while j < len(src) and (src[j].isdigit() or (src[j] == '.' and not has_dot)):
                    if src[j] == '.':
                        has_dot = True
                    j += 1
                symbols.append(MathSymbol(
                    form=src[i:j], domain="mathematical", valence=0,
                    category=MathCategory.NUMBER,
                ))
                i = j
                continue

            # Variable names (letters)
            if src[i].isalpha() and src[i] not in all_known:
                j = i
                while j < len(src) and (src[j].isalnum() or src[j] == '_'):
                    j += 1
                word = src[i:j]
                if word in all_known:
                    sym = self._inventory[word]
                    symbols.append(MathSymbol(
                        form=word, domain="mathematical",
                        valence=sym.valence,
                        category=sym.category if isinstance(sym, MathSymbol) else MathCategory.FUNCTION,
                    ))
                else:
                    symbols.append(MathSymbol(
                        form=word, domain="mathematical", valence=0,
                        category=MathCategory.VARIABLE,
                    ))
                i = j
                continue

            # Single special character
            symbols.append(MathSymbol(
                form=src[i], domain="mathematical", valence=0,
                category=MathCategory.VARIABLE,
            ))
            i += 1

        return Sequence(symbols)

    def decode(self, sequence: Sequence) -> str:
        """Reconstruct mathematical notation from symbol sequence."""
        parts: List[str] = []
        for idx, sym in enumerate(sequence):
            if idx > 0 and isinstance(sym, MathSymbol):
                prev = sequence[idx - 1]
                if isinstance(prev, MathSymbol):
                    if (prev.category == MathCategory.GROUPING and
                            prev.features.get("role") == "open"):
                        pass
                    elif (sym.category == MathCategory.GROUPING and
                          sym.features.get("role") == "close"):
                        pass
                    else:
                        parts.append(" ")
                else:
                    parts.append(" ")
            parts.append(sym.form)
        return "".join(parts)

    # -- analysis -----------------------------------------------------------

    def validate_expression(self, sequence: Sequence) -> List[str]:
        """Validate mathematical expression well-formedness."""
        errors: List[str] = []
        stack: List[Tuple[str, int]] = []

        for idx, sym in enumerate(sequence):
            if not isinstance(sym, MathSymbol):
                continue
            if sym.category == MathCategory.GROUPING:
                if sym.features.get("role") == "open":
                    stack.append((sym.form, idx))
                elif sym.features.get("role") == "close":
                    if not stack:
                        errors.append(f"Unmatched '{sym.form}' at position {idx}")
                    else:
                        opener, _ = stack.pop()
                        expected = GROUPING.get(opener, "")
                        if sym.form != expected:
                            errors.append(f"Mismatched: '{opener}' closed by '{sym.form}' at {idx}")
        for opener, pos in stack:
            errors.append(f"Unclosed '{opener}' at position {pos}")

        return errors

    def extract_variables(self, sequence: Sequence) -> Set[str]:
        """Extract all variable names from an expression."""
        return {
            sym.form for sym in sequence
            if isinstance(sym, MathSymbol) and sym.category == MathCategory.VARIABLE
        }

    def expression_depth(self, sequence: Sequence) -> int:
        """Compute nesting depth of mathematical expression."""
        max_depth = 0
        current = 0
        for sym in sequence:
            if isinstance(sym, MathSymbol) and sym.category == MathCategory.GROUPING:
                if sym.features.get("role") == "open":
                    current += 1
                    max_depth = max(max_depth, current)
                elif sym.features.get("role") == "close":
                    current = max(0, current - 1)
        return max_depth

    def classify_expression(self, sequence: Sequence) -> str:
        """Classify the type of mathematical expression."""
        forms = [sym.form for sym in sequence if isinstance(sym, MathSymbol)]

        if "∫" in forms or "∂" in forms:
            return "calculus"
        if "∀" in forms or "∃" in forms or "→" in forms:
            return "logic"
        if "∪" in forms or "∩" in forms or "∈" in forms:
            return "set_theory"
        if "=" in forms:
            return "equation"
        if any(r in forms for r in ("<", ">", "≤", "≥")):
            return "inequality"
        if "Σ" in forms or "Π" in forms:
            return "series"
        return "algebraic"
