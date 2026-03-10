"""
language.py — The MNEMO language specification.

MNEMO is a hyper-compressed encoding language where each 1-3 character
token encodes a grammatical concept across three dimensions:

    Position 1: Domain     — *what* substrate the operation targets
    Position 2: Operation  — *what* to do
    Position 3: Modifier   — *how* to do it (optional)

Example: ``"Lp+"`` = linguistic predict forward
         ``"Bf~"`` = biological forecast bidirectional
         ``"*c&"`` = universal compose parallel

The language provides three core transformations:

    encode(natural_language)  →  MNEMO string
    decode(mnemo_string)      →  structured instruction
    expand(mnemo_string)      →  full grammar operations
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from . import rules as _rules


# ---------------------------------------------------------------------------
# MnemoToken — a single MNEMO token
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class MnemoToken:
    """A single MNEMO token: 1-3 characters encoding a grammatical concept.

    Each character position encodes a different dimension:
        - ``domain``:    The substrate the operation targets (L, C, B, X, E, M, *)
        - ``operation``: What to do (p, r, f, t, i, d, c, s, a)
        - ``modifier``:  How to do it (+, -, ~, !, ?, @, #, &) — optional

    Attributes:
        raw:            The raw token string (e.g. ``"Lp+"``).
        domain_char:    First character.
        operation_char: Second character.
        modifier_char:  Third character or empty string.
        domain:         Resolved domain name.
        operation:      Resolved operation name.
        modifier:       Resolved modifier name (empty if unmodified).
    """

    raw: str
    domain_char: str = field(init=False, repr=False)
    operation_char: str = field(init=False, repr=False)
    modifier_char: str = field(init=False, repr=False)
    domain: str = field(init=False)
    operation: str = field(init=False)
    modifier: str = field(init=False)

    def __post_init__(self) -> None:
        # Frozen dataclass — must use object.__setattr__
        if len(self.raw) < 2 or len(self.raw) > 3:
            raise ValueError(
                f"MNEMO token must be 2-3 characters, got {len(self.raw)!r}: {self.raw!r}"
            )

        d_char = self.raw[0]
        o_char = self.raw[1]
        m_char = self.raw[2] if len(self.raw) == 3 else ""

        d_code = _rules.DOMAIN_CODES.get(d_char)
        o_code = _rules.OPERATION_CODES.get(o_char)
        m_code = _rules.MODIFIER_CODES.get(m_char) if m_char else None

        if d_code is None:
            raise ValueError(f"Unknown domain character {d_char!r} in token {self.raw!r}")
        if o_code is None:
            raise ValueError(f"Unknown operation character {o_char!r} in token {self.raw!r}")
        if m_char and m_code is None:
            raise ValueError(f"Unknown modifier character {m_char!r} in token {self.raw!r}")

        object.__setattr__(self, "domain_char", d_char)
        object.__setattr__(self, "operation_char", o_char)
        object.__setattr__(self, "modifier_char", m_char)
        object.__setattr__(self, "domain", d_code.name)
        object.__setattr__(self, "operation", o_code.name)
        object.__setattr__(self, "modifier", m_code.name if m_code else "")

    @property
    def description(self) -> str:
        """Human-readable description of this token."""
        parts = [self.domain, self.operation]
        if self.modifier:
            parts.append(self.modifier)
        return " ".join(parts)

    @property
    def is_meta(self) -> bool:
        """True if this is a meta-operation (M domain)."""
        return self.domain_char == "M"

    @property
    def is_universal(self) -> bool:
        """True if this is a universal/domain-agnostic operation (* domain)."""
        return self.domain_char == "*"

    def matches_domain(self, domain_name: str) -> bool:
        """True if this token applies to the given domain (or is universal)."""
        return self.is_universal or self.domain == domain_name

    def __str__(self) -> str:
        return self.raw


# ---------------------------------------------------------------------------
# MnemoGrammar — the grammar rules for MNEMO
# ---------------------------------------------------------------------------

# Keyword → (domain_char, operation_char, modifier_char) mapping.
# Used by ``encode()`` to translate natural language to MNEMO.

_DOMAIN_KEYWORDS: Dict[str, str] = {
    "linguistic": "L", "language": "L", "word": "L", "sentence": "L",
    "morphology": "L", "syntax": "L", "semantic": "L", "text": "L",
    "chemical": "C", "chemistry": "C", "molecule": "C", "reaction": "C",
    "compound": "C", "element": "C", "bond": "C", "formula": "C",
    "biological": "B", "biology": "B", "gene": "B", "protein": "B",
    "codon": "B", "dna": "B", "rna": "B", "sequence": "B", "cell": "B",
    "computational": "X", "compute": "X", "algorithm": "X", "code": "X",
    "program": "X", "function": "X", "software": "X", "formal": "X",
    "etymological": "E", "etymology": "E", "origin": "E", "root": "E",
    "historical": "E", "evolution": "E", "ancestor": "E",
    "meta": "M", "self": "M", "introspective": "M", "grammar": "M",
    "all": "*", "every": "*", "universal": "*", "any": "*", "cross": "*",
}

_OPERATION_KEYWORDS: Dict[str, str] = {
    "predict": "p", "generate": "p", "next": "p", "forward": "p",
    "reconstruct": "r", "recover": "r", "reverse": "r", "parse": "r",
    "forecast": "f", "extrapolate": "f", "future": "f", "project": "f",
    "translate": "t", "map": "t", "convert": "t", "transform": "t",
    "introspect": "i", "examine": "i", "inspect": "i", "reflect": "i",
    "derive": "d", "apply": "d", "produce": "d", "rule": "d",
    "compose": "c", "combine": "c", "merge": "c", "fugue": "c",
    "search": "s", "find": "s", "match": "s", "locate": "s", "query": "s",
    "analyze": "a", "analyse": "a", "structure": "a", "pattern": "a",
}

_MODIFIER_KEYWORDS: Dict[str, str] = {
    "forward": "+", "ahead": "+", "onwards": "+",
    "backward": "-", "back": "-", "reverse": "-", "previous": "-",
    "bidirectional": "~", "both": "~", "dual": "~", "symmetric": "~",
    "forced": "!", "force": "!", "override": "!", "must": "!",
    "query": "?", "question": "?", "candidate": "?", "possible": "?",
    "context": "@", "contextual": "@", "surrounding": "@", "ambient": "@",
    "count": "#", "number": "#", "quantity": "#", "how many": "#",
    "parallel": "&", "concurrent": "&", "simultaneous": "&",
}


@dataclass
class MnemoGrammar:
    """The grammar rules for MNEMO — how tokens combine and expand.

    The grammar knows how to:
    - Parse MNEMO strings into token sequences
    - Validate token combinations
    - Expand tokens into full grammar operation descriptors
    - Recognize compound operations (multi-token idioms)

    Attributes:
        vocabulary:          Reference to the full MNEMO vocabulary.
        compound_operations: Reference to compound operation definitions.
        meta_operations:     Reference to meta-operation definitions.
    """

    vocabulary: Dict[str, Dict[str, str]] = field(
        default_factory=lambda: _rules.MNEMO_VOCABULARY
    )
    compound_operations: Dict[str, _rules.CompoundOperation] = field(
        default_factory=lambda: _rules.COMPOUND_OPERATIONS
    )
    meta_operations: Dict[str, _rules.MetaOperation] = field(
        default_factory=lambda: _rules.META_OPERATIONS
    )

    # -- parsing ------------------------------------------------------------

    def tokenize(self, mnemo_string: str) -> List[MnemoToken]:
        """Parse a MNEMO string into a list of ``MnemoToken`` objects.

        Tokens in the string are separated by whitespace.

        Raises ``ValueError`` if any token is malformed.
        """
        raw_tokens = mnemo_string.strip().split()
        return [MnemoToken(raw=t) for t in raw_tokens]

    # -- compound detection -------------------------------------------------

    def find_compounds(self, tokens: List[MnemoToken]) -> List[Tuple[int, int, _rules.CompoundOperation]]:
        """Find compound operations in a token sequence.

        Returns a list of ``(start_index, end_index, CompoundOperation)``
        tuples for every recognized compound in the sequence.  Compounds
        may overlap; the caller decides resolution strategy.
        """
        results: List[Tuple[int, int, _rules.CompoundOperation]] = []
        raw_sequence = [t.raw for t in tokens]

        for key, compound in self.compound_operations.items():
            comp_tokens = list(compound.tokens)
            comp_len = len(comp_tokens)

            for i in range(len(raw_sequence) - comp_len + 1):
                if raw_sequence[i:i + comp_len] == comp_tokens:
                    results.append((i, i + comp_len, compound))

        return results

    # -- expansion ----------------------------------------------------------

    def expand_token(self, token: MnemoToken) -> Dict[str, Any]:
        """Expand a single token into a full grammar operation descriptor.

        Returns a dict describing the operation:
            domain:      Target domain name
            operation:   Operation name
            modifier:    Modifier name (or empty)
            direction:   Inferred temporal direction
            description: Human-readable expansion
            is_meta:     Whether this is a meta-operation
            grammar_actions: List of grammar engine actions to perform
        """
        direction = _infer_direction(token.modifier)
        actions = _infer_actions(token)

        return {
            "domain": token.domain,
            "operation": token.operation,
            "modifier": token.modifier,
            "direction": direction,
            "description": token.description,
            "is_meta": token.is_meta,
            "grammar_actions": actions,
        }

    def expand_sequence(self, tokens: List[MnemoToken]) -> List[Dict[str, Any]]:
        """Expand a token sequence, recognizing compounds.

        Compounds are expanded as unified operations; individual tokens
        that are not part of a compound are expanded individually.
        """
        compounds = self.find_compounds(tokens)
        # Build a set of indices consumed by compounds.
        consumed: set[int] = set()
        expanded: List[Dict[str, Any]] = []

        # Sort compounds by start index for deterministic output.
        compounds.sort(key=lambda c: c[0])

        idx = 0
        compound_idx = 0
        while idx < len(tokens):
            # Check if a compound starts here.
            matched_compound = False
            for start, end, compound in compounds:
                if start == idx and not any(i in consumed for i in range(start, end)):
                    # Expand as compound.
                    expanded.append({
                        "compound": compound.name,
                        "tokens": [t.raw for t in tokens[start:end]],
                        "description": compound.description,
                        "semantics": compound.semantics,
                        "steps": [self.expand_token(t) for t in tokens[start:end]],
                    })
                    for i in range(start, end):
                        consumed.add(i)
                    idx = end
                    matched_compound = True
                    break

            if not matched_compound:
                expanded.append(self.expand_token(tokens[idx]))
                idx += 1

        return expanded


# ---------------------------------------------------------------------------
# Module-level convenience functions
# ---------------------------------------------------------------------------

# Singleton grammar for the module-level functions.
_GRAMMAR = MnemoGrammar()


def encode(natural_language: str) -> str:
    """Convert a natural language instruction to a MNEMO string.

    Scans the input for keywords that map to MNEMO domain, operation,
    and modifier characters, then assembles the most specific token
    sequence possible.

    This is a heuristic encoder — it finds the best MNEMO representation
    of the *intent* expressed in natural language, not a lossless
    compression of the raw text (for that, see ``MnemoCodec``).

    Examples::

        >>> encode("predict the next word forward")
        'Lp+'
        >>> encode("analyze biological structure bidirectional")
        'Ba~'
        >>> encode("translate chemical to linguistic")
        'Lt+ Ct-'
    """
    words = _normalize_text(natural_language)

    # Collect all domain, operation, and modifier hits.
    domains: List[Tuple[int, str]] = []    # (position, char)
    operations: List[Tuple[int, str]] = [] # (position, char)
    modifiers: List[Tuple[int, str]] = []  # (position, char)

    for i, word in enumerate(words):
        if word in _DOMAIN_KEYWORDS:
            domains.append((i, _DOMAIN_KEYWORDS[word]))
        if word in _OPERATION_KEYWORDS:
            operations.append((i, _OPERATION_KEYWORDS[word]))
        if word in _MODIFIER_KEYWORDS:
            modifiers.append((i, _MODIFIER_KEYWORDS[word]))

    if not domains and not operations:
        # Fallback: universal analyze (catch-all).
        return "*a"

    # Default domain is universal, default operation is analyze.
    if not domains:
        domains = [(0, "*")]
    if not operations:
        operations = [(0, "a")]

    # Group operations with their nearest domain.
    # Strategy: for each operation, pair with the closest domain.
    tokens: List[str] = []
    used_ops: set[int] = set()

    for d_pos, d_char in domains:
        # Find the closest unused operation.
        best_op: Optional[Tuple[int, str]] = None
        best_dist = float("inf")
        for o_idx, (o_pos, o_char) in enumerate(operations):
            if o_idx in used_ops:
                continue
            dist = abs(o_pos - d_pos)
            if dist < best_dist:
                best_dist = dist
                best_op = (o_idx, o_char)
        if best_op is None:
            # No more operations — use analyze as default.
            o_char = "a"
        else:
            used_ops.add(best_op[0])
            o_char = best_op[1]

        # Find the closest modifier to this domain-operation pair.
        token = f"{d_char}{o_char}"
        best_mod: Optional[str] = None
        best_mod_dist = float("inf")
        for m_pos, m_char in modifiers:
            dist = abs(m_pos - d_pos)
            if dist < best_mod_dist:
                best_mod_dist = dist
                best_mod = m_char

        if best_mod is not None:
            token += best_mod

        tokens.append(token)

    # Handle remaining operations not paired with a domain.
    for o_idx, (o_pos, o_char) in enumerate(operations):
        if o_idx not in used_ops:
            token = f"*{o_char}"
            # Find closest modifier.
            best_mod = None
            best_mod_dist = float("inf")
            for m_pos, m_char in modifiers:
                dist = abs(m_pos - o_pos)
                if dist < best_mod_dist:
                    best_mod_dist = dist
                    best_mod = m_char
            if best_mod is not None:
                token += best_mod
            tokens.append(token)

    return " ".join(tokens)


def decode(mnemo_string: str) -> Dict[str, Any]:
    """Convert a MNEMO string back to a structured instruction.

    Returns a dict with:
        tokens:      List of parsed ``MnemoToken`` objects.
        description: Human-readable natural language description.
        operations:  List of expanded operation descriptors.
        compounds:   Any compound operations detected.
        is_valid:    Whether all tokens are valid.
    """
    valid, errors = _rules.validate(mnemo_string)

    if not valid:
        return {
            "tokens": [],
            "description": f"Invalid MNEMO: {'; '.join(errors)}",
            "operations": [],
            "compounds": [],
            "is_valid": False,
        }

    tokens = _GRAMMAR.tokenize(mnemo_string)
    compounds = _GRAMMAR.find_compounds(tokens)
    operations = _GRAMMAR.expand_sequence(tokens)

    # Build a natural language description.
    desc_parts: List[str] = []
    for op in operations:
        if "compound" in op:
            desc_parts.append(op["description"])
        else:
            desc_parts.append(op["description"])

    description = "; then ".join(desc_parts) if desc_parts else "no-op"

    return {
        "tokens": tokens,
        "description": description,
        "operations": operations,
        "compounds": [(s, e, c.name) for s, e, c in compounds],
        "is_valid": True,
    }


def expand(mnemo_string: str) -> List[Dict[str, Any]]:
    """Expand a MNEMO string into full grammar operations.

    This is the core function that turns a compressed MNEMO program into
    the sequence of grammar engine actions that the interpreter will execute.

    Each element in the returned list is an operation descriptor containing
    the domain, operation, modifier, direction, and specific grammar actions
    to perform.

    Example::

        >>> expand("Lp+ Xd~")
        [
            {
                'domain': 'linguistic',
                'operation': 'predict',
                'modifier': 'forward',
                'direction': 'forward',
                ...
                'grammar_actions': ['derive_forward']
            },
            {
                'domain': 'computational',
                'operation': 'derive',
                'modifier': 'bidirectional',
                'direction': 'bidirectional',
                ...
                'grammar_actions': ['derive_forward', 'derive_backward']
            }
        ]
    """
    valid, errors = _rules.validate(mnemo_string)
    if not valid:
        raise ValueError(f"Invalid MNEMO string: {'; '.join(errors)}")

    tokens = _GRAMMAR.tokenize(mnemo_string)
    return _GRAMMAR.expand_sequence(tokens)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _normalize_text(text: str) -> List[str]:
    """Normalize natural language text for keyword extraction."""
    text = text.lower().strip()
    # Remove punctuation except hyphens.
    text = re.sub(r"[^\w\s\-]", " ", text)
    # Collapse whitespace.
    text = re.sub(r"\s+", " ", text)
    return text.split()


def _infer_direction(modifier: str) -> str:
    """Infer the temporal direction from a modifier name."""
    direction_map = {
        "forward": "forward",
        "backward": "backward",
        "bidirectional": "bidirectional",
        "forced": "forward",       # forced defaults to forward
        "query": "none",           # queries don't have direction
        "context": "contextual",   # direction determined by context
        "count": "none",           # counting is non-directional
        "parallel": "parallel",    # parallel across all directions
    }
    return direction_map.get(modifier, "unspecified")


def _infer_actions(token: MnemoToken) -> List[str]:
    """Infer the grammar engine actions for a MNEMO token.

    Maps the operation + modifier combination to concrete engine methods.
    """
    op = token.operation
    mod = token.modifier

    # Base action per operation.
    action_map: Dict[str, List[str]] = {
        "predict":      ["derive_forward"],
        "reconstruct":  ["derive_backward"],
        "forecast":     ["derive_forward", "loop_project"],
        "translate":    ["cross_domain_map"],
        "introspect":   ["grammar_inspect"],
        "derive":       ["derive_forward"],
        "compose":      ["grammar_compose"],
        "search":       ["grammar_search"],
        "analyze":      ["structural_analysis"],
    }

    actions = list(action_map.get(op, ["unknown_action"]))

    # Modifier adjustments.
    if mod == "backward":
        actions = [a.replace("forward", "backward") for a in actions]
    elif mod == "bidirectional":
        expanded: List[str] = []
        for a in actions:
            if "forward" in a:
                expanded.append(a)
                expanded.append(a.replace("forward", "backward"))
            elif "backward" in a:
                expanded.append(a.replace("backward", "forward"))
                expanded.append(a)
            else:
                expanded.append(a)
        actions = expanded
    elif mod == "query":
        actions = [f"{a}_query" for a in actions]
    elif mod == "count":
        actions = [f"{a}_count" for a in actions]
    elif mod == "parallel":
        actions = [f"{a}_parallel" for a in actions]
    elif mod == "context":
        actions = [f"{a}_contextual" for a in actions]
    elif mod == "forced":
        actions = [f"{a}_forced" for a in actions]

    return actions
