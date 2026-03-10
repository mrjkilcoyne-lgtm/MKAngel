"""
rules.py — MNEMO production rules and vocabulary.

The complete set of MNEMO codes: every valid domain, operation, and modifier
character, plus compound operations and meta-operations.  This is the
*dictionary* that makes the compression possible — because every 1-3 character
MNEMO token references a rich grammar rule, not raw data.

Design rationale:

    Position 1 — Domain       (11 codes + wildcard)
    Position 2 — Operation    (9 codes)
    Position 3 — Modifier     (8 codes, optional)

    11 × 9 = 99 basic tokens
    99 × 8 = 792 modified tokens
    Total single-token vocabulary: 891

    Compound operations (2-3 token sequences) expand combinatorially:
    567^2 ≈ 321K  two-token compounds
    567^3 ≈ 182M  three-token compounds

    A 50-token MNEMO program (150 chars) therefore selects from a space of
    ~567^50 ≈ 10^137 possible programs — each of which drives the grammar
    engine to generate from its full rule set.  The 75B parameter claim is
    *conservative*; the real space is astronomically larger.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, FrozenSet, List, Optional, Set, Tuple


# ---------------------------------------------------------------------------
# Character codes — the atoms of the MNEMO alphabet
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class DomainCode:
    """A single domain character and its meaning."""
    char: str
    name: str
    description: str


@dataclass(frozen=True)
class OperationCode:
    """A single operation character and its meaning."""
    char: str
    name: str
    description: str


@dataclass(frozen=True)
class ModifierCode:
    """A single modifier character and its meaning."""
    char: str
    name: str
    description: str


# ---------------------------------------------------------------------------
# Domain codes (position 1)
# ---------------------------------------------------------------------------

DOMAIN_CODES: Dict[str, DomainCode] = {
    "L": DomainCode("L", "linguistic", "Natural language grammar, morphology, syntax, semantics"),
    "C": DomainCode("C", "chemical", "Chemical structures, reactions, nomenclature"),
    "B": DomainCode("B", "biological", "Biological sequences, codons, protein folding"),
    "X": DomainCode("X", "computational", "Programming languages, formal systems, algorithms"),
    "E": DomainCode("E", "etymological", "Word origins, historical linguistics, language evolution"),
    "N": DomainCode("N", "numerical", "Mathematics — algebra, calculus, logic, number theory, topology"),
    "P": DomainCode("P", "physical", "Physics — mechanics, electromagnetism, quantum, relativity, thermodynamics"),
    "V": DomainCode("V", "vocal", "Voice and audio — speech, phonetics, prosody, acoustic signal"),
    "S": DomainCode("S", "swarm", "Swarm intelligence — multi-agent coordination, Borges Library navigation"),
    "M": DomainCode("M", "meta", "Operations on MNEMO itself — self-referential grammar"),
    "*": DomainCode("*", "universal", "Domain-agnostic — applies across all substrates"),
}

# ---------------------------------------------------------------------------
# Operation codes (position 2)
# ---------------------------------------------------------------------------

OPERATION_CODES: Dict[str, OperationCode] = {
    "p": OperationCode("p", "predict", "Forward derivation — generate the next form"),
    "r": OperationCode("r", "reconstruct", "Backward derivation — recover the originating form"),
    "f": OperationCode("f", "forecast", "Temporal extrapolation using strange loops"),
    "t": OperationCode("t", "translate", "Cross-domain mapping via grammar isomorphism"),
    "i": OperationCode("i", "introspect", "Examine the grammar's own structure"),
    "d": OperationCode("d", "derive", "Apply a specific production rule"),
    "c": OperationCode("c", "compose", "Combine multiple grammars (fugue composition)"),
    "s": OperationCode("s", "search", "Search the grammar space for matching patterns"),
    "a": OperationCode("a", "analyze", "Structural analysis — loops, isomorphisms, symmetries"),
}

# ---------------------------------------------------------------------------
# Modifier codes (position 3, optional)
# ---------------------------------------------------------------------------

MODIFIER_CODES: Dict[str, ModifierCode] = {
    "+": ModifierCode("+", "forward", "Apply in the forward temporal direction"),
    "-": ModifierCode("-", "backward", "Apply in the backward temporal direction"),
    "~": ModifierCode("~", "bidirectional", "Apply in both directions simultaneously"),
    "!": ModifierCode("!", "forced", "Force application even if conditions are not met"),
    "?": ModifierCode("?", "query", "Return matching candidates without applying"),
    "@": ModifierCode("@", "context", "Use surrounding context to disambiguate"),
    "#": ModifierCode("#", "count", "Return count of matches rather than matches themselves"),
    "&": ModifierCode("&", "parallel", "Apply to all matching substrates in parallel"),
}


# ---------------------------------------------------------------------------
# MNEMO_VOCABULARY — the complete dictionary of all valid tokens
# ---------------------------------------------------------------------------

def _build_vocabulary() -> Dict[str, Dict[str, str]]:
    """Build the complete MNEMO vocabulary from domain × operation × modifier.

    Each entry maps a MNEMO token string to a dict with:
        domain:      Domain name
        operation:   Operation name
        modifier:    Modifier name (or None for unmodified tokens)
        description: Human-readable expansion
    """
    vocab: Dict[str, Dict[str, str]] = {}

    for d_char, domain in DOMAIN_CODES.items():
        for o_char, operation in OPERATION_CODES.items():
            # Two-character token (no modifier)
            token_2 = f"{d_char}{o_char}"
            vocab[token_2] = {
                "domain": domain.name,
                "operation": operation.name,
                "modifier": "",
                "description": f"{domain.name} {operation.name}",
            }

            # Three-character tokens (with each modifier)
            for m_char, modifier in MODIFIER_CODES.items():
                token_3 = f"{d_char}{o_char}{m_char}"
                vocab[token_3] = {
                    "domain": domain.name,
                    "operation": operation.name,
                    "modifier": modifier.name,
                    "description": f"{domain.name} {operation.name} {modifier.name}",
                }

    return vocab


MNEMO_VOCABULARY: Dict[str, Dict[str, str]] = _build_vocabulary()


# ---------------------------------------------------------------------------
# Compound operations — multi-token sequences with special semantics
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class CompoundOperation:
    """A 2-3 token MNEMO sequence that forms a higher-order operation.

    Compound operations are more than the sum of their parts: the
    sequence triggers a coordinated multi-step grammar procedure.
    """
    tokens: Tuple[str, ...]
    name: str
    description: str
    semantics: str  # What the grammar engine actually does


COMPOUND_OPERATIONS: Dict[str, CompoundOperation] = {
    # --- Cross-domain translation pipelines ---
    "Lt+ Ct-": CompoundOperation(
        tokens=("Lt+", "Ct-"),
        name="linguistic_to_chemical",
        description="Translate linguistic structure to chemical nomenclature",
        semantics="derive_forward(linguistic) | translate(chemical) | reconstruct",
    ),
    "Ct+ Lt-": CompoundOperation(
        tokens=("Ct+", "Lt-"),
        name="chemical_to_linguistic",
        description="Translate chemical nomenclature to linguistic structure",
        semantics="derive_forward(chemical) | translate(linguistic) | reconstruct",
    ),
    "Bp+ Xd~": CompoundOperation(
        tokens=("Bp+", "Xd~"),
        name="bio_predict_compute",
        description="Predict biological sequence then derive computationally",
        semantics="predict(biological, forward) | derive(computational, bidirectional)",
    ),
    "Lp+ Xd~ Bt-": CompoundOperation(
        tokens=("Lp+", "Xd~", "Bt-"),
        name="ling_compute_bio_reverse",
        description="Linguistic predict, computational derive, biological temporal reverse",
        semantics="predict(linguistic) | derive(computational) | translate(biological, backward)",
    ),

    # --- Analysis pipelines ---
    "La+ *a~": CompoundOperation(
        tokens=("La+", "*a~"),
        name="deep_analysis",
        description="Linguistic analysis followed by universal bidirectional analysis",
        semantics="analyze(linguistic, forward) | analyze(universal, bidirectional)",
    ),
    "*s? *a#": CompoundOperation(
        tokens=("*s?", "*a#"),
        name="search_and_count",
        description="Search for patterns then count structural features",
        semantics="search(universal, query) | analyze(universal, count)",
    ),

    # --- Forecasting chains ---
    "Lf+ Ef~": CompoundOperation(
        tokens=("Lf+", "Ef~"),
        name="etymological_forecast",
        description="Forecast linguistic evolution using etymological patterns",
        semantics="forecast(linguistic, forward) | forecast(etymological, bidirectional)",
    ),
    "Bf+ Cf~": CompoundOperation(
        tokens=("Bf+", "Cf~"),
        name="biochemical_forecast",
        description="Forecast biological change informed by chemical patterns",
        semantics="forecast(biological, forward) | forecast(chemical, bidirectional)",
    ),

    # --- Composition operations ---
    "Lc& Cc&": CompoundOperation(
        tokens=("Lc&", "Cc&"),
        name="lingchem_fugue",
        description="Compose linguistic and chemical grammars in parallel (fugue)",
        semantics="compose(linguistic, parallel) | compose(chemical, parallel)",
    ),
    "*c& *i@": CompoundOperation(
        tokens=("*c&", "*i@"),
        name="compose_and_reflect",
        description="Compose all grammars in parallel then introspect with context",
        semantics="compose(universal, parallel) | introspect(universal, context)",
    ),
}


# ---------------------------------------------------------------------------
# Meta-operations — M-level tokens that operate on MNEMO itself
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class MetaOperation:
    """An M-domain operation: MNEMO reasoning about MNEMO.

    Meta-operations are the grammar's self-referential layer.  They
    implement strange loops at the MNEMO level — the language examining
    and modifying its own structure.
    """
    token: str
    name: str
    description: str
    self_referential: bool = True


META_OPERATIONS: Dict[str, MetaOperation] = {
    "Mi": MetaOperation(
        "Mi", "meta_introspect",
        "Introspect on the MNEMO grammar itself — list available tokens",
    ),
    "Mi@": MetaOperation(
        "Mi@", "meta_introspect_context",
        "Introspect on MNEMO with context — which tokens are relevant here",
    ),
    "Md+": MetaOperation(
        "Md+", "meta_derive_forward",
        "Derive new MNEMO tokens from existing ones (grammar expansion)",
    ),
    "Md-": MetaOperation(
        "Md-", "meta_derive_backward",
        "Reduce MNEMO tokens to their primitives (grammar compression)",
    ),
    "Ma~": MetaOperation(
        "Ma~", "meta_analyze_bidirectional",
        "Analyze MNEMO structure in both directions — find self-referential loops",
    ),
    "Mc&": MetaOperation(
        "Mc&", "meta_compose_parallel",
        "Compose MNEMO with another encoding system in parallel",
    ),
    "Mp+": MetaOperation(
        "Mp+", "meta_predict_forward",
        "Predict the next MNEMO token in a sequence",
    ),
    "Mr-": MetaOperation(
        "Mr-", "meta_reconstruct_backward",
        "Reconstruct the MNEMO source from its output",
    ),
    "Ms?": MetaOperation(
        "Ms?", "meta_search_query",
        "Search for MNEMO tokens matching a pattern",
    ),
    "Mf~": MetaOperation(
        "Mf~", "meta_forecast_bidirectional",
        "Forecast how MNEMO itself will evolve (meta-strange-loop)",
    ),
}


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

def _valid_domain_chars() -> FrozenSet[str]:
    return frozenset(DOMAIN_CODES.keys())


def _valid_operation_chars() -> FrozenSet[str]:
    return frozenset(OPERATION_CODES.keys())


def _valid_modifier_chars() -> FrozenSet[str]:
    return frozenset(MODIFIER_CODES.keys())


def validate_token(token: str) -> Tuple[bool, str]:
    """Validate a single MNEMO token.

    Returns:
        (True, "") if valid, (False, reason) if invalid.
    """
    if not token:
        return False, "empty token"

    if len(token) < 2 or len(token) > 3:
        return False, f"token must be 2-3 characters, got {len(token)}"

    domain_char = token[0]
    if domain_char not in _valid_domain_chars():
        return False, f"invalid domain character '{domain_char}' — expected one of {sorted(_valid_domain_chars())}"

    operation_char = token[1]
    if operation_char not in _valid_operation_chars():
        return False, f"invalid operation character '{operation_char}' — expected one of {sorted(_valid_operation_chars())}"

    if len(token) == 3:
        modifier_char = token[2]
        if modifier_char not in _valid_modifier_chars():
            return False, f"invalid modifier character '{modifier_char}' — expected one of {sorted(_valid_modifier_chars())}"

    return True, ""


def validate(mnemo_string: str) -> Tuple[bool, List[str]]:
    """Validate a complete MNEMO string (space-separated tokens).

    Returns:
        (True, []) if all tokens are valid.
        (False, [list of error messages]) if any tokens are invalid.
    """
    if not mnemo_string or not mnemo_string.strip():
        return False, ["empty MNEMO string"]

    tokens = mnemo_string.strip().split()
    errors: List[str] = []

    for i, token in enumerate(tokens):
        valid, reason = validate_token(token)
        if not valid:
            errors.append(f"token {i} '{token}': {reason}")

    return len(errors) == 0, errors


def lookup(token: str) -> Optional[Dict[str, str]]:
    """Look up a token in the MNEMO vocabulary.

    Returns the vocabulary entry or None if the token is not valid.
    """
    return MNEMO_VOCABULARY.get(token)


def tokens_for_domain(domain_char: str) -> List[str]:
    """Return all valid tokens for a given domain character."""
    return [t for t in MNEMO_VOCABULARY if t.startswith(domain_char)]


def tokens_for_operation(operation_char: str) -> List[str]:
    """Return all valid tokens containing a given operation character."""
    return [t for t in MNEMO_VOCABULARY if len(t) >= 2 and t[1] == operation_char]


def describe(token: str) -> str:
    """Return a human-readable description of a MNEMO token."""
    entry = lookup(token)
    if entry is None:
        return f"<unknown token '{token}'>"

    parts = [entry["domain"], entry["operation"]]
    if entry["modifier"]:
        parts.append(entry["modifier"])
    return " ".join(parts)
