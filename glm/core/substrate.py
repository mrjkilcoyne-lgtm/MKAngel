"""
Substrate — the medium on which grammars operate.

Every symbolic domain has a substrate: language has phonemes and morphemes,
chemistry has atoms and bonds, music has notes and intervals, code has tokens
and syntax trees.  This module defines the universal abstractions that all
substrates share.

Conceptual framework
────────────────────
Strange loops:  A substrate can contain sequences that *encode* the rules of
    the substrate itself — DNA encodes proteins that read DNA, a compiler is
    written in its own language, a grammar describes its own meta-grammar.
    ``detect_self_reference`` finds these loops.

Fugues:  Multiple substrates can run in parallel, each following its own
    structural rules but moving in coordinated lockstep — like voices in a
    Bach fugue.  ``align`` discovers the shared skeleton.

Isomorphisms:  Wildly different substrates (sound, meaning, molecule, code)
    share deep structural patterns — valence, constituency, recursion,
    agreement.  The base classes here capture exactly those universals.
"""

from __future__ import annotations

import re
from abc import ABC, abstractmethod
from collections import Counter
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import (
    Any,
    Callable,
    Dict,
    Generic,
    Iterator,
    List,
    Optional,
    Sequence as SequenceType,
    Set,
    Tuple,
    TypeVar,
    Union,
)


# ---------------------------------------------------------------------------
# Symbol — the atom of any substrate
# ---------------------------------------------------------------------------

@dataclass
class Symbol:
    """The atomic, indivisible unit of a substrate.

    Generalises phoneme, morpheme, atom, token, note — any irreducible
    element that a grammar manipulates.

    Attributes
    ----------
    form : str
        The surface representation (e.g. 'p', 'un-', 'C', 'if').
    features : dict[str, Any]
        A feature bundle describing the symbol's properties.  What features
        exist depends on the domain (voiced/place/manner for phonemes,
        element/charge for atoms, keyword/operator for tokens).
    domain : str
        Which substrate this symbol belongs to ('phonological',
        'morphological', 'molecular', 'symbolic', …).
    valence : int
        How many bonds / connections this symbol can form.  Mirrors
        chemical valence, syntactic valence (verb argument structure),
        and connector arity in formal logic.
    """

    form: str
    features: Dict[str, Any] = field(default_factory=dict)
    domain: str = "generic"
    valence: int = 1

    # -- rich comparison by form + features so symbols can live in sets -----

    def __hash__(self) -> int:
        return hash((self.form, self.domain))

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Symbol):
            return NotImplemented
        return self.form == other.form and self.domain == other.domain

    def __repr__(self) -> str:
        tag = f":{self.domain}" if self.domain != "generic" else ""
        return f"Symbol({self.form!r}{tag})"

    # -- feature algebra ----------------------------------------------------

    def has_feature(self, key: str, value: Any = None) -> bool:
        """Check whether this symbol carries a specific feature (and value)."""
        if key not in self.features:
            return False
        return value is None or self.features[key] == value

    def feature_distance(self, other: Symbol) -> float:
        """Hamming-like distance over shared feature keys.

        Returns a float in [0, 1]: 0 = identical features, 1 = maximally
        different.  Only features present in *either* symbol are considered.
        """
        all_keys = set(self.features) | set(other.features)
        if not all_keys:
            return 0.0
        mismatches = sum(
            1
            for k in all_keys
            if self.features.get(k) != other.features.get(k)
        )
        return mismatches / len(all_keys)

    def matches(self, pattern: Symbol) -> bool:
        """Return True if *pattern*'s features are a subset of ours.

        A pattern symbol acts as a filter — it matches any symbol whose
        features include at least the pattern's features.
        """
        return all(
            self.features.get(k) == v for k, v in pattern.features.items()
        )

    def can_bond(self, other: Symbol) -> bool:
        """Return True if both symbols have remaining valence capacity.

        This is a *necessary* but not *sufficient* condition for bonding —
        domain-specific substrates add their own combination rules on top.
        """
        return self.valence > 0 and other.valence > 0


# ---------------------------------------------------------------------------
# Sequence — an ordered collection of symbols
# ---------------------------------------------------------------------------

class Sequence:
    """An ordered sequence of Symbols — the substrate analog of a string,
    a molecular chain, a melodic line, or a token stream.

    Supports slicing, concatenation, pattern matching, and self-similarity
    detection (a form of strange loop: a sequence that contains copies of
    its own sub-patterns).
    """

    __slots__ = ("_symbols",)

    def __init__(self, symbols: Optional[List[Symbol]] = None) -> None:
        self._symbols: List[Symbol] = list(symbols) if symbols else []

    # -- container protocol -------------------------------------------------

    def __len__(self) -> int:
        return len(self._symbols)

    def __getitem__(self, idx: Union[int, slice]) -> Union[Symbol, "Sequence"]:
        if isinstance(idx, slice):
            return Sequence(self._symbols[idx])
        return self._symbols[idx]

    def __iter__(self) -> Iterator[Symbol]:
        return iter(self._symbols)

    def __contains__(self, item: Union[Symbol, "Sequence"]) -> bool:
        if isinstance(item, Symbol):
            return item in self._symbols
        if isinstance(item, Sequence):
            return self._find_subsequence(item) >= 0
        return False

    def __add__(self, other: "Sequence") -> "Sequence":
        return Sequence(self._symbols + list(other))

    def __repr__(self) -> str:
        forms = " ".join(s.form for s in self._symbols)
        return f"Sequence([{forms}])"

    @property
    def symbols(self) -> List[Symbol]:
        return list(self._symbols)

    @property
    def forms(self) -> List[str]:
        """The surface forms of every symbol, as a plain list of strings."""
        return [s.form for s in self._symbols]

    def append(self, symbol: Symbol) -> None:
        self._symbols.append(symbol)

    def extend(self, symbols: SequenceType[Symbol]) -> None:
        self._symbols.extend(symbols)

    # -- subsequence search -------------------------------------------------

    def _find_subsequence(self, sub: "Sequence", start: int = 0) -> int:
        """Return the first index where *sub* occurs in self, or -1."""
        sub_len = len(sub)
        for i in range(start, len(self) - sub_len + 1):
            if all(self._symbols[i + j] == sub[j] for j in range(sub_len)):
                return i
        return -1

    def find_all(self, sub: "Sequence") -> List[int]:
        """Return every starting index where *sub* occurs."""
        positions: List[int] = []
        start = 0
        while True:
            idx = self._find_subsequence(sub, start)
            if idx < 0:
                break
            positions.append(idx)
            start = idx + 1
        return positions

    # -- pattern matching ---------------------------------------------------

    def match_pattern(self, pattern: List[Optional[Symbol]]) -> List[int]:
        """Match a pattern with optional wildcards (None) against the sequence.

        Returns a list of starting positions where the pattern matches.
        A None element in the pattern matches any symbol.
        """
        pat_len = len(pattern)
        hits: List[int] = []
        for i in range(len(self) - pat_len + 1):
            if all(
                p is None or self._symbols[i + j].matches(p)
                for j, p in enumerate(pattern)
            ):
                hits.append(i)
        return hits

    # -- self-similarity / strange-loop detection ---------------------------

    def find_repeating_patterns(
        self, min_length: int = 2, max_length: Optional[int] = None
    ) -> Dict[str, List[int]]:
        """Discover repeating sub-patterns within the sequence.

        This is a form of *self-similarity* — a hallmark of strange loops.
        When a sequence contains recurring motifs it is, in a loose sense,
        "quoting" itself.

        Returns a dict mapping the pattern's form-string to the list of
        starting positions where it occurs (only patterns occurring 2+
        times are returned).
        """
        if max_length is None:
            max_length = len(self) // 2
        max_length = min(max_length, len(self) // 2)
        result: Dict[str, List[int]] = {}
        for length in range(min_length, max_length + 1):
            for start in range(len(self) - length + 1):
                key = " ".join(
                    self._symbols[start + k].form for k in range(length)
                )
                if key not in result:
                    sub = Sequence(self._symbols[start : start + length])
                    positions = self.find_all(sub)
                    if len(positions) >= 2:
                        result[key] = positions
        return result

    # -- alignment ----------------------------------------------------------

    @staticmethod
    def align(
        seq_a: "Sequence",
        seq_b: "Sequence",
        match_score: float = 2.0,
        mismatch_penalty: float = -1.0,
        gap_penalty: float = -1.0,
        feature_weight: float = 1.0,
    ) -> Tuple[List[Optional[Symbol]], List[Optional[Symbol]], float]:
        """Needleman-Wunsch global alignment between two sequences.

        This mirrors sequence alignment in bioinformatics (DNA / protein)
        and phonological alignment in historical linguistics (cognate
        detection, sound-correspondence).

        Parameters
        ----------
        seq_a, seq_b : Sequence
            The two sequences to align.
        match_score : float
            Reward for an exact symbol match.
        mismatch_penalty : float
            Base penalty for mismatched symbols (modulated by feature
            distance when *feature_weight* > 0).
        gap_penalty : float
            Penalty for inserting a gap.
        feature_weight : float
            When > 0, mismatch penalty is softened for symbols that are
            featurally close (e.g. /p/ vs. /b/ differ only in voicing).

        Returns
        -------
        aligned_a : list[Symbol | None]
            Aligned version of seq_a (None = gap).
        aligned_b : list[Symbol | None]
            Aligned version of seq_b.
        score : float
            The alignment score.
        """
        n, m = len(seq_a), len(seq_b)

        # -- build score matrix ---------------------------------------------
        dp = [[0.0] * (m + 1) for _ in range(n + 1)]
        for i in range(1, n + 1):
            dp[i][0] = dp[i - 1][0] + gap_penalty
        for j in range(1, m + 1):
            dp[0][j] = dp[0][j - 1] + gap_penalty

        for i in range(1, n + 1):
            for j in range(1, m + 1):
                sa, sb = seq_a[i - 1], seq_b[j - 1]
                if sa == sb:
                    diag = dp[i - 1][j - 1] + match_score
                else:
                    dist = sa.feature_distance(sb) if feature_weight > 0 else 1.0
                    diag = dp[i - 1][j - 1] + mismatch_penalty * dist
                up = dp[i - 1][j] + gap_penalty
                left = dp[i][j - 1] + gap_penalty
                dp[i][j] = max(diag, up, left)

        # -- traceback ------------------------------------------------------
        aligned_a: List[Optional[Symbol]] = []
        aligned_b: List[Optional[Symbol]] = []
        i, j = n, m
        while i > 0 or j > 0:
            if i > 0 and j > 0:
                sa, sb = seq_a[i - 1], seq_b[j - 1]
                if sa == sb:
                    diag_score = dp[i - 1][j - 1] + match_score
                else:
                    dist = sa.feature_distance(sb) if feature_weight > 0 else 1.0
                    diag_score = dp[i - 1][j - 1] + mismatch_penalty * dist
                if dp[i][j] == diag_score:
                    aligned_a.append(sa)
                    aligned_b.append(sb)
                    i -= 1
                    j -= 1
                    continue
            if i > 0 and dp[i][j] == dp[i - 1][j] + gap_penalty:
                aligned_a.append(seq_a[i - 1])
                aligned_b.append(None)
                i -= 1
            else:
                aligned_a.append(None)
                aligned_b.append(seq_b[j - 1])
                j -= 1

        aligned_a.reverse()
        aligned_b.reverse()
        return aligned_a, aligned_b, dp[n][m]


# ---------------------------------------------------------------------------
# TransformationRule — a rewrite rule that operates on sequences
# ---------------------------------------------------------------------------

@dataclass
class TransformationRule:
    """A context-sensitive rewrite rule: A -> B / C _ D

    In linguistics this is a phonological or morphological rule.
    In chemistry it is a reaction rule.  In programming it is a
    source-to-source transformation.

    Attributes
    ----------
    name : str
        Human-readable label.
    pattern : list[Optional[Symbol]]
        The target pattern to match (None elements are wildcards).
    replacement : list[Symbol]
        What to substitute for the matched span.
    left_context : list[Optional[Symbol]] | None
        Required context to the left of the match (None = any).
    right_context : list[Optional[Symbol]] | None
        Required context to the right of the match (None = any).
    """

    name: str
    pattern: List[Optional[Symbol]]
    replacement: List[Symbol]
    left_context: Optional[List[Optional[Symbol]]] = None
    right_context: Optional[List[Optional[Symbol]]] = None

    def _context_matches(
        self,
        sequence: Sequence,
        match_start: int,
        match_end: int,
    ) -> bool:
        """Check whether left and right context constraints are satisfied."""
        if self.left_context is not None:
            ctx_len = len(self.left_context)
            ctx_start = match_start - ctx_len
            if ctx_start < 0:
                return False
            for k, pat in enumerate(self.left_context):
                if pat is not None and not sequence[ctx_start + k].matches(pat):
                    return False
        if self.right_context is not None:
            ctx_len = len(self.right_context)
            if match_end + ctx_len > len(sequence):
                return False
            for k, pat in enumerate(self.right_context):
                if pat is not None and not sequence[match_end + k].matches(pat):
                    return False
        return True

    def apply(self, sequence: Sequence) -> Sequence:
        """Apply this rule to the first matching site in *sequence*.

        Returns a new Sequence with the replacement performed.  If the
        pattern is not found, the original sequence is returned unchanged.
        """
        pat_len = len(self.pattern)
        for i in range(len(sequence) - pat_len + 1):
            if all(
                p is None or sequence[i + j].matches(p)
                for j, p in enumerate(self.pattern)
            ):
                if self._context_matches(sequence, i, i + pat_len):
                    new_symbols = (
                        sequence[:i].symbols
                        + list(self.replacement)
                        + sequence[i + pat_len :].symbols
                    )
                    return Sequence(new_symbols)
        return sequence

    def apply_all(self, sequence: Sequence) -> Sequence:
        """Apply this rule to every non-overlapping match, left to right."""
        pat_len = len(self.pattern)
        result: List[Symbol] = []
        i = 0
        while i < len(sequence):
            matched = False
            if i <= len(sequence) - pat_len:
                if all(
                    p is None or sequence[i + j].matches(p)
                    for j, p in enumerate(self.pattern)
                ):
                    if self._context_matches(sequence, i, i + pat_len):
                        result.extend(self.replacement)
                        i += pat_len
                        matched = True
            if not matched:
                result.append(sequence[i])
                i += 1
        return Sequence(result)


# ---------------------------------------------------------------------------
# Substrate — a complete symbolic system
# ---------------------------------------------------------------------------

class Substrate(ABC):
    """A complete symbolic system — the medium on which grammars operate.

    A Substrate is defined by:
    1. A *symbol inventory* — the set of atomic units.
    2. *Combination rules* — how symbols bond (phonotactics, chemical
       valence rules, syntax).
    3. A *feature system* — the dimensions along which symbols vary.
    4. *Transformation rules* — rewrite rules that map sequences to
       sequences (sound change, chemical reaction, compilation).

    Subclasses implement domain-specific encoding/decoding and can override
    alignment and pattern-finding with domain-aware algorithms.
    """

    def __init__(self, name: str, domain: str) -> None:
        self.name = name
        self.domain = domain
        self._inventory: Dict[str, Symbol] = {}
        self._rules: List[TransformationRule] = []
        self._combination_rules: List[Callable[[Symbol, Symbol], bool]] = []
        self._feature_system: Dict[str, Set[Any]] = {}

    # -- inventory management -----------------------------------------------

    @property
    def inventory(self) -> Dict[str, Symbol]:
        return dict(self._inventory)

    def add_symbol(self, symbol: Symbol) -> None:
        """Register a symbol in this substrate's inventory."""
        symbol.domain = self.domain
        self._inventory[symbol.form] = symbol

    def get_symbol(self, form: str) -> Optional[Symbol]:
        return self._inventory.get(form)

    def add_combination_rule(
        self, rule: Callable[[Symbol, Symbol], bool]
    ) -> None:
        """Add a predicate that decides whether two symbols may combine."""
        self._combination_rules.append(rule)

    def can_combine(self, a: Symbol, b: Symbol) -> bool:
        """Check all combination rules; default is True if no rules fail."""
        if not self._combination_rules:
            return a.can_bond(b)
        return all(rule(a, b) for rule in self._combination_rules)

    def add_feature(self, name: str, values: Set[Any]) -> None:
        """Declare a feature dimension and its legal values."""
        self._feature_system[name] = values

    def add_rule(self, rule: TransformationRule) -> None:
        self._rules.append(rule)

    # -- abstract interface -------------------------------------------------

    @abstractmethod
    def encode(self, raw_input: str) -> Sequence:
        """Convert raw external input into a Sequence of Symbols.

        Each substrate defines its own tokenisation / parsing strategy.
        """

    @abstractmethod
    def decode(self, sequence: Sequence) -> str:
        """Convert a Sequence of Symbols back into an external representation."""

    # -- alignment (default: Needleman-Wunsch) ------------------------------

    def align(
        self,
        seq_a: Sequence,
        seq_b: Sequence,
        **kwargs: Any,
    ) -> Tuple[List[Optional[Symbol]], List[Optional[Symbol]], float]:
        """Find the best structural alignment between two sequences.

        Like sequence alignment in bioinformatics or cognate alignment in
        historical linguistics.  Subclasses may override with domain-aware
        scoring.
        """
        return Sequence.align(seq_a, seq_b, **kwargs)

    # -- pattern finding ----------------------------------------------------

    def find_patterns(
        self,
        sequence: Sequence,
        min_length: int = 2,
        max_length: Optional[int] = None,
    ) -> Dict[str, List[int]]:
        """Find recurring patterns / motifs in a sequence.

        Returns a dict mapping pattern-form-string to the list of positions
        where it occurs.
        """
        return sequence.find_repeating_patterns(min_length, max_length)

    # -- strange loop detection ---------------------------------------------

    def detect_self_reference(self, sequence: Sequence) -> List[Tuple[int, int, str]]:
        """Find where a sequence references or encodes itself.

        A *strange loop* arises when a symbolic system describes its own
        structure — DNA encoding its own replication machinery, a quine
        printing its own source, a grammar whose rules are themselves
        sentences of the grammar.

        Default implementation: find every sub-sequence whose decoded form
        is a valid encoding back into the same (or a sub-) sequence.  This
        is necessarily heuristic; subclasses should override with domain-
        specific logic.

        Returns a list of (start, end, description) triples.
        """
        loops: List[Tuple[int, int, str]] = []
        n = len(sequence)
        if n == 0:
            return loops

        # Strategy 1: literal self-quoting — a sub-sequence whose forms
        # spell out the forms of the whole sequence (or a super-pattern).
        full_forms = " ".join(s.form for s in sequence)
        for length in range(2, n):
            for start in range(n - length + 1):
                sub_forms = " ".join(
                    sequence[start + k].form for k in range(length)
                )
                # Check if the sub-sequence's decoded form re-encodes to
                # something that appears elsewhere in the sequence.
                try:
                    decoded = self.decode(sequence[start : start + length])
                    re_encoded = self.encode(decoded)
                    re_forms = " ".join(s.form for s in re_encoded)
                    if re_forms in full_forms and len(re_encoded) > 0:
                        # Nontrivial: re-encoded form appears in the
                        # original and is not the sub-sequence itself
                        re_positions = sequence[0:0]  # dummy
                        outer_hits = full_forms.count(re_forms)
                        if outer_hits >= 2 or re_forms != sub_forms:
                            loops.append((
                                start,
                                start + length,
                                f"sub-sequence [{sub_forms}] re-encodes to "
                                f"[{re_forms}] which appears in the parent",
                            ))
                except Exception:
                    pass

        # Strategy 2: repeating motifs (self-similarity).
        patterns = self.find_patterns(sequence, min_length=2)
        for pat, positions in patterns.items():
            if len(positions) >= 3:
                loops.append((
                    positions[0],
                    positions[0] + len(pat.split()),
                    f"motif [{pat}] repeats {len(positions)} times "
                    f"(self-similar structure)",
                ))

        return loops

    # -- transformation -----------------------------------------------------

    def transform(
        self,
        sequence: Sequence,
        rule: Union[TransformationRule, str],
    ) -> Sequence:
        """Apply a transformation rule to a sequence.

        *rule* may be a ``TransformationRule`` instance or the *name* of a
        rule previously registered with ``add_rule``.
        """
        if isinstance(rule, str):
            for r in self._rules:
                if r.name == rule:
                    return r.apply(sequence)
            raise ValueError(f"Unknown rule: {rule!r}")
        return rule.apply(sequence)

    def transform_all(
        self,
        sequence: Sequence,
        rule: Union[TransformationRule, str],
    ) -> Sequence:
        """Apply a transformation rule to *every* matching site."""
        if isinstance(rule, str):
            for r in self._rules:
                if r.name == rule:
                    return r.apply_all(sequence)
            raise ValueError(f"Unknown rule: {rule!r}")
        return rule.apply_all(sequence)

    # -- utility ------------------------------------------------------------

    def __repr__(self) -> str:
        return (
            f"{type(self).__name__}({self.name!r}, "
            f"symbols={len(self._inventory)}, "
            f"rules={len(self._rules)})"
        )
