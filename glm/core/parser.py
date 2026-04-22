"""
Parser -- bridge natural language to abstract grammar categories.

The grammar engine works on abstract symbols (NP, VP, Det, N). Natural
language arrives as raw words ("the cat sat"). The Parser bridges the
two with two responsibilities:

1. **POS tagging** -- look up each word in the Lexicon and assign its
   category. Falls back to morphological heuristics for unknowns.

2. **Chart parsing** -- iteratively reduce category sequences using
   grammar productions, bottom-up. Finds a parse tree (or forest)
   that covers the full input.

Pure Python. No external parser library.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


# ---------------------------------------------------------------------------
# Heuristic POS tagger (fallback for words not in lexicon)
# ---------------------------------------------------------------------------

_DETERMINERS = {"the", "a", "an", "this", "that", "these", "those",
                "my", "your", "his", "her", "its", "our", "their",
                "some", "any", "no", "every", "each", "all"}
_PRONOUNS = {"i", "me", "you", "he", "him", "she", "her", "it",
             "we", "us", "they", "them", "who", "whom", "what"}
_AUX_VERBS = {"is", "are", "was", "were", "be", "been", "being",
              "am", "do", "does", "did", "have", "has", "had",
              "will", "would", "can", "could", "shall", "should",
              "may", "might", "must"}
_PREPOSITIONS = {"in", "on", "at", "by", "for", "with", "to", "from",
                 "of", "about", "into", "over", "under", "through",
                 "between", "among", "against", "during", "before",
                 "after", "above", "below", "near"}
_CONJUNCTIONS = {"and", "or", "but", "so", "because", "if", "when",
                 "while", "although", "though", "since", "as"}
_COMPLEMENTIZERS = {"that", "whether", "if"}
_WH_WORDS = {"who", "what", "where", "when", "why", "how", "which"}


def heuristic_tag(word: str) -> str:
    """Best-guess POS tag for a word not in the lexicon."""
    w = word.lower().strip(".,!?;:\"'()[]")
    if not w:
        return "X"

    if w in _WH_WORDS:
        return "Wh"
    if w in _DETERMINERS:
        return "Det"
    if w in _PRONOUNS:
        return "Pron"
    if w in _AUX_VERBS:
        return "Aux"
    if w in _PREPOSITIONS:
        return "P"
    if w in _CONJUNCTIONS:
        return "Conj"

    # Suffix-based heuristics
    if w.endswith("ly"):
        return "Adv"
    if w.endswith(("ing", "ed", "es")) and len(w) > 3:
        return "V"
    if w.endswith(("tion", "sion", "ment", "ness", "ity", "ence", "ance")):
        return "N"
    if w.endswith(("ful", "ous", "ive", "able", "ible", "al", "ic", "ish",
                    "ant", "ent", "lar", "ern")):
        return "Adj"
    if w.endswith("s") and len(w) > 2 and not w.endswith(("ss", "us", "is")):
        return "N"

    # Default: noun
    return "N"


# ---------------------------------------------------------------------------
# Parser
# ---------------------------------------------------------------------------

@dataclass
class ParseNode:
    """A node in a parse tree."""
    category: str
    span: tuple[int, int]  # (start, end) token indices
    children: list["ParseNode"] = field(default_factory=list)
    word: str | None = None  # leaf only
    rule_id: str = ""

    def to_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {"cat": self.category, "span": list(self.span)}
        if self.word is not None:
            d["word"] = self.word
        if self.children:
            d["children"] = [c.to_dict() for c in self.children]
        if self.rule_id:
            d["rule"] = self.rule_id
        return d

    def flatten(self) -> list[str]:
        """Return the leaf words in left-to-right order."""
        if self.word is not None:
            return [self.word]
        out = []
        for c in self.children:
            out.extend(c.flatten())
        return out


class Parser:
    """Bridges natural language and abstract grammar.

    Usage::

        parser = Parser(lexicon, grammar)
        tags = parser.tag(["the", "cat", "sat"])  # -> ["Det", "N", "V"]
        tree = parser.parse(["the", "cat", "sat"])  # -> ParseNode or None
    """

    def __init__(self, lexicon: Any = None, grammar: Any = None):
        self._lexicon = lexicon
        self._grammar = grammar

    # ------------------------------------------------------------------
    # POS tagging
    # ------------------------------------------------------------------

    def tag(self, tokens: list[str]) -> list[str]:
        """Assign a grammatical category to each token."""
        tags = []
        for tok in tokens:
            cat = self._lookup_category(tok)
            if not cat:
                cat = heuristic_tag(tok)
            tags.append(_normalise_category(cat))
        return tags

    def _lookup_category(self, word: str) -> str:
        """Look up word in lexicon; return its category or empty string.

        Lexicon lookup is substring-based by default, so we filter to
        exact matches and prefer categories that are already normalised
        grammatical tags (Det, Pron, N, V) over raw English like "noun".
        """
        if self._lexicon is None:
            return ""
        try:
            w = word.lower().strip(".,!?;:\"'()[]")
            entries = self._lexicon.lookup(form=w)
            if not entries:
                return ""
            exact = [e for e in entries if e.form == w]
            pool = exact if exact else entries
            # Prefer entries with tag-like categories (short, capitalised)
            pool.sort(key=lambda e: (
                0 if (e.category and e.category[0].isupper()) else 1,
                len(e.category),
            ))
            return pool[0].category
        except Exception:
            pass
        return ""

    # ------------------------------------------------------------------
    # Chart parsing (bottom-up, greedy leftmost reduction)
    # ------------------------------------------------------------------

    def parse(self, tokens: list[str]) -> ParseNode | None:
        """Parse raw tokens into a tree using the grammar.

        Returns the root ParseNode or None if parsing fails.
        Uses a bottom-up approach: POS-tag, then iteratively apply
        productions to reduce category sub-sequences.
        """
        if not tokens:
            return None

        tags = self.tag(tokens)

        # Build leaf nodes
        chart: list[ParseNode] = [
            ParseNode(category=tag, span=(i, i + 1), word=tok)
            for i, (tok, tag) in enumerate(zip(tokens, tags))
        ]

        if self._grammar is None:
            # No grammar -- return a flat "S" node wrapping leaves
            return ParseNode(
                category="S", span=(0, len(tokens)), children=chart
            )

        productions = list(self._grammar.all_productions())

        # Iteratively reduce until no more reductions possible
        changed = True
        max_iterations = 100
        iteration = 0
        while changed and iteration < max_iterations:
            changed = False
            iteration += 1

            # Try every production at every position (leftmost first)
            for prod in productions:
                rhs = prod.rhs
                if not isinstance(rhs, list):
                    continue

                n = len(rhs)
                # Scan for matching subsequence of length n
                for start in range(len(chart) - n + 1):
                    window = chart[start:start + n]
                    window_cats = [node.category for node in window]
                    if window_cats == list(rhs):
                        # Reduce
                        merged = ParseNode(
                            category=prod.lhs if isinstance(prod.lhs, str) else str(prod.lhs),
                            span=(window[0].span[0], window[-1].span[1]),
                            children=list(window),
                            rule_id=prod.id,
                        )
                        chart = chart[:start] + [merged] + chart[start + n:]
                        changed = True
                        break
                if changed:
                    break

        if len(chart) == 1:
            return chart[0]
        # Partial parse -- wrap whatever remains
        return ParseNode(
            category="S",
            span=(0, len(tokens)),
            children=chart,
        )

    # ------------------------------------------------------------------
    # Surface generation (reverse of parse)
    # ------------------------------------------------------------------

    def generate(self, start: str = "S", max_depth: int = 6) -> list[list[str]]:
        """Forward-generate surface category sequences from a start symbol.

        Returns a list of category sequences that the grammar can produce.
        Each sequence is a flat list of terminal categories.
        """
        if self._grammar is None:
            return []

        productions = list(self._grammar.all_productions())
        by_lhs: dict[str, list[Any]] = {}
        for prod in productions:
            lhs = prod.lhs if isinstance(prod.lhs, str) else str(prod.lhs)
            by_lhs.setdefault(lhs, []).append(prod.rhs)

        results: list[list[str]] = []

        def expand(seq: list[str], depth: int) -> None:
            if depth > max_depth:
                return
            # Check if all terminal (no entry in by_lhs)
            if all(sym not in by_lhs for sym in seq):
                results.append(list(seq))
                return
            # Expand the leftmost non-terminal
            for i, sym in enumerate(seq):
                if sym in by_lhs:
                    for rhs in by_lhs[sym]:
                        if isinstance(rhs, list):
                            new_seq = seq[:i] + list(rhs) + seq[i + 1:]
                        else:
                            new_seq = seq[:i] + [rhs] + seq[i + 1:]
                        expand(new_seq, depth + 1)
                    return

        expand([start], 0)
        # Deduplicate
        unique: list[list[str]] = []
        seen: set[tuple[str, ...]] = set()
        for seq in results:
            key = tuple(seq)
            if key not in seen:
                seen.add(key)
                unique.append(seq)
        return unique[:50]


# ---------------------------------------------------------------------------
# Category normalisation
# ---------------------------------------------------------------------------

_CATEGORY_ALIASES = {
    "noun": "N",
    "verb": "V",
    "adjective": "Adj",
    "adj": "Adj",
    "adverb": "Adv",
    "adv": "Adv",
    "determiner": "Det",
    "det": "Det",
    "pronoun": "Pron",
    "pron": "Pron",
    "preposition": "P",
    "prep": "P",
    "conjunction": "Conj",
    "conj": "Conj",
    "auxiliary": "Aux",
    "aux": "Aux",
    "complementizer": "C",
    "wh": "Wh",
}


def _normalise_category(cat: str) -> str:
    """Normalise a category string to the grammar's canonical form."""
    if not cat:
        return "N"
    c = cat.strip().lower()
    return _CATEGORY_ALIASES.get(c, cat.strip())
