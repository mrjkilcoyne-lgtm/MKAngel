"""
lexicon.py — Lexical units across all domains for the Grammar Language Model.

A lexicon is a structured inventory of the atomic *forms* a grammar
operates on.  In natural language these are words and morphemes; in
chemistry they are atoms and functional groups; in genetics they are
codons and motifs; in programming they are tokens and identifiers.

The GLM treats every domain uniformly: a ``LexicalEntry`` is a unit of
meaning that lives on one or more *substrates*, carries grammatical
category information, and — crucially — knows its own history.

Temporal metadata on each entry enables two powerful operations:

* **Etymology / derivation tracing** — follow an entry backward through
  time to discover what it derived from, all the way to the proto-form.

* **Derivative prediction** — follow an entry forward to forecast what
  new forms it is likely to produce, using the same grammatical rules
  that produced it in the first place.

Cross-domain cognates (shared structural roots) expose the *isomorphisms*
between grammars — the deep scales that the GLM is designed to learn.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set


# ---------------------------------------------------------------------------
# LexicalEntry — a single unit in any domain
# ---------------------------------------------------------------------------

@dataclass
class LexicalEntry:
    """A lexical unit: the atom a grammar transforms.

    A LexicalEntry is domain-agnostic.  It can represent a morpheme, an
    amino acid, a programming-language keyword, or a musical interval —
    whatever the grammar's substrate demands.

    Attributes:
        form:           The surface representation (string, symbol, structure).
        meaning:        Semantic content or functional role.
        id:             Unique identifier.
        category:       Grammatical category (e.g. "noun", "nucleotide",
                        "operator", "interval").
        substrates:     Names of substrates this entry belongs to.
        etymology:      Ordered derivation chain — a list of
                        ``(ancestor_form, rule_or_reason, timestamp)``
                        tuples tracing back through time.
        derivatives:    Known forward derivatives produced from this entry,
                        stored as ``(derived_form, rule_or_reason, timestamp)``.
        emerged_at:     Timestamp or epoch when this form first appeared.
        derived_from:   ID of the immediate parent entry (if any).
        predicts:       IDs of entries this form is expected to generate.
        metadata:       Arbitrary extra data.
    """

    form: Any
    meaning: Any = None
    id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])
    category: str = ""
    substrates: List[str] = field(default_factory=list)
    etymology: List[Dict[str, Any]] = field(default_factory=list)
    derivatives: List[Dict[str, Any]] = field(default_factory=list)
    emerged_at: Optional[float] = None
    derived_from: Optional[str] = None
    predicts: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    # -- derivation chain helpers -------------------------------------------

    def add_ancestor(
        self,
        ancestor_form: Any,
        rule: str = "",
        timestamp: Optional[float] = None,
    ) -> None:
        """Record an ancestor in the etymology chain."""
        self.etymology.append({
            "form": ancestor_form,
            "rule": rule,
            "timestamp": timestamp,
        })

    def add_derivative(
        self,
        derived_form: Any,
        rule: str = "",
        timestamp: Optional[float] = None,
    ) -> None:
        """Record a known derivative produced from this entry."""
        self.derivatives.append({
            "form": derived_form,
            "rule": rule,
            "timestamp": timestamp,
        })

    @property
    def root_form(self) -> Any:
        """The earliest known ancestor form, or self if none recorded."""
        if self.etymology:
            return self.etymology[0]["form"]
        return self.form

    def __hash__(self) -> int:
        return hash(self.id)

    def __eq__(self, other: object) -> bool:
        if isinstance(other, LexicalEntry):
            return self.id == other.id
        return NotImplemented

    def __repr__(self) -> str:
        return (
            f"LexicalEntry(form={self.form!r}, category={self.category!r}, "
            f"substrates={self.substrates!r})"
        )


# ---------------------------------------------------------------------------
# Lexicon — a structured collection of entries
# ---------------------------------------------------------------------------

@dataclass
class Lexicon:
    """A domain-spanning lexicon.

    The Lexicon is the GLM's memory of all known forms.  It supports four
    key operations aligned with the temporal-grammar framework:

    1. **lookup** — find entries by form, category, or substrate.
    2. **derive_etymology** — trace an entry backward through its
       ancestry chain.
    3. **predict_derivatives** — forecast what new forms an entry is
       likely to produce, using recorded derivatives and pattern matching.
    4. **find_cognates** — discover entries across different domains or
       substrates that share a common structural root, exposing the
       deep isomorphisms the GLM learns.

    Attributes:
        name:     Human-readable label.
        entries:  All lexical entries, keyed by ID.
        metadata: Arbitrary extra data.
    """

    name: str = "default"
    entries: Dict[str, LexicalEntry] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)

    # -- CRUD ---------------------------------------------------------------

    def add(self, entry: LexicalEntry) -> None:
        """Add an entry to the lexicon."""
        self.entries[entry.id] = entry

    def remove(self, entry_id: str) -> Optional[LexicalEntry]:
        """Remove and return an entry by ID."""
        return self.entries.pop(entry_id, None)

    def get(self, entry_id: str) -> Optional[LexicalEntry]:
        """Retrieve an entry by ID."""
        return self.entries.get(entry_id)

    @property
    def size(self) -> int:
        return len(self.entries)

    # -- lookup -------------------------------------------------------------

    def lookup(
        self,
        *,
        form: Any = None,
        category: str = "",
        substrate: str = "",
    ) -> List[LexicalEntry]:
        """Find entries matching any combination of form, category, substrate.

        All provided criteria must match (logical AND).  Omitted criteria
        are treated as wildcards.
        """
        results: List[LexicalEntry] = []
        for entry in self.entries.values():
            if form is not None and not _form_matches(entry.form, form):
                continue
            if category and entry.category != category:
                continue
            if substrate and substrate not in entry.substrates:
                continue
            results.append(entry)
        return results

    # -- etymology ----------------------------------------------------------

    def derive_etymology(self, entry_id: str) -> List[LexicalEntry]:
        """Trace the ancestry of an entry backward through time.

        Follows the ``derived_from`` chain, collecting every ancestor
        entry found in this lexicon.  The returned list is ordered from
        most recent ancestor to most distant.
        """
        chain: List[LexicalEntry] = []
        visited: Set[str] = set()
        current_id: Optional[str] = entry_id

        while current_id is not None:
            if current_id in visited:
                break  # strange loop in etymology — stop but don't crash
            visited.add(current_id)
            entry = self.entries.get(current_id)
            if entry is None:
                break
            if current_id != entry_id:
                chain.append(entry)
            current_id = entry.derived_from

        return chain

    # -- prediction ---------------------------------------------------------

    def predict_derivatives(self, entry_id: str) -> List[LexicalEntry]:
        """Forecast derivatives of an entry by following the ``predicts`` chain.

        Collects all entries this form is expected to generate (directly
        and transitively).  Stops at already-visited nodes to avoid
        infinite loops.
        """
        predictions: List[LexicalEntry] = []
        visited: Set[str] = set()
        frontier: List[str] = [entry_id]

        while frontier:
            current_id = frontier.pop(0)
            if current_id in visited:
                continue
            visited.add(current_id)

            entry = self.entries.get(current_id)
            if entry is None:
                continue

            for pred_id in entry.predicts:
                pred = self.entries.get(pred_id)
                if pred is not None and pred_id not in visited:
                    predictions.append(pred)
                    frontier.append(pred_id)

        return predictions

    # -- cognates -----------------------------------------------------------

    def find_cognates(
        self,
        entry_id: str,
        *,
        across_substrates: bool = True,
    ) -> List[LexicalEntry]:
        """Find cognates — entries sharing a common root with the given entry.

        Two entries are cognates if their etymology chains converge on
        the same root form.  When *across_substrates* is True (default),
        cognates from different substrates are included, exposing cross-
        domain structural isomorphisms.

        This is analogous to discovering that the English word "mother"
        and the Sanskrit "matar" share a Proto-Indo-European root, or
        that a codon and an amino acid map to each other through the
        genetic grammar.
        """
        target = self.entries.get(entry_id)
        if target is None:
            return []

        # Determine the root form of the target entry.
        target_root = self._root_form(entry_id)
        target_substrates = set(target.substrates)

        cognates: List[LexicalEntry] = []
        for eid, entry in self.entries.items():
            if eid == entry_id:
                continue
            if not across_substrates and not set(entry.substrates) & target_substrates:
                continue
            if self._root_form(eid) == target_root and target_root is not None:
                cognates.append(entry)

        return cognates

    # -- bulk operations ----------------------------------------------------

    def merge(self, other: "Lexicon") -> None:
        """Merge another lexicon into this one.

        Entries from *other* are added; collisions are resolved by keeping
        the entry with the longer etymology (more history preserved).
        """
        for eid, entry in other.entries.items():
            if eid not in self.entries:
                self.entries[eid] = entry
            else:
                existing = self.entries[eid]
                if len(entry.etymology) > len(existing.etymology):
                    self.entries[eid] = entry

    def all_substrates(self) -> Set[str]:
        """Return every substrate name represented in the lexicon."""
        subs: Set[str] = set()
        for entry in self.entries.values():
            subs.update(entry.substrates)
        return subs

    def all_categories(self) -> Set[str]:
        """Return every grammatical category present."""
        return {e.category for e in self.entries.values() if e.category}

    # -- internal -----------------------------------------------------------

    def _root_form(self, entry_id: str) -> Any:
        """Walk the etymology to find the ultimate root form."""
        entry = self.entries.get(entry_id)
        if entry is None:
            return None

        # Walk derived_from chain.
        visited: Set[str] = set()
        current = entry
        while current.derived_from and current.derived_from not in visited:
            visited.add(current.id)
            parent = self.entries.get(current.derived_from)
            if parent is None:
                break
            current = parent

        # If the entry has an etymology list, the first element is the oldest.
        if current.etymology:
            return current.etymology[0]["form"]
        return current.form

    def __len__(self) -> int:
        return len(self.entries)

    def __contains__(self, item: Any) -> bool:
        if isinstance(item, str):
            return item in self.entries
        if isinstance(item, LexicalEntry):
            return item.id in self.entries
        return False

    def __repr__(self) -> str:
        return f"Lexicon(name={self.name!r}, entries={len(self.entries)})"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _form_matches(entry_form: Any, query_form: Any) -> bool:
    """Flexible form matching: equality, substring, or type check.

    Short queries (< 3 chars) require exact match to prevent false
    positives like "i" matching "bind".  Longer queries still allow
    substring matching so that stems find their inflected forms.
    """
    if entry_form == query_form:
        return True
    if isinstance(entry_form, str) and isinstance(query_form, str):
        if len(query_form) < 3:
            # Short strings: exact only — avoids "i" ∈ "bind" etc.
            return False
        return query_form in entry_form
    return False
