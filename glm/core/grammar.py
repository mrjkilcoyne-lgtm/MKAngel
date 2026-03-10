"""
grammar.py — Foundational grammar system for the Grammar Language Model.

This module implements the core primitives that make a GLM possible:
rules, productions, grammars, and strange loops.  These are the *scales*
the model internalises so it can compose masterpieces from first principles.

Key concepts (after Hofstadter, *Gödel, Escher, Bach*):

* **Strange loops** — Following a chain of rules can return you to the
  starting symbol, but at a higher (or lower) level of abstraction.
  Self-reference is not a bug; it is the source of expressive power.

* **Fugues** — Multiple grammars (voices) can run in parallel over the
  same or different substrates, each following its own rule set but
  sharing a common temporal frame.  Where derivations converge we find
  *harmony*; where they diverge we find *counterpoint*.

* **Temporal derivation** — Every rule is bidirectional.  Applied forward
  it *predicts* (generates the next form); applied backward it
  *reconstructs* (recovers the originating form).  Time is not an axis
  the grammar sits on — it is a dimension the grammar *moves through*.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, FrozenSet, List, Optional, Set, Tuple


# ---------------------------------------------------------------------------
# Direction — the arrow of grammatical time
# ---------------------------------------------------------------------------

class Direction(Enum):
    """Which way the rule is applied along the temporal axis."""
    FORWARD = "forward"    # generate / predict
    BACKWARD = "backward"  # parse / reconstruct


# ---------------------------------------------------------------------------
# Rule — the atomic transformation
# ---------------------------------------------------------------------------

@dataclass
class Rule:
    """A single transformation rule: pattern → result.

    Rules are the atoms of the grammar.  Each rule says: "when you see
    *pattern* (and *conditions* are met), you may produce *result*."

    Rules can be **self-referential**: a rule's result may reference
    the rule itself (or another rule that eventually references it),
    creating a strange loop.

    Attributes:
        id:          Unique identifier.
        name:        Human-readable label.
        pattern:     What the rule matches (any hashable or structured form).
        result:      What the rule produces.
        conditions:  Optional predicate that must hold for the rule to fire.
        weight:      Confidence / probability weight in [0, 1].
        direction:   Whether the rule is meant to be applied forward,
                     backward, or both (None means both).
        references:  IDs of other rules this rule explicitly references
                     (enables loop detection).
        metadata:    Arbitrary extra data.
    """

    pattern: Any
    result: Any
    name: str = ""
    id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])
    conditions: Optional[Callable[..., bool]] = None
    weight: float = 1.0
    direction: Optional[Direction] = None  # None ⇒ bidirectional
    references: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    self_referential: bool = False

    def __post_init__(self) -> None:
        # Accept string directions from domain grammars
        if isinstance(self.direction, str):
            _dir_map = {
                "forward": Direction.FORWARD,
                "backward": Direction.BACKWARD,
                "bidirectional": None,
            }
            self.direction = _dir_map.get(self.direction.lower(), None)
        # If self_referential is set, record self-reference in references
        if self.self_referential and self.name and self.name not in self.references:
            self.references.append(self.name)

    # -- application --------------------------------------------------------

    def matches(self, form: Any, direction: Direction = Direction.FORWARD) -> bool:
        """Return True if *form* matches this rule's trigger side.

        For FORWARD application the trigger is ``pattern``;
        for BACKWARD it is ``result``.
        """
        if self.direction is not None and self.direction != direction:
            return False

        trigger = self.pattern if direction == Direction.FORWARD else self.result
        return self._match(trigger, form)

    def apply(self, form: Any, direction: Direction = Direction.FORWARD) -> Optional[Any]:
        """Apply the rule to *form*, returning the produced output or None.

        Checks ``conditions`` if present.  Returns ``None`` when the rule
        does not fire (no match or failed condition).
        """
        if not self.matches(form, direction):
            return None

        if self.conditions is not None:
            try:
                if not self.conditions(form):
                    return None
            except Exception:
                return None

        if direction == Direction.FORWARD:
            return self._produce(self.result, form)
        else:
            return self._produce(self.pattern, form)

    # -- internal helpers ---------------------------------------------------

    @staticmethod
    def _match(trigger: Any, form: Any) -> bool:
        """Flexible matching: equality, substring, or callable trigger."""
        if trigger is None:
            return True  # wildcard
        if callable(trigger) and not isinstance(trigger, type):
            try:
                return bool(trigger(form))
            except Exception:
                return False
        if isinstance(trigger, type):
            return isinstance(form, trigger)
        # String containment for string patterns.
        if isinstance(trigger, str) and isinstance(form, str):
            return trigger in form
        # Sequence prefix matching.
        if isinstance(trigger, (list, tuple)) and isinstance(form, (list, tuple)):
            if len(trigger) > len(form):
                return False
            return all(Rule._match(t, f) for t, f in zip(trigger, form))
        return trigger == form

    @staticmethod
    def _produce(template: Any, form: Any) -> Any:
        """Produce the output side given the template and matched form.

        If the template is callable it is invoked with *form*; otherwise
        the template is returned directly.
        """
        if callable(template) and not isinstance(template, type):
            try:
                return template(form)
            except Exception:
                return template
        return template

    def __hash__(self) -> int:
        return hash(self.id)

    def __eq__(self, other: object) -> bool:
        if isinstance(other, Rule):
            return self.id == other.id
        return NotImplemented


# ---------------------------------------------------------------------------
# Production — a formal grammar production LHS → RHS
# ---------------------------------------------------------------------------

@dataclass
class Production:
    """A production rule in a formal grammar: LHS → RHS.

    Productions are higher-level than raw Rules.  They track their
    **derivation history** — every time a production fires, the step is
    recorded so the full derivation tree can be reconstructed later.

    Productions are bidirectional:
      * Forward (generate/predict): expand LHS into RHS.
      * Backward (parse/reconstruct): reduce RHS back to LHS.

    Attributes:
        lhs:      The left-hand side (a symbol or sequence).
        rhs:      The right-hand side (a symbol, sequence, or list thereof).
        name:     Human-readable label.
        id:       Unique identifier.
        weight:   Probability weight.
        history:  List of (input, output, direction, context) derivation records.
    """

    lhs: Any
    rhs: Any
    name: str = ""
    id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])
    weight: float = 1.0
    history: List[Dict[str, Any]] = field(default_factory=list)

    def apply_forward(self, form: Any) -> Optional[Any]:
        """Expand: if *form* matches LHS, produce RHS."""
        if not self._lhs_matches(form):
            return None
        output = self._expand(form)
        self.history.append({
            "input": form,
            "output": output,
            "direction": Direction.FORWARD,
        })
        return output

    def apply_backward(self, form: Any) -> Optional[Any]:
        """Reduce: if *form* matches RHS, recover LHS."""
        if not self._rhs_matches(form):
            return None
        output = self._reduce(form)
        self.history.append({
            "input": form,
            "output": output,
            "direction": Direction.BACKWARD,
        })
        return output

    # -- matching helpers ---------------------------------------------------

    def _lhs_matches(self, form: Any) -> bool:
        return Rule._match(self.lhs, form)

    def _rhs_matches(self, form: Any) -> bool:
        """Check if form matches the RHS.

        If RHS is a list of alternatives, any match suffices.
        """
        if isinstance(self.rhs, list):
            return any(Rule._match(alt, form) for alt in self.rhs)
        return Rule._match(self.rhs, form)

    def _expand(self, form: Any) -> Any:
        """Produce RHS from the matched form."""
        if callable(self.rhs) and not isinstance(self.rhs, type):
            try:
                return self.rhs(form)
            except Exception:
                return self.rhs
        # If RHS is a list, return the full expansion.
        return self.rhs

    def _reduce(self, form: Any) -> Any:
        """Recover LHS from the matched form."""
        if callable(self.lhs) and not isinstance(self.lhs, type):
            try:
                return self.lhs(form)
            except Exception:
                return self.lhs
        return self.lhs

    def __hash__(self) -> int:
        return hash(self.id)

    def __eq__(self, other: object) -> bool:
        if isinstance(other, Production):
            return self.id == other.id
        return NotImplemented


# ---------------------------------------------------------------------------
# StrangeLoop — a self-referential cycle in the grammar
# ---------------------------------------------------------------------------

class StrangeLoop:
    """A detected self-referential cycle in a grammar.

    In Hofstadter's framing, a strange loop arises when moving through
    the levels of a hierarchical system unexpectedly returns you to where
    you started — but transformed.  In a grammar this means a chain of
    rule applications that maps a symbol back onto itself (possibly at a
    different level of derivation).

    Strange loops are **not** errors.  They are the grammar's mechanism
    for expressing recursion, self-similarity, and generative infinity
    with a finite rule set.

    Attributes:
        id:         Unique identifier.
        cycle:      The ordered list of rule/production IDs forming the loop.
        entry:      The symbol or form at which the loop begins/ends.
                    (Also accepted as ``entry_rule`` for compatibility.)
        level_delta: How many hierarchical levels the loop traverses before
                    returning.  Can be an int (+1/-1/0) **or** a descriptive
                    string (e.g. ``"upward_then_fold"``).
                    (Also accepted as ``level_shift`` for compatibility.)
        grammar_id: The grammar in which the loop was detected.
    """

    def __init__(
        self,
        cycle: List[str] | None = None,
        entry: Any = None,
        level_delta: Any = 0,
        id: str | None = None,
        grammar_id: str = "",
        # ---- aliases accepted by domain grammars ----
        entry_rule: Any = None,
        level_shift: Any = None,
    ) -> None:
        self.cycle: List[str] = cycle or []
        self.entry: Any = entry if entry is not None else entry_rule
        self.level_delta: Any = level_shift if level_shift is not None else level_delta
        self.id: str = id or uuid.uuid4().hex[:12]
        self.grammar_id: str = grammar_id

    # Alias so angel.py can access ``loop.entry_rule``
    @property
    def entry_rule(self) -> Any:
        return self.entry

    @property
    def level_shift(self) -> Any:
        return self.level_delta

    @property
    def length(self) -> int:
        """Number of steps in the cycle."""
        return len(self.cycle)

    @property
    def is_ascending(self) -> bool:
        """True if the loop moves up in abstraction."""
        return self.level_delta > 0

    @property
    def is_descending(self) -> bool:
        """True if the loop moves down in abstraction."""
        return self.level_delta < 0

    @property
    def is_flat(self) -> bool:
        """True if the loop stays at the same level (simple recursion)."""
        return self.level_delta == 0

    def __hash__(self) -> int:
        return hash(self.id)


# ---------------------------------------------------------------------------
# Grammar — a named collection of rules
# ---------------------------------------------------------------------------

@dataclass
class Grammar:
    """A grammar: a named, composable collection of rules and productions.

    Grammars form a **tangled hierarchy** — they can contain sub-grammars,
    derive from parent grammars, and reference sibling grammars.  This
    mirrors Hofstadter's insight that formal systems powerful enough to
    talk about themselves inevitably contain self-referential tangles.

    Composition follows the **fugue** metaphor: two grammars can run in
    parallel over the same input.  Where their derivations agree we find
    harmony; where they diverge we find counterpoint.

    Attributes:
        name:           Human-readable name.
        domain:         The domain this grammar describes (e.g. "english",
                        "organic_chemistry", "python3").
        id:             Unique identifier.
        rules:          The atomic rules.
        productions:    The formal productions.
        sub_grammars:   Child grammars (tangled hierarchy).
        parent_ids:     IDs of grammars this grammar derives from.
        metadata:       Arbitrary extra data.
    """

    name: str
    domain: str = ""
    id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])
    rules: List[Rule] = field(default_factory=list)
    productions: List[Production] = field(default_factory=list)
    sub_grammars: List["Grammar"] = field(default_factory=list)
    parent_ids: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    strange_loops: List["StrangeLoop"] = field(default_factory=list)

    def __post_init__(self) -> None:
        # Domain grammars sometimes put StrangeLoop objects in sub_grammars.
        # Separate them out into the strange_loops list.
        clean_subs: List["Grammar"] = []
        for item in self.sub_grammars:
            if isinstance(item, StrangeLoop):
                self.strange_loops.append(item)
            else:
                clean_subs.append(item)
        self.sub_grammars = clean_subs

    # -- rule management ----------------------------------------------------

    def add_rule(self, rule: Rule) -> None:
        """Add a rule to this grammar."""
        self.rules.append(rule)

    def add_production(self, production: Production) -> None:
        """Add a production to this grammar."""
        self.productions.append(production)

    def add_sub_grammar(self, grammar: "Grammar") -> None:
        """Nest *grammar* inside this one (tangled hierarchy)."""
        grammar.parent_ids.append(self.id)
        self.sub_grammars.append(grammar)

    # -- collection helpers -------------------------------------------------

    def all_rules(self) -> List[Rule]:
        """Return all rules including those from sub-grammars (depth-first)."""
        collected: List[Rule] = list(self.rules)
        for sg in self.sub_grammars:
            collected.extend(sg.all_rules())
        return collected

    def all_productions(self) -> List[Production]:
        """Return all productions including sub-grammars."""
        collected: List[Production] = list(self.productions)
        for sg in self.sub_grammars:
            collected.extend(sg.all_productions())
        return collected

    def rule_index(self) -> Dict[str, Rule]:
        """Map rule-id → Rule for all rules in this grammar tree."""
        idx: Dict[str, Rule] = {}
        for r in self.all_rules():
            idx[r.id] = r
        return idx

    def production_index(self) -> Dict[str, Production]:
        """Map production-id → Production for all productions."""
        idx: Dict[str, Production] = {}
        for p in self.all_productions():
            idx[p.id] = p
        return idx

    # -- application --------------------------------------------------------

    def apply_forward(self, form: Any, *, max_steps: int = 100) -> List[Tuple[Any, str]]:
        """Apply grammar rules forward (predict/generate).

        Returns a list of ``(derived_form, rule_id)`` pairs for every
        rule that fires on *form*, cascading up to *max_steps* times.
        """
        results: List[Tuple[Any, str]] = []
        frontier: List[Any] = [form]
        seen: Set[int] = set()
        steps = 0

        while frontier and steps < max_steps:
            current = frontier.pop(0)
            form_key = id(current) if not isinstance(current, (str, int, float, tuple)) else hash(current)
            if form_key in seen:
                continue
            seen.add(form_key)

            for rule in self.all_rules():
                output = rule.apply(current, Direction.FORWARD)
                if output is not None:
                    results.append((output, rule.id))
                    frontier.append(output)
                    steps += 1
                    if steps >= max_steps:
                        break

            if steps >= max_steps:
                break

            for prod in self.all_productions():
                output = prod.apply_forward(current)
                if output is not None:
                    results.append((output, prod.id))
                    frontier.append(output)
                    steps += 1
                    if steps >= max_steps:
                        break

        return results

    def apply_backward(self, form: Any, *, max_steps: int = 100) -> List[Tuple[Any, str]]:
        """Apply grammar rules backward (reconstruct/parse).

        Returns ``(reconstructed_form, rule_id)`` pairs.
        """
        results: List[Tuple[Any, str]] = []
        frontier: List[Any] = [form]
        seen: Set[int] = set()
        steps = 0

        while frontier and steps < max_steps:
            current = frontier.pop(0)
            form_key = id(current) if not isinstance(current, (str, int, float, tuple)) else hash(current)
            if form_key in seen:
                continue
            seen.add(form_key)

            for rule in self.all_rules():
                output = rule.apply(current, Direction.BACKWARD)
                if output is not None:
                    results.append((output, rule.id))
                    frontier.append(output)
                    steps += 1
                    if steps >= max_steps:
                        break

            if steps >= max_steps:
                break

            for prod in self.all_productions():
                output = prod.apply_backward(current)
                if output is not None:
                    results.append((output, prod.id))
                    frontier.append(output)
                    steps += 1
                    if steps >= max_steps:
                        break

        return results

    # -- loop detection -----------------------------------------------------

    def find_loops(self) -> List[StrangeLoop]:
        """Detect self-referential cycles (strange loops) in this grammar.

        Strategy: build a directed graph of rule references and production
        chains, then find all cycles using depth-first search.

        A strange loop exists when a chain of rule applications maps a
        form back onto itself — possibly at a different level.  We detect
        two kinds:

        1. **Explicit reference loops** — rules whose ``references`` field
           forms a cycle.
        2. **Derivation loops** — productions whose LHS appears (directly
           or transitively) in their own RHS expansion.
        """
        loops: List[StrangeLoop] = []

        # --- 1. Explicit reference graph -----------------------------------
        ref_graph: Dict[str, List[str]] = {}
        rule_idx = self.rule_index()
        for r in self.all_rules():
            valid_refs = [ref for ref in r.references if ref in rule_idx]
            ref_graph[r.id] = valid_refs

        for cycle in _find_cycles(ref_graph):
            entry_rule = rule_idx.get(cycle[0])
            entry = entry_rule.pattern if entry_rule else cycle[0]
            loops.append(StrangeLoop(
                cycle=cycle,
                entry=entry,
                level_delta=len(cycle) - 1,  # heuristic: longer loops traverse more levels
                grammar_id=self.id,
            ))

        # --- 2. Derivation loops (LHS reachable from RHS) -----------------
        prod_graph: Dict[str, List[str]] = {}
        prod_idx = self.production_index()
        for p in self.all_productions():
            targets: List[str] = []
            for q in self.all_productions():
                if p.id == q.id:
                    continue
                # Does q's LHS appear inside p's RHS?
                if _form_contains(p.rhs, q.lhs):
                    targets.append(q.id)
            # Self-loop: does p's LHS appear in its own RHS?
            if _form_contains(p.rhs, p.lhs):
                targets.append(p.id)
            prod_graph[p.id] = targets

        for cycle in _find_cycles(prod_graph):
            entry_prod = prod_idx.get(cycle[0])
            entry = entry_prod.lhs if entry_prod else cycle[0]
            loops.append(StrangeLoop(
                cycle=cycle,
                entry=entry,
                level_delta=0 if len(cycle) == 1 else len(cycle) - 1,
                grammar_id=self.id,
            ))

        return loops

    # -- composition (fugue) ------------------------------------------------

    def compose(self, other: "Grammar", *, name: Optional[str] = None) -> "Grammar":
        """Compose two grammars like voices in a fugue.

        The resulting grammar contains both sets of rules.  Rules from
        each voice retain their identity so that derivations can later be
        attributed to a specific voice.

        The composed grammar's sub-grammars are the two originals,
        preserving the tangled hierarchy.
        """
        composed = Grammar(
            name=name or f"{self.name}+{other.name}",
            domain=f"{self.domain},{other.domain}" if self.domain != other.domain else self.domain,
        )
        composed.add_sub_grammar(self)
        composed.add_sub_grammar(other)
        return composed

    # -- introspection ------------------------------------------------------

    def symbols(self) -> Set[Any]:
        """Return the set of all symbols appearing in this grammar.

        Symbols are extracted from rule patterns/results and production
        LHS/RHS.  Only hashable forms are included.
        """
        syms: Set[Any] = set()
        for r in self.all_rules():
            for side in (r.pattern, r.result):
                _collect_hashable(side, syms)
        for p in self.all_productions():
            _collect_hashable(p.lhs, syms)
            if isinstance(p.rhs, list):
                for item in p.rhs:
                    _collect_hashable(item, syms)
            else:
                _collect_hashable(p.rhs, syms)
        return syms

    def __contains__(self, rule_or_id: Any) -> bool:
        if isinstance(rule_or_id, str):
            return rule_or_id in self.rule_index() or rule_or_id in self.production_index()
        if isinstance(rule_or_id, Rule):
            return rule_or_id in self.all_rules()
        if isinstance(rule_or_id, Production):
            return rule_or_id in self.all_productions()
        return False

    def __repr__(self) -> str:
        return (
            f"Grammar(name={self.name!r}, domain={self.domain!r}, "
            f"rules={len(self.rules)}, productions={len(self.productions)}, "
            f"sub_grammars={len(self.sub_grammars)})"
        )


# ---------------------------------------------------------------------------
# Module-level helpers
# ---------------------------------------------------------------------------

def _find_cycles(graph: Dict[str, List[str]]) -> List[List[str]]:
    """Find all simple cycles in a directed graph (Johnson's-style DFS).

    Returns each cycle as a list of node IDs in traversal order.
    """
    cycles: List[List[str]] = []
    visited: Set[str] = set()

    def _dfs(node: str, path: List[str], on_stack: Set[str]) -> None:
        visited.add(node)
        on_stack.add(node)
        path.append(node)

        for neighbour in graph.get(node, []):
            if neighbour == path[0] and len(path) > 0:
                # Found a cycle back to the starting node.
                cycles.append(list(path))
            elif neighbour not in on_stack and neighbour not in visited:
                _dfs(neighbour, path, on_stack)

        path.pop()
        on_stack.discard(node)

    for start in graph:
        visited.clear()
        _dfs(start, [], set())

    # Deduplicate — two rotations of the same cycle are identical.
    unique: List[List[str]] = []
    seen_sets: List[FrozenSet[str]] = []
    for c in cycles:
        fs = frozenset(c)
        if fs not in seen_sets:
            seen_sets.append(fs)
            unique.append(c)
    return unique


def _form_contains(haystack: Any, needle: Any) -> bool:
    """Return True if *needle* appears somewhere inside *haystack*.

    Works for strings (substring), sequences (element), and equality.
    """
    if haystack is None or needle is None:
        return False
    if haystack == needle:
        return True
    if isinstance(haystack, str) and isinstance(needle, str):
        return needle in haystack
    if isinstance(haystack, (list, tuple)):
        return any(_form_contains(item, needle) for item in haystack)
    return False


def _collect_hashable(form: Any, into: Set[Any]) -> None:
    """Add *form* to *into* if it is hashable; recurse into sequences."""
    if form is None or callable(form):
        return
    if isinstance(form, (list, tuple)):
        for item in form:
            _collect_hashable(item, into)
        return
    try:
        hash(form)
        into.add(form)
    except TypeError:
        pass
