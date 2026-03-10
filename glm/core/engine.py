"""
engine.py — The derivation engine for the Grammar Language Model.

The ``DerivationEngine`` is the core runtime that applies grammars to
inputs, producing derivations forward (prediction) and backward
(reconstruction).  It is the performer who takes the *scales* (grammar)
and the *score* (input) and plays the music.

Key capabilities:

* **derive** — Apply a grammar to an input in either temporal direction.
* **superforecast** — Extrapolate patterns into the future by exploiting
  strange loops: recursive grammatical structures that project forward
  with increasing confidence as more of the pattern is confirmed.
* **reconstruct** — Work backward from an output to recover the
  originating forms, like an archaeologist reassembling a language from
  its modern descendants.
* **detect_loops** — Survey a grammar for strange loops — the self-
  referential cycles that give the grammar its generative power.
* **compose_fugue** — Run multiple grammars in parallel over (possibly
  different) inputs, finding harmonic convergences and contrapuntal
  divergences, exactly like voices in a Bach fugue.
* **find_isomorphisms** — Discover structural mappings between grammars
  from different domains, revealing the deep shared scales.
"""

from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple

from .grammar import Direction, Grammar, Rule, Production, StrangeLoop


# ---------------------------------------------------------------------------
# Derivation — a single step record
# ---------------------------------------------------------------------------

@dataclass
class Derivation:
    """A record of one derivation step.

    Every time a rule or production fires, a ``Derivation`` is created
    to capture what happened, preserving the full audit trail.

    Attributes:
        id:         Unique identifier.
        rule_id:    ID of the rule or production that fired.
        input:      The form before the rule was applied.
        output:     The form after the rule was applied.
        direction:  FORWARD (predict) or BACKWARD (reconstruct).
        timestamp:  Wall-clock time of the derivation.
        metadata:   Arbitrary extra data.
    """

    rule_id: str
    input: Any
    output: Any
    direction: Direction
    id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])
    timestamp: float = field(default_factory=time.time)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __repr__(self) -> str:
        arrow = "→" if self.direction == Direction.FORWARD else "←"
        return f"Derivation({self.input!r} {arrow} {self.output!r} [{self.rule_id}])"


# ---------------------------------------------------------------------------
# DerivationTree — full derivation history
# ---------------------------------------------------------------------------

@dataclass
class DerivationTree:
    """A tree of derivations showing the full generative/reconstructive history.

    The root is the original input.  Each child is a derivation step
    that was applied to the parent node.  Branches occur when multiple
    rules match the same form (ambiguity — a feature, not a bug).

    Attributes:
        form:       The form at this node.
        step:       The derivation step that produced this node (None for root).
        children:   Child derivation trees (further derivations from this form).
        depth:      Depth in the tree (root = 0).
    """

    form: Any
    step: Optional[Derivation] = None
    children: List["DerivationTree"] = field(default_factory=list)
    depth: int = 0

    def add_child(self, child: "DerivationTree") -> None:
        """Attach a child derivation."""
        child.depth = self.depth + 1
        self.children.append(child)

    @property
    def is_leaf(self) -> bool:
        return len(self.children) == 0

    def leaves(self) -> List["DerivationTree"]:
        """Return all leaf nodes (terminal derivations)."""
        if self.is_leaf:
            return [self]
        result: List[DerivationTree] = []
        for child in self.children:
            result.extend(child.leaves())
        return result

    def all_forms(self) -> List[Any]:
        """Collect every form in the tree (breadth-first)."""
        forms: List[Any] = [self.form]
        for child in self.children:
            forms.extend(child.all_forms())
        return forms

    def paths(self) -> List[List[Derivation]]:
        """Return all root-to-leaf derivation paths."""
        if self.is_leaf:
            return [[self.step]] if self.step else [[]]
        result: List[List[Derivation]] = []
        prefix = [self.step] if self.step else []
        for child in self.children:
            for path in child.paths():
                result.append(prefix + path)
        return result

    @property
    def height(self) -> int:
        """Maximum depth from this node to any leaf."""
        if self.is_leaf:
            return 0
        return 1 + max(child.height for child in self.children)

    def __repr__(self) -> str:
        return (
            f"DerivationTree(form={self.form!r}, "
            f"children={len(self.children)}, height={self.height})"
        )


# ---------------------------------------------------------------------------
# DerivationEngine — the core runtime
# ---------------------------------------------------------------------------

class DerivationEngine:
    """The engine that applies grammars to substrates.

    This is the performer.  It takes grammars (the scales) and inputs
    (the score) and produces derivations (the music).

    The engine maintains a history of all derivations it has performed,
    enabling meta-analysis and pattern discovery across runs.
    """

    def __init__(self) -> None:
        self.history: List[Derivation] = []

    # -- derive -------------------------------------------------------------

    def derive(
        self,
        input: Any,
        grammar: Grammar,
        direction: str = "forward",
        *,
        max_steps: int = 200,
    ) -> DerivationTree:
        """Apply *grammar* to *input*, producing a full derivation tree.

        Parameters:
            input:      The starting form.
            grammar:    The grammar whose rules/productions to apply.
            direction:  ``"forward"`` (predict/generate) or
                        ``"backward"`` (reconstruct/parse).
            max_steps:  Safety limit on total derivation steps.

        Returns:
            A ``DerivationTree`` rooted at *input* with branches for
            every rule that fired.
        """
        dir_enum = Direction.FORWARD if direction == "forward" else Direction.BACKWARD
        root = DerivationTree(form=input)
        self._expand(root, grammar, dir_enum, max_steps, _steps_taken=[0])
        return root

    def _expand(
        self,
        node: DerivationTree,
        grammar: Grammar,
        direction: Direction,
        max_steps: int,
        _steps_taken: List[int],
    ) -> None:
        """Recursively expand a derivation tree node."""
        if _steps_taken[0] >= max_steps:
            return

        form = node.form
        fired: List[Tuple[Any, str]] = []

        # Try all rules.
        for rule in grammar.all_rules():
            output = rule.apply(form, direction)
            if output is not None:
                fired.append((output, rule.id))

        # Try all productions.
        for prod in grammar.all_productions():
            if direction == Direction.FORWARD:
                output = prod.apply_forward(form)
            else:
                output = prod.apply_backward(form)
            if output is not None:
                fired.append((output, prod.id))

        for output, rule_id in fired:
            if _steps_taken[0] >= max_steps:
                break

            step = Derivation(
                rule_id=rule_id,
                input=form,
                output=output,
                direction=direction,
            )
            self.history.append(step)
            _steps_taken[0] += 1

            child = DerivationTree(form=output, step=step)
            node.add_child(child)

            # Recurse — but avoid re-deriving the same form to prevent
            # infinite loops on flat strange loops.  We still *detect*
            # loops; we just don't follow them infinitely.
            if not self._seen_on_path(node, output):
                self._expand(child, grammar, direction, max_steps, _steps_taken)

    @staticmethod
    def _seen_on_path(node: DerivationTree, form: Any) -> bool:
        """Walk up from *node* to root checking if *form* already appeared."""
        current: Optional[DerivationTree] = node
        while current is not None:
            try:
                if current.form == form:
                    return True
            except Exception:
                if current.form is form:
                    return True
            # Walk up via step's input — but DerivationTree doesn't store
            # parent, so we check the form at each level via depth.
            # Simple approach: only check the immediate node.
            break
        return False

    # -- superforecast ------------------------------------------------------

    def superforecast(
        self,
        input: Any,
        grammar: Grammar,
        context: Optional[List[Any]] = None,
        horizon: int = 5,
    ) -> List[Dict[str, Any]]:
        """Extrapolate patterns into the future using grammatical structure.

        The superforecast combines:
        1. Direct forward derivation from the input.
        2. Strange-loop detection — recursive patterns that, once
           identified, can be projected forward with confidence.
        3. Context integration — if prior forms are provided, they are
           used to weight which derivation paths are most likely.

        Parameters:
            input:    The current form to forecast from.
            grammar:  The grammar encoding the structural rules.
            context:  Optional list of prior forms (temporal context).
            horizon:  How many steps into the future to project.

        Returns:
            A list of forecast dicts, each with:
              - ``step``: how many steps ahead
              - ``form``: the predicted form
              - ``confidence``: estimated probability [0, 1]
              - ``loop``: the strange loop powering the prediction, if any
              - ``path``: the derivation path that produced it
        """
        context = context or []
        forecasts: List[Dict[str, Any]] = []

        # --- Phase 1: direct forward derivation ----------------------------
        tree = self.derive(input, grammar, "forward", max_steps=horizon * 10)
        leaves = tree.leaves()

        # Score each leaf by how much context it matches.
        for leaf in leaves:
            confidence = self._context_score(leaf, context, grammar)
            forecasts.append({
                "step": leaf.depth,
                "form": leaf.form,
                "confidence": confidence,
                "loop": None,
                "path": [s for s in self._path_to(leaf, tree)],
            })

        # --- Phase 2: strange-loop projection ------------------------------
        loops = self.detect_loops(grammar)
        for loop in loops:
            # A loop implies periodicity.  If the input matches the loop's
            # entry point, we can project the cycle forward.
            if self._matches_loop_entry(input, loop, grammar):
                cycle_rules = self._resolve_cycle_rules(loop, grammar)
                if not cycle_rules:
                    continue

                current = input
                for step_num in range(1, horizon + 1):
                    # Apply each rule in the cycle sequentially.
                    rule = cycle_rules[(step_num - 1) % len(cycle_rules)]
                    output = rule.apply(current, Direction.FORWARD)
                    if output is None:
                        break
                    # Loop-based predictions get a confidence boost that
                    # increases with each confirmed cycle.
                    base_conf = rule.weight
                    cycle_bonus = min(0.3, 0.1 * (step_num // loop.length))
                    confidence = min(1.0, base_conf + cycle_bonus)

                    forecasts.append({
                        "step": step_num,
                        "form": output,
                        "confidence": confidence,
                        "loop": loop,
                        "path": [rule.id],
                    })
                    current = output

        # Sort by step, then descending confidence.
        forecasts.sort(key=lambda f: (f["step"], -f["confidence"]))

        # Trim to horizon.
        forecasts = [f for f in forecasts if f["step"] <= horizon]

        return forecasts

    # -- reconstruct --------------------------------------------------------

    def reconstruct(
        self,
        output: Any,
        grammar: Grammar,
        *,
        max_steps: int = 200,
    ) -> DerivationTree:
        """Given an output, work backward to find originating forms.

        This is the temporal inverse of ``derive(..., direction="forward")``.
        """
        return self.derive(output, grammar, "backward", max_steps=max_steps)

    # -- loop detection -----------------------------------------------------

    def detect_loops(self, grammar: Grammar) -> List[StrangeLoop]:
        """Find all strange loops in *grammar*.

        Delegates to ``Grammar.find_loops()`` but also performs a runtime
        derivation-based check: if deriving forward from any symbol
        eventually produces that symbol again, it is a loop even if the
        static graph analysis missed it.
        """
        static_loops = grammar.find_loops()
        seen_entries = {
            (frozenset(l.cycle), self._hashable(l.entry))
            for l in static_loops
        }

        # Runtime derivation check on a sample of symbols.
        symbols = grammar.symbols()
        for sym in list(symbols)[:50]:  # cap to avoid runaway
            tree = self.derive(sym, grammar, "forward", max_steps=30)
            for leaf in tree.leaves():
                try:
                    if leaf.form == sym and leaf.depth > 0:
                        # Collect rule IDs along the path.
                        path_rules = self._path_to(leaf, tree)
                        cycle = [d.rule_id for d in path_rules if d is not None]
                        key = (frozenset(cycle), self._hashable(sym))
                        if key not in seen_entries and cycle:
                            seen_entries.add(key)
                            static_loops.append(StrangeLoop(
                                cycle=cycle,
                                entry=sym,
                                level_delta=0,
                                grammar_id=grammar.id,
                            ))
                except Exception:
                    continue

        return static_loops

    # -- fugue composition --------------------------------------------------

    def compose_fugue(
        self,
        grammars: List[Grammar],
        inputs: List[Any],
    ) -> Dict[str, Any]:
        """Run multiple grammars in parallel, like voices in a fugue.

        Each grammar is applied to its corresponding input (or to the
        same input if fewer inputs than grammars are provided).  The
        engine then analyses the derivations for:

        * **Harmonies** — forms produced by more than one voice.
        * **Counterpoints** — forms unique to one voice, providing
          complementary information.
        * **Canons** — identical derivation chains offset in time.

        Parameters:
            grammars:  The "voices" — one grammar per voice.
            inputs:    The inputs for each voice (cycled if fewer than voices).

        Returns:
            A dict with keys:
              - ``voices``:       list of per-voice derivation trees
              - ``harmonies``:    forms shared across multiple voices
              - ``counterpoints``: forms unique to each voice
              - ``canons``:       derivation chains that appear in multiple
                                  voices (possibly offset)
        """
        if not grammars:
            return {"voices": [], "harmonies": [], "counterpoints": [], "canons": []}

        # Pair each grammar with an input, cycling inputs if necessary.
        paired_inputs = [
            inputs[i % len(inputs)] if inputs else None
            for i in range(len(grammars))
        ]

        # Derive each voice.
        voice_trees: List[DerivationTree] = []
        voice_forms: List[Set[Any]] = []

        for grammar, inp in zip(grammars, paired_inputs):
            tree = self.derive(inp, grammar, "forward", max_steps=100)
            voice_trees.append(tree)

            forms: Set[Any] = set()
            for f in tree.all_forms():
                try:
                    hash(f)
                    forms.add(f)
                except TypeError:
                    pass
            voice_forms.append(forms)

        # Find harmonies — forms appearing in >= 2 voices.
        all_forms_flat: Dict[Any, int] = {}
        for fs in voice_forms:
            for f in fs:
                all_forms_flat[f] = all_forms_flat.get(f, 0) + 1

        harmonies = [f for f, count in all_forms_flat.items() if count >= 2]

        # Find counterpoints — forms unique to exactly one voice.
        counterpoints: List[Dict[str, Any]] = []
        for idx, fs in enumerate(voice_forms):
            unique = fs - set().union(*(
                voice_forms[j] for j in range(len(voice_forms)) if j != idx
            )) if len(voice_forms) > 1 else fs
            counterpoints.append({
                "voice": idx,
                "grammar": grammars[idx].name,
                "unique_forms": list(unique),
            })

        # Find canons — identical derivation paths in multiple voices.
        canons = self._find_canons(voice_trees, grammars)

        return {
            "voices": voice_trees,
            "harmonies": harmonies,
            "counterpoints": counterpoints,
            "canons": canons,
        }

    # -- isomorphisms -------------------------------------------------------

    def find_isomorphisms(
        self,
        grammar_a: Grammar,
        grammar_b: Grammar,
    ) -> List[Dict[str, Any]]:
        """Find structural mappings between two grammars from different domains.

        An isomorphism exists when two grammars have the same *shape* —
        the same graph of rule dependencies, the same cycle structure,
        the same branching pattern — even though they operate on
        completely different substrates.

        This is the formal incarnation of the GLM's core insight: that
        English morphology and organic-chemistry nomenclature and
        genetic codon tables are, at some level of abstraction, the
        *same grammar* wearing different costumes.

        Returns a list of mappings, each with:
          - ``type``: kind of isomorphism detected
          - ``a_elements``: elements from grammar A
          - ``b_elements``: elements from grammar B
          - ``confidence``: strength of the mapping
        """
        mappings: List[Dict[str, Any]] = []

        # --- 1. Rule-count / production-count similarity -------------------
        a_rules = grammar_a.all_rules()
        b_rules = grammar_b.all_rules()
        a_prods = grammar_a.all_productions()
        b_prods = grammar_b.all_productions()

        # --- 2. Loop structure isomorphism ---------------------------------
        a_loops = grammar_a.find_loops()
        b_loops = grammar_b.find_loops()

        for la in a_loops:
            for lb in b_loops:
                if la.length == lb.length and la.level_delta == lb.level_delta:
                    mappings.append({
                        "type": "loop_isomorphism",
                        "a_elements": la.cycle,
                        "b_elements": lb.cycle,
                        "confidence": _loop_similarity(la, lb),
                    })

        # --- 3. Production-graph shape similarity --------------------------
        a_shape = _production_shape(a_prods)
        b_shape = _production_shape(b_prods)

        if a_shape and b_shape:
            # Compare out-degree distributions.
            shared_degrees = set(a_shape.keys()) & set(b_shape.keys())
            if shared_degrees:
                total_a = sum(a_shape.values())
                total_b = sum(b_shape.values())
                if total_a > 0 and total_b > 0:
                    similarity = sum(
                        min(a_shape[d] / total_a, b_shape[d] / total_b)
                        for d in shared_degrees
                    )
                    if similarity > 0.3:
                        mappings.append({
                            "type": "shape_isomorphism",
                            "a_elements": list(a_shape.keys()),
                            "b_elements": list(b_shape.keys()),
                            "confidence": min(1.0, similarity),
                        })

        # --- 4. Rule weight distribution similarity ------------------------
        a_weights = sorted(r.weight for r in a_rules)
        b_weights = sorted(r.weight for r in b_rules)

        if a_weights and b_weights:
            weight_sim = _distribution_similarity(a_weights, b_weights)
            if weight_sim > 0.5:
                mappings.append({
                    "type": "weight_distribution_isomorphism",
                    "a_elements": a_weights,
                    "b_elements": b_weights,
                    "confidence": weight_sim,
                })

        # --- 5. Direct rule-pattern matching (structural analogy) ----------
        for ra in a_rules:
            for rb in b_rules:
                sim = _rule_structural_similarity(ra, rb)
                if sim > 0.7:
                    mappings.append({
                        "type": "rule_analogy",
                        "a_elements": [ra.id],
                        "b_elements": [rb.id],
                        "confidence": sim,
                    })

        return mappings

    # -- internal helpers ---------------------------------------------------

    def _context_score(
        self,
        leaf: DerivationTree,
        context: List[Any],
        grammar: Grammar,
    ) -> float:
        """Score a derivation leaf based on how well it fits context.

        Context forms that can derive to (or from) the leaf's form
        increase confidence.
        """
        if not context:
            return 0.5  # no context → neutral

        score = 0.0
        for ctx_form in context:
            # Check if any rule connects the context form to the leaf.
            for rule in grammar.all_rules():
                if rule.matches(ctx_form, Direction.FORWARD):
                    output = rule.apply(ctx_form, Direction.FORWARD)
                    try:
                        if output == leaf.form:
                            score += rule.weight
                    except Exception:
                        pass

        # Normalise to [0, 1].
        return min(1.0, max(0.1, score / max(len(context), 1)))

    def _matches_loop_entry(
        self,
        form: Any,
        loop: StrangeLoop,
        grammar: Grammar,
    ) -> bool:
        """Check if *form* matches the entry point of a strange loop."""
        try:
            return form == loop.entry or Rule._match(loop.entry, form)
        except Exception:
            return False

    def _resolve_cycle_rules(
        self,
        loop: StrangeLoop,
        grammar: Grammar,
    ) -> List[Rule]:
        """Resolve the rule IDs in a loop to actual Rule objects."""
        idx = grammar.rule_index()
        rules: List[Rule] = []
        for rid in loop.cycle:
            rule = idx.get(rid)
            if rule is not None:
                rules.append(rule)
        return rules

    def _path_to(
        self,
        target: DerivationTree,
        root: DerivationTree,
    ) -> List[Derivation]:
        """Find the derivation path from *root* to *target*."""
        path: List[Derivation] = []
        if self._dfs_path(root, target, path):
            return path
        return []

    def _dfs_path(
        self,
        node: DerivationTree,
        target: DerivationTree,
        path: List[Derivation],
    ) -> bool:
        """DFS to find path from node to target, collecting derivation steps."""
        if node is target:
            return True
        for child in node.children:
            if child.step:
                path.append(child.step)
            if self._dfs_path(child, target, path):
                return True
            if child.step and path and path[-1] is child.step:
                path.pop()
        return False

    def _find_canons(
        self,
        voice_trees: List[DerivationTree],
        grammars: List[Grammar],
    ) -> List[Dict[str, Any]]:
        """Find canonical patterns — identical derivation chains across voices."""
        canons: List[Dict[str, Any]] = []

        # Extract rule-id sequences from each voice.
        voice_sequences: List[List[List[str]]] = []
        for tree in voice_trees:
            paths = tree.paths()
            sequences: List[List[str]] = []
            for path in paths:
                seq = [d.rule_id for d in path if isinstance(d, Derivation)]
                if seq:
                    sequences.append(seq)
            voice_sequences.append(sequences)

        # Compare all pairs of voices for shared subsequences.
        for i in range(len(voice_sequences)):
            for j in range(i + 1, len(voice_sequences)):
                for seq_a in voice_sequences[i]:
                    for seq_b in voice_sequences[j]:
                        common = _longest_common_subsequence(seq_a, seq_b)
                        if len(common) >= 2:
                            canons.append({
                                "voice_a": i,
                                "voice_b": j,
                                "shared_rules": common,
                                "grammar_a": grammars[i].name,
                                "grammar_b": grammars[j].name,
                            })

        return canons

    @staticmethod
    def _hashable(form: Any) -> Any:
        """Return a hashable version of *form* for set membership checks."""
        try:
            hash(form)
            return form
        except TypeError:
            return str(form)


# ---------------------------------------------------------------------------
# Module-level helpers
# ---------------------------------------------------------------------------

def _loop_similarity(la: StrangeLoop, lb: StrangeLoop) -> float:
    """Compute similarity between two strange loops."""
    if la.length == 0 or lb.length == 0:
        return 0.0
    length_sim = 1.0 - abs(la.length - lb.length) / max(la.length, lb.length)
    delta_sim = 1.0 if la.level_delta == lb.level_delta else 0.5
    return length_sim * delta_sim


def _production_shape(productions: List[Production]) -> Dict[int, int]:
    """Compute the out-degree distribution of a production set.

    For each production, the "out-degree" is the number of other
    productions whose LHS appears in its RHS (a rough graph shape metric).
    """
    shape: Dict[int, int] = {}
    lhs_set = []
    for p in productions:
        try:
            lhs_set.append(p.lhs)
        except Exception:
            pass

    for p in productions:
        degree = 0
        for lhs in lhs_set:
            try:
                rhs = p.rhs
                if isinstance(rhs, list):
                    if any(_contains(alt, lhs) for alt in rhs):
                        degree += 1
                elif _contains(rhs, lhs):
                    degree += 1
            except Exception:
                pass
        shape[degree] = shape.get(degree, 0) + 1

    return shape


def _contains(haystack: Any, needle: Any) -> bool:
    """Check if needle is in haystack."""
    if haystack is None or needle is None:
        return False
    try:
        if haystack == needle:
            return True
    except Exception:
        pass
    if isinstance(haystack, str) and isinstance(needle, str):
        return needle in haystack
    if isinstance(haystack, (list, tuple)):
        return any(_contains(h, needle) for h in haystack)
    return False


def _distribution_similarity(a: List[float], b: List[float]) -> float:
    """Compare two sorted numerical distributions.

    Uses a simple normalized overlap of binned values.
    """
    if not a or not b:
        return 0.0

    # Normalise both to [0, 1] range.
    def normalise(vals: List[float]) -> List[float]:
        lo, hi = min(vals), max(vals)
        span = hi - lo
        if span == 0:
            return [0.5] * len(vals)
        return [(v - lo) / span for v in vals]

    na = normalise(a)
    nb = normalise(b)

    # Resample the shorter list to match the longer.
    if len(na) != len(nb):
        shorter, longer = (na, nb) if len(na) < len(nb) else (nb, na)
        resampled: List[float] = []
        for i in range(len(longer)):
            src_idx = i * (len(shorter) - 1) / max(len(longer) - 1, 1)
            lo_idx = int(src_idx)
            hi_idx = min(lo_idx + 1, len(shorter) - 1)
            frac = src_idx - lo_idx
            resampled.append(shorter[lo_idx] * (1 - frac) + shorter[hi_idx] * frac)
        na, nb = longer, resampled

    # Compute 1 - mean absolute difference.
    diffs = [abs(x - y) for x, y in zip(na, nb)]
    return 1.0 - (sum(diffs) / len(diffs))


def _rule_structural_similarity(ra: Rule, rb: Rule) -> float:
    """Compare two rules for structural similarity.

    Looks at: direction match, weight similarity, whether patterns/results
    are the same type (callable, string, sequence, etc.).
    """
    score = 0.0
    total = 0.0

    # Direction match.
    total += 1.0
    if ra.direction == rb.direction:
        score += 1.0
    elif ra.direction is None or rb.direction is None:
        score += 0.5

    # Weight similarity.
    total += 1.0
    score += 1.0 - abs(ra.weight - rb.weight)

    # Pattern type match.
    total += 1.0
    if type(ra.pattern) == type(rb.pattern):
        score += 1.0
    elif (callable(ra.pattern) and callable(rb.pattern)):
        score += 0.8

    # Result type match.
    total += 1.0
    if type(ra.result) == type(rb.result):
        score += 1.0
    elif (callable(ra.result) and callable(rb.result)):
        score += 0.8

    # Reference count similarity.
    total += 1.0
    if len(ra.references) == len(rb.references):
        score += 1.0
    elif abs(len(ra.references) - len(rb.references)) <= 1:
        score += 0.5

    return score / total if total > 0 else 0.0


def _longest_common_subsequence(a: List[str], b: List[str]) -> List[str]:
    """Find the longest common subsequence of two string lists (DP)."""
    m, n = len(a), len(b)
    if m == 0 or n == 0:
        return []

    # DP table.
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if a[i - 1] == b[j - 1]:
                dp[i][j] = dp[i - 1][j - 1] + 1
            else:
                dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])

    # Backtrack.
    result: List[str] = []
    i, j = m, n
    while i > 0 and j > 0:
        if a[i - 1] == b[j - 1]:
            result.append(a[i - 1])
            i -= 1
            j -= 1
        elif dp[i - 1][j] > dp[i][j - 1]:
            i -= 1
        else:
            j -= 1

    result.reverse()
    return result
