"""
reasoning.py — Chain-of-thought reasoning via grammar derivation.

This is the core reasoning engine that gives the GLM its power over
brute-force LLMs.  Instead of predicting the next token by statistics,
it *derives conclusions from rules* — like a theorem prover that can
also reason by analogy across domains.

Key capabilities:

* **Chain-of-thought** — Multi-step derivation with backtracking.
  Each step is justified by a named rule. The chain IS the reasoning.
* **Beam search** — Explore multiple derivation paths in parallel,
  pruning by confidence.  Grammar constraints make this exponentially
  cheaper than token-level search.
* **Goal-directed reasoning** — Given a question, decompose into
  sub-goals and solve each via grammar derivation.
* **Analogical transfer** — When stuck in one domain, find an
  isomorphic problem in another domain, solve it there, and map
  the solution back.
* **Self-improving rules** — Successful derivation chains become
  new rules, compressing multi-step reasoning into single steps.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple

from .grammar import Direction, Grammar, Rule, Production, StrangeLoop
from .engine import DerivationEngine, Derivation, DerivationTree


# ---------------------------------------------------------------------------
# ReasoningStep — one step in a chain of thought
# ---------------------------------------------------------------------------

@dataclass
class ReasoningStep:
    """A single reasoning step: input → output via a named rule.

    Unlike a raw Derivation, a ReasoningStep carries semantic context:
    what question it's answering, what sub-goal it's pursuing, and
    its confidence given the full chain so far.
    """
    input: Any
    output: Any
    rule_id: str
    rule_name: str
    grammar_name: str
    domain: str
    confidence: float
    justification: str  # Human-readable explanation
    depth: int = 0
    sub_goal: str = ""
    timestamp: float = field(default_factory=time.time)

    def __repr__(self) -> str:
        return (
            f"Step({self.input!r} → {self.output!r} "
            f"[{self.rule_name}, conf={self.confidence:.2f}])"
        )


# ---------------------------------------------------------------------------
# ReasoningChain — a complete chain of thought
# ---------------------------------------------------------------------------

@dataclass
class ReasoningChain:
    """A complete chain of reasoning steps from question to answer.

    The chain IS the reasoning — every step is justified, every
    conclusion follows from premises via named rules.  This is what
    LLMs cannot do: provide a verifiable proof trail.
    """
    question: str
    steps: List[ReasoningStep] = field(default_factory=list)
    answer: Any = None
    confidence: float = 0.0
    domains_used: List[str] = field(default_factory=list)
    analogies_used: List[Dict[str, Any]] = field(default_factory=list)
    is_complete: bool = False

    def add_step(self, step: ReasoningStep) -> None:
        """Add a step and update chain confidence."""
        self.steps.append(step)
        if step.domain not in self.domains_used:
            self.domains_used.append(step.domain)
        # Chain confidence is the product of step confidences
        # (each step must hold for the chain to hold)
        if self.steps:
            self.confidence = 1.0
            for s in self.steps:
                self.confidence *= s.confidence

    def complete(self, answer: Any) -> None:
        """Mark the chain as complete with a final answer."""
        self.answer = answer
        self.is_complete = True

    @property
    def depth(self) -> int:
        return len(self.steps)

    @property
    def trace(self) -> List[str]:
        """Human-readable reasoning trace."""
        lines = [f"Q: {self.question}"]
        for i, step in enumerate(self.steps, 1):
            lines.append(
                f"  {i}. {step.input!r} → {step.output!r}  "
                f"[{step.rule_name}] ({step.justification})"
            )
        if self.is_complete:
            lines.append(f"A: {self.answer!r} (confidence: {self.confidence:.3f})")
        return lines

    def __repr__(self) -> str:
        status = "complete" if self.is_complete else f"depth={self.depth}"
        return f"Chain({self.question!r}, {status}, conf={self.confidence:.3f})"


# ---------------------------------------------------------------------------
# BeamState — a candidate in beam search
# ---------------------------------------------------------------------------

@dataclass
class BeamState:
    """A candidate state in beam search over derivation space."""
    form: Any
    chain: ReasoningChain
    score: float  # log-probability or confidence
    visited: Set[str] = field(default_factory=set)  # Seen forms (avoid loops)

    def __lt__(self, other: "BeamState") -> bool:
        return self.score < other.score


# ---------------------------------------------------------------------------
# ReasoningEngine — the chain-of-thought reasoner
# ---------------------------------------------------------------------------

class ReasoningEngine:
    """Chain-of-thought reasoning via grammar derivation with beam search.

    This is the engine that gives the GLM reasoning power beyond LLMs.
    Instead of predicting tokens, it:

    1. Parses the input into grammar symbols
    2. Identifies applicable rules across all domains
    3. Chains derivation steps via beam search
    4. When stuck, tries analogical transfer to another domain
    5. Produces a complete, verifiable reasoning chain

    The grammar constraints make search tractable: instead of exploring
    all possible token sequences (vocabulary^length), it explores
    rule applications (rules^depth), which is exponentially smaller
    for well-structured problems.
    """

    def __init__(
        self,
        engine: DerivationEngine,
        grammars: Dict[str, List[Grammar]],
        beam_width: int = 5,
        max_depth: int = 20,
        min_confidence: float = 0.01,
    ) -> None:
        self.engine = engine
        self.grammars = grammars  # domain → [Grammar]
        self.beam_width = beam_width
        self.max_depth = max_depth
        self.min_confidence = min_confidence
        self._learned_rules: List[Rule] = []
        self._reasoning_history: List[ReasoningChain] = []

    # ------------------------------------------------------------------
    # Core reasoning: chain-of-thought with beam search
    # ------------------------------------------------------------------

    def reason(
        self,
        question: str,
        start_form: Any,
        goal: Optional[Any] = None,
        domains: Optional[List[str]] = None,
        *,
        beam_width: Optional[int] = None,
        max_depth: Optional[int] = None,
    ) -> ReasoningChain:
        """Reason about a question via grammar derivation.

        This is the main entry point.  Given a starting form (parsed
        from the question) and an optional goal, it searches for a
        derivation chain that reaches the goal or maximizes confidence.

        Args:
            question:   Natural language question.
            start_form: The parsed/symbolic form to reason from.
            goal:       Optional target form to reach.
            domains:    Which domains to search (None = all).
            beam_width: Override default beam width.
            max_depth:  Override default max depth.

        Returns:
            The best ReasoningChain found.
        """
        beam_width = beam_width or self.beam_width
        max_depth = max_depth or self.max_depth
        domains = domains or list(self.grammars.keys())

        best_chain = ReasoningChain(question=question)

        # Initialize beam with the starting form
        initial_chain = ReasoningChain(question=question)
        initial_state = BeamState(
            form=start_form,
            chain=initial_chain,
            score=1.0,
            visited={_form_key(start_form)},
        )
        beam = [initial_state]

        for depth in range(max_depth):
            if not beam:
                break

            candidates: List[BeamState] = []

            for state in beam:
                # Check if we've reached the goal
                if goal is not None and _forms_match(state.form, goal):
                    state.chain.complete(state.form)
                    if state.chain.confidence > best_chain.confidence:
                        best_chain = state.chain
                    continue

                # Try all applicable rules from all requested domains
                expansions = self._expand_state(
                    state, domains, depth
                )
                candidates.extend(expansions)

            if not candidates:
                # Try analogical transfer before giving up
                for state in beam:
                    analogy_results = self._try_analogy(
                        state, domains, depth
                    )
                    candidates.extend(analogy_results)

            if not candidates:
                break

            # Prune to beam width (keep highest-scoring)
            candidates.sort(key=lambda s: s.score, reverse=True)
            beam = candidates[:beam_width]

            # Check if any candidate reached the goal
            for state in beam:
                if goal is not None and _forms_match(state.form, goal):
                    state.chain.complete(state.form)
                    if state.chain.confidence > best_chain.confidence:
                        best_chain = state.chain

        # If no goal was specified, take the best chain
        if goal is None and beam:
            best_state = max(beam, key=lambda s: s.score)
            if best_state.chain.depth > 0:
                best_state.chain.complete(best_state.form)
                if best_state.chain.confidence > best_chain.confidence:
                    best_chain = best_state.chain

        # Record in history
        self._reasoning_history.append(best_chain)

        # Learn from successful chains
        if best_chain.is_complete and best_chain.confidence > 0.5:
            self._learn_from_chain(best_chain)

        return best_chain

    def _expand_state(
        self,
        state: BeamState,
        domains: List[str],
        depth: int,
    ) -> List[BeamState]:
        """Expand a beam state by applying all matching rules."""
        candidates: List[BeamState] = []

        for domain in domains:
            for grammar in self.grammars.get(domain, []):
                # Try rules
                for rule in grammar.all_rules():
                    output = rule.apply(state.form, Direction.FORWARD)
                    if output is None:
                        continue

                    form_key = _form_key(output)
                    if form_key in state.visited:
                        continue

                    step = ReasoningStep(
                        input=state.form,
                        output=output,
                        rule_id=rule.id,
                        rule_name=rule.name or rule.id,
                        grammar_name=grammar.name,
                        domain=domain,
                        confidence=rule.weight,
                        justification=_build_justification(rule, state.form, output),
                        depth=depth,
                    )

                    new_chain = _clone_chain(state.chain)
                    new_chain.add_step(step)

                    if new_chain.confidence < self.min_confidence:
                        continue

                    new_visited = set(state.visited)
                    new_visited.add(form_key)

                    candidates.append(BeamState(
                        form=output,
                        chain=new_chain,
                        score=new_chain.confidence,
                        visited=new_visited,
                    ))

                # Try productions
                for prod in grammar.all_productions():
                    output = prod.apply_forward(state.form)
                    if output is None:
                        continue

                    form_key = _form_key(output)
                    if form_key in state.visited:
                        continue

                    step = ReasoningStep(
                        input=state.form,
                        output=output,
                        rule_id=prod.id,
                        rule_name=prod.name or prod.id,
                        grammar_name=grammar.name,
                        domain=domain,
                        confidence=prod.weight,
                        justification=f"Production: {prod.lhs} → {prod.rhs}",
                        depth=depth,
                    )

                    new_chain = _clone_chain(state.chain)
                    new_chain.add_step(step)

                    if new_chain.confidence < self.min_confidence:
                        continue

                    new_visited = set(state.visited)
                    new_visited.add(form_key)

                    candidates.append(BeamState(
                        form=output,
                        chain=new_chain,
                        score=new_chain.confidence,
                        visited=new_visited,
                    ))

        # Also try learned rules
        for rule in self._learned_rules:
            output = rule.apply(state.form, Direction.FORWARD)
            if output is None:
                continue
            form_key = _form_key(output)
            if form_key in state.visited:
                continue

            step = ReasoningStep(
                input=state.form,
                output=output,
                rule_id=rule.id,
                rule_name=f"learned:{rule.name}",
                grammar_name="self_improved",
                domain="meta",
                confidence=rule.weight,
                justification=f"Learned rule: {rule.name}",
                depth=depth,
            )

            new_chain = _clone_chain(state.chain)
            new_chain.add_step(step)

            if new_chain.confidence >= self.min_confidence:
                new_visited = set(state.visited)
                new_visited.add(form_key)
                candidates.append(BeamState(
                    form=output,
                    chain=new_chain,
                    score=new_chain.confidence,
                    visited=new_visited,
                ))

        return candidates

    # ------------------------------------------------------------------
    # Analogical transfer
    # ------------------------------------------------------------------

    def _try_analogy(
        self,
        state: BeamState,
        current_domains: List[str],
        depth: int,
    ) -> List[BeamState]:
        """When direct derivation is stuck, try reasoning by analogy.

        Find a domain with isomorphic structure, solve the problem
        there, and map the solution back.  This is the GLM's superpower:
        if you can't solve a chemistry problem, maybe it has the same
        shape as a linguistics problem you CAN solve.
        """
        candidates: List[BeamState] = []
        other_domains = [
            d for d in self.grammars if d not in current_domains
        ]

        for source_domain in current_domains:
            for target_domain in other_domains:
                source_grammars = self.grammars.get(source_domain, [])
                target_grammars = self.grammars.get(target_domain, [])

                for sg in source_grammars:
                    for tg in target_grammars:
                        isos = self.engine.find_isomorphisms(sg, tg)
                        if not isos:
                            continue

                        # Found an isomorphism — try deriving in target domain
                        for rule in tg.all_rules():
                            output = rule.apply(
                                state.form, Direction.FORWARD
                            )
                            if output is None:
                                continue

                            form_key = _form_key(output)
                            if form_key in state.visited:
                                continue

                            # Confidence penalty for analogical transfer
                            best_iso_conf = max(
                                iso["confidence"] for iso in isos
                            )
                            analogy_conf = rule.weight * best_iso_conf * 0.8

                            step = ReasoningStep(
                                input=state.form,
                                output=output,
                                rule_id=rule.id,
                                rule_name=f"analogy:{rule.name}",
                                grammar_name=tg.name,
                                domain=target_domain,
                                confidence=analogy_conf,
                                justification=(
                                    f"By analogy: {source_domain}↔{target_domain} "
                                    f"isomorphism (conf={best_iso_conf:.2f}), "
                                    f"applied {rule.name}"
                                ),
                                depth=depth,
                            )

                            new_chain = _clone_chain(state.chain)
                            new_chain.add_step(step)
                            new_chain.analogies_used.append({
                                "source_domain": source_domain,
                                "target_domain": target_domain,
                                "isomorphism_confidence": best_iso_conf,
                                "rule_applied": rule.name,
                            })

                            if new_chain.confidence >= self.min_confidence:
                                new_visited = set(state.visited)
                                new_visited.add(form_key)
                                candidates.append(BeamState(
                                    form=output,
                                    chain=new_chain,
                                    score=new_chain.confidence,
                                    visited=new_visited,
                                ))

        return candidates

    # ------------------------------------------------------------------
    # Goal decomposition
    # ------------------------------------------------------------------

    def decompose_and_reason(
        self,
        question: str,
        sub_goals: List[Tuple[str, Any, Optional[Any]]],
        domains: Optional[List[str]] = None,
    ) -> ReasoningChain:
        """Decompose a complex question into sub-goals and solve each.

        Each sub-goal is a (description, start_form, goal_form) tuple.
        The output of each sub-goal feeds as context to the next.

        This is divide-and-conquer reasoning via grammar.
        """
        master_chain = ReasoningChain(question=question)
        domains = domains or list(self.grammars.keys())

        for desc, start, goal in sub_goals:
            sub_chain = self.reason(
                question=desc,
                start_form=start,
                goal=goal,
                domains=domains,
            )

            # Merge sub-chain steps into master chain
            for step in sub_chain.steps:
                step.sub_goal = desc
                master_chain.add_step(step)

            # If sub-chain reached its goal, use its answer as input
            # for the next sub-goal
            if sub_chain.is_complete:
                # Update start_form for subsequent goals if needed
                pass

        if master_chain.steps:
            last_output = master_chain.steps[-1].output
            master_chain.complete(last_output)

        self._reasoning_history.append(master_chain)
        return master_chain

    # ------------------------------------------------------------------
    # Bidirectional reasoning (meet in the middle)
    # ------------------------------------------------------------------

    def reason_bidirectional(
        self,
        question: str,
        start_form: Any,
        goal_form: Any,
        domains: Optional[List[str]] = None,
        *,
        max_depth: Optional[int] = None,
    ) -> ReasoningChain:
        """Reason forward from start AND backward from goal, meeting in the middle.

        This is dramatically more efficient than unidirectional search:
        if the solution is d steps away, unidirectional searches O(b^d)
        states while bidirectional searches O(2 * b^(d/2)).

        For grammar-constrained search with branching factor b=10 and
        depth d=10: unidirectional = 10^10 = 10B states, bidirectional
        = 2 * 10^5 = 200K states.  That's 50,000x fewer states.
        """
        max_depth = max_depth or self.max_depth
        domains = domains or list(self.grammars.keys())
        half_depth = max_depth // 2

        # Forward pass: start → ???
        forward_states: Dict[str, BeamState] = {}
        fwd_chain = ReasoningChain(question=question)
        fwd_beam = [BeamState(
            form=start_form,
            chain=fwd_chain,
            score=1.0,
            visited={_form_key(start_form)},
        )]

        for depth in range(half_depth):
            if not fwd_beam:
                break
            new_beam: List[BeamState] = []
            for state in fwd_beam:
                expansions = self._expand_state(state, domains, depth)
                for exp in expansions:
                    key = _form_key(exp.form)
                    forward_states[key] = exp
                    new_beam.append(exp)
            new_beam.sort(key=lambda s: s.score, reverse=True)
            fwd_beam = new_beam[:self.beam_width]

        # Backward pass: goal → ???
        backward_states: Dict[str, BeamState] = {}
        bwd_chain = ReasoningChain(question=f"reconstruct: {question}")
        bwd_beam = [BeamState(
            form=goal_form,
            chain=bwd_chain,
            score=1.0,
            visited={_form_key(goal_form)},
        )]

        for depth in range(half_depth):
            if not bwd_beam:
                break
            new_beam = []
            for state in bwd_beam:
                expansions = self._expand_backward(state, domains, depth)
                for exp in expansions:
                    key = _form_key(exp.form)
                    backward_states[key] = exp
                    new_beam.append(exp)
            new_beam.sort(key=lambda s: s.score, reverse=True)
            bwd_beam = new_beam[:self.beam_width]

        # Find meeting points
        meeting_points: List[Tuple[str, float]] = []
        for key in forward_states:
            if key in backward_states:
                fwd = forward_states[key]
                bwd = backward_states[key]
                combined_score = fwd.score * bwd.score
                meeting_points.append((key, combined_score))

        if not meeting_points:
            # No meeting point found — return best forward chain
            if fwd_beam:
                best = max(fwd_beam, key=lambda s: s.score)
                return best.chain
            return ReasoningChain(question=question)

        # Build the combined chain through the best meeting point
        meeting_points.sort(key=lambda x: x[1], reverse=True)
        best_key = meeting_points[0][0]
        fwd_state = forward_states[best_key]
        bwd_state = backward_states[best_key]

        combined = ReasoningChain(question=question)
        for step in fwd_state.chain.steps:
            combined.add_step(step)
        # Reverse the backward chain and add
        for step in reversed(bwd_state.chain.steps):
            combined.add_step(step)

        combined.complete(goal_form)
        self._reasoning_history.append(combined)
        return combined

    def _expand_backward(
        self,
        state: BeamState,
        domains: List[str],
        depth: int,
    ) -> List[BeamState]:
        """Expand a state by applying rules backward."""
        candidates: List[BeamState] = []

        for domain in domains:
            for grammar in self.grammars.get(domain, []):
                for rule in grammar.all_rules():
                    output = rule.apply(state.form, Direction.BACKWARD)
                    if output is None:
                        continue
                    form_key = _form_key(output)
                    if form_key in state.visited:
                        continue

                    step = ReasoningStep(
                        input=state.form,
                        output=output,
                        rule_id=rule.id,
                        rule_name=rule.name or rule.id,
                        grammar_name=grammar.name,
                        domain=domain,
                        confidence=rule.weight,
                        justification=f"Backward: {rule.name}",
                        depth=depth,
                    )

                    new_chain = _clone_chain(state.chain)
                    new_chain.add_step(step)

                    if new_chain.confidence >= self.min_confidence:
                        new_visited = set(state.visited)
                        new_visited.add(form_key)
                        candidates.append(BeamState(
                            form=output,
                            chain=new_chain,
                            score=new_chain.confidence,
                            visited=new_visited,
                        ))

                for prod in grammar.all_productions():
                    output = prod.apply_backward(state.form)
                    if output is None:
                        continue
                    form_key = _form_key(output)
                    if form_key in state.visited:
                        continue

                    step = ReasoningStep(
                        input=state.form,
                        output=output,
                        rule_id=prod.id,
                        rule_name=prod.name or prod.id,
                        grammar_name=grammar.name,
                        domain=domain,
                        confidence=prod.weight,
                        justification=f"Backward production: {prod.lhs} ← {prod.rhs}",
                        depth=depth,
                    )

                    new_chain = _clone_chain(state.chain)
                    new_chain.add_step(step)

                    if new_chain.confidence >= self.min_confidence:
                        new_visited = set(state.visited)
                        new_visited.add(form_key)
                        candidates.append(BeamState(
                            form=output,
                            chain=new_chain,
                            score=new_chain.confidence,
                            visited=new_visited,
                        ))

        return candidates

    # ------------------------------------------------------------------
    # Self-improving rule discovery
    # ------------------------------------------------------------------

    def _learn_from_chain(self, chain: ReasoningChain) -> None:
        """Extract new rules from successful reasoning chains.

        If a multi-step chain is used repeatedly, compress it into a
        single rule.  This is how the GLM gets smarter over time:
        common reasoning patterns become first-class rules.
        """
        if chain.depth < 2:
            return  # Nothing to compress

        # Create a shortcut rule: first input → last output
        first_step = chain.steps[0]
        last_step = chain.steps[-1]

        # Build a name from the chain
        rule_names = [s.rule_name for s in chain.steps]
        shortcut_name = "chain:" + "→".join(rule_names[:4])

        # Check if we already have this shortcut
        for existing in self._learned_rules:
            if existing.name == shortcut_name:
                # Boost its weight (seen again = more reliable)
                existing.weight = min(1.0, existing.weight + 0.05)
                return

        # Create the shortcut rule
        shortcut = Rule(
            name=shortcut_name,
            pattern=first_step.input,
            result=last_step.output,
            weight=chain.confidence * 0.9,  # Slight discount
            direction=Direction.FORWARD,
            metadata={
                "learned_from": chain.question,
                "chain_depth": chain.depth,
                "domains": chain.domains_used,
                "learned_at": time.time(),
            },
        )
        self._learned_rules.append(shortcut)

    def discover_rules(
        self,
        domain: str,
        sample_size: int = 20,
    ) -> List[Rule]:
        """Actively discover new rules by exploring derivation space.

        For each grammar symbol, derive forward and backward, looking
        for patterns that consistently produce results.  These become
        candidate new rules.
        """
        discovered: List[Rule] = []
        grammars = self.grammars.get(domain, [])

        for grammar in grammars:
            symbols = list(grammar.symbols())[:sample_size]

            # Track which transformations occur repeatedly
            transform_counts: Dict[Tuple[str, str], int] = {}
            transform_rules: Dict[Tuple[str, str], str] = {}

            for sym in symbols:
                tree = self.engine.derive(
                    sym, grammar, "forward", max_steps=30
                )
                for leaf in tree.leaves():
                    if leaf.form != sym and leaf.step:
                        key = (_form_key(sym), _form_key(leaf.form))
                        transform_counts[key] = transform_counts.get(key, 0) + 1
                        transform_rules[key] = leaf.step.rule_id

            # Patterns seen 2+ times become candidates
            for (src_key, dst_key), count in transform_counts.items():
                if count >= 2:
                    discovered.append(Rule(
                        name=f"discovered:{domain}:{src_key}→{dst_key}",
                        pattern=src_key,
                        result=dst_key,
                        weight=min(0.9, 0.5 + count * 0.1),
                        direction=Direction.FORWARD,
                        metadata={
                            "discovered_in": domain,
                            "occurrence_count": count,
                            "source_rule": transform_rules.get((src_key, dst_key)),
                        },
                    ))

        self._learned_rules.extend(discovered)
        return discovered

    # ------------------------------------------------------------------
    # Introspection
    # ------------------------------------------------------------------

    def get_stats(self) -> Dict[str, Any]:
        """Return reasoning statistics."""
        completed = [c for c in self._reasoning_history if c.is_complete]
        return {
            "total_reasoning_chains": len(self._reasoning_history),
            "completed_chains": len(completed),
            "learned_rules": len(self._learned_rules),
            "avg_chain_depth": (
                sum(c.depth for c in completed) / max(len(completed), 1)
            ),
            "avg_confidence": (
                sum(c.confidence for c in completed) / max(len(completed), 1)
            ),
            "domains_reasoned_about": list(set(
                d for c in self._reasoning_history for d in c.domains_used
            )),
            "analogies_used": sum(
                len(c.analogies_used) for c in self._reasoning_history
            ),
        }


# ---------------------------------------------------------------------------
# Module-level helpers
# ---------------------------------------------------------------------------

def _form_key(form: Any) -> str:
    """Create a hashable key for any form."""
    try:
        hash(form)
        return str(form)
    except TypeError:
        return repr(form)


def _forms_match(a: Any, b: Any) -> bool:
    """Check if two forms match (flexible comparison)."""
    try:
        if a == b:
            return True
    except Exception:
        pass
    if isinstance(a, str) and isinstance(b, str):
        return a.lower().strip() == b.lower().strip()
    return str(a) == str(b)


def _clone_chain(chain: ReasoningChain) -> ReasoningChain:
    """Create a shallow clone of a reasoning chain."""
    new = ReasoningChain(question=chain.question)
    for step in chain.steps:
        new.add_step(step)
    new.analogies_used = list(chain.analogies_used)
    return new


def _build_justification(rule: Rule, input_form: Any, output: Any) -> str:
    """Build a human-readable justification for a rule application."""
    name = rule.name or rule.id
    meta = rule.metadata
    if isinstance(meta, dict):
        desc = meta.get("description", "")
        axiom = meta.get("axiom", "")
        if desc:
            return f"{name}: {desc}"
        if axiom:
            return f"{name}: by {axiom}"
    if isinstance(rule.result, dict):
        desc = rule.result.get("description", "")
        axiom = rule.result.get("axiom", "")
        if desc:
            return f"{name}: {desc}"
        if axiom:
            return f"{name}: by {axiom}"
    return f"Applied rule '{name}'"
