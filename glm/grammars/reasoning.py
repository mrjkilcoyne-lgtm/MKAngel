"""
reasoning.py — Operational reasoning rules that actually fire.

Unlike the domain grammars (which encode knowledge as dict patterns),
these rules work with string-form symbolic expressions that the
derivation engine can chain.  This is what gives the GLM its
reasoning power: composable, chainable inference rules.

The key insight: LLMs predict the next token from statistics.
These rules derive conclusions from premises via named axioms.
Each step is justified.  Each chain is verifiable.

Categories of rules:
  1. Logical inference (modus ponens, syllogisms, etc.)
  2. Mathematical reasoning (arithmetic, algebraic manipulation)
  3. Causal reasoning (if-then chains, counterfactuals)
  4. Structural reasoning (part-whole, classification, analogy)
  5. Meta-reasoning (reasoning about reasoning)
"""

from glm.core.grammar import Rule, Production, Grammar, StrangeLoop, Direction


# ---------------------------------------------------------------------------
# Logical inference rules — the backbone of reasoning
# ---------------------------------------------------------------------------

def build_inference_grammar() -> Grammar:
    """Operational inference rules that fire on symbolic strings.

    These are the workhorses of reasoning.  Each rule takes a
    structured string input and produces a structured string output.
    The derivation engine can chain these to build proofs.
    """

    rules = [
        # --- Transitivity of implication (checked BEFORE modus ponens) ---
        Rule(
            name="transitivity_op",
            pattern=lambda form: (
                isinstance(form, tuple) and len(form) == 2
                and isinstance(form[0], str) and isinstance(form[1], str)
                and " implies " in form[0] and " implies " in form[1]
                and _can_chain_implications(form[0], form[1])
            ),
            result=lambda form: _apply_transitivity(form),
            weight=0.95,
            direction=Direction.FORWARD,
        ),

        # --- Modus Ponens (operational) ---
        # Matches (fact, "fact implies conclusion") where fact is NOT itself an implication
        Rule(
            name="modus_ponens_op",
            pattern=lambda form: (
                isinstance(form, tuple) and len(form) == 2
                and isinstance(form[0], str) and isinstance(form[1], str)
                and " implies " in form[1]
                and " implies " not in form[0]
            ),
            result=lambda form: _apply_modus_ponens(form),
            weight=0.95,
            direction=Direction.FORWARD,
        ),

        # --- Contrapositive ---
        Rule(
            name="contrapositive_op",
            pattern=lambda form: (
                isinstance(form, str) and " implies " in form
            ),
            result=lambda form: _apply_contrapositive(form),
            weight=0.95,
            direction=Direction.FORWARD,
        ),

        # --- Conjunction introduction ---
        Rule(
            name="conjunction_intro",
            pattern=lambda form: (
                isinstance(form, tuple) and len(form) == 2
                and isinstance(form[0], str) and isinstance(form[1], str)
                and "and" not in form[0] and "and" not in form[1]
            ),
            result=lambda form: f"{form[0]} and {form[1]}",
            weight=0.95,
            direction=Direction.FORWARD,
        ),

        # --- Conjunction elimination ---
        Rule(
            name="conjunction_elim_left",
            pattern=lambda form: isinstance(form, str) and " and " in form,
            result=lambda form: form.split(" and ")[0].strip(),
            weight=0.95,
            direction=Direction.FORWARD,
        ),
        Rule(
            name="conjunction_elim_right",
            pattern=lambda form: isinstance(form, str) and " and " in form,
            result=lambda form: form.split(" and ")[-1].strip(),
            weight=0.95,
            direction=Direction.FORWARD,
        ),

        # --- Disjunction introduction ---
        Rule(
            name="disjunction_intro",
            pattern=lambda form: isinstance(form, str) and " or " not in form,
            result=lambda form: f"{form} or [unknown]",
            weight=0.5,  # Low confidence — disjunction is weak
            direction=Direction.FORWARD,
        ),

        # --- Double negation elimination ---
        Rule(
            name="double_negation_elim",
            pattern=lambda form: isinstance(form, str) and form.startswith("not not "),
            result=lambda form: form[8:],  # Remove "not not "
            weight=0.95,
            direction=Direction.FORWARD,
        ),

        # --- Negation of implication ---
        Rule(
            name="deny_consequent",
            pattern=lambda form: (
                isinstance(form, tuple) and len(form) == 2
                and isinstance(form[0], str) and isinstance(form[1], str)
                and " implies " in form[0]
                and form[1].startswith("not ")
            ),
            result=lambda form: _apply_modus_tollens(form),
            weight=0.95,
            direction=Direction.FORWARD,
        ),

        # --- Universal to particular ---
        Rule(
            name="universal_instantiate",
            pattern=lambda form: (
                isinstance(form, tuple) and len(form) == 2
                and isinstance(form[0], str) and isinstance(form[1], str)
                and form[0].startswith("for all ")
            ),
            result=lambda form: _instantiate_universal(form),
            weight=0.9,
            direction=Direction.FORWARD,
        ),
    ]

    return Grammar(
        name="operational_inference",
        domain="reasoning",
        rules=rules,
    )


# ---------------------------------------------------------------------------
# Causal reasoning rules
# ---------------------------------------------------------------------------

def build_causal_grammar() -> Grammar:
    """Rules for causal reasoning: if-then chains, explanations."""

    rules = [
        # --- Causal chain ---
        Rule(
            name="causal_chain",
            pattern=lambda form: (
                isinstance(form, tuple) and len(form) == 2
                and isinstance(form[0], str) and isinstance(form[1], str)
                and " causes " in form[0] and " causes " in form[1]
            ),
            result=lambda form: _chain_causes(form),
            weight=0.85,
            direction=Direction.FORWARD,
        ),

        # --- Effect propagation ---
        Rule(
            name="effect_propagation",
            pattern=lambda form: (
                isinstance(form, str) and " causes " in form
            ),
            result=lambda form: _propagate_effect(form),
            weight=0.8,
            direction=Direction.FORWARD,
        ),

        # --- Counterfactual (forward: A causes B → if not A then not B) ---
        Rule(
            name="counterfactual",
            pattern=lambda form: (
                isinstance(form, str) and " causes " in form
            ),
            result=lambda form: _counterfactual(form),
            weight=0.7,
            direction=Direction.FORWARD,
        ),
    ]

    return Grammar(
        name="causal_reasoning",
        domain="reasoning",
        rules=rules,
    )


# ---------------------------------------------------------------------------
# Structural reasoning rules
# ---------------------------------------------------------------------------

def build_structural_grammar() -> Grammar:
    """Rules for structural reasoning: classification, part-whole, analogy."""

    rules = [
        # --- Classification (is-a) ---
        Rule(
            name="classify_isa",
            pattern=lambda form: (
                isinstance(form, str) and " is a " in form
            ),
            result=lambda form: _classify(form),
            weight=0.9,
            direction=Direction.FORWARD,
        ),

        # --- Part-whole ---
        Rule(
            name="part_whole",
            pattern=lambda form: (
                isinstance(form, str) and " has " in form
            ),
            result=lambda form: _part_whole(form),
            weight=0.85,
            direction=Direction.FORWARD,
        ),

        # --- Analogy (A is to B as C is to D) ---
        Rule(
            name="analogy",
            pattern=lambda form: (
                isinstance(form, tuple) and len(form) == 3
                and all(isinstance(x, str) for x in form)
            ),
            result=lambda form: _analogy(form),
            weight=0.7,
            direction=Direction.FORWARD,
        ),

        # --- Property inheritance ---
        Rule(
            name="property_inherit",
            pattern=lambda form: (
                isinstance(form, tuple) and len(form) == 2
                and isinstance(form[0], str) and isinstance(form[1], str)
                and " is a " in form[0] and " has " in form[1]
            ),
            result=lambda form: _inherit_property(form),
            weight=0.8,
            direction=Direction.FORWARD,
        ),
    ]

    return Grammar(
        name="structural_reasoning",
        domain="reasoning",
        rules=rules,
    )


# ---------------------------------------------------------------------------
# Meta-reasoning rules
# ---------------------------------------------------------------------------

def build_meta_grammar() -> Grammar:
    """Rules for reasoning about reasoning — the strange loop."""

    rules = [
        # --- Confidence aggregation ---
        Rule(
            name="confidence_combine",
            pattern=lambda form: (
                isinstance(form, tuple) and len(form) == 2
                and all(isinstance(x, (int, float)) for x in form)
            ),
            result=lambda form: form[0] * form[1],
            weight=1.0,
            direction=Direction.FORWARD,
        ),

        # --- Contradiction detection ---
        Rule(
            name="detect_contradiction",
            pattern=lambda form: (
                isinstance(form, tuple) and len(form) == 2
                and isinstance(form[0], str) and isinstance(form[1], str)
                and _is_contradiction(form[0], form[1])
            ),
            result=lambda form: f"CONTRADICTION: '{form[0]}' vs '{form[1]}'",
            weight=1.0,
            direction=Direction.FORWARD,
        ),

        # --- Certainty decay ---
        Rule(
            name="certainty_decay",
            pattern=lambda form: (
                isinstance(form, dict) and "confidence" in form
                and "depth" in form
            ),
            result=lambda form: {
                **form,
                "confidence": form["confidence"] * (0.95 ** form["depth"]),
                "note": "confidence decays with reasoning depth",
            },
            weight=1.0,
            direction=Direction.FORWARD,
        ),
    ]

    strange_loop = StrangeLoop(
        entry_rule="meta_reasoning",
        cycle=[
            "reason_about_conclusion",
            "assess_reasoning_quality",
            "adjust_confidence",
            "reason_again",
        ],
        level_shift=1,  # Goes up one level of abstraction
    )

    return Grammar(
        name="meta_reasoning",
        domain="reasoning",
        rules=rules,
        sub_grammars=[strange_loop],
    )


# ---------------------------------------------------------------------------
# Combined reasoning grammar
# ---------------------------------------------------------------------------

def build_reasoning_grammar() -> Grammar:
    """Build the combined reasoning grammar with all operational rules."""
    inference = build_inference_grammar()
    causal = build_causal_grammar()
    structural = build_structural_grammar()
    meta = build_meta_grammar()

    combined = Grammar(
        name="reasoning",
        domain="reasoning",
        sub_grammars=[inference, causal, structural, meta],
    )
    return combined


# ---------------------------------------------------------------------------
# Rule implementation helpers
# ---------------------------------------------------------------------------

def _can_chain_implications(imp1: str, imp2: str) -> bool:
    """Check if two implications can be chained: A→B, B→C."""
    parts1 = imp1.split(" implies ")
    parts2 = imp2.split(" implies ")
    if len(parts1) == 2 and len(parts2) == 2:
        return parts1[1].strip().lower() == parts2[0].strip().lower()
    return False


def _apply_modus_ponens(form: tuple) -> str:
    """P, P implies Q → Q"""
    p = form[0]
    implication = form[1]
    parts = implication.split(" implies ")
    if len(parts) == 2:
        antecedent, consequent = parts
        if p.strip().lower() == antecedent.strip().lower():
            return consequent.strip()
    return f"cannot derive from {p!r} and {implication!r}"


def _apply_transitivity(form: tuple) -> str:
    """A implies B, B implies C → A implies C"""
    imp1 = form[0]
    imp2 = form[1]
    parts1 = imp1.split(" implies ")
    parts2 = imp2.split(" implies ")
    if len(parts1) == 2 and len(parts2) == 2:
        a, b1 = parts1
        b2, c = parts2
        if b1.strip().lower() == b2.strip().lower():
            return f"{a.strip()} implies {c.strip()}"
    return f"no transitivity between {imp1!r} and {imp2!r}"


def _apply_contrapositive(form: str) -> str:
    """P implies Q → not Q implies not P"""
    parts = form.split(" implies ")
    if len(parts) == 2:
        p, q = parts
        not_q = f"not {q.strip()}" if not q.strip().startswith("not ") else q.strip()[4:]
        not_p = f"not {p.strip()}" if not p.strip().startswith("not ") else p.strip()[4:]
        return f"{not_q} implies {not_p}"
    return form


def _apply_modus_tollens(form: tuple) -> str:
    """P implies Q, not Q → not P"""
    implication = form[0]
    negation = form[1]
    parts = implication.split(" implies ")
    if len(parts) == 2:
        p, q = parts
        neg_q = negation[4:] if negation.startswith("not ") else negation
        if neg_q.strip().lower() == q.strip().lower():
            return f"not {p.strip()}"
    return f"cannot apply modus tollens"


def _instantiate_universal(form: tuple) -> str:
    """for all X, P(X) + instance a → P(a)"""
    universal = form[0]
    instance = form[1]
    # Parse "for all X, P(X)"
    rest = universal[8:]  # Remove "for all "
    if ", " in rest:
        var, predicate = rest.split(", ", 1)
        # Replace variable with instance
        return predicate.replace(var.strip(), instance.strip())
    return f"cannot instantiate {universal!r} with {instance!r}"


def _chain_causes(form: tuple) -> str:
    """A causes B, B causes C → A causes C"""
    c1 = form[0]
    c2 = form[1]
    parts1 = c1.split(" causes ")
    parts2 = c2.split(" causes ")
    if len(parts1) == 2 and len(parts2) == 2:
        a, b1 = parts1
        b2, c = parts2
        if b1.strip().lower() == b2.strip().lower():
            return f"{a.strip()} causes {c.strip()}"
    return f"no causal chain between {c1!r} and {c2!r}"


def _propagate_effect(form: str) -> str:
    """A causes B → therefore B"""
    parts = form.split(" causes ")
    if len(parts) == 2:
        return f"therefore {parts[1].strip()}"
    return form


def _counterfactual(form: str) -> str:
    """A causes B → if not A then not B"""
    parts = form.split(" causes ")
    if len(parts) == 2:
        return f"if not {parts[0].strip()} then not {parts[1].strip()}"
    return form


def _classify(form: str) -> str:
    """X is a Y → X has properties of Y"""
    parts = form.split(" is a ")
    if len(parts) == 2:
        return f"{parts[0].strip()} inherits properties of {parts[1].strip()}"
    return form


def _part_whole(form: str) -> str:
    """X has Y → Y is part of X"""
    parts = form.split(" has ")
    if len(parts) == 2:
        return f"{parts[1].strip()} is part of {parts[0].strip()}"
    return form


def _analogy(form: tuple) -> str:
    """(A, B, relation) → A relates to B by relation"""
    a, b, relation = form
    return f"{a} {relation} {b}"


def _inherit_property(form: tuple) -> str:
    """X is a Y, Y has Z → X has Z"""
    isa = form[0]
    has = form[1]
    isa_parts = isa.split(" is a ")
    has_parts = has.split(" has ")
    if len(isa_parts) == 2 and len(has_parts) == 2:
        x, y1 = isa_parts
        y2, z = has_parts
        if y1.strip().lower() == y2.strip().lower():
            return f"{x.strip()} has {z.strip()}"
    return f"cannot inherit"


def _is_contradiction(a: str, b: str) -> bool:
    """Check if two statements contradict each other."""
    a_low = a.lower().strip()
    b_low = b.lower().strip()
    # Direct negation
    if a_low == f"not {b_low}" or b_low == f"not {a_low}":
        return True
    # Negation within
    if a_low.startswith("not ") and a_low[4:] == b_low:
        return True
    if b_low.startswith("not ") and b_low[4:] == a_low:
        return True
    return False
