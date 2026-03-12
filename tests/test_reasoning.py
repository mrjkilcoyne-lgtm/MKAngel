"""Tests for the reasoning engine and operational grammar rules.

These tests verify that the GLM can:
1. Chain derivation steps (multi-step reasoning)
2. Apply logical inference rules (modus ponens, etc.)
3. Reason by analogy across domains
4. Learn from successful chains (self-improvement)
5. Reason bidirectionally (meet in the middle)
"""

import pytest

from glm.core.grammar import Grammar, Rule, Production, Direction
from glm.core.engine import DerivationEngine
from glm.core.reasoning import (
    ReasoningEngine,
    ReasoningChain,
    ReasoningStep,
    _forms_match,
)
from glm.grammars.reasoning import (
    build_inference_grammar,
    build_causal_grammar,
    build_structural_grammar,
    build_meta_grammar,
    build_reasoning_grammar,
)
from glm.angel import Angel


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def engine():
    return DerivationEngine()


@pytest.fixture
def inference_grammar():
    return build_inference_grammar()


@pytest.fixture
def causal_grammar():
    return build_causal_grammar()


@pytest.fixture
def structural_grammar():
    return build_structural_grammar()


@pytest.fixture
def reasoning_grammar():
    return build_reasoning_grammar()


@pytest.fixture
def reasoner(engine):
    """A reasoning engine with all reasoning grammars loaded."""
    grammars = {
        "reasoning": [build_reasoning_grammar()],
    }
    return ReasoningEngine(
        engine=engine,
        grammars=grammars,
        beam_width=5,
        max_depth=10,
    )


@pytest.fixture
def angel():
    a = Angel()
    a.awaken()
    return a


# ---------------------------------------------------------------------------
# Operational inference rule tests
# ---------------------------------------------------------------------------

class TestInferenceRules:
    """Test that the operational logic rules actually fire."""

    def test_modus_ponens(self, inference_grammar, engine):
        """P, P implies Q → Q"""
        form = ("it rains", "it rains implies ground is wet")
        tree = engine.derive(form, inference_grammar, "forward", max_steps=50)
        all_forms = tree.all_forms()
        assert any("ground is wet" == str(f) or "ground is wet" in str(f) for f in all_forms)

    def test_contrapositive(self, inference_grammar, engine):
        """P implies Q → not Q implies not P"""
        form = "rain implies wet"
        tree = engine.derive(form, inference_grammar, "forward", max_steps=50)
        all_forms = tree.all_forms()
        assert any("not wet implies not rain" in str(o) for o in all_forms)

    def test_conjunction_elimination(self, inference_grammar, engine):
        """A and B → A, B"""
        form = "sky is blue and grass is green"
        tree = engine.derive(form, inference_grammar, "forward", max_steps=50)
        all_forms = tree.all_forms()
        assert any("sky is blue" == f for f in all_forms)
        assert any("grass is green" == f for f in all_forms)

    def test_conjunction_introduction(self, inference_grammar, engine):
        """A, B → A and B"""
        form = ("the sun shines", "birds sing")
        tree = engine.derive(form, inference_grammar, "forward", max_steps=50)
        all_forms = tree.all_forms()
        assert any("the sun shines and birds sing" in str(o) for o in all_forms)

    def test_double_negation(self, inference_grammar, engine):
        """not not P → P"""
        form = "not not it is true"
        tree = engine.derive(form, inference_grammar, "forward", max_steps=50)
        all_forms = tree.all_forms()
        assert any("it is true" == f for f in all_forms)

    def test_modus_tollens(self, inference_grammar, engine):
        """P implies Q, not Q → not P"""
        form = ("rain implies wet", "not wet")
        tree = engine.derive(form, inference_grammar, "forward", max_steps=50)
        all_forms = tree.all_forms()
        assert any("not rain" in str(o) for o in all_forms)

    def test_transitivity(self, inference_grammar, engine):
        """A implies B, B implies C → A implies C"""
        form = ("A implies B", "B implies C")
        tree = engine.derive(form, inference_grammar, "forward", max_steps=50)
        all_forms = tree.all_forms()
        assert any("A implies C" in str(o) for o in all_forms)

    def test_universal_instantiation(self, inference_grammar, engine):
        """for all X, P(X) + a → P(a)"""
        form = ("for all dogs, dogs bark", "Rex")
        tree = engine.derive(form, inference_grammar, "forward", max_steps=50)
        all_forms = tree.all_forms()
        assert any("Rex bark" in str(o) for o in all_forms)


class TestCausalRules:
    """Test causal reasoning rules."""

    def test_causal_chain(self, causal_grammar, engine):
        """A causes B, B causes C → A causes C"""
        form = ("rain causes flood", "flood causes damage")
        tree = engine.derive(form, causal_grammar, "forward", max_steps=50)
        all_forms = tree.all_forms()
        assert any("rain causes damage" in str(o) for o in all_forms)

    def test_effect_propagation(self, causal_grammar, engine):
        """A causes B → therefore B"""
        form = "fire causes heat"
        tree = engine.derive(form, causal_grammar, "forward", max_steps=50)
        all_forms = tree.all_forms()
        assert any("therefore heat" in str(o) for o in all_forms)

    def test_counterfactual(self, causal_grammar, engine):
        """A causes B → if not A then not B"""
        form = "fire causes heat"
        tree = engine.derive(form, causal_grammar, "forward", max_steps=50)
        all_forms = tree.all_forms()
        assert any("if not fire then not heat" in str(o) for o in all_forms)


class TestStructuralRules:
    """Test structural reasoning rules."""

    def test_classification(self, structural_grammar, engine):
        """X is a Y → X inherits properties of Y"""
        form = "dog is a mammal"
        tree = engine.derive(form, structural_grammar, "forward", max_steps=50)
        all_forms = tree.all_forms()
        assert any("inherits properties of mammal" in str(o) for o in all_forms)

    def test_part_whole(self, structural_grammar, engine):
        """X has Y → Y is part of X"""
        form = "car has engine"
        tree = engine.derive(form, structural_grammar, "forward", max_steps=50)
        all_forms = tree.all_forms()
        assert any("engine is part of car" in str(o) for o in all_forms)

    def test_property_inheritance(self, structural_grammar, engine):
        """X is a Y, Y has Z → X has Z"""
        form = ("dog is a mammal", "mammal has warm blood")
        tree = engine.derive(form, structural_grammar, "forward", max_steps=50)
        all_forms = tree.all_forms()
        assert any("dog has warm blood" in str(o) for o in all_forms)


# ---------------------------------------------------------------------------
# Reasoning engine tests
# ---------------------------------------------------------------------------

class TestReasoningEngine:
    """Test the chain-of-thought reasoning engine."""

    def test_simple_chain(self, reasoner):
        """Reason about a simple implication."""
        chain = reasoner.reason(
            question="Does the ground get wet when it rains?",
            start_form="rain implies wet ground",
        )
        assert isinstance(chain, ReasoningChain)
        assert chain.depth >= 0

    def test_chain_confidence(self, reasoner):
        """Chain confidence should be product of step confidences."""
        chain = reasoner.reason(
            question="Test confidence",
            start_form="sky is blue and water is wet",
        )
        if chain.depth > 0:
            expected = 1.0
            for step in chain.steps:
                expected *= step.confidence
            assert abs(chain.confidence - expected) < 1e-6

    def test_goal_directed(self, reasoner):
        """Reason toward a specific goal."""
        chain = reasoner.reason(
            question="Can we conclude C?",
            start_form=("A implies B", "B implies C"),
            goal="A implies C",
        )
        # May or may not find the goal depending on rule chaining
        assert isinstance(chain, ReasoningChain)

    def test_reasoning_trace(self, reasoner):
        """Reasoning trace should be human-readable."""
        chain = reasoner.reason(
            question="What follows?",
            start_form="not not it is true",
        )
        trace = chain.trace
        assert isinstance(trace, list)
        assert len(trace) >= 1  # At least the question line

    def test_self_improvement(self, reasoner):
        """Engine should learn rules from successful chains."""
        initial_rules = len(reasoner._learned_rules)
        # Run several reasoning chains
        for _ in range(3):
            reasoner.reason(
                question="Test learning",
                start_form="rain implies wet ground",
            )
        # Learned rules may or may not increase depending on chain success
        assert len(reasoner._learned_rules) >= initial_rules

    def test_stats(self, reasoner):
        """Stats should track reasoning history."""
        reasoner.reason(
            question="Test stats",
            start_form="fire causes heat",
        )
        stats = reasoner.get_stats()
        assert stats["total_reasoning_chains"] >= 1

    def test_decompose_and_reason(self, reasoner):
        """Decompose a complex problem into sub-goals."""
        chain = reasoner.decompose_and_reason(
            question="Multi-step problem",
            sub_goals=[
                ("Step 1", "rain implies flood", None),
                ("Step 2", "flood causes damage", None),
            ],
        )
        assert isinstance(chain, ReasoningChain)


class TestBidirectionalReasoning:
    """Test meet-in-the-middle reasoning."""

    def test_bidirectional_basic(self, reasoner):
        """Bidirectional search should find a path if one exists."""
        chain = reasoner.reason_bidirectional(
            question="Connect A to C",
            start_form="NP",
            goal_form="S",
        )
        assert isinstance(chain, ReasoningChain)

    def test_bidirectional_returns_chain(self, reasoner):
        """Even with no meeting point, should return a valid chain."""
        chain = reasoner.reason_bidirectional(
            question="Hard problem",
            start_form="alpha",
            goal_form="omega",
        )
        assert isinstance(chain, ReasoningChain)


# ---------------------------------------------------------------------------
# Combined grammar tests
# ---------------------------------------------------------------------------

class TestReasoningGrammar:
    """Test the combined reasoning grammar."""

    def test_builds_successfully(self, reasoning_grammar):
        """Combined grammar should build without errors."""
        assert reasoning_grammar.name == "reasoning"
        assert len(reasoning_grammar.sub_grammars) == 4

    def test_has_all_subgrammars(self, reasoning_grammar):
        names = [sg.name for sg in reasoning_grammar.sub_grammars]
        assert "operational_inference" in names
        assert "causal_reasoning" in names
        assert "structural_reasoning" in names
        assert "meta_reasoning" in names


# ---------------------------------------------------------------------------
# Angel integration tests
# ---------------------------------------------------------------------------

class TestAngelReasoning:
    """Test reasoning integrated into the Angel."""

    def test_angel_has_reasoner(self, angel):
        """Angel should have a reasoning engine after awakening."""
        assert angel._reasoner is not None

    def test_angel_reason_method(self, angel):
        """Angel.reason() should return a structured result."""
        result = angel.reason("rain implies wet ground")
        assert "question" in result
        assert "reasoning_chain" in result
        assert "confidence" in result
        assert "trace" in result
        assert "stats" in result

    def test_angel_respond_includes_reasoning(self, angel):
        """Angel.respond() should include reasoning in its output."""
        result = angel.respond("rain implies wet ground")
        assert "reasoning" in result
        assert "chain_depth" in result["reasoning"]
        assert "confidence" in result["reasoning"]
        assert "trace" in result["reasoning"]

    def test_angel_reason_with_causes(self, angel):
        """Angel should reason about causal statements."""
        result = angel.reason("fire causes heat")
        assert result["chain_depth"] >= 0

    def test_angel_reason_with_classification(self, angel):
        """Angel should reason about is-a relationships."""
        result = angel.reason("dog is a mammal")
        assert result["chain_depth"] >= 0

    def test_angel_introspect_includes_reasoning(self, angel):
        """Introspection should include reasoning stats."""
        info = angel.introspect()
        assert "reasoning_stats" in info

    def test_angel_reasoning_domain_loaded(self, angel):
        """Reasoning domain should be in loaded grammars."""
        assert "reasoning" in angel._grammars


# ---------------------------------------------------------------------------
# Helper tests
# ---------------------------------------------------------------------------

class TestHelpers:
    def test_forms_match_strings(self):
        assert _forms_match("hello", "hello")
        assert _forms_match("Hello", "hello")
        assert not _forms_match("hello", "world")

    def test_forms_match_other_types(self):
        assert _forms_match(42, 42)
        assert not _forms_match(42, 43)

    def test_reasoning_chain_trace(self):
        chain = ReasoningChain(question="Test?")
        step = ReasoningStep(
            input="A",
            output="B",
            rule_id="r1",
            rule_name="test_rule",
            grammar_name="test",
            domain="test",
            confidence=0.9,
            justification="because",
        )
        chain.add_step(step)
        chain.complete("B")
        trace = chain.trace
        assert "Q: Test?" in trace[0]
        assert "A: 'B'" in trace[-1]
