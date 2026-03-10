"""Tests for core grammar primitives."""

import pytest
from glm.core.grammar import Rule, Production, Grammar, StrangeLoop
from glm.core.substrate import Substrate, Symbol, Sequence
from glm.core.lexicon import Lexicon, LexicalEntry
from glm.core.engine import DerivationEngine


class TestRule:
    def test_create_rule(self):
        rule = Rule(
            name="assimilation",
            pattern="n+p",
            result="m+p",
            direction="forward",
        )
        assert rule.name == "assimilation"
        assert rule.direction == "forward"

    def test_bidirectional_rule(self):
        rule = Rule(
            name="vowel_shift",
            pattern="a:",
            result="eɪ",
            direction="bidirectional",
        )
        assert rule.direction == "bidirectional"

    def test_self_referential_rule(self):
        rule = Rule(
            name="recursion",
            pattern="S",
            result="S+S",
            self_referential=True,
        )
        assert rule.self_referential is True


class TestProduction:
    def test_create_production(self):
        prod = Production(lhs="S", rhs=["NP", "VP"])
        assert prod.lhs == "S"
        assert prod.rhs == ["NP", "VP"]

    def test_production_with_grammar(self):
        prod = Production(lhs="NP", rhs=["Det", "N"], grammar_name="syntax")
        assert prod.grammar_name == "syntax"


class TestGrammar:
    def test_create_grammar(self):
        rules = [
            Rule(name="r1", pattern="a", result="b"),
        ]
        prods = [
            Production(lhs="S", rhs=["a", "b"]),
        ]
        grammar = Grammar(name="test", domain="test", rules=rules, productions=prods)
        assert grammar.name == "test"
        assert len(grammar.rules) == 1
        assert len(grammar.productions) == 1

    def test_grammar_with_sub_grammars(self):
        sub = Grammar(name="sub", domain="test")
        parent = Grammar(name="parent", domain="test", sub_grammars=[sub])
        assert len(parent.sub_grammars) == 1


class TestSymbol:
    def test_create_symbol(self):
        sym = Symbol(form="a", domain="phonological")
        assert sym.form == "a"
        assert sym.domain == "phonological"


class TestSequence:
    def test_create_sequence(self):
        syms = [Symbol(form="a"), Symbol(form="b"), Symbol(form="c")]
        seq = Sequence(symbols=syms)
        assert len(seq) == 3

    def test_sequence_pattern_detection(self):
        syms = [Symbol(form=c) for c in "abcabc"]
        seq = Sequence(symbols=syms)
        patterns = seq.find_patterns(min_length=2, max_length=4)
        assert len(patterns) > 0


class TestDerivationEngine:
    def test_create_engine(self):
        engine = DerivationEngine()
        assert engine is not None

    def test_derive_forward(self):
        rule = Rule(name="r1", pattern="a", result="b", direction="forward")
        grammar = Grammar(name="test", domain="test", rules=[rule])
        engine = DerivationEngine()
        derivations = engine.derive(["a"], grammar, direction="forward")
        assert len(derivations) > 0

    def test_detect_loops(self):
        rules = [
            Rule(name="r1", pattern="A", result="B", direction="forward"),
            Rule(name="r2", pattern="B", result="C", direction="forward"),
            Rule(name="r3", pattern="C", result="A", direction="forward"),
        ]
        grammar = Grammar(name="loop_test", domain="test", rules=rules)
        engine = DerivationEngine()
        loops = engine.detect_loops(grammar)
        assert len(loops) > 0

    def test_find_isomorphisms(self):
        r1 = Rule(name="r1", pattern="A", result="B")
        r2 = Rule(name="r2", pattern="X", result="Y")
        g1 = Grammar(name="g1", domain="d1", rules=[r1])
        g2 = Grammar(name="g2", domain="d2", rules=[r2])
        engine = DerivationEngine()
        isos = engine.find_isomorphisms(g1, g2)
        assert isinstance(isos, list)


class TestLexicon:
    def test_create_lexicon(self):
        lexicon = Lexicon()
        assert len(lexicon) == 0

    def test_add_entry(self):
        lexicon = Lexicon()
        entry = LexicalEntry(
            form="water",
            meaning="H2O",
            domain="linguistic",
            category="noun",
        )
        lexicon.add(entry)
        assert len(lexicon) == 1

    def test_lookup(self):
        lexicon = Lexicon()
        entry = LexicalEntry(form="water", meaning="H2O", domain="linguistic")
        lexicon.add(entry)
        found = lexicon.lookup("water")
        assert found is not None
        assert found.meaning == "H2O"
