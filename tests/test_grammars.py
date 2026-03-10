"""Tests for domain-specific grammars."""

from glm.grammars.linguistic import (
    build_syntactic_grammar,
    build_phonological_grammar,
    build_morphological_grammar,
)
from glm.grammars.etymological import (
    build_etymology_grammar,
    build_substrate_transfer_grammar,
    build_cognate_detection_grammar,
)
from glm.grammars.chemical import (
    build_bonding_grammar,
    build_reaction_grammar,
    build_molecular_grammar,
)
from glm.grammars.biological import (
    build_genetic_grammar,
    build_protein_grammar,
    build_evolutionary_grammar,
)
from glm.grammars.computational import (
    build_syntax_grammar,
    build_type_grammar,
    build_pattern_grammar,
)


class TestLinguisticGrammars:
    def test_syntactic_grammar(self):
        g = build_syntactic_grammar()
        assert g.name
        assert len(g.rules) > 0 or len(g.productions) > 0
        assert g.domain == "linguistics"

    def test_phonological_grammar(self):
        g = build_phonological_grammar()
        assert g.name
        assert len(g.rules) > 0

    def test_morphological_grammar(self):
        g = build_morphological_grammar()
        assert g.name
        assert len(g.rules) > 0


class TestEtymologicalGrammars:
    def test_etymology_grammar(self):
        g = build_etymology_grammar()
        assert g.domain == "etymology"
        assert len(g.rules) > 0

    def test_substrate_transfer(self):
        g = build_substrate_transfer_grammar()
        assert len(g.rules) > 0

    def test_cognate_detection(self):
        g = build_cognate_detection_grammar()
        assert len(g.rules) > 0


class TestChemicalGrammars:
    def test_bonding_grammar(self):
        g = build_bonding_grammar()
        assert g.domain == "chemistry"

    def test_reaction_grammar(self):
        g = build_reaction_grammar()
        assert len(g.rules) > 0

    def test_molecular_grammar(self):
        g = build_molecular_grammar()
        assert len(g.rules) > 0


class TestBiologicalGrammars:
    def test_genetic_grammar(self):
        g = build_genetic_grammar()
        assert g.domain == "biology"

    def test_protein_grammar(self):
        g = build_protein_grammar()
        assert len(g.rules) > 0

    def test_evolutionary_grammar(self):
        g = build_evolutionary_grammar()
        assert len(g.rules) > 0


class TestComputationalGrammars:
    def test_syntax_grammar(self):
        g = build_syntax_grammar()
        assert g.domain == "computation"

    def test_type_grammar(self):
        g = build_type_grammar()
        assert len(g.rules) > 0

    def test_pattern_grammar(self):
        g = build_pattern_grammar()
        assert len(g.rules) > 0
