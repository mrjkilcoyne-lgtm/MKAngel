"""Tests for the 8th domain response grammar."""

import pytest
from glm.nlg.response_grammar import (
    UtteranceNode, DomainAnalysisNode,
    EvidentialMarkerNode, build_response_grammar,
)
from glm.core.mnemo_substrate import MnemoSubstrate, Tier


class TestResponseGrammar:
    def test_build_grammar(self):
        grammar = build_response_grammar()
        assert grammar.name == "response"
        assert grammar.domain == "response"
        assert len(grammar.productions) > 0

    def test_utterance_production(self):
        grammar = build_response_grammar()
        utterance_prods = [p for p in grammar.productions if p.lhs == "Utterance"]
        assert len(utterance_prods) > 0

    def test_multi_domain_composition(self):
        grammar = build_response_grammar()
        multi_prods = [
            p for p in grammar.productions
            if p.lhs == "DomainAnalysis" and "Conjunction" in (p.rhs or [])
        ]
        assert len(multi_prods) > 0


class TestUtteranceNode:
    def test_create_utterance(self):
        node = UtteranceNode(
            domain_results=["mathematical"],
            evidential_source="obs",
            evidential_confidence="cert",
            evidential_temporal="obs_pres",
        )
        assert node.domain_results == ["mathematical"]
        assert node.evidential_source == "obs"

    def test_utterance_to_mnemo(self):
        substrate = MnemoSubstrate()
        node = UtteranceNode(
            domain_results=["mathematical"],
            evidential_source="obs",
            evidential_confidence="cert",
            evidential_temporal="obs_pres",
        )
        seq = node.to_mnemo_sequence(substrate)
        assert seq.has_evidential_marking()

    def test_multi_domain_utterance(self):
        node = UtteranceNode(
            domain_results=["mathematical", "linguistic"],
            evidential_source="inf",
            evidential_confidence="prob",
            evidential_temporal="pred_fut",
        )
        assert len(node.domain_results) == 2
