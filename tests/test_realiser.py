"""Tests for the reverse-production realiser."""

import pytest
from glm.nlg.realiser import Realiser
from glm.nlg.templates import TemplateRegistry
from glm.nlg.templates.en import register_english
from glm.core.mnemo_substrate import MnemoSubstrate


class TestRealiser:
    def setup_method(self):
        self.registry = TemplateRegistry()
        register_english(self.registry)
        self.substrate = MnemoSubstrate()
        self.realiser = Realiser(registry=self.registry, substrate=self.substrate)

    def test_create_realiser(self):
        assert self.realiser is not None

    def test_realise_simple_math(self):
        candidates = self.realiser.realise(
            domain="mathematical",
            slots={"result": "42"},
            evidential_source="comp",
            evidential_confidence="cert",
            evidential_temporal="obs_pres",
        )
        assert len(candidates) > 0
        assert any("42" in c.text for c in candidates)

    def test_realise_with_evidential_hedging(self):
        candidates = self.realiser.realise(
            domain="linguistic",
            slots={"word": "run", "analysis": "is polysemous"},
            evidential_source="inf",
            evidential_confidence="prob",
            evidential_temporal="obs_pres",
        )
        assert len(candidates) > 0
        best = candidates[0]
        assert "suggest" in best.text.lower() or "run" in best.text.lower()

    def test_realise_returns_scored_candidates(self):
        candidates = self.realiser.realise(
            domain="chemical",
            slots={"compound": "H2O", "formula": "H2O"},
            evidential_source="obs",
            evidential_confidence="cert",
            evidential_temporal="obs_pres",
        )
        assert all(hasattr(c, "score") for c in candidates)
        scores = [c.score for c in candidates]
        assert scores == sorted(scores, reverse=True)
