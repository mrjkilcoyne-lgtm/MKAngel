"""Tests for the NLG encoder: natural language -> MNEMO."""

import pytest
from glm.nlg.encoder import MnemoEncoder
from glm.core.mnemo_substrate import MnemoSubstrate, MnemoSequence, Tier


class TestMnemoEncoder:
    def test_create_encoder(self):
        encoder = MnemoEncoder()
        assert encoder is not None

    def test_encode_simple_text(self):
        encoder = MnemoEncoder()
        result = encoder.encode("water is a molecule")
        assert isinstance(result, MnemoSequence)
        assert len(result) > 0

    def test_encode_attaches_domain(self):
        encoder = MnemoEncoder()
        result = encoder.encode("the verb agrees with the noun")
        codes = [g.code for g in result.glyphs]
        has_ling_domain = any("scale_00" in c for c in codes)
        assert has_ling_domain

    def test_encode_adds_evidential_defaults(self):
        encoder = MnemoEncoder()
        result = encoder.encode("some text input")
        assert result.has_evidential_marking()

    def test_encode_empty_input(self):
        encoder = MnemoEncoder()
        result = encoder.encode("")
        assert isinstance(result, MnemoSequence)

    def test_domain_detection_math(self):
        encoder = MnemoEncoder()
        domain = encoder.detect_domain("solve the equation x + 2 = 5")
        assert domain == "mathematical"

    def test_domain_detection_chemistry(self):
        encoder = MnemoEncoder()
        domain = encoder.detect_domain("the molecule bonds with oxygen")
        assert domain == "chemical"
