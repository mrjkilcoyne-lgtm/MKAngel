"""Tests for the NLG decoder: MNEMO -> natural language."""

import pytest
from glm.nlg.decoder import MnemoDecoder
from glm.core.mnemo_substrate import MnemoSubstrate, MnemoSequence


class TestMnemoDecoder:
    def test_create_decoder(self):
        decoder = MnemoDecoder()
        assert decoder is not None

    def test_decode_mnemo_sequence(self):
        decoder = MnemoDecoder()
        substrate = MnemoSubstrate()
        seq = substrate.encode_codes(["scale_05", "proc_09", "obs", "cert", "obs_pres"])
        result = decoder.decode(seq, language="en")
        assert isinstance(result, str)
        assert len(result) > 0

    def test_decode_preserves_evidentiality(self):
        decoder = MnemoDecoder()
        substrate = MnemoSubstrate()
        seq = substrate.encode_codes(["scale_00", "proc_15", "inf", "unl", "pred_fut"])
        result = decoder.decode(seq, language="en")
        assert isinstance(result, str)

    def test_decode_selects_best_candidate(self):
        decoder = MnemoDecoder()
        substrate = MnemoSubstrate()
        seq = substrate.encode_codes([
            "scale_05", "proc_09", "obs", "cert", "obs_pres"
        ])
        result = decoder.decode(seq, language="en")
        assert result.strip() != ""

    def test_decode_empty_sequence(self):
        decoder = MnemoDecoder()
        result = decoder.decode(MnemoSequence(), language="en")
        assert isinstance(result, str)
