"""Tests for the unified 270-glyph MNEMO substrate."""

import pytest
from glm.core.mnemo_substrate import (
    MnemoGlyph, MnemoSequence, MnemoSubstrate,
    Tier, GLYPH_REGISTRY,
    EvidentialSource, EvidentialConfidence, EvidentialTemporal,
)


class TestGlyphRegistry:
    def test_registry_has_270_glyphs(self):
        assert len(GLYPH_REGISTRY) == 270

    def test_tiers_are_complete(self):
        counts = {}
        for g in GLYPH_REGISTRY.values():
            counts[g.tier] = counts.get(g.tier, 0) + 1
        assert counts[Tier.ONTOLOGICAL] == 50
        assert counts[Tier.PROCESS] == 40
        assert counts[Tier.STATE] == 25
        assert counts[Tier.RELATIONAL] == 35
        assert counts[Tier.EPISTEMIC] == 25
        assert counts[Tier.TEMPORAL] == 30
        assert counts[Tier.SCALE] == 35
        assert counts[Tier.META] == 30

    def test_glyph_lookup_by_code(self):
        g = GLYPH_REGISTRY["obs"]
        assert g.tier == Tier.STATE
        assert g.concept == "observed"

    def test_glyph_lookup_by_concept(self):
        substrate = MnemoSubstrate()
        g = substrate.lookup_concept("observed")
        assert g.code == "obs"


class TestEvidentialMarkers:
    def test_source_glyphs_exist(self):
        for code in ("obs", "inf", "comp", "rep", "trad", "spec", "ctr"):
            assert code in GLYPH_REGISTRY

    def test_confidence_glyphs_exist(self):
        for code in ("cert", "prob", "poss", "unl", "unk"):
            assert code in GLYPH_REGISTRY

    def test_temporal_glyphs_exist(self):
        for code in ("ver_past", "obs_pres", "pred_fut", "hyp", "timeless"):
            assert code in GLYPH_REGISTRY


class TestMnemoSequence:
    def test_create_sequence(self):
        substrate = MnemoSubstrate()
        seq = substrate.encode_codes(["obs", "cert", "obs_pres"])
        assert len(seq) == 3

    def test_has_evidential_marking(self):
        substrate = MnemoSubstrate()
        seq = substrate.encode_codes(["obs", "cert", "obs_pres"])
        assert seq.has_evidential_marking()

    def test_missing_evidential_detected(self):
        substrate = MnemoSubstrate()
        seq = substrate.encode_codes(["obs"])
        assert not seq.has_evidential_marking()

    def test_get_evidential_triple(self):
        substrate = MnemoSubstrate()
        seq = substrate.encode_codes(["obs", "cert", "obs_pres"])
        source, conf, temp = seq.get_evidential_triple()
        assert source.code == "obs"
        assert conf.code == "cert"
        assert temp.code == "obs_pres"


class TestMnemoSubstrate:
    def test_encode_decode_roundtrip(self):
        substrate = MnemoSubstrate()
        codes = ["obs", "cert", "obs_pres"]
        seq = substrate.encode_codes(codes)
        result = substrate.decode_codes(seq)
        assert result == codes

    def test_validate_rejects_bad_sequence(self):
        substrate = MnemoSubstrate()
        assert not substrate.validate_sequence(["INVALID", "obs"])

    def test_validate_accepts_good_sequence(self):
        substrate = MnemoSubstrate()
        assert substrate.validate_sequence(["obs", "cert", "obs_pres"])

    def test_session_mapping(self):
        s1 = MnemoSubstrate(session_seed="session_a")
        s2 = MnemoSubstrate(session_seed="session_b")
        g1 = s1.lookup_concept("observed")
        g2 = s2.lookup_concept("observed")
        assert g1.concept == g2.concept
