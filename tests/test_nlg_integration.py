"""Integration tests for the full NLG pipeline: ENCODE -> PROCESS -> DECODE."""

import pytest
from glm.nlg.encoder import MnemoEncoder
from glm.nlg.decoder import MnemoDecoder
from glm.core.mnemo_substrate import MnemoSubstrate, MnemoSequence


class TestFullPipeline:
    """End-to-end tests: natural language in, natural language out."""

    def setup_method(self):
        self.substrate = MnemoSubstrate()
        self.encoder = MnemoEncoder(substrate=self.substrate)
        self.decoder = MnemoDecoder(substrate=self.substrate)

    def test_roundtrip_math(self):
        text = "solve the equation for x"
        mnemo = self.encoder.encode(text)
        assert mnemo.has_evidential_marking()
        output = self.decoder.decode(mnemo, language="en")
        assert isinstance(output, str)
        assert len(output) > 0

    def test_roundtrip_chemistry(self):
        text = "the molecule bonds with oxygen atoms"
        mnemo = self.encoder.encode(text)
        assert mnemo.has_evidential_marking()
        output = self.decoder.decode(mnemo, language="en")
        assert isinstance(output, str)

    def test_roundtrip_linguistics(self):
        text = "the verb agrees with the noun in this sentence"
        mnemo = self.encoder.encode(text)
        # Should route through linguistic domain
        codes = [g.code for g in mnemo.glyphs]
        assert "scale_00" in codes  # linguistic_space
        output = self.decoder.decode(mnemo, language="en")
        assert isinstance(output, str)

    def test_roundtrip_biology(self):
        text = "the gene encodes a protein"
        mnemo = self.encoder.encode(text)
        output = self.decoder.decode(mnemo, language="en")
        assert isinstance(output, str)

    def test_evidential_markers_preserved(self):
        """Evidential markers must survive the full pipeline."""
        text = "the algorithm compares two values"
        mnemo = self.encoder.encode(text)
        source, conf, temp = mnemo.get_evidential_triple()
        assert source is not None
        assert conf is not None
        assert temp is not None
        # Default evidentials from encoder
        assert source.code == "rep"   # reported (user told us)
        assert conf.code == "prob"    # probable
        assert temp.code == "obs_pres"  # present

    def test_decoder_with_extra_slots(self):
        text = "solve the equation for x"
        mnemo = self.encoder.encode(text)
        output = self.decoder.decode(
            mnemo,
            language="en",
            extra_slots={"result": "4", "content": text},
        )
        assert isinstance(output, str)
        assert "4" in output

    def test_empty_input_handled(self):
        mnemo = self.encoder.encode("")
        output = self.decoder.decode(mnemo, language="en")
        assert isinstance(output, str)
        assert output == ""

    def test_domain_detection_routing(self):
        """Each domain should route through different scale glyphs."""
        domains = {
            "solve the equation": "scale_05",    # mathematical
            "the molecule reacts": "scale_01",   # chemical
            "the gene mutates": "scale_02",      # biological
            "the algorithm loops": "scale_03",   # computational
            "from latin root": "scale_04",       # etymological
            "force and energy": "scale_06",      # physical
        }
        for text, expected_scale in domains.items():
            mnemo = self.encoder.encode(text)
            codes = [g.code for g in mnemo.glyphs]
            assert expected_scale in codes, f"Expected {expected_scale} for '{text}', got {codes}"


class TestNLGProviderIntegration:
    """Test the NLGProvider as wired into the app."""

    def test_provider_full_pipeline(self):
        from app.providers import NLGProvider
        provider = NLGProvider()
        result = provider.generate("the verb agrees with the noun")
        assert "[NLG" in result
        assert "domain:" in result

    def test_provider_different_domains(self):
        from app.providers import NLGProvider
        provider = NLGProvider()
        for prompt in [
            "solve x + 2 = 5",
            "the molecule bonds",
            "the gene encodes a protein",
            "the algorithm iterates",
        ]:
            result = provider.generate(prompt)
            assert isinstance(result, str)
            assert len(result) > 0
