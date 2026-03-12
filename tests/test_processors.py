"""Tests for domain-specific processors (Stage 2 of the NLG pipeline).

All HTTP calls are mocked — no network required.
"""

import json
import pytest
from unittest.mock import patch, MagicMock
from io import BytesIO

from glm.core.mnemo_substrate import MnemoSubstrate, MnemoSequence
from glm.nlg.encoder import MnemoEncoder
from glm.nlg.processors import (
    DomainProcessor,
    ProcessorDispatcher,
    create_default_dispatcher,
)
from glm.nlg.processors._http import fetch_json, clear_cache
from glm.nlg.processors.mathematical import MathProcessor
from glm.nlg.processors.linguistic import LinguisticProcessor
from glm.nlg.processors.chemical import ChemicalProcessor
from glm.nlg.processors.biological import BiologicalProcessor
from glm.nlg.processors.physical import PhysicalProcessor
from glm.nlg.processors.computational import ComputationalProcessor
from glm.nlg.processors.etymological import EtymologicalProcessor


@pytest.fixture(autouse=True)
def _clear_http_cache():
    """Clear HTTP cache before each test."""
    clear_cache()
    yield
    clear_cache()


@pytest.fixture
def substrate():
    return MnemoSubstrate()


@pytest.fixture
def encoder(substrate):
    return MnemoEncoder(substrate=substrate)


def _mock_urlopen(data):
    """Create a mock for urllib.request.urlopen returning JSON data."""
    mock_resp = MagicMock()
    mock_resp.read.return_value = json.dumps(data).encode("utf-8")
    mock_resp.__enter__ = lambda s: s
    mock_resp.__exit__ = MagicMock(return_value=False)
    return mock_resp


# ---------------------------------------------------------------------------
# Infrastructure
# ---------------------------------------------------------------------------


class TestHTTPHelper:
    def test_fetch_json_returns_none_on_failure(self):
        with patch("glm.nlg.processors._http.urllib.request.urlopen", side_effect=Exception("no network")):
            result = fetch_json("https://example.com/api")
            assert result is None

    def test_fetch_json_caches_results(self):
        mock_resp = _mock_urlopen({"key": "value"})
        with patch("glm.nlg.processors._http.urllib.request.urlopen", return_value=mock_resp) as mock_open:
            r1 = fetch_json("https://example.com/cached")
            r2 = fetch_json("https://example.com/cached")
            assert r1 == {"key": "value"}
            assert r2 == {"key": "value"}
            # Only one actual HTTP call
            assert mock_open.call_count == 1


class TestDispatcher:
    def test_create_default_dispatcher(self):
        dispatcher = create_default_dispatcher()
        assert len(dispatcher.domains) == 7
        assert "mathematical" in dispatcher.domains
        assert "physical" in dispatcher.domains

    def test_unknown_domain_returns_empty(self):
        dispatcher = ProcessorDispatcher()
        result = dispatcher.process("nonexistent", "text", MnemoSequence([]))
        assert result == {}

    def test_processor_exception_returns_empty(self):
        class BadProcessor(DomainProcessor):
            domain = "bad"
            def process(self, text, mnemo_seq):
                raise RuntimeError("boom")

        dispatcher = ProcessorDispatcher()
        dispatcher.register(BadProcessor())
        result = dispatcher.process("bad", "text", MnemoSequence([]))
        assert result == {}


# ---------------------------------------------------------------------------
# Mathematical
# ---------------------------------------------------------------------------


class TestMathProcessor:
    def test_process_with_newton_api(self, encoder):
        seq = encoder.encode("solve x + 2 = 5")
        proc = MathProcessor()
        mock_resp = _mock_urlopen({
            "operation": "zeroes",
            "expression": "x + 2 = 5",
            "result": "3",
        })
        with patch("glm.nlg.processors._http.urllib.request.urlopen", return_value=mock_resp):
            slots = proc.process("solve x + 2 = 5", seq)
            assert slots["result"] == "3"
            assert slots["operation"] == "zeroes"
            assert "expression" in slots

    def test_process_api_failure_returns_expression(self, encoder):
        seq = encoder.encode("simplify 2 + 2")
        proc = MathProcessor()
        with patch("glm.nlg.processors._http.urllib.request.urlopen", side_effect=Exception):
            slots = proc.process("simplify 2 + 2", seq)
            assert "operation" in slots
            assert slots["operation"] == "simplify"


# ---------------------------------------------------------------------------
# Linguistic
# ---------------------------------------------------------------------------


class TestLinguisticProcessor:
    def test_process_with_dictionary_api(self, encoder):
        seq = encoder.encode("the verb agrees with the noun")
        proc = LinguisticProcessor()
        mock_resp = _mock_urlopen([{
            "word": "agrees",
            "phonetic": "/əˈɡriːz/",
            "meanings": [{
                "partOfSpeech": "verb",
                "definitions": [{"definition": "to have the same opinion"}],
            }],
        }])
        with patch("glm.nlg.processors._http.urllib.request.urlopen", return_value=mock_resp):
            slots = proc.process("the verb agrees with the noun", seq)
            assert slots["word"] == "agrees"
            assert slots["part_of_speech"] == "verb"
            assert "definition" in slots
            assert slots["role"] == "verb"

    def test_process_api_failure_returns_word(self, encoder):
        seq = encoder.encode("the verb agrees")
        proc = LinguisticProcessor()
        with patch("glm.nlg.processors._http.urllib.request.urlopen", side_effect=Exception):
            slots = proc.process("the verb agrees", seq)
            assert "word" in slots
            assert "analysis" in slots


# ---------------------------------------------------------------------------
# Chemical
# ---------------------------------------------------------------------------


class TestChemicalProcessor:
    def test_process_with_pubchem_api(self, encoder):
        seq = encoder.encode("the molecule bonds with oxygen")
        proc = ChemicalProcessor()
        mock_resp = _mock_urlopen({
            "PropertyTable": {
                "Properties": [{
                    "MolecularFormula": "O2",
                    "MolecularWeight": 31.998,
                    "IUPACName": "molecular oxygen",
                }]
            }
        })
        with patch("glm.nlg.processors._http.urllib.request.urlopen", return_value=mock_resp):
            slots = proc.process("the molecule bonds with oxygen", seq)
            assert slots["compound"] == "oxygen"
            assert slots["formula"] == "O2"
            assert "molecular_weight" in slots

    def test_process_api_failure_returns_compound(self, encoder):
        seq = encoder.encode("water molecule")
        proc = ChemicalProcessor()
        with patch("glm.nlg.processors._http.urllib.request.urlopen", side_effect=Exception):
            slots = proc.process("water molecule", seq)
            assert slots["compound"] == "water"


# ---------------------------------------------------------------------------
# Biological
# ---------------------------------------------------------------------------


class TestBiologicalProcessor:
    def test_process_with_uniprot_api(self, encoder):
        seq = encoder.encode("the gene encodes a protein")
        proc = BiologicalProcessor()
        mock_resp = _mock_urlopen({
            "results": [{
                "genes": [{"geneName": {"value": "TP53"}}],
                "proteinDescription": {
                    "recommendedName": {
                        "fullName": {"value": "Cellular tumor antigen p53"},
                    }
                },
                "organism": {"scientificName": "Homo sapiens"},
            }]
        })
        with patch("glm.nlg.processors._http.urllib.request.urlopen", return_value=mock_resp):
            slots = proc.process("the gene encodes a protein", seq)
            assert slots["gene"] == "TP53"
            assert "p53" in slots["protein"].lower() or "p53" in slots["protein"]
            assert slots["organism"] == "Homo sapiens"

    def test_process_api_failure_returns_term(self, encoder):
        seq = encoder.encode("the gene mutates")
        proc = BiologicalProcessor()
        with patch("glm.nlg.processors._http.urllib.request.urlopen", side_effect=Exception):
            slots = proc.process("the gene mutates", seq)
            assert slots["gene"] == "gene"


# ---------------------------------------------------------------------------
# Physical (no API — built-in constants)
# ---------------------------------------------------------------------------


class TestPhysicalProcessor:
    def test_speed_of_light(self, encoder):
        seq = encoder.encode("the speed of light")
        proc = PhysicalProcessor()
        slots = proc.process("the speed of light", seq)
        assert slots["symbol"] == "c"
        assert slots["value"] == "299792458"
        assert slots["unit"] == "m/s"
        assert slots["quantity"] == "speed of light"

    def test_gravity(self, encoder):
        seq = encoder.encode("what is gravity")
        proc = PhysicalProcessor()
        slots = proc.process("what is gravity", seq)
        assert slots["symbol"] == "g"
        assert "9.80665" in slots["value"]

    def test_unknown_concept_returns_description(self, encoder):
        seq = encoder.encode("what is dark matter")
        proc = PhysicalProcessor()
        slots = proc.process("what is dark matter", seq)
        assert "description" in slots


# ---------------------------------------------------------------------------
# Computational
# ---------------------------------------------------------------------------


class TestComputationalProcessor:
    def test_number_fact(self, encoder):
        seq = encoder.encode("the number 42")
        proc = ComputationalProcessor()
        slots = proc.process("the number 42", seq)
        assert "42" in slots.get("result", "")
        assert "Catalan" in slots.get("fact", "")

    def test_prime_detection(self, encoder):
        seq = encoder.encode("number 17")
        proc = ComputationalProcessor()
        slots = proc.process("number 17", seq)
        assert "prime" in slots.get("fact", "").lower() or "prime" in slots.get("result", "").lower()

    def test_algorithm_concept(self, encoder):
        seq = encoder.encode("the algorithm sorts the list")
        proc = ComputationalProcessor()
        slots = proc.process("the algorithm sorts the list", seq)
        assert slots["concept"] == "sort"
        assert "O(n log n)" in slots["complexity"]


# ---------------------------------------------------------------------------
# Etymological
# ---------------------------------------------------------------------------


class TestEtymologicalProcessor:
    def test_process_with_datamuse_api(self, encoder):
        seq = encoder.encode("from latin root")
        proc = EtymologicalProcessor()
        mock_resp = _mock_urlopen([{
            "word": "latin",
            "score": 100,
            "defs": ["n\trelating to the language of ancient Rome"],
        }])
        with patch("glm.nlg.processors._http.urllib.request.urlopen", return_value=mock_resp):
            slots = proc.process("from latin root", seq)
            assert slots["word"] == "latin"
            assert slots["origin"] == "Latin"
            assert "definition" in slots

    def test_root_language_detection(self, encoder):
        seq = encoder.encode("derived from greek")
        proc = EtymologicalProcessor()
        with patch("glm.nlg.processors._http.urllib.request.urlopen", side_effect=Exception):
            slots = proc.process("derived from greek", seq)
            assert slots["origin"] == "Greek"


# ---------------------------------------------------------------------------
# Full pipeline integration (mocked APIs)
# ---------------------------------------------------------------------------


class TestProcessorPipelineIntegration:
    """Test that processors wire correctly through the full NLG pipeline."""

    def test_math_through_pipeline(self, encoder):
        from glm.nlg.decoder import MnemoDecoder
        decoder = MnemoDecoder(substrate=encoder.substrate)
        dispatcher = create_default_dispatcher()

        text = "solve x + 2 = 5"
        seq = encoder.encode(text)
        domain = encoder.detect_domain(text)

        mock_resp = _mock_urlopen({
            "operation": "zeroes",
            "expression": "x + 2 = 5",
            "result": "3",
        })
        with patch("glm.nlg.processors._http.urllib.request.urlopen", return_value=mock_resp):
            api_slots = dispatcher.process(domain, text, seq)
            extra = {"content": text}
            extra.update(api_slots)
            output = decoder.decode(seq, language="en", extra_slots=extra)
            assert "3" in output

    def test_physics_through_pipeline(self, encoder):
        from glm.nlg.decoder import MnemoDecoder
        decoder = MnemoDecoder(substrate=encoder.substrate)
        dispatcher = create_default_dispatcher()

        text = "speed of light and energy"
        seq = encoder.encode(text)
        domain = encoder.detect_domain(text)

        api_slots = dispatcher.process(domain, text, seq)
        extra = {"content": text}
        extra.update(api_slots)
        output = decoder.decode(seq, language="en", extra_slots=extra)
        assert isinstance(output, str)
        assert len(output) > 0
