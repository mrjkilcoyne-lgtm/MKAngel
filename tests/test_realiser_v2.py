import pytest
from glm.realiser_v2 import GenerativeRealiser


def test_realiser_streams_tokens():
    r = GenerativeRealiser()
    tokens = list(r.stream(None, "What is gravity?"))
    assert len(tokens) > 5
    full = "".join(tokens)
    assert len(full) > 50


def test_realiser_produces_long_form():
    r = GenerativeRealiser()
    # Simulate a rich pipeline result with multiple claims
    from glm.pipeline import ReasoningPipeline
    pipeline = ReasoningPipeline()
    result = pipeline.run("Explain the relationship between etymology and molecular biology")
    tokens = list(r.stream(result, "Explain the relationship between etymology and molecular biology"))
    full = "".join(tokens)
    # Should produce substantial output for a cross-domain question
    assert len(full) > 200


def test_realiser_includes_connectives():
    r = GenerativeRealiser()
    from glm.pipeline import ReasoningPipeline
    pipeline = ReasoningPipeline()
    result = pipeline.run("Compare photosynthesis and cellular respiration")
    full = "".join(r.stream(result, "Compare photosynthesis and cellular respiration"))
    # Should have structural connectives
    has_connective = any(c in full for c in ["Furthermore", "However", "Therefore", "Additionally", "In contrast"])
    assert has_connective


def test_realiser_ends_with_signature():
    r = GenerativeRealiser()
    full = "".join(r.stream(None, "Hello"))
    assert "CANZUK-AI" in full
