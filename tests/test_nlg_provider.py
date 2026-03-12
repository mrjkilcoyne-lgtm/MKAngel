"""Tests for the NLGProvider."""

import pytest
from app.providers import NLGProvider


class TestNLGProvider:
    def test_create_provider(self):
        provider = NLGProvider()
        assert provider.name == "nlg"

    def test_is_available(self):
        provider = NLGProvider()
        assert provider.is_available() is True

    def test_generate_returns_string(self):
        provider = NLGProvider()
        result = provider.generate("What is water?")
        assert isinstance(result, str)
        assert len(result) > 0

    def test_generate_with_domain_hint(self):
        provider = NLGProvider()
        result = provider.generate("the molecule bonds with oxygen")
        assert isinstance(result, str)

    def test_generate_includes_evidential(self):
        provider = NLGProvider()
        result = provider.generate("solve x + 2 = 5")
        assert isinstance(result, str)
        assert len(result) > 0
