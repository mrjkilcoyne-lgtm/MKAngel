"""Tests for English surface templates."""

import pytest
from glm.nlg.templates import TemplateRegistry
from glm.nlg.templates.en import ENGLISH_TEMPLATES, register_english


class TestTemplateRegistry:
    def test_create_registry(self):
        reg = TemplateRegistry()
        assert len(reg) == 0

    def test_register_english(self):
        reg = TemplateRegistry()
        register_english(reg)
        assert len(reg) > 0

    def test_lookup_by_domain(self):
        reg = TemplateRegistry()
        register_english(reg)
        templates = reg.for_domain("mathematical")
        assert len(templates) > 0

    def test_lookup_by_domain_and_pattern(self):
        reg = TemplateRegistry()
        register_english(reg)
        templates = reg.for_domain("mathematical")
        assert any("result" in t.slots for t in templates)


class TestEvidentialTemplates:
    def test_english_hedging_for_inference(self):
        reg = TemplateRegistry()
        register_english(reg)
        templates = reg.for_evidential("inf", "prob")
        assert len(templates) > 0
        assert any("suggest" in t.template or "evidence" in t.template for t in templates)

    def test_english_certainty_for_observed(self):
        reg = TemplateRegistry()
        register_english(reg)
        templates = reg.for_evidential("obs", "cert")
        assert len(templates) > 0


class TestEnglishTemplates:
    def test_all_seven_domains_covered(self):
        reg = TemplateRegistry()
        register_english(reg)
        for domain in ("mathematical", "linguistic", "biological",
                       "chemical", "physical", "computational", "etymological"):
            templates = reg.for_domain(domain)
            assert len(templates) > 0, f"No templates for {domain}"
