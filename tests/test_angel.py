"""Tests for the Angel — the heart of MKAngel."""

import tempfile
import os

from glm.angel import Angel, AngelConfig


class TestAngel:
    def test_create_angel(self):
        angel = Angel()
        assert repr(angel).startswith("Angel(dormant")

    def test_awaken(self):
        angel = Angel()
        angel.awaken()
        assert repr(angel).startswith("Angel(awake")

    def test_introspect(self):
        angel = Angel()
        angel.awaken()
        info = angel.introspect()
        assert info["self_referential"] is True
        assert info["total_grammars"] > 0
        assert info["total_rules"] > 0
        assert len(info["domains_loaded"]) > 0

    def test_predict(self):
        angel = Angel()
        angel.awaken()
        predictions = angel.predict(["the", "cat"], domain="linguistic")
        assert isinstance(predictions, list)

    def test_reconstruct(self):
        angel = Angel()
        angel.awaken()
        reconstructions = angel.reconstruct(
            ["water"], domain="linguistic"
        )
        assert isinstance(reconstructions, list)

    def test_superforecast(self):
        angel = Angel()
        angel.awaken()
        forecast = angel.superforecast(
            ["the", "pattern"],
            context={"domain_hint": "linguistic"},
            domain="linguistic",
        )
        assert "predictions" in forecast
        assert "strange_loops" in forecast
        assert "reasoning" in forecast

    def test_compose_fugue(self):
        angel = Angel()
        angel.awaken()
        fugue = angel.compose_fugue(
            theme=["pattern", "transform", "repeat"],
        )
        assert "voices" in fugue
        assert "harmonics" in fugue
        assert fugue["num_voices"] > 0

    def test_translate(self):
        angel = Angel()
        angel.awaken()
        translations = angel.translate(
            ["bond", "transform"],
            source_domain="chemical",
            target_domain="linguistic",
        )
        assert isinstance(translations, list)

    def test_save_and_load(self):
        angel = Angel()
        angel.awaken()
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".json", delete=False
        ) as f:
            path = f.name
        try:
            angel.save_state(path)
            loaded = Angel.load_state(path)
            assert repr(loaded).startswith("Angel(awake")
        finally:
            os.unlink(path)

    def test_auto_awaken(self):
        """Angel auto-awakens when needed — a form of self-reference."""
        angel = Angel()
        # predict should trigger awaken
        predictions = angel.predict(["test"])
        assert isinstance(predictions, list)
