"""Tests for the skill management system."""

import json
import tempfile
from pathlib import Path

from app.skills import Skill, SkillManager


class TestSkill:
    def test_create_skill(self):
        s = Skill("test", "A test skill", "test_trigger", "prompt")
        assert s.name == "test"
        assert s.enabled is True

    def test_matches_keyword(self):
        s = Skill("test", "", "summarize", "prompt")
        assert s.matches("please summarize this") is True
        assert s.matches("do something else") is False

    def test_matches_multi_word(self):
        s = Skill("test", "", "predict next", "prompt")
        assert s.matches("predict the next element") is True
        assert s.matches("predict something") is False

    def test_to_dict(self):
        s = Skill("test", "desc", "trigger", "prompt", {"key": "val"})
        d = s.to_dict()
        assert d["name"] == "test"
        assert d["config"]["key"] == "val"

    def test_from_dict(self):
        data = {"name": "test", "description": "desc", "trigger": "go",
                "action": "transform", "config": {"transform": "uppercase"}}
        s = Skill.from_dict(data)
        assert s.name == "test"
        assert s.action == "transform"

    def test_from_dict_defaults(self):
        s = Skill.from_dict({})
        assert s.name == "unnamed"
        assert s.action == "prompt"
        assert s.enabled is True


class TestSkillManager:
    def _make_manager(self):
        d = tempfile.mkdtemp()
        return SkillManager(skills_dir=d), d

    def test_create_manager(self):
        mgr, _ = self._make_manager()
        skills = mgr.list_skills()
        # Should have 4 defaults: summarize, translate, analyze, predict
        assert len(skills) >= 4

    def test_default_skills_present(self):
        mgr, _ = self._make_manager()
        names = [s.name for s in mgr.list_skills()]
        assert "summarize" in names
        assert "translate" in names
        assert "analyze" in names
        assert "predict" in names

    def test_create_skill(self):
        mgr, _ = self._make_manager()
        s = mgr.create_skill("custom", "custom trigger", "prompt", "My skill")
        assert s.name == "custom"
        assert mgr.get_skill("custom") is not None

    def test_create_duplicate_raises(self):
        mgr, _ = self._make_manager()
        mgr.create_skill("unique", "trigger")
        try:
            mgr.create_skill("unique", "trigger")
            assert False, "Should have raised ValueError"
        except ValueError:
            pass

    def test_delete_skill(self):
        mgr, _ = self._make_manager()
        mgr.create_skill("deleteme", "trigger")
        assert mgr.delete_skill("deleteme") is True
        assert mgr.get_skill("deleteme") is None

    def test_delete_nonexistent(self):
        mgr, _ = self._make_manager()
        assert mgr.delete_skill("nope") is False

    def test_find_matching_skills(self):
        mgr, _ = self._make_manager()
        matches = mgr.find_matching_skills("please summarize this text")
        names = [s.name for s in matches]
        assert "summarize" in names

    def test_toggle_skill(self):
        mgr, _ = self._make_manager()
        state = mgr.toggle_skill("summarize")
        assert state is False  # was True, now False
        state = mgr.toggle_skill("summarize")
        assert state is True  # back to True

    def test_toggle_nonexistent(self):
        mgr, _ = self._make_manager()
        assert mgr.toggle_skill("nope") is None

    def test_execute_skill_not_found(self):
        mgr, _ = self._make_manager()
        result = mgr.execute_skill("nonexistent", "test")
        assert "not found" in result.lower()

    def test_execute_transform_skill(self):
        mgr, _ = self._make_manager()
        mgr.create_skill("upper", "upper", "transform", config={"transform": "uppercase"})
        result = mgr.execute_skill("upper", "hello")
        assert result == "HELLO"

    def test_execute_prompt_skill_no_provider(self):
        mgr, _ = self._make_manager()
        result = mgr.execute_skill("summarize", "test input")
        assert isinstance(result, str)
        assert len(result) > 0

    def test_persistence(self):
        d = tempfile.mkdtemp()
        mgr1 = SkillManager(skills_dir=d)
        mgr1.create_skill("persistent", "trigger", "prompt", "Test persistence")

        # Create new manager pointing to same dir
        mgr2 = SkillManager(skills_dir=d)
        assert mgr2.get_skill("persistent") is not None

    def test_skill_files_are_json(self):
        d = tempfile.mkdtemp()
        mgr = SkillManager(skills_dir=d)
        mgr.create_skill("jsontest", "trigger")
        path = Path(d) / "jsontest.json"
        assert path.exists()
        data = json.loads(path.read_text())
        assert data["name"] == "jsontest"
