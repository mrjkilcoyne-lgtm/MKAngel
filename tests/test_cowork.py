"""Tests for the multi-agent collaboration system."""

from unittest.mock import MagicMock

from app.cowork import AgentResult, CoworkResult, CoworkSession


class TestAgentResult:
    def test_create_result(self):
        r = AgentResult("linguistic", "output", [], 0, 1.0)
        assert r.domain == "linguistic"
        assert r.error is None

    def test_create_result_with_error(self):
        r = AgentResult("math", "err", [], 0, 0.5, error="failed")
        assert r.error == "failed"

    def test_skills_used_default_empty(self):
        r = AgentResult("test", "", [], 0, 0.0)
        assert r.skills_used == []


class TestCoworkResult:
    def test_create_result(self):
        r = CoworkResult("task", [], [], "synthesis", 1.0)
        assert r.task == "task"
        assert r.total_duration == 1.0


class TestCoworkSession:
    def test_create_session(self):
        s = CoworkSession()
        assert repr(s) == "CoworkSession(idle, agents=0)"

    def test_start_session(self):
        s = CoworkSession()
        msg = s.start_session("test task")
        assert "Cowork session started" in msg
        assert "linguistic" in msg

    def test_add_agent(self):
        s = CoworkSession()
        msg = s.add_agent("mathematical")
        assert "Agent added" in msg

    def test_add_duplicate_agent(self):
        s = CoworkSession()
        s.add_agent("linguistic")
        msg = s.add_agent("linguistic")
        assert "already exists" in msg

    def test_remove_agent(self):
        s = CoworkSession()
        s.add_agent("physics")
        msg = s.remove_agent("physics")
        assert "Agent removed" in msg

    def test_remove_nonexistent(self):
        s = CoworkSession()
        msg = s.remove_agent("nope")
        assert "No agent found" in msg

    def test_list_agents(self):
        s = CoworkSession()
        s.add_agent("linguistic")
        s.add_agent("computational")
        agents = s.list_agents()
        assert len(agents) == 2
        assert agents[0]["domain"] == "linguistic"

    def test_list_agents_has_skilled_field(self):
        s = CoworkSession()
        s.add_agent("linguistic")
        agents = s.list_agents()
        assert "skilled" in agents[0]

    def test_run_no_task(self):
        s = CoworkSession()
        result = s.run()
        assert "No task specified" in result.synthesis

    def test_run_with_task(self):
        s = CoworkSession()
        s.start_session("analyse grammar patterns")
        result = s.run()
        assert isinstance(result, CoworkResult)
        assert result.task == "analyse grammar patterns"
        assert len(result.agents) > 0

    def test_run_override_task(self):
        s = CoworkSession()
        s.start_session("original task")
        result = s.run("new task")
        assert result.task == "new task"

    def test_collect_results_alias(self):
        s = CoworkSession()
        s.start_session("test")
        r1 = s.run()
        # collect_results should work the same
        r2 = s.collect_results()
        assert isinstance(r2, CoworkResult)

    def test_run_with_provider(self):
        provider = MagicMock()
        provider.generate.return_value = "enriched analysis"
        s = CoworkSession(provider=provider)
        s.start_session("test")
        result = s.run()
        assert isinstance(result, CoworkResult)

    def test_synthesis_without_provider(self):
        s = CoworkSession()
        s.start_session("test grammar")
        result = s.run()
        assert "Task:" in result.synthesis
        assert "predictions" in result.synthesis.lower() or "Agents:" in result.synthesis

    def test_agents_have_duration(self):
        s = CoworkSession()
        s.start_session("test")
        result = s.run()
        for agent in result.agents:
            assert agent.duration >= 0
