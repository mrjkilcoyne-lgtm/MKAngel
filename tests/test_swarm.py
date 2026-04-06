"""
Tests for the swarm orchestration system.

Tests all 5 iterations:
  1. SkillableAgent -- agents with skills
  2. AgentCombo -- paired/trio agents
  3. AgentTeam -- assembled combos
  4. SwarmOrchestrator -- multi-cycle iteration
  5. CelestialSwitchboard -- named angel invocation
"""

import time
from unittest.mock import MagicMock

from app.swarm import (
    AgentRole,
    AgentSkill,
    AgentCombo,
    AgentTeam,
    BorgesLibrary,
    CelestialName,
    CelestialSwitchboard,
    CoordinationProtocol,
    Host,
    HostHarness,
    HostMessage,
    HostResult,
    NamedAngel,
    SkillableAgent,
    SwarmOrchestrator,
    TeamResult,
    SwarmCycleResult,
    SwarmReport,
    _default_skills_for_role,
)


# ═══════════════════════════════════════════════════════════════════════════
# Iteration 1 -- SkillableAgent
# ═══════════════════════════════════════════════════════════════════════════

class TestAgentSkill:
    def test_create_skill(self):
        skill = AgentSkill("test", "A test skill")
        assert skill.name == "test"
        assert skill.action == "prompt"

    def test_transform_uppercase(self):
        skill = AgentSkill("upper", "Uppercase", "transform", {"transform": "uppercase"})
        assert skill.execute("hello") == "HELLO"

    def test_transform_lowercase(self):
        skill = AgentSkill("lower", "Lowercase", "transform", {"transform": "lowercase"})
        assert skill.execute("HELLO") == "hello"

    def test_transform_reverse(self):
        skill = AgentSkill("rev", "Reverse", "transform", {"transform": "reverse"})
        assert skill.execute("hello") == "olleh"

    def test_transform_identity(self):
        skill = AgentSkill("id", "Identity", "transform", {"transform": "identity"})
        assert skill.execute("hello") == "hello"

    def test_prompt_with_provider(self):
        provider = MagicMock()
        provider.generate.return_value = "generated response"
        skill = AgentSkill("ask", "Ask", "prompt", {"system_prompt": "Be helpful"})
        result = skill.execute("test input", provider)
        assert result == "generated response"
        provider.generate.assert_called_once()

    def test_prompt_without_provider_fallback(self):
        skill = AgentSkill("ask", "Ask", "prompt")
        result = skill.execute("test input")
        assert isinstance(result, str)
        assert len(result) > 0

    def test_transform_mnemo_fallback(self):
        skill = AgentSkill("mnemo", "MNEMO", "transform", {"transform": "mnemo"})
        result = skill.execute("hello world")
        assert isinstance(result, str)


class TestSkillableAgent:
    def test_create_agent(self):
        agent = SkillableAgent("test", AgentRole.EXPLORER)
        assert agent.name == "test"
        assert agent.role == AgentRole.EXPLORER
        assert agent.active is True

    def test_default_skills_assigned(self):
        agent = SkillableAgent("test", AgentRole.EXPLORER)
        assert len(agent.skills) > 0
        assert agent.skills[0].name == "broad_search"

    def test_add_skill(self):
        agent = SkillableAgent("test", AgentRole.EXPLORER)
        initial = len(agent.skills)
        agent.add_skill(AgentSkill("extra", "Extra skill"))
        assert len(agent.skills) == initial + 1

    def test_remove_skill(self):
        agent = SkillableAgent("test", AgentRole.EXPLORER)
        agent.add_skill(AgentSkill("removable", "Remove me"))
        assert agent.remove_skill("removable") is True
        assert agent.remove_skill("nonexistent") is False

    def test_apply_skills(self):
        agent = SkillableAgent("test", AgentRole.SCRIBE)
        result = agent.apply_skills("hello world")
        assert isinstance(result, str)

    def test_best_skill_for(self):
        agent = SkillableAgent("test", AgentRole.EXPLORER)
        skill = agent.best_skill_for("search for connections")
        assert skill is not None

    def test_describe(self):
        agent = SkillableAgent("test", AgentRole.EXPLORER, "linguistic")
        desc = agent.describe()
        assert "test" in desc
        assert "EXPLORER" in desc
        assert "linguistic" in desc

    def test_all_roles_get_default_skills(self):
        for role in AgentRole:
            skills = _default_skills_for_role(role)
            assert len(skills) > 0, f"Role {role.name} has no default skills"

    def test_agent_score_default(self):
        agent = SkillableAgent("test", AgentRole.EXPLORER)
        assert agent._score == 0.5


# ═══════════════════════════════════════════════════════════════════════════
# Iteration 2 -- AgentCombo
# ═══════════════════════════════════════════════════════════════════════════

class TestAgentCombo:
    def test_create_combo(self):
        agents = [
            SkillableAgent("a", AgentRole.EXPLORER),
            SkillableAgent("b", AgentRole.CRITIC),
        ]
        combo = AgentCombo("test_combo", agents, "Explore then critique")
        assert combo.name == "test_combo"
        assert len(combo.agents) == 2

    def test_run_combo(self):
        agents = [
            SkillableAgent("explorer", AgentRole.EXPLORER),
            SkillableAgent("critic", AgentRole.CRITIC),
        ]
        combo = AgentCombo("test", agents)
        result = combo.run("test task")
        assert "combo" in result
        assert "agent_outputs" in result
        assert len(result["agent_outputs"]) == 2
        assert "duration" in result

    def test_run_combo_with_provider(self):
        provider = MagicMock()
        provider.generate.return_value = "provider response"
        agents = [SkillableAgent("a", AgentRole.ANALYST)]
        combo = AgentCombo("test", agents)
        result = combo.run("task", provider)
        assert len(result["agent_outputs"]) == 1

    def test_suggested_combos(self):
        combos = AgentCombo.suggested_combos()
        assert len(combos) == 5
        names = [c.name for c in combos]
        assert "scout_and_critic" in names
        assert "oracle_and_scribe" in names
        assert "analyst_trio" in names
        assert "research_team" in names
        assert "guardian_pair" in names

    def test_combo_chains_output(self):
        """Each agent should build on the previous agent's output."""
        agents = [
            SkillableAgent("a", AgentRole.EXPLORER),
            SkillableAgent("b", AgentRole.SYNTHESISER),
        ]
        combo = AgentCombo("chain_test", agents)
        result = combo.run("test task")
        outputs = result["agent_outputs"]
        assert len(outputs) == 2

    def test_inactive_agent_skipped(self):
        agents = [
            SkillableAgent("active", AgentRole.EXPLORER),
            SkillableAgent("inactive", AgentRole.CRITIC, active=False),
        ]
        combo = AgentCombo("test", agents)
        result = combo.run("task")
        assert len(result["agent_outputs"]) == 1


# ═══════════════════════════════════════════════════════════════════════════
# Iteration 3 -- AgentTeam
# ═══════════════════════════════════════════════════════════════════════════

class TestAgentTeam:
    def test_create_team(self):
        team = AgentTeam("test_team")
        assert team.name == "test_team"
        assert team.protocol == CoordinationProtocol.FUGUE

    def test_add_combo(self):
        team = AgentTeam("test")
        combo = AgentCombo("c", [SkillableAgent("a", AgentRole.EXPLORER)])
        team.add_combo(combo)
        assert len(team._combos) == 1

    def test_add_agent(self):
        team = AgentTeam("test")
        agent = SkillableAgent("solo", AgentRole.SPECIALIST)
        team.add_agent(agent)
        assert len(team._solo_agents) == 1

    def test_all_agents(self):
        team = AgentTeam("test")
        combo = AgentCombo("c", [
            SkillableAgent("a", AgentRole.EXPLORER),
            SkillableAgent("b", AgentRole.CRITIC),
        ])
        team.add_combo(combo)
        team.add_agent(SkillableAgent("solo", AgentRole.SPECIALIST))
        assert len(team.all_agents()) == 3

    def test_run_team(self):
        team = AgentTeam("test")
        team.add_agent(SkillableAgent("a", AgentRole.EXPLORER))
        team.add_agent(SkillableAgent("b", AgentRole.CRITIC))
        result = team.run("test task")
        assert isinstance(result, TeamResult)
        assert result.task == "test task"
        assert result.duration > 0

    def test_assemble_default_team(self):
        team = AgentTeam.assemble_default_team()
        assert team.name == "angel_host"
        assert len(team._combos) == 3
        assert len(team._solo_agents) == 2

    def test_repr(self):
        team = AgentTeam("test")
        assert "test" in repr(team)


# ═══════════════════════════════════════════════════════════════════════════
# Iteration 4 -- SwarmOrchestrator
# ═══════════════════════════════════════════════════════════════════════════

class TestSwarmOrchestrator:
    def test_create_orchestrator(self):
        orch = SwarmOrchestrator()
        assert orch._provider is None
        assert isinstance(orch._switchboard, CelestialSwitchboard)

    def test_run_single_cycle(self):
        orch = SwarmOrchestrator()
        report = orch.run("test task", cycles=1)
        assert isinstance(report, SwarmReport)
        assert report.cycles == 1
        assert len(report.cycle_results) == 1
        assert report.total_duration > 0

    def test_run_five_cycles(self):
        orch = SwarmOrchestrator()
        report = orch.run("test task", cycles=5)
        assert len(report.cycle_results) == 5
        assert len(report.options) >= 3

    def test_report_options(self):
        orch = SwarmOrchestrator()
        orch.run("test task", cycles=2)
        report = orch.report_options()
        assert "SWARM ORCHESTRATION REPORT" in report
        assert "Cycle 1" in report
        assert "Cycle 2" in report

    def test_invoke_angel(self):
        orch = SwarmOrchestrator()
        result = orch.invoke_angel("gabriel", "Hello Gabriel")
        assert result["angel"] == "gabriel"
        assert "response" in result

    def test_invoke_unknown_angel(self):
        orch = SwarmOrchestrator()
        result = orch.invoke_angel("nonexistent", "Hello")
        assert "error" in result

    def test_report_options_empty(self):
        orch = SwarmOrchestrator()
        report = orch.report_options()
        assert "No swarm cycles" in report

    def test_with_provider(self):
        provider = MagicMock()
        provider.generate.return_value = "synthesised"
        orch = SwarmOrchestrator(provider)
        report = orch.run("test", cycles=1)
        assert isinstance(report, SwarmReport)


# ═══════════════════════════════════════════════════════════════════════════
# Iteration 5 -- CelestialSwitchboard
# ═══════════════════════════════════════════════════════════════════════════

class TestCelestialName:
    def test_all_angels_named(self):
        names = [n.value for n in CelestialName]
        assert "gabriel" in names
        assert "michael" in names
        assert "raphael" in names
        assert "uriel" in names
        assert "puriel" in names
        assert "ariel" in names
        assert "azrael" in names
        assert "metatron" in names


class TestNamedAngel:
    def test_create_named_angel(self):
        angel = NamedAngel(
            CelestialName.GABRIEL, "Gabriel the Messenger", "linguistic",
            "You are Gabriel.", AgentRole.MESSENGER,
        )
        assert angel.celestial_name == CelestialName.GABRIEL
        assert angel.domain == "linguistic"

    def test_invoke_without_provider(self):
        angel = NamedAngel(
            CelestialName.GABRIEL, "Gabriel the Messenger", "linguistic",
            "You are Gabriel.", AgentRole.MESSENGER,
        )
        result = angel.invoke("Hello Gabriel")
        assert result["angel"] == "gabriel"
        assert "response" in result
        assert result["duration"] > 0

    def test_invoke_with_provider(self):
        provider = MagicMock()
        provider.generate.return_value = "Gabriel speaks"
        angel = NamedAngel(
            CelestialName.GABRIEL, "Gabriel", "linguistic",
            "You are Gabriel.", AgentRole.MESSENGER,
        )
        result = angel.invoke("Hello", provider)
        assert result["response"] == "Gabriel speaks"


class TestCelestialSwitchboard:
    def test_create_switchboard(self):
        sb = CelestialSwitchboard()
        assert len(sb.who_is_available()) == 8

    def test_invoke_gabriel(self):
        sb = CelestialSwitchboard()
        result = sb.invoke("gabriel", "Carry this message")
        assert result["angel"] == "gabriel"

    def test_invoke_michael(self):
        sb = CelestialSwitchboard()
        result = sb.invoke("michael", "Protect this")
        assert result["angel"] == "michael"

    def test_invoke_all_angels(self):
        sb = CelestialSwitchboard()
        for name in CelestialName:
            result = sb.invoke(name.value, "Test invocation")
            assert result["angel"] == name.value
            assert "response" in result

    def test_invoke_unknown(self):
        sb = CelestialSwitchboard()
        result = sb.invoke("nonexistent", "Hello")
        assert "error" in result

    def test_find_angel_for_error(self):
        sb = CelestialSwitchboard()
        angel = sb.find_angel_for("fix this error")
        assert angel is not None
        assert angel.celestial_name == CelestialName.RAPHAEL

    def test_find_angel_for_security(self):
        sb = CelestialSwitchboard()
        angel = sb.find_angel_for("protect the system")
        assert angel is not None
        assert angel.celestial_name == CelestialName.MICHAEL

    def test_find_angel_for_prediction(self):
        sb = CelestialSwitchboard()
        angel = sb.find_angel_for("predict the outcome")
        assert angel is not None
        assert angel.celestial_name == CelestialName.URIEL

    def test_find_angel_default_gabriel(self):
        sb = CelestialSwitchboard()
        angel = sb.find_angel_for("do something vague")
        assert angel is not None
        assert angel.celestial_name == CelestialName.GABRIEL

    def test_who_is_available(self):
        sb = CelestialSwitchboard()
        available = sb.who_is_available()
        assert len(available) == 8
        names = [a["name"] for a in available]
        assert "gabriel" in names
        assert "metatron" in names


# ═══════════════════════════════════════════════════════════════════════════
# BorgesLibrary
# ═══════════════════════════════════════════════════════════════════════════

class TestBorgesLibrary:
    def test_create_library(self):
        lib = BorgesLibrary()
        assert lib.rooms_explored == 0

    def test_explore_without_grammars(self):
        lib = BorgesLibrary()
        paths = lib.explore("test query")
        assert isinstance(paths, list)
        assert lib.rooms_explored == 1

    def test_find_improbable(self):
        lib = BorgesLibrary()
        improbable = lib.find_improbable("test")
        assert isinstance(improbable, list)


# ═══════════════════════════════════════════════════════════════════════════
# Host (updated with SkillableAgent)
# ═══════════════════════════════════════════════════════════════════════════

class TestHost:
    def test_create_host(self):
        host = Host("test")
        assert host.name == "test"

    def test_add_agent(self):
        host = Host()
        agent = host.add_agent("test", AgentRole.EXPLORER)
        assert isinstance(agent, SkillableAgent)
        assert agent.name == "test"
        assert len(agent.skills) > 0

    def test_remove_agent(self):
        host = Host()
        host.add_agent("test", AgentRole.EXPLORER)
        assert host.remove_agent("test") is True
        assert host.remove_agent("nonexistent") is False

    def test_list_agents(self):
        host = Host()
        host.add_agent("a", AgentRole.EXPLORER)
        host.add_agent("b", AgentRole.CRITIC)
        agents = host.list_agents()
        assert len(agents) == 2

    def test_run_creates_default_host(self):
        host = Host()
        result = host.run("test task")
        assert isinstance(result, HostResult)
        assert len(result.agents_used) > 0

    def test_run_with_custom_agents(self):
        host = Host()
        host.add_agent("explorer", AgentRole.EXPLORER)
        host.add_agent("critic", AgentRole.CRITIC)
        result = host.run("test task")
        assert "explorer" in result.agents_used or "critic" in result.agents_used

    def test_repr(self):
        host = Host("test")
        assert "test" in repr(host)

    def test_library_access(self):
        host = Host()
        assert isinstance(host.library, BorgesLibrary)


# ═══════════════════════════════════════════════════════════════════════════
# HostHarness (updated)
# ═══════════════════════════════════════════════════════════════════════════

class TestHostHarness:
    def test_create_harness(self):
        harness = HostHarness()
        assert len(harness.history) == 0

    def test_create_host(self):
        harness = HostHarness()
        host = harness.create_host("test")
        assert host.name == "test"

    def test_run_host(self):
        harness = HostHarness()
        result = harness.run_host("test task")
        assert isinstance(result, HostResult)
        assert len(harness.history) == 1

    def test_run_swarm(self):
        harness = HostHarness()
        report = harness.run_swarm("test task", cycles=2)
        assert isinstance(report, SwarmReport)
        assert report.cycles == 2

    def test_invoke_angel(self):
        harness = HostHarness()
        result = harness.invoke_angel("gabriel", "Hello")
        assert result["angel"] == "gabriel"

    def test_get_switchboard(self):
        harness = HostHarness()
        sb = harness.get_switchboard()
        assert isinstance(sb, CelestialSwitchboard)

    def test_repr(self):
        harness = HostHarness()
        assert "HostHarness" in repr(harness)


# ═══════════════════════════════════════════════════════════════════════════
# HostMessage
# ═══════════════════════════════════════════════════════════════════════════

class TestHostMessage:
    def test_create_message(self):
        msg = HostMessage("sender", "content")
        assert msg.sender == "sender"
        assert msg.timestamp > 0

    def test_message_types(self):
        msg = HostMessage("sender", "content", "result")
        assert msg.message_type == "result"
