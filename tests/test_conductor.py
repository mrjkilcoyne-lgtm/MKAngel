"""Tests for the AngelConductor -- boot, process, shutdown pipeline."""

from app.conductor import AngelConductor


class TestConductorBoot:
    def test_create_conductor(self):
        c = AngelConductor()
        assert not c._awake
        assert repr(c) == "AngelConductor(dormant)"

    def test_awaken(self):
        c = AngelConductor()
        result = c.awaken()
        assert result is c  # chaining
        assert c._awake
        assert c.session_id != ""

    def test_subsystems_loaded(self):
        c = AngelConductor().awaken()
        status = c.get_status()
        assert status["awake"] is True
        # Angel should load (it's pure Python, no deps)
        assert status["subsystems"]["angel"] == "active"

    def test_skills_loaded(self):
        c = AngelConductor().awaken()
        status = c.get_status()
        assert status["subsystems"]["skills"] == "active"

    def test_swarm_loaded(self):
        c = AngelConductor().awaken()
        status = c.get_status()
        assert status["subsystems"]["swarm"] == "active"

    def test_cowork_loaded(self):
        c = AngelConductor().awaken()
        status = c.get_status()
        assert status["subsystems"]["cowork"] == "active"


class TestConductorProcess:
    def test_process_requires_awaken(self):
        c = AngelConductor()
        result = c.process("hello")
        assert "not yet awake" in result.lower()

    def test_process_returns_string(self):
        c = AngelConductor().awaken()
        result = c.process("hello world")
        assert isinstance(result, str)
        assert len(result) > 0

    def test_process_multiple_messages(self):
        c = AngelConductor().awaken()
        r1 = c.process("hello")
        r2 = c.process("what is grammar")
        assert isinstance(r1, str)
        assert isinstance(r2, str)


class TestConductorShutdown:
    def test_shutdown(self):
        c = AngelConductor().awaken()
        msg = c.shutdown()
        assert isinstance(msg, str)
        assert not c._awake

    def test_shutdown_without_awaken(self):
        c = AngelConductor()
        msg = c.shutdown()
        assert isinstance(msg, str)


class TestConductorCommands:
    def test_unknown_command(self):
        c = AngelConductor().awaken()
        result = c.handle_command("/nonexistent")
        assert result is None

    def test_consent_command(self):
        c = AngelConductor().awaken()
        result = c.handle_command("/consent")
        assert isinstance(result, str)

    def test_growth_command(self):
        c = AngelConductor().awaken()
        result = c.handle_command("/growth")
        assert isinstance(result, str)

    def test_health_command(self):
        c = AngelConductor().awaken()
        result = c.handle_command("/health")
        assert isinstance(result, str)

    def test_language_command_no_arg(self):
        c = AngelConductor().awaken()
        result = c.handle_command("/language")
        assert isinstance(result, str)


class TestConductorSwarmIntegration:
    def test_run_swarm(self):
        c = AngelConductor().awaken()
        report = c.run_swarm("test task", cycles=1)
        assert report is not None

    def test_invoke_angel(self):
        c = AngelConductor().awaken()
        result = c.invoke_angel("gabriel", "Hello Gabriel")
        assert "angel" in result or "error" not in result

    def test_execute_skill(self):
        c = AngelConductor().awaken()
        result = c.execute_skill("summarize", "This is a test.")
        assert isinstance(result, str)

    def test_start_cowork(self):
        c = AngelConductor().awaken()
        result = c.start_cowork("analyse this")
        assert "Cowork session started" in result


class TestConductorStatus:
    def test_get_status_dormant(self):
        c = AngelConductor()
        status = c.get_status()
        assert status["awake"] is False

    def test_get_status_awake(self):
        c = AngelConductor().awaken()
        status = c.get_status()
        assert status["awake"] is True
        assert "subsystems" in status
        # Should have 17 subsystems now
        assert len(status["subsystems"]) >= 14

    def test_repr_awake(self):
        c = AngelConductor().awaken()
        r = repr(c)
        assert "awake" in r
        assert "/17" in r


class TestConductorProperties:
    def test_properties_before_awaken(self):
        c = AngelConductor()
        assert c.angel is None
        assert c.router is None
        assert c.provider is None
        assert c.skills is None
        assert c.swarm is None
        assert c.cowork is None

    def test_properties_after_awaken(self):
        c = AngelConductor().awaken()
        # Angel should be available (pure Python)
        assert c.angel is not None
        assert c.skills is not None
        assert c.swarm is not None
        assert c.cowork is not None
