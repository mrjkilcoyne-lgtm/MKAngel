"""Tests for the discovery ledger and hash chain."""

import tempfile
from pathlib import Path

from app.discovery import DiscoveryEvent, DiscoveryLedger, AlgorandBridge


class TestDiscoveryEvent:
    def test_create_event(self):
        e = DiscoveryEvent(description="test discovery")
        assert e.description == "test discovery"
        assert e.discovery_id != ""
        assert e.timestamp > 0

    def test_compute_hash(self):
        e = DiscoveryEvent(description="test", domains_crossed=["physics", "chemistry"])
        h = e.compute_hash()
        assert len(h) == 64  # SHA-256 hex
        assert e.content_hash == h

    def test_hash_deterministic(self):
        e1 = DiscoveryEvent(discovery_id="abc", timestamp=1.0,
                            description="test", domains_crossed=["a"])
        e2 = DiscoveryEvent(discovery_id="abc", timestamp=1.0,
                            description="test", domains_crossed=["a"])
        assert e1.compute_hash() == e2.compute_hash()

    def test_hash_changes_with_content(self):
        e1 = DiscoveryEvent(discovery_id="x", timestamp=1.0, description="a")
        e2 = DiscoveryEvent(discovery_id="x", timestamp=1.0, description="b")
        assert e1.compute_hash() != e2.compute_hash()

    def test_citation_key(self):
        e = DiscoveryEvent(discovery_id="abcdef1234567890", angel_name="uriel")
        key = e.citation_key()
        assert "DISC-" in key
        assert "URI" in key

    def test_to_dict_and_back(self):
        e = DiscoveryEvent(description="test", domains_crossed=["physics"],
                           angel_name="uriel", confidence=0.9)
        d = e.to_dict()
        e2 = DiscoveryEvent.from_dict(d)
        assert e2.description == "test"
        assert e2.angel_name == "uriel"


class TestDiscoveryLedger:
    def _make_ledger(self):
        d = tempfile.mkdtemp()
        return DiscoveryLedger(db_path=Path(d) / "test_ledger.db")

    def test_create_ledger(self):
        ledger = self._make_ledger()
        assert ledger.count() == 0
        assert ledger.chain_tip() == "GENESIS"

    def test_record_discovery(self):
        ledger = self._make_ledger()
        event = DiscoveryEvent(
            description="Wave equation isomorphism",
            domains_crossed=["physics", "phonology"],
            rules_fired=["wave_equation", "phonological_wave"],
            confidence=0.9,
            angel_name="uriel",
            human_partner="mk",
        )
        recorded = ledger.record(event)
        assert recorded.content_hash != ""
        assert recorded.previous_hash == "GENESIS"
        assert ledger.count() == 1

    def test_chain_links(self):
        ledger = self._make_ledger()
        e1 = DiscoveryEvent(description="first", domains_crossed=["a"])
        e2 = DiscoveryEvent(description="second", domains_crossed=["b"])
        r1 = ledger.record(e1)
        r2 = ledger.record(e2)
        assert r1.previous_hash == "GENESIS"
        assert r2.previous_hash == r1.content_hash

    def test_verify_chain_intact(self):
        ledger = self._make_ledger()
        for i in range(5):
            ledger.record(DiscoveryEvent(
                description=f"discovery {i}", domains_crossed=["test"],
            ))
        intact, checked, broken = ledger.verify_chain()
        assert intact is True
        assert checked == 5

    def test_search_by_domain(self):
        ledger = self._make_ledger()
        ledger.record(DiscoveryEvent(
            description="physics thing", domains_crossed=["physics"],
        ))
        ledger.record(DiscoveryEvent(
            description="biology thing", domains_crossed=["biology"],
        ))
        results = ledger.search(domain="physics")
        assert len(results) == 1
        assert results[0].description == "physics thing"

    def test_search_by_query(self):
        ledger = self._make_ledger()
        ledger.record(DiscoveryEvent(
            description="wave equation isomorphism", domains_crossed=["physics"],
        ))
        results = ledger.search(query="wave")
        assert len(results) == 1

    def test_pending_on_chain(self):
        ledger = self._make_ledger()
        ledger.record(DiscoveryEvent(description="pending", domains_crossed=["a"]))
        pending = ledger.pending_on_chain()
        assert len(pending) == 1

    def test_mark_on_chain(self):
        ledger = self._make_ledger()
        event = DiscoveryEvent(description="to submit", domains_crossed=["a"])
        recorded = ledger.record(event)
        ledger.mark_on_chain(recorded.discovery_id, "TX123")
        pending = ledger.pending_on_chain()
        assert len(pending) == 0


class TestAlgorandBridge:
    def test_create_bridge_offline(self):
        bridge = AlgorandBridge()
        assert not bridge.is_connected
        assert "offline" in repr(bridge)

    def test_submit_without_connection(self):
        bridge = AlgorandBridge()
        event = DiscoveryEvent(description="test", domains_crossed=["a"])
        event.compute_hash()
        result = bridge.submit_discovery(event)
        assert result is None


class TestSwarmDiscoveryIntegration:
    def test_switchboard_record_discovery(self):
        from app.swarm import CelestialSwitchboard
        sb = CelestialSwitchboard()
        result = sb.record_discovery(
            description="SHM equation identical in mechanics and LC circuits",
            domains_crossed=["physics", "computational"],
            derivation_chain=[{"rule": "simple_harmonic_motion"}, {"rule": "lc_resonance"}],
            rules_fired=["simple_harmonic_motion", "lc_resonance"],
            confidence=1.0,
            angel_name="uriel",
            human_partner="mk",
        )
        assert "discovery_id" in result
        assert "content_hash" in result

    def test_switchboard_verify_ledger(self):
        from app.swarm import CelestialSwitchboard
        sb = CelestialSwitchboard()
        result = sb.verify_ledger()
        assert "intact" in result

    def test_switchboard_memory_report(self):
        from app.swarm import CelestialSwitchboard
        sb = CelestialSwitchboard()
        report = sb.memory_report()
        assert "gabriel" in report
        assert "_ledger" in report
