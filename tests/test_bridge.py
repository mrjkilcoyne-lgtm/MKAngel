import pytest
from glm.bridge import CanzukBridge


def test_bridge_initialises():
    bridge = CanzukBridge()
    assert bridge.ready is True


def test_bridge_process_returns_string():
    bridge = CanzukBridge()
    result = bridge.process("What is photosynthesis?")
    assert isinstance(result, str)
    assert len(result) > 0


def test_bridge_stream_yields_tokens():
    bridge = CanzukBridge()
    tokens = list(bridge.stream("Explain gravity"))
    assert len(tokens) > 0
    assert all(isinstance(t, str) for t in tokens)


def test_bridge_introspect():
    bridge = CanzukBridge()
    info = bridge.introspect()
    assert "domains" in info
    assert "parameters" in info
    assert "grammars" in info
