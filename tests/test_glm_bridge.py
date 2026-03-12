"""Tests for the GLM Bridge server."""

import json
import pytest
from glm_server import GLMBridgeServer


class TestGLMBridge:
    def setup_method(self):
        self.server = GLMBridgeServer()

    def test_encode(self):
        req = {"jsonrpc": "2.0", "method": "encode", "params": {"text": "the verb agrees"}, "id": 1}
        resp = self.server.handle_request(req)
        assert resp["id"] == 1
        assert "result" in resp
        assert resp["result"]["has_evidential"] is True
        assert resp["result"]["length"] > 0

    def test_decode(self):
        req = {"jsonrpc": "2.0", "method": "decode",
               "params": {"codes": ["scale_05", "proc_09", "obs", "cert", "obs_pres"]}, "id": 2}
        resp = self.server.handle_request(req)
        assert "result" in resp
        assert isinstance(resp["result"]["text"], str)

    def test_derive(self):
        req = {"jsonrpc": "2.0", "method": "derive",
               "params": {"text": "solve the equation"}, "id": 3}
        resp = self.server.handle_request(req)
        result = resp["result"]
        assert result["domain"] == "mathematical"
        assert result["output"] != ""
        assert result["evidential"]["source"] is not None

    def test_route(self):
        req = {"jsonrpc": "2.0", "method": "route",
               "params": {"text": "the gene encodes a protein"}, "id": 4}
        resp = self.server.handle_request(req)
        assert resp["result"]["domain"] == "biological"

    def test_realise(self):
        req = {"jsonrpc": "2.0", "method": "realise",
               "params": {"domain": "mathematical", "slots": {"result": "42"},
                          "evidential_source": "comp", "evidential_confidence": "cert"}, "id": 5}
        resp = self.server.handle_request(req)
        assert resp["result"]["count"] > 0
        assert any("42" in c["text"] for c in resp["result"]["candidates"])

    def test_mnemo(self):
        req = {"jsonrpc": "2.0", "method": "mnemo", "params": {}, "id": 6}
        resp = self.server.handle_request(req)
        assert resp["result"]["total_glyphs"] == 270
        assert len(resp["result"]["tiers"]) == 8

    def test_unknown_method(self):
        req = {"jsonrpc": "2.0", "method": "nonexistent", "params": {}, "id": 7}
        resp = self.server.handle_request(req)
        assert "error" in resp
        assert resp["error"]["code"] == -32601
