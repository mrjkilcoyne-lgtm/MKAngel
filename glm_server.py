"""
GLM Bridge Server — JSON-over-stdio protocol for TARDAI integration.

The Python side of the TARDAI <-> MKAngel bridge. Reads JSON-RPC requests
from stdin, processes them through the MNEMO NLG pipeline, and writes
JSON-RPC responses to stdout.

Protocol:
    Request:  {"jsonrpc": "2.0", "method": "<method>", "params": {...}, "id": <n>}
    Response: {"jsonrpc": "2.0", "result": {...}, "id": <n>}
    Error:    {"jsonrpc": "2.0", "error": {"code": <n>, "message": "..."}, "id": <n>}

Methods:
    encode   -- NL text -> MNEMO glyph codes
    decode   -- MNEMO glyph codes -> NL text
    derive   -- Full pipeline: NL -> MNEMO -> NL
    route    -- Domain detection only
    realise  -- Slot fills + domain + evidentials -> NL text
    mnemo    -- Raw MNEMO substrate info (registry, tiers)
"""

from __future__ import annotations

import json
import sys
from typing import Any, Dict, List, Optional

from glm.core.mnemo_substrate import (
    GLYPH_REGISTRY, MnemoSubstrate, MnemoSequence, Tier,
)
from glm.nlg.encoder import MnemoEncoder
from glm.nlg.decoder import MnemoDecoder
from glm.nlg.realiser import Realiser
from glm.nlg.templates import TemplateRegistry
from glm.nlg.templates.en import register_english
from glm.nlg.templates.fr import register_french
from glm.nlg.templates.es import register_spanish
from glm.nlg.templates.de import register_german
from glm.nlg.templates.tr import register_turkish
from glm.nlg.templates.cy import register_welsh
from glm.nlg.processors import create_default_dispatcher
from glm.nlg.data_crawler import crawl


class GLMBridgeServer:
    """JSON-RPC server for the GLM bridge."""

    def __init__(self) -> None:
        self._substrate = MnemoSubstrate()
        self._encoder = MnemoEncoder(substrate=self._substrate)
        self._decoder = MnemoDecoder(substrate=self._substrate)
        self._registry = TemplateRegistry()
        register_english(self._registry)
        register_french(self._registry)
        register_spanish(self._registry)
        register_german(self._registry)
        register_turkish(self._registry)
        register_welsh(self._registry)
        self._dispatcher = create_default_dispatcher()

    def handle_request(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """Route a JSON-RPC request to the appropriate handler."""
        method = request.get("method", "")
        params = request.get("params", {})
        req_id = request.get("id")

        handlers = {
            "encode": self._handle_encode,
            "decode": self._handle_decode,
            "derive": self._handle_derive,
            "route": self._handle_route,
            "realise": self._handle_realise,
            "mnemo": self._handle_mnemo,
        }

        handler = handlers.get(method)
        if handler is None:
            return self._error(req_id, -32601, f"Method not found: {method}")

        try:
            result = handler(params)
            return {"jsonrpc": "2.0", "result": result, "id": req_id}
        except Exception as exc:
            return self._error(req_id, -32000, str(exc))

    def _handle_encode(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Encode natural language to MNEMO glyph codes."""
        text = params.get("text", "")
        seq = self._encoder.encode(text)
        codes = [g.code for g in seq.glyphs]
        concepts = [g.concept for g in seq.glyphs]
        tiers = [g.tier.name for g in seq.glyphs]
        return {
            "codes": codes,
            "concepts": concepts,
            "tiers": tiers,
            "has_evidential": seq.has_evidential_marking(),
            "length": len(codes),
        }

    def _handle_decode(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Decode MNEMO glyph codes to natural language."""
        codes = params.get("codes", [])
        language = params.get("language", "en")
        extra_slots = params.get("slots", {})
        seq = self._substrate.encode_codes(codes)
        text = self._decoder.decode(seq, language=language, extra_slots=extra_slots)
        return {"text": text, "language": language}

    def _handle_derive(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Full pipeline: NL -> MNEMO -> NL (with API enrichment)."""
        text = params.get("text", "")
        language = params.get("language", "en")

        # Stage 1: Encode
        seq = self._encoder.encode(text)
        codes = [g.code for g in seq.glyphs]
        domain = self._encoder.detect_domain(text)

        # Stage 2: Process — domain-specific API enrichment
        api_slots = self._dispatcher.process(domain, text, seq)

        # Stage 2b: Data crawler fallback — if processor returned nothing,
        # try crawling free public APIs for real data
        if not api_slots:
            try:
                api_slots = crawl(text, domain=domain, lang=language)
            except Exception:
                api_slots = {}

        extra_slots = {"content": text}
        extra_slots.update(api_slots)

        # Stage 3: Decode
        output = self._decoder.decode(
            seq, language=language, extra_slots=extra_slots,
        )

        # Extract evidentials
        source, conf, temp = seq.get_evidential_triple()

        return {
            "input": text,
            "output": output,
            "domain": domain,
            "mnemo_codes": codes,
            "api_slots": api_slots,
            "evidential": {
                "source": source.code if source else None,
                "confidence": conf.code if conf else None,
                "temporal": temp.code if temp else None,
            },
        }

    def _handle_route(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Domain detection only."""
        text = params.get("text", "")
        domain = self._encoder.detect_domain(text)
        return {"domain": domain, "text": text}

    def _handle_realise(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Realise from slot fills + domain + evidentials."""
        domain = params.get("domain", "general")
        slots = params.get("slots", {})
        source = params.get("evidential_source", "inf")
        confidence = params.get("evidential_confidence", "prob")
        temporal = params.get("evidential_temporal", "obs_pres")
        language = params.get("language", "en")

        realiser = Realiser(
            registry=self._registry,
            substrate=self._substrate,
            language=language,
        )
        candidates = realiser.realise(
            domain=domain,
            slots=slots,
            evidential_source=source,
            evidential_confidence=confidence,
            evidential_temporal=temporal,
        )
        return {
            "candidates": [
                {"text": c.text, "score": c.score, "template": c.template_used}
                for c in candidates
            ],
            "count": len(candidates),
        }

    def _handle_mnemo(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Raw MNEMO substrate info."""
        return {
            "total_glyphs": len(GLYPH_REGISTRY),
            "tiers": [t.name for t in Tier],
            "tier_counts": {
                t.name: sum(1 for g in GLYPH_REGISTRY.values() if g.tier == t)
                for t in Tier
            },
        }

    def _error(
        self, req_id: Any, code: int, message: str
    ) -> Dict[str, Any]:
        return {
            "jsonrpc": "2.0",
            "error": {"code": code, "message": message},
            "id": req_id,
        }


def main() -> None:
    """Run the bridge server: read JSON-RPC from stdin, write to stdout."""
    server = GLMBridgeServer()

    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue

        try:
            request = json.loads(line)
        except json.JSONDecodeError as exc:
            response = {
                "jsonrpc": "2.0",
                "error": {"code": -32700, "message": f"Parse error: {exc}"},
                "id": None,
            }
            print(json.dumps(response), flush=True)
            continue

        response = server.handle_request(request)
        print(json.dumps(response), flush=True)


if __name__ == "__main__":
    main()
