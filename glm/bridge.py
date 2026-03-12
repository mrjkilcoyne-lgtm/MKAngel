"""
CANZUK-AI Bridge — entry point for Chaquopy calls from Kotlin.

All Kotlin→Python communication flows through CanzukBridge.
Methods return plain types (str, dict, list) for Chaquopy serialisation.
"""
from typing import Iterator

from glm.angel import Angel
from glm.pipeline import ReasoningPipeline
from glm.realiser_v2 import GenerativeRealiser


class CanzukBridge:
    """Single entry point for the Kotlin UI to call the GLM engine."""

    def __init__(self):
        self._angel = Angel()
        self._angel.awaken()
        self._pipeline = ReasoningPipeline()
        self._realiser = GenerativeRealiser(self._angel)
        self.ready = True

    def process(self, text: str) -> str:
        """Process input and return complete response as string."""
        return "".join(self.stream(text))

    def stream(self, text: str) -> list:
        """Return response tokens from grammar engine as a list.

        Returns a list (not a generator) for Chaquopy Java interop —
        PyObject.asList() requires a concrete Python list.

        1. Pipeline decomposes input (Skeleton→DAG→Disconfirm→Synthesis)
        2. Realiser walks the validated derivation tree
        3. Returns all tokens as a list
        """
        pipeline_result = self._pipeline.run(text)
        return list(self._realiser.stream(pipeline_result, text))

    def introspect(self) -> dict:
        """Return live model stats for the Introspection screen."""
        info = self._angel.introspect() if hasattr(self._angel, 'introspect') else {}
        return {
            "domains": info.get("domains", []),
            "grammars": info.get("grammar_count", 0),
            "rules": info.get("rule_count", 0),
            "strange_loops": info.get("strange_loops", 0),
            "parameters": info.get("parameters", 303754),
            "version": "0.1.0",
            "name": "CANZUK-AI",
        }

    def get_domains(self) -> list:
        """List available grammar domains."""
        return self.introspect().get("domains", [])
