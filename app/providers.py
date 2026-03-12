"""
LLM provider abstraction for MKAngel.

Three modes of operation:
    LocalProvider   -- uses the Grammar Language Model offline (no API)
    APIProvider     -- routes to external APIs (OpenAI, Anthropic, etc.)
    HybridProvider  -- local GLM for structure, API for generation

When offline, everything gracefully falls back to LocalProvider.
"""

from __future__ import annotations

import json
import textwrap
import urllib.error
import urllib.request
from abc import ABC, abstractmethod
from typing import Any

from app.settings import Settings


# ---------------------------------------------------------------------------
# ANSI colour helpers (reused across the app)
# ---------------------------------------------------------------------------

class _C:
    """ANSI colour codes for terminal output."""
    RESET   = "\033[0m"
    BOLD    = "\033[1m"
    DIM     = "\033[2m"
    CYAN    = "\033[36m"
    GREEN   = "\033[32m"
    YELLOW  = "\033[33m"
    RED     = "\033[31m"
    MAGENTA = "\033[35m"
    BLUE    = "\033[34m"
    WHITE   = "\033[37m"


# ---------------------------------------------------------------------------
# Abstract base
# ---------------------------------------------------------------------------

class Provider(ABC):
    """Abstract base for LLM providers."""

    name: str = "base"

    @abstractmethod
    def generate(
        self,
        prompt: str,
        *,
        system: str = "",
        temperature: float = 0.7,
        max_tokens: int = 1024,
    ) -> str:
        """Generate a text completion from the given prompt."""

    @abstractmethod
    def is_available(self) -> bool:
        """Return True if this provider can currently serve requests."""

    def __repr__(self) -> str:
        avail = "available" if self.is_available() else "unavailable"
        return f"{self.__class__.__name__}({avail})"


# ---------------------------------------------------------------------------
# Local provider -- offline GLM
# ---------------------------------------------------------------------------

class LocalProvider(Provider):
    """Uses the Grammar Language Model locally -- no network required.

    This is the default, always-available provider.  It uses the GLM's
    grammar engine to produce structured responses by deriving outputs
    from grammatical rules.
    """

    name = "local"

    def __init__(self, angel=None) -> None:
        self._angel = angel

    def _get_angel(self):
        """Lazy-load the Angel to avoid import-time overhead."""
        if self._angel is None:
            from glm.angel import Angel
            self._angel = Angel()
            self._angel.awaken()
        return self._angel

    def generate(
        self,
        prompt: str,
        *,
        system: str = "",
        temperature: float = 0.7,
        max_tokens: int = 1024,
    ) -> str:
        angel = self._get_angel()

        # Extract the actual user text (strip conversation context framing)
        user_text = prompt
        if "\nUser: " in prompt:
            user_text = prompt.rsplit("\nUser: ", 1)[-1].strip()
        elif "User: " in prompt:
            user_text = prompt.split("User: ", 1)[-1].strip()

        # Use the Angel's full respond() method
        try:
            result = angel.respond(user_text)
            return self._format_response(result, user_text)
        except Exception:
            pass

        # Fallback: simple prediction
        tokens = user_text.lower().split()
        try:
            predictions = angel.predict(tokens, domain="linguistic", horizon=8)
        except Exception:
            predictions = []

        if predictions:
            predicted_tokens = [p.get("predicted", "") for p in predictions[:6]]
            predicted_str = " ".join(str(t) for t in predicted_tokens if t)
            grammar_names = list({p.get("grammar", "?") for p in predictions})
            confidence = predictions[0].get("confidence", 0.0)

            lines = []
            lines.append(f"[GLM local | grammars: {', '.join(grammar_names)}]")
            lines.append("")
            lines.append(
                f"Grammatical derivation (confidence {confidence:.2f}):"
            )
            lines.append(f"  {predicted_str}")
            return "\n".join(lines)

        # Introspection-based fallback
        try:
            info = angel.introspect()
            return (
                f"I processed your input through {info['total_rules']} "
                f"grammatical rules across {len(info['domains_loaded'])} "
                f"domains. My grammars are still learning the structure "
                f"of what you said."
            )
        except Exception:
            return "I hear you. Let me process that through my grammars."

    def _format_response(self, result: dict, user_text: str) -> str:
        """Format an Angel.respond() result into natural text."""
        lines = []
        predictions = result.get("predictions", [])
        new_words = result.get("new_words", [])
        morpho = result.get("morphological", [])
        cats = result.get("categories", [])
        cat_struct = result.get("category_structure", {})

        # Build a natural conversational response
        if predictions:
            pred_strs = []
            for p in predictions[:3]:
                predicted = p.get("predicted", "")
                grammar = p.get("grammar", "")
                conf = p.get("confidence", 0)
                if predicted:
                    pred_strs.append(f"{predicted} ({grammar}, {conf:.0%})")
            if pred_strs:
                lines.append(
                    f"Grammatical derivation: {'; '.join(pred_strs)}"
                )
                lines.append("")

        # Report structural understanding
        if cat_struct:
            structure_parts = []
            cat_names = {
                "N": "noun", "V": "verb", "Adj": "adjective",
                "Adv": "adverb", "Det": "determiner", "P": "preposition",
                "NP": "pronoun", "Wh": "interrogative", "I": "modal",
                "C": "complementiser", "Conj": "conjunction",
            }
            for cat, count in sorted(
                cat_struct.items(), key=lambda x: -x[1]
            ):
                name = cat_names.get(cat, cat)
                if count > 1:
                    structure_parts.append(f"{count} {name}s")
                else:
                    structure_parts.append(f"{count} {name}")
            lines.append(f"I see: {', '.join(structure_parts)}.")

        # Morphological insight
        if morpho:
            for m in morpho[:2]:
                analysis = m.get("analysis", "")
                if analysis:
                    lines.append(f"  {m.get('word', '')}: {analysis}")

        # New words — show learning
        if new_words:
            if len(new_words) == 1:
                lines.append(
                    f"New word: '{new_words[0]}'. "
                    f"Use it again and I'll learn it."
                )
            elif len(new_words) <= 3:
                lines.append(
                    f"New words: {', '.join(repr(w) for w in new_words)}. "
                    f"I'll learn them with repetition."
                )
            else:
                lines.append(
                    f"{len(new_words)} new words detected. "
                    f"I'm learning as we talk."
                )

        # If we produced nothing meaningful, give a contextual response
        if not lines:
            word_count = len(user_text.split())
            known = len(result.get("known_words", []))
            total = known + len(new_words)
            if total > 0:
                pct = known / total * 100
                lines.append(
                    f"I recognised {known}/{total} words ({pct:.0f}%). "
                    f"My grammars are still learning the structure of "
                    f"what you said."
                )
            else:
                lines.append(
                    "I hear you. My grammars are still learning "
                    "the structure of what you said."
                )

        return "\n".join(lines)

    def is_available(self) -> bool:
        return True


# ---------------------------------------------------------------------------
# API provider -- external LLM services
# ---------------------------------------------------------------------------

# Endpoint configuration for supported API providers
_API_ENDPOINTS: dict[str, dict[str, str]] = {
    "openai": {
        "url": "https://api.openai.com/v1/chat/completions",
        "model": "gpt-4o-mini",
        "auth_header": "Authorization",
        "auth_prefix": "Bearer ",
    },
    "anthropic": {
        "url": "https://api.anthropic.com/v1/messages",
        "model": "claude-sonnet-4-20250514",
        "auth_header": "x-api-key",
        "auth_prefix": "",
    },
    "groq": {
        "url": "https://api.groq.com/openai/v1/chat/completions",
        "model": "llama-3.1-8b-instant",
        "auth_header": "Authorization",
        "auth_prefix": "Bearer ",
    },
    "mistral": {
        "url": "https://api.mistral.ai/v1/chat/completions",
        "model": "mistral-small-latest",
        "auth_header": "Authorization",
        "auth_prefix": "Bearer ",
    },
    "google": {
        "url": "https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent",
        "model": "gemini-1.5-flash",
        "auth_header": "",
        "auth_prefix": "",
    },
}


class APIProvider(Provider):
    """Routes requests to external LLM APIs when online.

    Supports OpenAI, Anthropic, Google, Mistral, and Groq.
    Falls back gracefully if the network is unavailable.
    """

    def __init__(self, provider_name: str, api_key: str):
        self.name = provider_name
        self._api_key = api_key
        self._config = _API_ENDPOINTS.get(provider_name, {})

    def generate(
        self,
        prompt: str,
        *,
        system: str = "",
        temperature: float = 0.7,
        max_tokens: int = 1024,
    ) -> str:
        if not self._config:
            return f"[Error] No API configuration for provider '{self.name}'."

        try:
            if self.name == "anthropic":
                return self._call_anthropic(prompt, system, temperature, max_tokens)
            elif self.name == "google":
                return self._call_google(prompt, system, temperature, max_tokens)
            else:
                return self._call_openai_compatible(prompt, system, temperature, max_tokens)
        except urllib.error.URLError as exc:
            return f"[Network error] Could not reach {self.name} API: {exc.reason}"
        except Exception as exc:
            return f"[Error] {self.name} API call failed: {exc}"

    def _call_openai_compatible(
        self, prompt: str, system: str, temperature: float, max_tokens: int,
    ) -> str:
        """Call OpenAI-compatible APIs (OpenAI, Groq, Mistral)."""
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})

        payload = json.dumps({
            "model": self._config["model"],
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }).encode("utf-8")

        req = urllib.request.Request(
            self._config["url"],
            data=payload,
            headers={
                "Content-Type": "application/json",
                self._config["auth_header"]: (
                    self._config["auth_prefix"] + self._api_key
                ),
            },
            method="POST",
        )

        with urllib.request.urlopen(req, timeout=30) as resp:
            data = json.loads(resp.read().decode("utf-8"))

        return data["choices"][0]["message"]["content"]

    def _call_anthropic(
        self, prompt: str, system: str, temperature: float, max_tokens: int,
    ) -> str:
        """Call the Anthropic Messages API."""
        body: dict[str, Any] = {
            "model": self._config["model"],
            "max_tokens": max_tokens,
            "temperature": temperature,
            "messages": [{"role": "user", "content": prompt}],
        }
        if system:
            body["system"] = system

        payload = json.dumps(body).encode("utf-8")

        req = urllib.request.Request(
            self._config["url"],
            data=payload,
            headers={
                "Content-Type": "application/json",
                "x-api-key": self._api_key,
                "anthropic-version": "2023-06-01",
            },
            method="POST",
        )

        with urllib.request.urlopen(req, timeout=30) as resp:
            data = json.loads(resp.read().decode("utf-8"))

        return data["content"][0]["text"]

    def _call_google(
        self, prompt: str, system: str, temperature: float, max_tokens: int,
    ) -> str:
        """Call the Google Gemini API."""
        model = self._config["model"]
        url = (
            f"https://generativelanguage.googleapis.com/v1beta/"
            f"models/{model}:generateContent?key={self._api_key}"
        )
        parts = []
        if system:
            parts.append({"text": system + "\n\n"})
        parts.append({"text": prompt})

        payload = json.dumps({
            "contents": [{"parts": parts}],
            "generationConfig": {
                "temperature": temperature,
                "maxOutputTokens": max_tokens,
            },
        }).encode("utf-8")

        req = urllib.request.Request(
            url,
            data=payload,
            headers={"Content-Type": "application/json"},
            method="POST",
        )

        with urllib.request.urlopen(req, timeout=30) as resp:
            data = json.loads(resp.read().decode("utf-8"))

        return data["candidates"][0]["content"]["parts"][0]["text"]

    def is_available(self) -> bool:
        """Check if the API is reachable (lightweight connectivity test)."""
        if not self._api_key or not self._config:
            return False
        try:
            url = self._config["url"].split("/v1")[0]
            req = urllib.request.Request(url, method="HEAD")
            urllib.request.urlopen(req, timeout=5)
            return True
        except Exception:
            return False


# ---------------------------------------------------------------------------
# Hybrid provider -- local GLM + API
# ---------------------------------------------------------------------------

class HybridProvider(Provider):
    """Combines local GLM for grammar/structure with an API for generation.

    The local GLM analyses the input structurally, then the API provider
    generates a rich response informed by that analysis.
    """

    name = "hybrid"

    def __init__(self, local: LocalProvider, api: APIProvider):
        self._local = local
        self._api = api

    def generate(
        self,
        prompt: str,
        *,
        system: str = "",
        temperature: float = 0.7,
        max_tokens: int = 1024,
    ) -> str:
        # Phase 1: local grammar analysis
        try:
            from glm.angel import Angel
            angel = self._local._get_angel()
            tokens = prompt.lower().split()
            predictions = angel.predict(tokens, domain="linguistic", horizon=4)
            grammar_context = ""
            if predictions:
                pred_summary = "; ".join(
                    f"{p.get('grammar','?')}: {p.get('predicted','?')} "
                    f"(conf={p.get('confidence',0):.2f})"
                    for p in predictions[:3]
                )
                grammar_context = (
                    f"\n\n[Grammar analysis: {pred_summary}]\n"
                    f"[Strange loops active: "
                    f"{len(angel._strange_loops)}]"
                )
        except Exception:
            grammar_context = ""

        # Phase 2: API generation enriched with grammar context
        enriched_system = system
        if grammar_context:
            enriched_system = (system + grammar_context) if system else grammar_context

        if self._api.is_available():
            return self._api.generate(
                prompt,
                system=enriched_system,
                temperature=temperature,
                max_tokens=max_tokens,
            )

        # Fallback to local if API is unreachable
        return self._local.generate(
            prompt, system=system, temperature=temperature, max_tokens=max_tokens
        )

    def is_available(self) -> bool:
        return True  # Always available -- local fallback


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

def get_provider(settings: Settings, angel=None) -> Provider:
    """Return the appropriate provider based on current settings.

    - If offline_mode is True or provider is 'local', use LocalProvider.
    - If an API key is configured for the active provider, use APIProvider.
    - If provider is 'hybrid', combine local + API.
    - Falls back to LocalProvider if anything is missing.

    If *angel* is provided, the LocalProvider will reuse it instead of
    creating a second Angel instance.
    """
    provider_name = settings.model_provider

    if settings.offline_mode or provider_name == "local":
        return LocalProvider(angel=angel)

    api_key = settings.get_api_key(provider_name)

    if provider_name == "hybrid":
        local = LocalProvider(angel=angel)
        # Try to find any configured API key
        for p in ("anthropic", "openai", "groq", "mistral", "google"):
            key = settings.get_api_key(p)
            if key:
                api = APIProvider(p, key)
                return HybridProvider(local, api)
        # No API key found, fall back to local
        return local

    if api_key:
        return APIProvider(provider_name, api_key)

    # No key configured -- fall back to local
    return LocalProvider(angel=angel)
