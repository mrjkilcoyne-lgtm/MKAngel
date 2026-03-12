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
import time
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

    def __init__(self) -> None:
        self._angel = None

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
        tokens = prompt.lower().split()

        # Use the grammar engine to produce a structured response
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
            lines.append("")
            lines.append(
                "Note: Local GLM produces structural derivations, not "
                "free-form text. For richer generation, configure an "
                "API provider with /settings."
            )
            return "\n".join(lines)

        # Fallback: introspection-based response
        try:
            info = angel.introspect()
            return (
                f"[GLM local | {info['total_grammars']} grammars, "
                f"{info['strange_loops_detected']} strange loops]\n\n"
                f"I processed your input through {info['total_rules']} "
                f"grammatical rules across {len(info['domains_loaded'])} "
                f"domains but could not derive a strong prediction.\n\n"
                f"Try a more specific query, or use /predict, /reconstruct, "
                f"or /forecast for targeted grammar operations."
            )
        except Exception:
            return (
                "[GLM local]\n\n"
                "Grammar engine active. No derivation found for this input.\n"
                "Try /predict <tokens> or /help for available commands."
            )

    def is_available(self) -> bool:
        return True


# ---------------------------------------------------------------------------
# NLG provider -- MNEMO-native generation
# ---------------------------------------------------------------------------

class NLGProvider(Provider):
    """Uses the MNEMO NLG engine for native generation.

    Three-stage pipeline: ENCODE (NL->MNEMO) -> PROCESS -> DECODE (MNEMO->NL).
    No API dependency — the GLM IS the engine.
    """

    name = "nlg"

    def __init__(self) -> None:
        self._encoder = None
        self._decoder = None
        self._substrate = None
        self._dispatcher = None

    def _ensure_loaded(self) -> None:
        """Lazy-load NLG components."""
        if self._encoder is None:
            from glm.nlg.encoder import MnemoEncoder
            from glm.nlg.decoder import MnemoDecoder
            from glm.core.mnemo_substrate import MnemoSubstrate
            from glm.nlg.processors import create_default_dispatcher
            self._substrate = MnemoSubstrate()
            self._encoder = MnemoEncoder(substrate=self._substrate)
            self._decoder = MnemoDecoder(substrate=self._substrate)
            self._dispatcher = create_default_dispatcher()

    def generate(
        self,
        prompt: str,
        *,
        system: str = "",
        temperature: float = 0.7,
        max_tokens: int = 1024,
    ) -> str:
        """Generate via MNEMO NLG pipeline.

        1. ENCODE: Natural language -> MNEMO
        2. PROCESS: (future: grammar derivation + attention in MNEMO space)
        3. DECODE: MNEMO -> Natural language
        """
        self._ensure_loaded()

        # Stage 1: ENCODE
        mnemo_seq = self._encoder.encode(prompt)

        # Stage 2: PROCESS — domain-specific API enrichment
        domain = self._encoder.detect_domain(prompt)
        api_slots = self._dispatcher.process(domain, prompt, mnemo_seq)
        extra_slots = {"content": prompt}
        extra_slots.update(api_slots)

        # Stage 3: DECODE
        result = self._decoder.decode(
            mnemo_seq,
            language="en",
            extra_slots=extra_slots,
        )

        if result:
            domain = self._encoder.detect_domain(prompt)
            source, confidence, temporal = self._decoder._extract_evidentials(mnemo_seq)
            header = f"[NLG | domain: {domain} | {source}/{confidence}/{temporal}]"
            return f"{header}\n\n{result}"

        return "[NLG] Processed through MNEMO pipeline. No strong derivation."

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
# Orchestra provider -- smart multi-provider routing
# ---------------------------------------------------------------------------

class OrchestraProvider(Provider):
    """The Choir -- multiple providers singing in harmony.

    The Orchestra manages multiple AI providers, routing requests
    to the best available one based on:
    - Task type (code, chat, translation, etc.)
    - Provider availability and latency
    - Cost efficiency
    - Fallback chains for reliability

    Like a choir director: each voice has its strength,
    the conductor knows which voice to call on.
    """

    name = "orchestra"

    def __init__(self, settings: Settings | None = None):
        self._providers: dict[str, Provider] = {}
        self._fallback_chains: dict[str, list[str]] = {
            "default": ["anthropic", "openai", "google", "groq", "mistral", "local"],
            "code": ["anthropic", "openai", "local"],
            "chat": ["anthropic", "openai", "google", "local"],
            "search": ["google", "openai", "anthropic", "local"],
            "translate": ["google", "anthropic", "openai", "local"],
            "predict": ["local", "anthropic"],
            "create": ["openai", "anthropic", "google", "local"],
            "fast": ["groq", "mistral", "openai", "local"],
        }
        self._usage: list[dict] = []
        self._latency_cache: dict[str, float] = {}
        if settings:
            self._auto_configure(settings)

    def add_provider(self, name: str, provider: Provider) -> None:
        """Register a provider under the given name."""
        self._providers[name] = provider

    def generate(
        self,
        prompt: str,
        *,
        system: str = "",
        temperature: float = 0.7,
        max_tokens: int = 1024,
    ) -> str:
        return self.generate_with_preference(
            prompt,
            preference="default",
            system=system,
            temperature=temperature,
            max_tokens=max_tokens,
        )

    def generate_with_preference(
        self,
        prompt: str,
        preference: str = "default",
        *,
        system: str = "",
        temperature: float = 0.7,
        max_tokens: int = 1024,
    ) -> str:
        """Generate using the preferred fallback chain.

        Tries each provider in the chain until one succeeds.
        Records usage and latency for future optimization.
        """
        chain = self._fallback_chains.get(
            preference, self._fallback_chains["default"]
        )

        for provider_name in chain:
            provider = self._providers.get(provider_name)
            if not provider or not provider.is_available():
                continue

            start = time.time()
            try:
                result = provider.generate(
                    prompt,
                    system=system,
                    temperature=temperature,
                    max_tokens=max_tokens,
                )
                latency = time.time() - start
                self._record_usage(provider_name, preference, latency, True)
                self._latency_cache[provider_name] = latency
                return result
            except Exception as exc:
                latency = time.time() - start
                self._record_usage(
                    provider_name, preference, latency, False, str(exc)
                )
                continue

        # All providers failed -- use local as absolute fallback
        local = self._providers.get("local") or LocalProvider()
        return local.generate(
            prompt,
            system=system,
            temperature=temperature,
            max_tokens=max_tokens,
        )

    def is_available(self) -> bool:
        return True  # Local fallback always available

    def _auto_configure(self, settings: Settings) -> None:
        """Auto-discover and configure providers from settings."""
        # Always add local
        self._providers["local"] = LocalProvider()

        # Try each API provider
        for name in ("anthropic", "openai", "google", "groq", "mistral"):
            key = settings.get_api_key(name)
            if key:
                self._providers[name] = APIProvider(name, key)

        # Add hybrid if any API is available
        api_providers = [
            n
            for n in ("anthropic", "openai", "google", "groq", "mistral")
            if n in self._providers
        ]
        if api_providers:
            self._providers["hybrid"] = HybridProvider(
                self._providers["local"],
                self._providers[api_providers[0]],
            )

    def _record_usage(
        self,
        provider: str,
        preference: str,
        latency: float,
        success: bool,
        error: str = "",
    ) -> None:
        """Record a usage entry for analytics."""
        self._usage.append(
            {
                "provider": provider,
                "preference": preference,
                "latency": latency,
                "success": success,
                "error": error,
                "timestamp": time.time(),
            }
        )
        # Keep last 1000 entries
        if len(self._usage) > 1000:
            self._usage = self._usage[-500:]

    def get_stats(self) -> dict[str, Any]:
        """Return usage statistics for all providers."""
        stats: dict[str, dict] = {}
        for entry in self._usage:
            name = entry["provider"]
            if name not in stats:
                stats[name] = {
                    "calls": 0,
                    "successes": 0,
                    "total_latency": 0.0,
                }
            stats[name]["calls"] += 1
            if entry["success"]:
                stats[name]["successes"] += 1
            stats[name]["total_latency"] += entry["latency"]

        for name, s in stats.items():
            s["avg_latency"] = s["total_latency"] / max(s["calls"], 1)
            s["success_rate"] = s["successes"] / max(s["calls"], 1)

        return stats

    def available_providers(self) -> list[str]:
        """Return names of all currently available providers."""
        return [n for n, p in self._providers.items() if p.is_available()]

    def set_fallback_chain(self, preference: str, chain: list[str]) -> None:
        """Override or add a fallback chain for a given preference."""
        self._fallback_chains[preference] = chain


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

def get_provider(settings: Settings) -> Provider:
    """Return the appropriate provider based on current settings.

    - If offline_mode is True or provider is 'local', use LocalProvider.
    - If an API key is configured for the active provider, use APIProvider.
    - If provider is 'hybrid', combine local + API.
    - Falls back to LocalProvider if anything is missing.
    """
    provider_name = settings.model_provider

    if settings.offline_mode or provider_name == "local":
        return LocalProvider()

    api_key = settings.get_api_key(provider_name)

    if provider_name == "orchestra":
        return OrchestraProvider(settings)

    if provider_name == "hybrid":
        local = LocalProvider()
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
    return LocalProvider()
