"""
The Universal Router — the switchboard of the EVERYTHING app.

The Router sits between the user and the Choir (multiple AI providers).
It is the conductor's ear: before the fugue plays, someone must listen
to the opening phrase and decide which voices should sing, in what key,
and at what tempo.

The Router uses the GLM's grammar engine to classify intent — not by
statistical n-grams, but by structural analysis.  A request for code is
recognisable not just by keywords but by the *grammar* of the request:
imperative verbs, technical nouns, structural specificity.  A request
for prediction carries the grammar of futures: conditionals, temporal
markers, uncertainty quantifiers.

The pipeline:
    1. Classify  — Grammar-based intent recognition
    2. Enrich    — Add GLM context to the prompt (structural hints)
    3. Select    — Pick the best available provider for this intent
    4. Route     — Deliver the enriched prompt to the chosen provider

Like a telephone exchange that not only connects the call but also
translates the language and adds context for the listener.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from glm.angel import Angel
    from app.providers import Provider


# ---------------------------------------------------------------------------
# Intent taxonomy
# ---------------------------------------------------------------------------

class IntentCategory(Enum):
    """The ten fundamental intents — the phonemes of purpose.

    Every user utterance, no matter how complex, decomposes into one
    (or a combination) of these structural intents.  Like phonemes
    composing words, these intents compose tasks.
    """

    CHAT = auto()       # General conversation — the unmarked case
    CODE = auto()       # Code generation, analysis, debugging
    SEARCH = auto()     # Web search, information retrieval
    CREATE = auto()     # Content creation: docs, images, drafts
    ANALYZE = auto()    # Data analysis, reasoning, logic
    TRANSLATE = auto()  # Language translation, cross-domain mapping
    PREDICT = auto()    # Forecasting, prediction, futures
    EXPLAIN = auto()    # Explanation, teaching, pedagogy
    COMMAND = auto()    # System commands, app control, settings
    MULTI = auto()      # Multi-step, compound tasks — a fugue of intents


# ---------------------------------------------------------------------------
# Route — a routing decision
# ---------------------------------------------------------------------------

@dataclass
class Route:
    """A routing decision — the score for the upcoming performance.

    Contains everything needed to direct the user's request to the
    right provider with the right context: the classified intent,
    the confidence of classification, the preferred providers, the
    grammar context to enrich the prompt, and the tools required.
    """

    intent: IntentCategory
    confidence: float
    provider_preference: list[str] = field(default_factory=list)
    grammar_context: dict[str, Any] = field(default_factory=dict)
    tools_needed: list[str] = field(default_factory=list)
    domains: list[str] = field(default_factory=list)

    def __repr__(self) -> str:
        return (
            f"Route({self.intent.name}, confidence={self.confidence:.2f}, "
            f"providers={self.provider_preference}, "
            f"domains={self.domains})"
        )


# ---------------------------------------------------------------------------
# Keyword heuristics — the surface grammar of intent
# ---------------------------------------------------------------------------

# Each pattern set maps to an intent category.  These are the *surface*
# markers — shallow but fast.  The GLM grammar analysis provides depth.

_INTENT_KEYWORDS: dict[IntentCategory, list[str]] = {
    IntentCategory.CODE: [
        "code", "function", "debug", "error", "implement", "refactor",
        "class", "api", "bug", "compile", "syntax", "variable", "method",
        "algorithm", "program", "script", "import", "module", "exception",
        "stacktrace", "traceback", "lint", "test", "unittest", "deploy",
        "git", "commit", "merge", "pull request", "regex", "parse",
        "def ", "return ", "async", "await", "lambda",
    ],
    IntentCategory.SEARCH: [
        "search", "find", "look up", "lookup", "what is", "who is",
        "when did", "where is", "how many", "how much", "define",
        "wikipedia", "google", "latest news", "current", "recent",
    ],
    IntentCategory.CREATE: [
        "write", "create", "draft", "compose", "design", "make",
        "generate", "build", "produce", "author", "outline", "sketch",
        "brainstorm", "template", "format", "document", "essay", "poem",
        "story", "letter", "email", "report", "presentation", "blog",
    ],
    IntentCategory.TRANSLATE: [
        "translate", "translation", "in french", "in spanish",
        "in german", "in italian", "in portuguese", "in japanese",
        "in chinese", "in korean", "in arabic", "in hindi",
        "in russian", "en espanol", "en francais", "auf deutsch",
        "to english", "from english", "language", "localize",
        "localise", "i18n", "multilingual",
    ],
    IntentCategory.COMMAND: [
        "/", "settings", "configure", "open", "close", "toggle",
        "enable", "disable", "switch", "mode", "set ", "reset",
        "clear", "exit", "quit", "restart", "update", "install",
        "uninstall", "preferences",
    ],
    IntentCategory.PREDICT: [
        "predict", "forecast", "future", "will ", "next ",
        "projection", "trend", "anticipate", "expect", "outlook",
        "scenario", "likelihood", "probability", "extrapolate",
        "what if", "what happens",
    ],
    IntentCategory.EXPLAIN: [
        "explain", "why", "how does", "how do", "teach", "tutorial",
        "understand", "meaning", "concept", "learn", "clarify",
        "elaborate", "describe", "detail", "break down", "eli5",
        "in simple terms", "step by step", "walkthrough",
    ],
    IntentCategory.ANALYZE: [
        "analyze", "analyse", "compare", "contrast", "evaluate",
        "assess", "review", "inspect", "examine", "breakdown",
        "statistics", "data", "metric", "measure", "benchmark",
        "pros and cons", "tradeoff", "trade-off", "reasoning",
        "logic", "deduce", "infer", "calculate", "compute",
    ],
}

# Domain mappings — which grammar domains are relevant for each intent
_INTENT_DOMAINS: dict[IntentCategory, list[str]] = {
    IntentCategory.CHAT: ["linguistic"],
    IntentCategory.CODE: ["computational", "mathematical"],
    IntentCategory.SEARCH: ["linguistic", "etymological"],
    IntentCategory.CREATE: ["linguistic", "etymological"],
    IntentCategory.ANALYZE: ["mathematical", "computational", "linguistic"],
    IntentCategory.TRANSLATE: ["linguistic", "etymological"],
    IntentCategory.PREDICT: ["mathematical", "physics", "biological"],
    IntentCategory.EXPLAIN: ["linguistic", "etymological"],
    IntentCategory.COMMAND: ["computational"],
    IntentCategory.MULTI: ["linguistic", "computational"],
}

# Provider preferences — the Choir's seating chart
_PROVIDER_PREFERENCES: dict[IntentCategory, list[str]] = {
    IntentCategory.CHAT: ["anthropic", "openai", "google", "local"],
    IntentCategory.CODE: ["anthropic", "openai", "local"],
    IntentCategory.SEARCH: ["google", "openai", "anthropic"],
    IntentCategory.CREATE: ["openai", "anthropic", "google"],
    IntentCategory.ANALYZE: ["anthropic", "openai", "google"],
    IntentCategory.TRANSLATE: ["google", "anthropic", "openai"],
    IntentCategory.PREDICT: ["local", "anthropic"],
    IntentCategory.EXPLAIN: ["anthropic", "openai", "google"],
    IntentCategory.COMMAND: ["local"],
    IntentCategory.MULTI: ["hybrid", "anthropic", "openai"],
}

# Tools that may be needed for each intent
_INTENT_TOOLS: dict[IntentCategory, list[str]] = {
    IntentCategory.CHAT: [],
    IntentCategory.CODE: ["coder", "file_read", "file_write"],
    IntentCategory.SEARCH: ["web_search"],
    IntentCategory.CREATE: ["file_write"],
    IntentCategory.ANALYZE: ["file_read", "calculator"],
    IntentCategory.TRANSLATE: ["translator"],
    IntentCategory.PREDICT: ["glm_predict", "glm_superforecast"],
    IntentCategory.EXPLAIN: [],
    IntentCategory.COMMAND: ["app_control"],
    IntentCategory.MULTI: [],
}


# ---------------------------------------------------------------------------
# The Router
# ---------------------------------------------------------------------------

class Router:
    """The Universal Router — switchboard of the EVERYTHING app.

    The Router listens to the user's opening phrase and decides how
    the performance should unfold.  It is not a simple keyword matcher;
    it uses the GLM's grammar engine to understand the *structure* of
    intent, then routes to the best available provider with grammar-
    enriched context.

    Like a conductor who reads the score and assigns the voices:
    the Router reads the user's intent and assigns the providers.
    """

    def __init__(self, angel: "Angel | None" = None):
        self._angel = angel
        self._angel_loaded = angel is not None

    # ------------------------------------------------------------------
    # Angel access — lazy loading
    # ------------------------------------------------------------------

    def _get_angel(self) -> "Angel | None":
        """Lazy-load the Angel — awaken it only when needed.

        The Angel is heavy; we do not awaken it for simple keyword
        classification.  Only when grammar analysis is required does
        the Angel open its eyes.
        """
        if self._angel is not None:
            return self._angel

        if self._angel_loaded:
            # Already tried and failed; do not retry
            return None

        try:
            from glm.angel import Angel
            self._angel = Angel()
            self._angel.awaken()
            self._angel_loaded = True
            return self._angel
        except Exception:
            self._angel_loaded = True
            return None

    # ------------------------------------------------------------------
    # Classification — reading the opening phrase
    # ------------------------------------------------------------------

    def classify(self, user_input: str) -> Route:
        """Classify intent using GLM grammar analysis.

        Two-phase classification:
            Phase 1 — Keyword heuristics (fast, shallow)
            Phase 2 — Grammar derivation (deep, structural)

        The final intent is a weighted blend.  If grammar analysis
        disagrees with keywords, we trust the grammar — structure
        is deeper than surface.

        Args:
            user_input: The raw user input string.

        Returns:
            A Route containing the classified intent and all
            routing metadata.
        """
        text = user_input.strip()
        lower = text.lower()

        # Phase 1: Keyword heuristics — fast surface scan
        keyword_scores = self._score_keywords(lower)

        # Phase 2: Grammar analysis — deep structure (if Angel available)
        grammar_scores = self._score_grammar(text)

        # Blend the scores: grammar gets 60% weight when available
        blended = self._blend_scores(keyword_scores, grammar_scores)

        # Special case: commands always win if input starts with "/"
        if lower.startswith("/"):
            blended[IntentCategory.COMMAND] = 1.0

        # Find the winner
        if not blended:
            best_intent = IntentCategory.CHAT
            best_score = 0.3
        else:
            best_intent = max(blended, key=blended.get)  # type: ignore[arg-type]
            best_score = blended[best_intent]

        # Check for MULTI — if multiple intents score highly, it is
        # a compound task (a fugue of intents)
        high_scorers = [
            cat for cat, score in blended.items()
            if score > 0.4 and cat != best_intent
        ]
        if len(high_scorers) >= 2 and best_score < 0.7:
            best_intent = IntentCategory.MULTI
            best_score = 0.6

        # Gather grammar context from the Angel if available
        grammar_context = self._gather_grammar_context(text, best_intent)

        return Route(
            intent=best_intent,
            confidence=min(best_score, 1.0),
            provider_preference=list(
                _PROVIDER_PREFERENCES.get(best_intent, ["local"])
            ),
            grammar_context=grammar_context,
            tools_needed=list(
                _INTENT_TOOLS.get(best_intent, [])
            ),
            domains=list(
                _INTENT_DOMAINS.get(best_intent, ["linguistic"])
            ),
        )

    def _score_keywords(self, lower_input: str) -> dict[IntentCategory, float]:
        """Score each intent category by keyword matches.

        Returns a dict of category -> score in [0, 1].
        The score is the fraction of that category's keywords found
        in the input, boosted by early-position matches.
        """
        scores: dict[IntentCategory, float] = {}

        for category, keywords in _INTENT_KEYWORDS.items():
            hits = 0
            total = len(keywords)
            if total == 0:
                continue

            for kw in keywords:
                if kw in lower_input:
                    hits += 1
                    # Bonus for keywords appearing at the start
                    if lower_input.startswith(kw):
                        hits += 0.5

            if hits > 0:
                # Normalise but cap at 1.0; even 2-3 hits is strong signal
                score = min(hits / max(total * 0.15, 1.0), 1.0)
                scores[category] = score

        return scores

    def _score_grammar(self, text: str) -> dict[IntentCategory, float]:
        """Score intents using GLM grammar analysis.

        The Angel derives the input through relevant grammars and
        measures how well it fits each domain's structural patterns.
        A good derivation = high confidence that this domain is
        relevant = evidence for the corresponding intent.
        """
        angel = self._get_angel()
        if angel is None:
            return {}

        scores: dict[IntentCategory, float] = {}
        tokens = text.lower().split()

        if not tokens:
            return {}

        # Try derivation in each domain and score based on prediction
        # confidence.  Higher confidence = better structural fit.
        domain_to_intents: dict[str, list[IntentCategory]] = {}
        for intent, domains in _INTENT_DOMAINS.items():
            for domain in domains:
                if domain not in domain_to_intents:
                    domain_to_intents[domain] = []
                domain_to_intents[domain].append(intent)

        for domain, intents in domain_to_intents.items():
            try:
                predictions = angel.predict(
                    tokens, domain=domain, horizon=3
                )
                if predictions:
                    # Average confidence across predictions
                    avg_conf = sum(
                        p.get("confidence", 0.0) for p in predictions
                    ) / len(predictions)
                    for intent in intents:
                        current = scores.get(intent, 0.0)
                        scores[intent] = max(current, avg_conf)
            except Exception:
                continue

        return scores

    def _blend_scores(
        self,
        keyword_scores: dict[IntentCategory, float],
        grammar_scores: dict[IntentCategory, float],
    ) -> dict[IntentCategory, float]:
        """Blend keyword and grammar scores.

        Grammar gets 60% weight when available — structure is
        deeper than surface.  Keywords get 40%.  When grammar
        is unavailable, keywords get full weight.
        """
        has_grammar = bool(grammar_scores)
        kw_weight = 0.4 if has_grammar else 1.0
        gr_weight = 0.6 if has_grammar else 0.0

        all_categories = set(keyword_scores) | set(grammar_scores)
        blended: dict[IntentCategory, float] = {}

        for cat in all_categories:
            kw = keyword_scores.get(cat, 0.0)
            gr = grammar_scores.get(cat, 0.0)
            blended[cat] = (kw * kw_weight) + (gr * gr_weight)

        return blended

    def _gather_grammar_context(
        self,
        text: str,
        intent: IntentCategory,
    ) -> dict[str, Any]:
        """Gather grammar context from the Angel for prompt enrichment.

        The context includes derivation results, detected patterns,
        and strange loop information — all the structural intelligence
        the GLM can bring to bear on this input.
        """
        angel = self._get_angel()
        if angel is None:
            return {}

        tokens = text.lower().split()
        if not tokens:
            return {}

        context: dict[str, Any] = {}
        domains = _INTENT_DOMAINS.get(intent, ["linguistic"])

        for domain in domains:
            try:
                predictions = angel.predict(
                    tokens, domain=domain, horizon=4
                )
                if predictions:
                    context[f"{domain}_predictions"] = [
                        {
                            "predicted": p.get("predicted", ""),
                            "confidence": p.get("confidence", 0.0),
                            "grammar": p.get("grammar", ""),
                        }
                        for p in predictions[:3]
                    ]
            except Exception:
                continue

        # For PREDICT and ANALYZE, use superforecast for richer context
        if intent in (IntentCategory.PREDICT, IntentCategory.ANALYZE):
            primary_domain = domains[0] if domains else "linguistic"
            try:
                forecast = angel.superforecast(
                    tokens, domain=primary_domain, horizon=4
                )
                context["superforecast"] = {
                    "overall_confidence": forecast.get(
                        "overall_confidence", 0.0
                    ),
                    "reasoning": forecast.get("reasoning", []),
                    "strange_loops": len(
                        forecast.get("strange_loops", [])
                    ),
                }
            except Exception:
                pass

        # For TRANSLATE, invoke the Angel's translate method
        if intent == IntentCategory.TRANSLATE and len(domains) >= 2:
            try:
                translations = angel.translate(
                    tokens,
                    source_domain=domains[0],
                    target_domain=domains[1],
                )
                if translations:
                    context["cross_domain_translations"] = [
                        {
                            "source_grammar": t.get("source_grammar", ""),
                            "target_grammar": t.get("target_grammar", ""),
                        }
                        for t in translations[:3]
                    ]
            except Exception:
                pass

        return context

    # ------------------------------------------------------------------
    # Enrichment — adding grammar context to the prompt
    # ------------------------------------------------------------------

    def enrich(self, user_input: str, route: Route) -> str:
        """Enrich the prompt with grammar context for the provider.

        The grammar context is woven into the system-level framing,
        giving the provider structural hints that improve its
        generation.  Like a musician being told "this is in C minor,
        the theme inverts at bar 12" before improvising.

        Args:
            user_input: The original user input.
            route: The routing decision containing grammar context.

        Returns:
            The enriched prompt string with grammar hints prepended.
        """
        if not route.grammar_context:
            return user_input

        # Build a concise grammar context block
        context_lines = ["[GLM Grammar Context]"]

        for key, value in route.grammar_context.items():
            if key.endswith("_predictions") and isinstance(value, list):
                domain = key.replace("_predictions", "")
                preds = [
                    f"{p.get('predicted', '?')} "
                    f"(conf={p.get('confidence', 0):.2f})"
                    for p in value
                ]
                if preds:
                    context_lines.append(
                        f"  {domain}: {', '.join(preds)}"
                    )
            elif key == "superforecast" and isinstance(value, dict):
                conf = value.get("overall_confidence", 0.0)
                loops = value.get("strange_loops", 0)
                reasoning = value.get("reasoning", [])
                context_lines.append(
                    f"  Superforecast: confidence={conf:.2f}, "
                    f"strange_loops={loops}"
                )
                for reason in reasoning[:2]:
                    context_lines.append(f"    - {reason}")
            elif key == "cross_domain_translations" and isinstance(value, list):
                for t in value:
                    context_lines.append(
                        f"  Translation: {t.get('source_grammar', '?')} "
                        f"-> {t.get('target_grammar', '?')}"
                    )

        # Only prepend if we actually gathered useful context
        if len(context_lines) <= 1:
            return user_input

        context_block = "\n".join(context_lines)
        return f"{context_block}\n\n{user_input}"

    # ------------------------------------------------------------------
    # Provider selection — assigning the voice
    # ------------------------------------------------------------------

    def select_provider(
        self,
        route: Route,
        available_providers: dict[str, "Provider"],
    ) -> "Provider":
        """Select the best available provider for this route.

        Walks the preference list in order and returns the first
        provider that is both present in the available dict and
        reports itself as available.  If none match, falls back
        to whatever is available.

        Like casting a play: you want your first-choice actor, but
        the show must go on with whoever is in the building.

        Args:
            route: The routing decision with provider preferences.
            available_providers: Dict of provider_name -> Provider.

        Returns:
            The selected Provider instance.

        Raises:
            RuntimeError: If no providers are available at all.
        """
        # Try each preferred provider in order
        for pref in route.provider_preference:
            provider = available_providers.get(pref)
            if provider is not None and provider.is_available():
                return provider

        # Preference list exhausted — try any available provider
        for name, provider in available_providers.items():
            if provider.is_available():
                return provider

        # Nothing available — last resort
        raise RuntimeError(
            "No providers available. Configure an API key with "
            "/settings or ensure LocalProvider is loaded."
        )

    # ------------------------------------------------------------------
    # Post-processing — the GLM's final word
    # ------------------------------------------------------------------

    def post_process(self, response: str, route: Route) -> str:
        """Post-process the response through GLM if beneficial.

        Not every response needs post-processing.  The Router only
        intervenes when grammar analysis can add value: structural
        validation for code, pattern detection for predictions,
        cross-domain enrichment for analysis.

        Like an editor who only touches the manuscript where it
        genuinely improves — the lightest hand that serves the work.

        Args:
            response: The raw response from the provider.
            route: The routing decision.

        Returns:
            The (possibly enriched) response string.
        """
        # Only post-process when grammar adds value
        if route.intent not in (
            IntentCategory.PREDICT,
            IntentCategory.ANALYZE,
            IntentCategory.CODE,
        ):
            return response

        angel = self._get_angel()
        if angel is None:
            return response

        # For predictions, append GLM structural analysis
        if route.intent == IntentCategory.PREDICT:
            return self._post_process_prediction(response, angel, route)

        # For analysis, append cross-domain harmonics
        if route.intent == IntentCategory.ANALYZE:
            return self._post_process_analysis(response, angel, route)

        # For code, append structural notes from computational grammar
        if route.intent == IntentCategory.CODE:
            return self._post_process_code(response, angel, route)

        return response

    def _post_process_prediction(
        self,
        response: str,
        angel: "Angel",
        route: Route,
    ) -> str:
        """Append GLM structural forecast to a prediction response."""
        forecast_info = route.grammar_context.get("superforecast")
        if not forecast_info:
            return response

        reasoning = forecast_info.get("reasoning", [])
        if not reasoning:
            return response

        addendum_lines = [
            "",
            "---",
            "[GLM Structural Analysis]",
        ]
        for reason in reasoning:
            addendum_lines.append(f"  {reason}")

        confidence = forecast_info.get("overall_confidence", 0.0)
        addendum_lines.append(
            f"  Overall structural confidence: {confidence:.2f}"
        )

        return response + "\n".join(addendum_lines)

    def _post_process_analysis(
        self,
        response: str,
        angel: "Angel",
        route: Route,
    ) -> str:
        """Append cross-domain pattern notes to an analysis response."""
        context = route.grammar_context
        # Look for any domain predictions that reveal structural patterns
        pattern_notes = []
        for key, value in context.items():
            if key.endswith("_predictions") and isinstance(value, list):
                domain = key.replace("_predictions", "")
                high_conf = [
                    p for p in value
                    if p.get("confidence", 0) > 0.6
                ]
                if high_conf:
                    pattern_notes.append(
                        f"  {domain}: {len(high_conf)} high-confidence "
                        f"structural patterns detected"
                    )

        if not pattern_notes:
            return response

        addendum = "\n".join(
            ["", "---", "[GLM Pattern Detection]"] + pattern_notes
        )
        return response + addendum

    def _post_process_code(
        self,
        response: str,
        angel: "Angel",
        route: Route,
    ) -> str:
        """Append structural notes from computational grammar to code."""
        comp_preds = route.grammar_context.get(
            "computational_predictions", []
        )
        if not comp_preds:
            return response

        notes = []
        for pred in comp_preds:
            grammar = pred.get("grammar", "")
            conf = pred.get("confidence", 0.0)
            if conf > 0.5:
                notes.append(
                    f"  {grammar}: structural pattern match "
                    f"(confidence {conf:.2f})"
                )

        if not notes:
            return response

        addendum = "\n".join(
            ["", "---", "[GLM Code Structure Analysis]"] + notes
        )
        return response + addendum

    # ------------------------------------------------------------------
    # Full routing pipeline — the complete performance
    # ------------------------------------------------------------------

    def route(
        self,
        user_input: str,
        providers: dict[str, "Provider"],
    ) -> tuple["Provider", str, Route]:
        """Full routing pipeline: classify, enrich, select.

        This is the main entry point.  It takes raw user input and
        a dict of available providers, and returns everything needed
        to make the call: the provider, the enriched prompt, and the
        routing decision (for logging and post-processing).

        Like a conductor raising the baton: in one gesture, the
        voices are assigned, the key is set, and the tempo is given.

        Args:
            user_input: Raw user input string.
            providers: Dict mapping provider names to Provider instances.

        Returns:
            A tuple of (selected_provider, enriched_prompt, route).

        Raises:
            RuntimeError: If no providers are available.
        """
        # 1. Classify intent
        classified_route = self.classify(user_input)

        # 2. Enrich the prompt with grammar context
        enriched_prompt = self.enrich(user_input, classified_route)

        # 3. Select the best available provider
        provider = self.select_provider(classified_route, providers)

        return provider, enriched_prompt, classified_route

    # ------------------------------------------------------------------
    # String representation
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        angel_status = "connected" if self._angel is not None else "disconnected"
        return f"Router(angel={angel_status})"
