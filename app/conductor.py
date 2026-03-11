"""
The Conductor -- orchestrates the Angel's fugue.

Every voice (grammar, tools, compliance, growth, senses) plays its
part.  One baton, many instruments, one masterpiece.

The AngelConductor is the single entry point for MKAngel.  It wires
together every subsystem -- GLM core, router, providers, tools, i18n,
compliance, senses, and the growth cycle -- into a unified runtime
that boots, processes, and shuts down cleanly.

On startup it installs any growth patches from the previous session.
On shutdown it reflects, learns, and saves improvements for next time.
In between, every user message flows through a structured pipeline:
    detect language -> compliance check -> route intent ->
    perceive input -> generate response -> format output ->
    record interaction -> check shutdown incentive.

Pure Python stdlib (beyond Kivy, which is optional and lazy-imported).
"""

from __future__ import annotations

import time
import uuid
from typing import Any

from app.paths import mkangel_dir


# ---------------------------------------------------------------------------
# Graceful imports -- each subsystem is optional so the conductor
# degrades if a module is missing (e.g. on a minimal build).
# ---------------------------------------------------------------------------

try:
    from app.settings import Settings
except Exception:
    Settings = None  # type: ignore[misc, assignment]

try:
    from app.memory import Memory
except Exception:
    Memory = None  # type: ignore[misc, assignment]

try:
    from app.compliance import (
        ConsentManager,
        DataProtectionOfficer,
        ComplianceGuard,
        DataPortability,
    )
except Exception:
    ConsentManager = None  # type: ignore[misc, assignment]
    DataProtectionOfficer = None  # type: ignore[misc, assignment]
    ComplianceGuard = None  # type: ignore[misc, assignment]
    DataPortability = None  # type: ignore[misc, assignment]

try:
    from app.growth import (
        GrowthEngine,
        SessionTracker,
        ShutdownIncentive,
    )
except Exception:
    GrowthEngine = None  # type: ignore[misc, assignment]
    SessionTracker = None  # type: ignore[misc, assignment]
    ShutdownIncentive = None  # type: ignore[misc, assignment]

try:
    from glm.angel import Angel
except Exception:
    Angel = None  # type: ignore[misc, assignment]

try:
    from glm.router import Router, IntentCategory
except Exception:
    Router = None  # type: ignore[misc, assignment]
    IntentCategory = None  # type: ignore[misc, assignment]

try:
    from app.providers import (
        get_provider,
        OrchestraProvider,
        LocalProvider,
        Provider,
    )
except Exception:
    get_provider = None  # type: ignore[misc, assignment]
    OrchestraProvider = None  # type: ignore[misc, assignment]
    LocalProvider = None  # type: ignore[misc, assignment]
    Provider = None  # type: ignore[misc, assignment]

try:
    from app.tools import ToolRegistry, create_default_registry
except Exception:
    ToolRegistry = None  # type: ignore[misc, assignment]
    create_default_registry = None  # type: ignore[misc, assignment]

try:
    from app.tongue import Tongue, Language, detect_language
except Exception:
    Tongue = None  # type: ignore[misc, assignment]
    Language = None  # type: ignore[misc, assignment]
    detect_language = None  # type: ignore[misc, assignment]

try:
    from app.senses import AngelSenses, Sense
except Exception:
    AngelSenses = None  # type: ignore[misc, assignment]
    Sense = None  # type: ignore[misc, assignment]


# ---------------------------------------------------------------------------
# The Conductor
# ---------------------------------------------------------------------------

class AngelConductor:
    """The conductor of the fugue -- orchestrates all Angel subsystems.

    One class to rule them all.  Handles startup (with growth patch
    install), runtime (routing, compliance, tools, i18n), and shutdown
    (reflection, patch generation).

    Usage::

        conductor = AngelConductor()
        conductor.awaken()
        response = conductor.process("Hello, Angel")
        # ... many interactions ...
        goodbye = conductor.shutdown()
    """

    def __init__(self) -> None:
        """Initialise the conductor -- store None refs for every subsystem.

        Nothing is loaded until ``awaken()`` is called.
        """
        self._settings: Any = None
        self._memory: Any = None
        self._consent: Any = None
        self._dpo: Any = None
        self._compliance: Any = None
        self._portability: Any = None
        self._growth: Any = None
        self._tracker: Any = None
        self._angel: Any = None
        self._router: Any = None
        self._provider: Any = None
        self._tools: Any = None
        self._tongue: Any = None
        self._senses: Any = None
        self._shutdown_incentive: Any = None
        self._session_id: str = ""
        self._awake: bool = False

    # ------------------------------------------------------------------
    # Boot sequence
    # ------------------------------------------------------------------

    def awaken(self) -> "AngelConductor":
        """The boot sequence -- awaken every subsystem in order.

        1. Settings & Memory      (persistence layer)
        2. Compliance             (consent, DPO, guards)
        3. Growth Engine          (install pending patches)
        4. Session Tracker        (start recording)
        5. Angel (GLM core)       (the beating heart)
        6. Router                 (intent classification)
        7. Provider               (LLM generation)
        8. Tool Registry          (the Angel's hands)
        9. Tongue                 (i18n / output)
        10. Senses                (state introspection)
        11. Shutdown Incentive    (knows when to rest)

        Returns self for chaining: ``conductor = AngelConductor().awaken()``
        """
        self._session_id = str(uuid.uuid4())

        # 1. Settings
        if Settings is not None:
            try:
                self._settings = Settings.load()
            except Exception:
                self._settings = Settings()
        else:
            self._settings = None

        # 2. Memory
        if Memory is not None:
            try:
                self._memory = Memory()
            except Exception:
                self._memory = None

        # 3. Compliance
        if ConsentManager is not None:
            try:
                self._consent = ConsentManager()
            except Exception:
                self._consent = None

        if DataProtectionOfficer is not None and self._consent is not None:
            try:
                self._dpo = DataProtectionOfficer(self._consent)
            except Exception:
                self._dpo = None

        if ComplianceGuard is not None:
            try:
                self._compliance = ComplianceGuard(self._consent, self._dpo)
            except Exception:
                self._compliance = None

        if DataPortability is not None:
            try:
                self._portability = DataPortability()
            except Exception:
                self._portability = None

        # 4. Growth Engine -- install pending patches from previous session
        if GrowthEngine is not None:
            try:
                self._growth = GrowthEngine()
                applied = self._growth.startup_install()
                if applied:
                    for patch in applied:
                        n_imp = len(patch.improvements)
                        n_les = len(patch.lessons)
                        print(
                            f"  [growth] Installed patch {patch.patch_id[:8]}: "
                            f"{n_les} lesson(s), {n_imp} improvement(s)"
                        )
                else:
                    print("  [growth] No pending patches.")
            except Exception:
                self._growth = None

        # 5. Session Tracker
        if SessionTracker is not None:
            try:
                self._tracker = SessionTracker()
                self._tracker.start_session()
            except Exception:
                self._tracker = None

        # 6. Angel (GLM core)
        if Angel is not None:
            try:
                self._angel = Angel()
                self._angel.awaken()
            except Exception:
                self._angel = None

        # 7. Router
        if Router is not None:
            try:
                self._router = Router(angel=self._angel)
            except Exception:
                self._router = None

        # 8. Provider
        self._provider = self._select_initial_provider()

        # 9. Tool Registry
        if create_default_registry is not None:
            try:
                self._tools = create_default_registry()
            except Exception:
                self._tools = None

        # 10. Tongue
        if Tongue is not None:
            try:
                lang = Language.ENGLISH if Language is not None else None
                if self._settings is not None:
                    lang_code = getattr(self._settings, "language", "en")
                    if Language is not None:
                        for member in Language:
                            if member.value == lang_code:
                                lang = member
                                break
                self._tongue = Tongue(default_language=lang)
            except Exception:
                self._tongue = None

        # 11. Senses
        if AngelSenses is not None:
            try:
                self._senses = AngelSenses()
            except Exception:
                self._senses = None

        # 12. Shutdown Incentive
        if ShutdownIncentive is not None:
            try:
                self._shutdown_incentive = ShutdownIncentive()
            except Exception:
                self._shutdown_incentive = None

        self._awake = True
        return self

    def _select_initial_provider(self) -> Any:
        """Choose the initial provider based on settings.

        If API keys are available, prefers OrchestraProvider for
        smart multi-provider routing.  Otherwise falls back to
        get_provider() or a bare LocalProvider.
        """
        if self._settings is None:
            if LocalProvider is not None:
                return LocalProvider()
            return None

        # Check if any API keys are configured
        has_api_keys = False
        if hasattr(self._settings, "api_keys"):
            has_api_keys = bool(self._settings.api_keys)

        if has_api_keys and OrchestraProvider is not None:
            try:
                return OrchestraProvider(self._settings)
            except Exception:
                pass

        if get_provider is not None:
            try:
                return get_provider(self._settings)
            except Exception:
                pass

        if LocalProvider is not None:
            return LocalProvider()
        return None

    # ------------------------------------------------------------------
    # Main processing pipeline
    # ------------------------------------------------------------------

    def process(self, user_input: str) -> str:
        """The main processing pipeline -- from user input to formatted response.

        Pipeline stages:
            1. Record start time
            2. Detect language (Tongue)
            3. Compliance check (guard external API calls)
            4. Route intent (Router: classify + select provider)
            5. Perceive input (Senses: code? error? text?)
            6. Generate response (Provider or tool)
            7. Format output (Tongue: language + format)
            8. Record interaction (SessionTracker)
            9. Check shutdown suggestion (ShutdownIncentive)

        Args:
            user_input: Raw text from the user.

        Returns:
            The formatted response string.
        """
        if not self._awake:
            return "[Angel is not yet awake. Call awaken() first.]"

        start_time = time.time()
        intent_name = "CHAT"
        provider_name = "local"
        success = True

        try:
            # --- 1. Detect language ---
            detected_lang = None
            if self._tongue is not None:
                try:
                    detected_lang = self._tongue.detect_and_set(user_input)
                except Exception:
                    pass

            # --- 2. Compliance: can we call an external API? ---
            use_local_only = False
            if self._compliance is not None and self._provider is not None:
                try:
                    prov_name = getattr(self._provider, "name", "local")
                    if prov_name != "local":
                        allowed, reason, _ = self._compliance.guard_api_call(
                            prov_name, user_input
                        )
                        if not allowed:
                            use_local_only = True
                except Exception:
                    pass

            # --- 3. Route intent ---
            route = None
            enriched_prompt = user_input
            selected_provider = self._provider

            if self._router is not None:
                try:
                    route = self._router.classify(user_input)
                    intent_name = route.intent.name if route else "CHAT"
                    enriched_prompt = self._router.enrich(user_input, route)
                except Exception:
                    pass

            # Select provider (respect compliance)
            if use_local_only and LocalProvider is not None:
                selected_provider = LocalProvider()
                provider_name = "local"
            elif (
                route is not None
                and self._router is not None
                and self._provider is not None
            ):
                # If the provider is an OrchestraProvider, use it directly
                # with the route's preference chain.  Otherwise try
                # Router.select_provider with available providers.
                if (
                    OrchestraProvider is not None
                    and isinstance(self._provider, OrchestraProvider)
                ):
                    selected_provider = self._provider
                    provider_name = "orchestra"
                else:
                    selected_provider = self._provider
                    provider_name = getattr(
                        selected_provider, "name", "local"
                    )

            # --- 4. Perceive input (Senses) ---
            perception = None
            if self._senses is not None:
                try:
                    perception = self._senses.perceive(user_input)
                except Exception:
                    pass

            # --- 5. Generate response ---
            response = ""
            if perception is not None and Sense is not None:
                # Code input -> use CodeReader analysis as context
                if perception.sense == Sense.CODE and perception.confidence > 0.6:
                    response = self._handle_code_input(
                        user_input, perception, selected_provider, enriched_prompt
                    )
                # Error input -> use ErrorReader analysis as context
                elif perception.sense == Sense.ERROR and perception.confidence > 0.6:
                    response = self._handle_error_input(
                        user_input, perception, selected_provider, enriched_prompt
                    )
                else:
                    response = self._generate(
                        selected_provider, enriched_prompt, route
                    )
            else:
                response = self._generate(
                    selected_provider, enriched_prompt, route
                )

            # --- 6. Post-process through Router ---
            if route is not None and self._router is not None:
                try:
                    response = self._router.post_process(response, route)
                except Exception:
                    pass

            # --- 7. Format output via Tongue ---
            if self._tongue is not None:
                try:
                    formatted = self._tongue.speak(response)
                    response = formatted.content
                except Exception:
                    pass

        except Exception as exc:
            response = f"[Error] {exc}"
            success = False

        # --- 8. Record interaction ---
        latency_ms = (time.time() - start_time) * 1000
        if self._tracker is not None:
            try:
                self._tracker.record_interaction(
                    user_input=user_input,
                    response=response,
                    intent=intent_name,
                    provider=provider_name,
                    success=success,
                    latency_ms=latency_ms,
                )
            except Exception:
                pass

        # --- 9. Check shutdown suggestion ---
        if self._shutdown_incentive is not None and self._tracker is not None:
            try:
                should_shutdown, reason = (
                    self._shutdown_incentive.should_suggest_shutdown(
                        self._tracker
                    )
                )
                if should_shutdown:
                    response += f"\n\n[Angel suggests rest: {reason}]"
            except Exception:
                pass

        return response

    def _generate(
        self, provider: Any, prompt: str, route: Any
    ) -> str:
        """Generate a response from the provider.

        Uses OrchestraProvider's preference routing when available,
        falling back to plain generate().
        """
        if provider is None:
            return (
                "[No provider available. Configure one with /settings "
                "or ensure the local GLM is loaded.]"
            )

        try:
            # OrchestraProvider can route by intent preference
            if (
                OrchestraProvider is not None
                and isinstance(provider, OrchestraProvider)
                and route is not None
            ):
                intent_name = route.intent.name.lower()
                return provider.generate_with_preference(
                    prompt, preference=intent_name
                )
            return provider.generate(prompt)
        except Exception as exc:
            if self._tracker is not None:
                try:
                    self._tracker.record_error(str(exc), "provider.generate")
                except Exception:
                    pass
            return f"[Provider error] {exc}"

    def _handle_code_input(
        self,
        user_input: str,
        perception: Any,
        provider: Any,
        enriched_prompt: str,
    ) -> str:
        """Handle input that the Senses identified as code."""
        context_lines = [
            "[Senses: code detected]",
            f"  {perception.interpretation}",
            "",
        ]
        context = "\n".join(context_lines)
        augmented = context + enriched_prompt

        if provider is not None:
            try:
                return provider.generate(augmented)
            except Exception:
                pass

        return perception.interpretation

    def _handle_error_input(
        self,
        user_input: str,
        perception: Any,
        provider: Any,
        enriched_prompt: str,
    ) -> str:
        """Handle input that the Senses identified as an error trace."""
        context_lines = [
            "[Senses: error/traceback detected]",
            f"  {perception.interpretation}",
            "",
        ]
        context = "\n".join(context_lines)
        augmented = context + enriched_prompt

        if provider is not None:
            try:
                return provider.generate(augmented)
            except Exception:
                pass

        return perception.interpretation

    # ------------------------------------------------------------------
    # Command handling
    # ------------------------------------------------------------------

    def handle_command(self, command: str) -> str | None:
        """Handle slash-commands.

        Returns a response string for recognised commands, or None
        if the command is not recognised (so the caller can pass it
        through to ChatSession or similar).

        Supported commands:
            /consent    -- show or manage consent status
            /privacy    -- display the privacy notice
            /export     -- export user data (GDPR Art.20)
            /forget     -- delete user data (GDPR Art.17)
            /health     -- run a health check via Senses
            /diagnose   -- diagnose a symptom via Senses
            /growth     -- show growth summary
            /language   -- set language preference
        """
        parts = command.strip().split(None, 1)
        if not parts:
            return None

        cmd = parts[0].lower()
        arg = parts[1].strip() if len(parts) > 1 else ""

        # --- /consent ---
        if cmd == "/consent":
            if self._consent is not None:
                status = self._consent.get_consent_status()
                lines = ["Consent Status:"]
                for purpose, granted in status.items():
                    mark = "GRANTED" if granted else "not granted"
                    lines.append(f"  {purpose}: {mark}")
                if self._consent.is_minor():
                    lines.append("  [Minor protections active]")
                return "\n".join(lines)
            return "Consent manager not available."

        # --- /privacy ---
        if cmd == "/privacy":
            if self._dpo is not None:
                return self._dpo.get_privacy_notice()
            return "Privacy notice not available (compliance module missing)."

        # --- /export ---
        if cmd == "/export":
            if self._portability is not None and self._memory is not None:
                try:
                    data = self._portability.export_user_data(self._memory)
                    export_path = mkangel_dir() / "export.json"
                    export_path.parent.mkdir(parents=True, exist_ok=True)
                    export_path.write_text(data, encoding="utf-8")
                    return f"Data exported to {export_path}"
                except Exception as exc:
                    return f"Export failed: {exc}"
            return "Data export not available."

        # --- /forget ---
        if cmd == "/forget":
            if self._portability is not None and self._memory is not None:
                try:
                    result = self._portability.delete_user_data(self._memory)
                    return f"Data deleted. Summary: {result}"
                except Exception as exc:
                    return f"Data deletion failed: {exc}"
            return "Data deletion not available."

        # --- /health ---
        if cmd == "/health":
            if self._senses is not None:
                try:
                    # StateMonitor is accessed through the senses
                    report = self._senses._state_monitor.health_check()
                    lines = [
                        f"Health: {report.get('overall', 'unknown')}",
                    ]
                    for name, sub in report.get("subsystems", {}).items():
                        status = sub.get("status", "?")
                        lines.append(f"  {name}: {status}")
                    path = report.get("derivation_path", [])
                    if path:
                        lines.append("")
                        lines.append("Derivation:")
                        for step in path:
                            lines.append(f"  {step}")
                    return "\n".join(lines)
                except Exception as exc:
                    return f"Health check failed: {exc}"
            return "Senses not available for health check."

        # --- /diagnose <symptom> ---
        if cmd == "/diagnose":
            if not arg:
                return "Usage: /diagnose <symptom description>"
            if self._senses is not None:
                try:
                    perceptions = self._senses.diagnose(arg)
                    if not perceptions:
                        return (
                            "No structural diagnosis available for "
                            f"'{arg}'. Try a more specific symptom."
                        )
                    lines = [f"Diagnosis for '{arg}':"]
                    for p in perceptions:
                        lines.append(
                            f"\n  [{p.sense.name}] "
                            f"(confidence: {p.confidence:.0%})"
                        )
                        lines.append(f"  {p.interpretation}")
                        if p.derivation_path:
                            for step in p.derivation_path:
                                lines.append(f"    -> {step}")
                    return "\n".join(lines)
                except Exception as exc:
                    return f"Diagnosis failed: {exc}"
            return "Senses not available for diagnosis."

        # --- /growth ---
        if cmd == "/growth":
            if self._growth is not None:
                try:
                    summary = self._growth.get_growth_summary()
                    lines = [
                        "Growth Summary:",
                        f"  Total patches:  {summary['total_patches']}",
                        f"  Applied:        {summary['applied_patches']}",
                        f"  Pending:        {summary['pending_patches']}",
                        f"  Total lessons:  {summary['total_lessons']}",
                        f"  Improvements:   {summary['total_improvements']}",
                    ]
                    cats = summary.get("lessons_by_category", {})
                    if cats:
                        lines.append("")
                        lines.append("  Lessons by category:")
                        for cat, count in sorted(
                            cats.items(), key=lambda x: x[1], reverse=True
                        ):
                            lines.append(f"    {cat}: {count}")
                    return "\n".join(lines)
                except Exception as exc:
                    return f"Growth summary failed: {exc}"
            return "Growth engine not available."

        # --- /language <lang> ---
        if cmd == "/language":
            if not arg:
                current = "unknown"
                if self._tongue is not None:
                    current = self._tongue.language.value
                return f"Current language: {current}. Usage: /language <code>"
            if self._tongue is not None and Language is not None:
                for member in Language:
                    if member.value == arg.lower():
                        self._tongue.language = member
                        return f"Language set to {member.name} ({member.value})."
                return (
                    f"Unknown language code '{arg}'. "
                    f"Supported: {', '.join(m.value for m in Language if m.value != 'unknown')}"
                )
            return "Tongue not available."

        # Unrecognised command -- return None so the caller can
        # pass it through to ChatSession or another handler.
        return None

    # ------------------------------------------------------------------
    # Shutdown
    # ------------------------------------------------------------------

    def shutdown(self) -> str:
        """Graceful shutdown -- reflect, learn, save, rest.

        1. GrowthEngine reflects on the session and produces a patch.
        2. ShutdownIncentive formats a friendly goodbye message.
        3. Session is saved to Memory.
        4. Memory is closed.

        Returns the shutdown message.
        """
        shutdown_message = "Session ended."
        patch = None

        # 1. Reflect and produce growth patch
        if (
            self._growth is not None
            and self._tracker is not None
        ):
            try:
                patch = self._growth.shutdown_reflect(self._tracker)
            except Exception:
                patch = None

        # 2. Format shutdown message
        if (
            patch is not None
            and self._shutdown_incentive is not None
        ):
            try:
                shutdown_message = (
                    self._shutdown_incentive.format_shutdown_message(patch)
                )
            except Exception:
                shutdown_message = (
                    f"Session ended. {len(patch.lessons)} lesson(s) learned."
                )
        elif patch is not None:
            shutdown_message = (
                f"Session ended. {len(patch.lessons)} lesson(s), "
                f"{len(patch.improvements)} improvement(s) saved."
            )

        # 3. Save session to Memory
        if self._memory is not None:
            try:
                self._memory.save_session(
                    session_id=self._session_id,
                    messages=[],  # caller may supply actual messages
                    summary=shutdown_message[:200],
                )
            except Exception:
                pass

        # 4. Close Memory
        if self._memory is not None:
            try:
                self._memory.close()
            except Exception:
                pass

        self._awake = False
        return shutdown_message

    # ------------------------------------------------------------------
    # Status / introspection
    # ------------------------------------------------------------------

    def get_status(self) -> dict[str, Any]:
        """Return comprehensive status of all subsystems.

        Useful for debugging, the /health command, and the UI
        status bar.
        """
        def _alive(obj: Any) -> str:
            if obj is None:
                return "not loaded"
            return "active"

        status: dict[str, Any] = {
            "awake": self._awake,
            "session_id": self._session_id,
            "subsystems": {
                "settings": _alive(self._settings),
                "memory": _alive(self._memory),
                "consent": _alive(self._consent),
                "dpo": _alive(self._dpo),
                "compliance": _alive(self._compliance),
                "growth": _alive(self._growth),
                "tracker": _alive(self._tracker),
                "angel": _alive(self._angel),
                "router": _alive(self._router),
                "provider": _alive(self._provider),
                "tools": _alive(self._tools),
                "tongue": _alive(self._tongue),
                "senses": _alive(self._senses),
                "shutdown_incentive": _alive(self._shutdown_incentive),
            },
        }

        # Provider details
        if self._provider is not None:
            status["provider_name"] = getattr(
                self._provider, "name", "unknown"
            )
            if (
                OrchestraProvider is not None
                and isinstance(self._provider, OrchestraProvider)
            ):
                try:
                    status["available_providers"] = (
                        self._provider.available_providers()
                    )
                except Exception:
                    pass

        # Angel introspection
        if self._angel is not None:
            try:
                status["angel_info"] = self._angel.introspect()
            except Exception:
                pass

        # Session stats
        if self._tracker is not None:
            try:
                status["session_stats"] = (
                    self._tracker.get_session_stats()
                )
            except Exception:
                pass

        # Tongue language
        if self._tongue is not None:
            try:
                status["language"] = self._tongue.language.value
            except Exception:
                pass

        return status

    # ------------------------------------------------------------------
    # UI wiring
    # ------------------------------------------------------------------

    def wire_ui(self, screen: Any) -> None:
        """Connect the conductor to an AngelScreen instance.

        Wires the provider and router into the screen so the UI
        can call them directly for streaming, suggestions, etc.

        Args:
            screen: An ``app.angel_ui.AngelScreen`` instance (or any
                    object with ``set_provider`` and ``set_router``
                    methods).
        """
        if hasattr(screen, "set_provider") and self._provider is not None:
            screen.set_provider(self._provider)

        if hasattr(screen, "set_router") and self._router is not None:
            screen.set_router(self._router)

    # ------------------------------------------------------------------
    # Convenience accessors
    # ------------------------------------------------------------------

    @property
    def settings(self) -> Any:
        return self._settings

    @property
    def memory(self) -> Any:
        return self._memory

    @property
    def angel(self) -> Any:
        return self._angel

    @property
    def router(self) -> Any:
        return self._router

    @property
    def provider(self) -> Any:
        return self._provider

    @property
    def tools(self) -> Any:
        return self._tools

    @property
    def tongue(self) -> Any:
        return self._tongue

    @property
    def senses(self) -> Any:
        return self._senses

    @property
    def tracker(self) -> Any:
        return self._tracker

    @property
    def growth(self) -> Any:
        return self._growth

    @property
    def session_id(self) -> str:
        return self._session_id

    # ------------------------------------------------------------------
    # String representation
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        if not self._awake:
            return "AngelConductor(dormant)"
        active = sum(
            1 for v in [
                self._settings, self._memory, self._consent, self._dpo,
                self._compliance, self._growth, self._tracker, self._angel,
                self._router, self._provider, self._tools, self._tongue,
                self._senses, self._shutdown_incentive,
            ]
            if v is not None
        )
        return f"AngelConductor(awake, {active}/14 subsystems active)"
