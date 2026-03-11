"""
Chat interface for MKAngel.

Manages interactive conversations with the Angel through a
terminal-based interface with ANSI colour formatting, command
handling, and message history.
"""

from __future__ import annotations

import json
import shutil
import textwrap
import time
import uuid
from typing import Any


# ---------------------------------------------------------------------------
# ANSI colour palette
# ---------------------------------------------------------------------------

class C:
    """Terminal colour codes."""
    RESET   = "\033[0m"
    BOLD    = "\033[1m"
    DIM     = "\033[2m"
    ITALIC  = "\033[3m"
    UNDER   = "\033[4m"

    BLACK   = "\033[30m"
    RED     = "\033[31m"
    GREEN   = "\033[32m"
    YELLOW  = "\033[33m"
    BLUE    = "\033[34m"
    MAGENTA = "\033[35m"
    CYAN    = "\033[36m"
    WHITE   = "\033[37m"

    BG_BLACK  = "\033[40m"
    BG_RED    = "\033[41m"
    BG_BLUE   = "\033[44m"

    BRIGHT_CYAN    = "\033[96m"
    BRIGHT_GREEN   = "\033[92m"
    BRIGHT_YELLOW  = "\033[93m"
    BRIGHT_MAGENTA = "\033[95m"
    BRIGHT_WHITE   = "\033[97m"


# ---------------------------------------------------------------------------
# Formatting helpers
# ---------------------------------------------------------------------------

def _term_width() -> int:
    """Get terminal width, defaulting to 80."""
    return shutil.get_terminal_size((80, 24)).columns


def _wrap(text: str, indent: int = 2) -> str:
    """Word-wrap text to terminal width with indent."""
    width = max(_term_width() - indent - 2, 40)
    prefix = " " * indent
    lines = text.splitlines()
    wrapped = []
    for line in lines:
        if not line.strip():
            wrapped.append("")
        else:
            for wl in textwrap.wrap(line, width=width):
                wrapped.append(prefix + wl)
    return "\n".join(wrapped)


def _header(text: str) -> str:
    """Format a section header."""
    return f"\n{C.BOLD}{C.CYAN}{text}{C.RESET}"


def _info(text: str) -> str:
    """Format an info message."""
    return f"{C.DIM}{C.WHITE}{text}{C.RESET}"


def _success(text: str) -> str:
    """Format a success message."""
    return f"{C.BRIGHT_GREEN}{text}{C.RESET}"


def _warn(text: str) -> str:
    """Format a warning message."""
    return f"{C.YELLOW}{text}{C.RESET}"


def _error(text: str) -> str:
    """Format an error message."""
    return f"{C.RED}{C.BOLD}{text}{C.RESET}"


def _angel_response(text: str) -> str:
    """Format an Angel response."""
    lines = text.splitlines()
    formatted = []
    for line in lines:
        formatted.append(f"  {C.WHITE}{line}{C.RESET}")
    return "\n".join(formatted)


def _prompt_str() -> str:
    """The input prompt string."""
    return f"{C.BRIGHT_CYAN}{C.BOLD}> {C.RESET}"


# ---------------------------------------------------------------------------
# Help text
# ---------------------------------------------------------------------------

_COMMANDS: dict[str, tuple[str, str]] = {
    "/help":        ("",            "Show this help message"),
    "/status":      ("",            "Show Angel status and system info"),
    "/predict":     ("<tokens>",    "Predict next elements from grammatical structure"),
    "/reconstruct": ("<tokens>",    "Reconstruct origins / trace backward"),
    "/forecast":    ("<tokens>",    "Superforecast using grammar + strange loops"),
    "/fugue":       ("<tokens>",    "Compose a fugue across all domains"),
    "/translate":   ("<src> <dst> <tokens>", "Translate patterns between domains"),
    "/introspect":  ("",            "Angel examines its own structure"),
    "/memory":      ("[search <q>]","View or search persistent memory"),
    "/settings":    ("[key val]",   "View or change settings"),
    "/code":        ("<action> ...", "Code operations: generate, analyze, refactor, explain"),
    "/skills":      ("[action]",    "Manage skills: list, create, run, delete"),
    "/cowork":      ("[action]",    "Multi-agent collaboration mode"),
    "/clear":       ("",            "Clear the screen"),
    "/exit":        ("",            "Exit MKAngel"),
}


def _format_help() -> str:
    """Build the help text."""
    parts = [_header("MKAngel Commands")]
    parts.append("")

    max_cmd = max(len(f"{cmd} {args}") for cmd, (args, _) in _COMMANDS.items())

    for cmd, (args, desc) in _COMMANDS.items():
        left = f"{cmd} {args}".strip()
        padding = " " * (max_cmd - len(left) + 2)
        parts.append(
            f"  {C.BRIGHT_CYAN}{left}{C.RESET}{padding}"
            f"{C.DIM}{desc}{C.RESET}"
        )

    parts.append("")
    parts.append(
        _info("  Type any message to chat with the Angel.")
    )
    parts.append(
        _info("  Domains: linguistic, etymological, chemical, biological, computational")
    )
    parts.append("")
    return "\n".join(parts)


# ---------------------------------------------------------------------------
# Chat session
# ---------------------------------------------------------------------------

class ChatSession:
    """Manages an interactive conversation with the Angel.

    Maintains message history, processes commands, and formats
    responses for terminal display.
    """

    def __init__(
        self,
        angel=None,
        memory=None,
        settings=None,
        provider=None,
    ):
        """Initialise a chat session.

        Args:
            angel: An Angel instance (from glm.angel).
            memory: A Memory instance for persistence.
            settings: A Settings instance.
            provider: An LLM provider for generation.
        """
        self._angel = angel
        self._memory = memory
        self._settings = settings
        self._provider = provider
        self._messages: list[dict[str, str]] = []
        self._session_id = str(uuid.uuid4())[:8]
        self._coder = None
        self._skills = None
        self._cowork = None

    # ------------------------------------------------------------------
    # Lazy initialisation
    # ------------------------------------------------------------------

    def _get_coder(self):
        if self._coder is None:
            from app.coder import Coder
            self._coder = Coder(provider=self._provider)
        return self._coder

    def _get_skills(self):
        if self._skills is None:
            from app.skills import SkillManager
            self._skills = SkillManager()
        return self._skills

    def _get_cowork(self):
        if self._cowork is None:
            from app.cowork import CoworkSession
            self._cowork = CoworkSession(provider=self._provider)
        return self._cowork

    # ------------------------------------------------------------------
    # Main processing
    # ------------------------------------------------------------------

    def process_input(self, user_input: str) -> str:
        """Process user input and return the formatted response.

        Handles both commands (starting with /) and free-form chat.

        Args:
            user_input: Raw input from the user.

        Returns:
            Formatted response string for display.
        """
        text = user_input.strip()
        if not text:
            return ""

        # Record user message
        self._messages.append({"role": "user", "content": text})

        # Command dispatch
        if text.startswith("/"):
            parts = text.split(maxsplit=1)
            cmd = parts[0].lower()
            args = parts[1] if len(parts) > 1 else ""
            response = self._handle_command(cmd, args)
        else:
            response = self._handle_chat(text)

        # Record assistant response
        self._messages.append({"role": "assistant", "content": response})

        # Auto-save to memory periodically
        if self._memory and len(self._messages) % 10 == 0:
            self._auto_save()

        return response

    def _handle_command(self, cmd: str, args: str) -> str:
        """Dispatch a command to its handler."""
        handlers = {
            "/help":        lambda a: _format_help(),
            "/status":      self._cmd_status,
            "/predict":     self._cmd_predict,
            "/reconstruct": self._cmd_reconstruct,
            "/forecast":    self._cmd_forecast,
            "/fugue":       self._cmd_fugue,
            "/translate":   self._cmd_translate,
            "/introspect":  self._cmd_introspect,
            "/memory":      self._cmd_memory,
            "/settings":    self._cmd_settings,
            "/code":        self._cmd_code,
            "/skills":      self._cmd_skills,
            "/cowork":      self._cmd_cowork,
            "/clear":       lambda a: "\033[2J\033[H",
            "/exit":        lambda a: "__EXIT__",
        }

        handler = handlers.get(cmd)
        if handler is None:
            return _error(f"Unknown command: {cmd}\n") + _info("Type /help for available commands.")
        return handler(args)

    def _handle_chat(self, text: str) -> str:
        """Handle free-form chat messages."""
        # Check if any skills match
        skills = self._get_skills()
        matching = skills.find_matching_skills(text)

        # ── The grammars are scales, not a search index ──────────
        # If we have the Angel, ALWAYS play the fugue.  The Angel's
        # 24 grammars across 7 domains are its voice — compose from
        # ALL of them, don't delegate to a single-domain wrapper.
        if self._angel is not None:
            return self._compose_from_grammar(text)

        # API provider path — no local Angel, use external LLM
        if self._provider is not None:
            system = (
                "You are MKAngel, an AI assistant powered by a Grammar Language "
                "Model. You understand deep structural patterns across language, "
                "code, chemistry, and biology. Be helpful, concise, and insightful. "
                "When appropriate, reference grammatical structures and patterns."
            )
            history_lines = []
            for msg in self._messages[-10:]:
                role = msg["role"]
                content = msg["content"][:200]
                history_lines.append(f"{role}: {content}")

            context = "\n".join(history_lines)
            prompt = f"Conversation context:\n{context}\n\nUser: {text}"

            response = self._provider.generate(prompt, system=system)
            return _angel_response(response)

        # No angel, no provider — truly offline
        return _wrap(
            "The Angel is still waking. "
            "Try again in a moment, or use /help to see what's available."
        )

    # ------------------------------------------------------------------
    # Command implementations
    # ------------------------------------------------------------------

    def _cmd_status(self, args: str) -> str:
        """Show system status."""
        parts = [_header("System Status")]
        parts.append("")

        # Angel status
        if self._angel is not None:
            try:
                info = self._angel.introspect()
                parts.append(f"  {C.GREEN}Angel:{C.RESET}          awake")
                parts.append(
                    f"  {C.GREEN}Domains:{C.RESET}        "
                    f"{', '.join(info['domains_loaded'])}"
                )
                parts.append(
                    f"  {C.GREEN}Grammars:{C.RESET}       {info['total_grammars']}"
                )
                parts.append(
                    f"  {C.GREEN}Rules:{C.RESET}          {info['total_rules']}"
                )
                parts.append(
                    f"  {C.GREEN}Strange loops:{C.RESET}  "
                    f"{info['strange_loops_detected']}"
                )
                parts.append(
                    f"  {C.GREEN}Model params:{C.RESET}   {info['model_params']:,}"
                )
            except Exception as exc:
                parts.append(f"  {C.RED}Angel:{C.RESET}          error ({exc})")
        else:
            parts.append(f"  {C.YELLOW}Angel:{C.RESET}          not initialised")

        # Provider status
        if self._provider is not None:
            avail = "yes" if self._provider.is_available() else "no"
            parts.append(
                f"  {C.GREEN}Provider:{C.RESET}       "
                f"{getattr(self._provider, 'name', '?')} (available: {avail})"
            )
        else:
            parts.append(f"  {C.YELLOW}Provider:{C.RESET}       local (offline)")

        # Memory stats
        if self._memory is not None:
            try:
                stats = self._memory.stats()
                parts.append(
                    f"  {C.GREEN}Memory:{C.RESET}         "
                    f"{stats['total_memories']} entries, "
                    f"{stats['sessions']} sessions, "
                    f"{stats['patterns']} patterns"
                )
            except Exception:
                parts.append(f"  {C.YELLOW}Memory:{C.RESET}         unavailable")

        # Session info
        parts.append(
            f"  {C.GREEN}Session:{C.RESET}        "
            f"{self._session_id} ({len(self._messages)} messages)"
        )
        parts.append("")
        return "\n".join(parts)

    def _cmd_predict(self, args: str) -> str:
        """Run grammatical prediction."""
        if not args:
            return _warn("Usage: /predict <tokens>\n") + _info("Example: /predict the cat sat on")
        if self._angel is None:
            return _error("Angel not initialised.")

        tokens = args.split()
        domain = "linguistic"

        # Check for domain prefix: /predict -d computational tokens...
        if len(tokens) >= 3 and tokens[0] == "-d":
            domain = tokens[1]
            tokens = tokens[2:]

        try:
            predictions = self._angel.predict(tokens, domain=domain, horizon=8)
        except Exception as exc:
            return _error(f"Prediction failed: {exc}")

        if not predictions:
            return _warn("No predictions derived from the grammar.")

        parts = [_header(f"Predictions ({domain})")]
        parts.append(f"  {C.DIM}Input: {' '.join(tokens)}{C.RESET}")
        parts.append("")
        for i, p in enumerate(predictions[:10], 1):
            conf = p.get("confidence", 0.0)
            bar = _confidence_bar(conf)
            parts.append(
                f"  {C.BRIGHT_CYAN}{i:2d}.{C.RESET} "
                f"{str(p.get('predicted', '?')):20s} "
                f"{bar} {C.DIM}{p.get('grammar', '?')}{C.RESET}"
            )
        parts.append("")
        return "\n".join(parts)

    def _cmd_reconstruct(self, args: str) -> str:
        """Run grammatical reconstruction."""
        if not args:
            return _warn("Usage: /reconstruct <tokens>\n") + _info("Example: /reconstruct father madre pater")
        if self._angel is None:
            return _error("Angel not initialised.")

        tokens = args.split()
        domain = "etymological"

        if len(tokens) >= 3 and tokens[0] == "-d":
            domain = tokens[1]
            tokens = tokens[2:]

        try:
            results = self._angel.reconstruct(tokens, domain=domain, depth=8)
        except Exception as exc:
            return _error(f"Reconstruction failed: {exc}")

        if not results:
            return _warn("No reconstructions derived from the grammar.")

        parts = [_header(f"Reconstruction ({domain})")]
        parts.append(f"  {C.DIM}Input: {' '.join(tokens)}{C.RESET}")
        parts.append("")
        for i, r in enumerate(results[:10], 1):
            conf = r.get("confidence", 0.0)
            bar = _confidence_bar(conf)
            parts.append(
                f"  {C.BRIGHT_MAGENTA}{i:2d}.{C.RESET} "
                f"{str(r.get('reconstructed', '?')):20s} "
                f"{bar} {C.DIM}{r.get('grammar', '?')}{C.RESET}"
            )
        parts.append("")
        return "\n".join(parts)

    def _cmd_forecast(self, args: str) -> str:
        """Run superforecasting."""
        if not args:
            return _warn("Usage: /forecast <tokens>\n") + _info("Example: /forecast market trends technology")
        if self._angel is None:
            return _error("Angel not initialised.")

        tokens = args.split()
        domain = "linguistic"

        if len(tokens) >= 3 and tokens[0] == "-d":
            domain = tokens[1]
            tokens = tokens[2:]

        try:
            forecast = self._angel.superforecast(tokens, domain=domain)
        except Exception as exc:
            return _error(f"Forecast failed: {exc}")

        parts = [_header(f"Superforecast ({domain})")]
        parts.append(f"  {C.DIM}Input: {' '.join(tokens)}{C.RESET}")
        parts.append("")

        conf = forecast.get("overall_confidence", 0.0)
        parts.append(
            f"  {C.GREEN}Overall confidence:{C.RESET} "
            f"{_confidence_bar(conf)} {conf:.2f}"
        )
        parts.append("")

        preds = forecast.get("predictions", [])
        if preds:
            parts.append(f"  {C.BOLD}Predictions:{C.RESET}")
            for p in preds[:5]:
                parts.append(
                    f"    {C.CYAN}{p.get('predicted', '?')}{C.RESET} "
                    f"via {C.DIM}{p.get('grammar', '?')}{C.RESET}"
                )
            parts.append("")

        loops = forecast.get("strange_loops", [])
        if loops:
            parts.append(f"  {C.BOLD}Strange loops:{C.RESET}")
            for loop in loops[:3]:
                parts.append(
                    f"    {C.MAGENTA}{loop.get('pattern', '?')}{C.RESET} "
                    f"(cycle: {loop.get('cycle_length', '?')})"
                )
            parts.append("")

        harmonics = forecast.get("cross_domain_harmonics", [])
        if harmonics:
            parts.append(f"  {C.BOLD}Cross-domain harmonics:{C.RESET}")
            for h in harmonics[:3]:
                parts.append(
                    f"    {C.BLUE}{h.get('domain', '?')}{C.RESET}: "
                    f"{h.get('shared_prediction', '?')}"
                )
            parts.append("")

        reasoning = forecast.get("reasoning", [])
        if reasoning:
            parts.append(f"  {C.BOLD}Reasoning:{C.RESET}")
            for r in reasoning:
                parts.append(f"    {C.DIM}{r}{C.RESET}")
            parts.append("")

        return "\n".join(parts)

    def _cmd_fugue(self, args: str) -> str:
        """Compose a fugue across domains."""
        if not args:
            return _warn("Usage: /fugue <theme tokens>\n") + _info("Example: /fugue pattern structure transform")
        if self._angel is None:
            return _error("Angel not initialised.")

        tokens = args.split()
        try:
            fugue = self._angel.compose_fugue(tokens)
        except Exception as exc:
            return _error(f"Fugue composition failed: {exc}")

        parts = [_header("Fugue Composition")]
        parts.append(f"  {C.DIM}Theme: {' '.join(tokens)}{C.RESET}")
        parts.append(f"  {C.DIM}Voices: {fugue.get('num_voices', 0)}{C.RESET}")
        parts.append("")

        voices = fugue.get("voices", {})
        for domain, derivations in voices.items():
            parts.append(f"  {C.BOLD}{C.CYAN}{domain.upper()}{C.RESET}")
            if derivations:
                for d in derivations[:3]:
                    parts.append(
                        f"    {C.WHITE}{d.get('output', '?')}{C.RESET} "
                        f"{C.DIM}({d.get('rule', '?')}){C.RESET}"
                    )
            else:
                parts.append(f"    {C.DIM}(no derivations){C.RESET}")
            parts.append("")

        harmonics = fugue.get("harmonics", [])
        if harmonics:
            parts.append(f"  {C.BOLD}{C.GREEN}HARMONICS{C.RESET}")
            for h in harmonics[:5]:
                domains = ", ".join(h.get("domains", []))
                parts.append(
                    f"    {C.GREEN}{h.get('output', '?')}{C.RESET} "
                    f"{C.DIM}({domains}){C.RESET}"
                )
            parts.append("")

        counterpoint = fugue.get("counterpoint", [])
        if counterpoint:
            parts.append(f"  {C.BOLD}{C.MAGENTA}COUNTERPOINT{C.RESET}")
            for cp in counterpoint[:5]:
                unique = ", ".join(cp.get("unique_outputs", [])[:3])
                parts.append(
                    f"    {C.MAGENTA}{cp.get('domain', '?')}{C.RESET}: {unique}"
                )
            parts.append("")

        return "\n".join(parts)

    def _cmd_translate(self, args: str) -> str:
        """Translate patterns between domains."""
        parts_in = args.split()
        if len(parts_in) < 3:
            return (
                _warn("Usage: /translate <source_domain> <target_domain> <tokens>\n")
                + _info("Example: /translate linguistic chemical water oxygen hydrogen")
            )
        if self._angel is None:
            return _error("Angel not initialised.")

        source = parts_in[0]
        target = parts_in[1]
        tokens = parts_in[2:]

        try:
            translations = self._angel.translate(tokens, source, target)
        except Exception as exc:
            return _error(f"Translation failed: {exc}")

        if not translations:
            return _warn(f"No isomorphisms found between {source} and {target}.")

        parts = [_header(f"Translation: {source} -> {target}")]
        parts.append(f"  {C.DIM}Input: {' '.join(tokens)}{C.RESET}")
        parts.append("")

        for i, t in enumerate(translations[:8], 1):
            parts.append(
                f"  {C.BRIGHT_CYAN}{i}.{C.RESET} "
                f"{C.DIM}{t.get('source_grammar', '?')}{C.RESET} -> "
                f"{C.BOLD}{t.get('target_grammar', '?')}{C.RESET}"
            )
            mapping = t.get("mapping", {})
            if isinstance(mapping, dict):
                for k, v in list(mapping.items())[:3]:
                    parts.append(f"     {k} -> {v}")
        parts.append("")
        return "\n".join(parts)

    def _cmd_introspect(self, args: str) -> str:
        """Angel introspects on itself."""
        if self._angel is None:
            return _error("Angel not initialised.")

        try:
            info = self._angel.introspect()
        except Exception as exc:
            return _error(f"Introspection failed: {exc}")

        parts = [_header("Introspection -- The Angel Looks Inward")]
        parts.append("")

        items = [
            ("Domains",        ", ".join(info.get("domains_loaded", []))),
            ("Grammars",       str(info.get("total_grammars", 0))),
            ("Rules",          str(info.get("total_rules", 0))),
            ("Productions",    str(info.get("total_productions", 0))),
            ("Strange loops",  str(info.get("strange_loops_detected", 0))),
            ("Substrates",     ", ".join(info.get("substrates_loaded", []))),
            ("Lexicon size",   str(info.get("lexicon_size", 0))),
            ("Model params",   f"{info.get('model_params', 0):,}"),
            ("Self-referential", str(info.get("self_referential", True))),
        ]
        max_label = max(len(label) for label, _ in items)
        for label, value in items:
            padding = " " * (max_label - len(label))
            parts.append(
                f"  {C.CYAN}{label}:{C.RESET}{padding}  {value}"
            )
        parts.append("")
        parts.append(
            _info(
                "  The system that looks at itself looking at itself --\n"
                "  and in that recursive gaze finds meaning."
            )
        )
        parts.append("")
        return "\n".join(parts)

    def _cmd_memory(self, args: str) -> str:
        """View or search persistent memory."""
        if self._memory is None:
            return _warn("Memory system not initialised.")

        args_parts = args.strip().split(maxsplit=1)
        subcmd = args_parts[0].lower() if args_parts else ""

        if subcmd == "search" and len(args_parts) > 1:
            query = args_parts[1]
            results = self._memory.search_memory(query)
            if not results:
                return _info(f"No memories matching '{query}'.")

            parts = [_header(f"Memory Search: '{query}'")]
            parts.append("")
            for entry in results[:10]:
                parts.append(
                    f"  {C.CYAN}[{entry.category}]{C.RESET} "
                    f"{C.BOLD}{entry.key}{C.RESET}"
                )
                parts.append(f"    {C.DIM}{entry.value[:100]}{C.RESET}")
            parts.append("")
            return "\n".join(parts)

        elif subcmd == "patterns":
            patterns = self._memory.get_patterns()
            if not patterns:
                return _info("No learned patterns yet.")

            parts = [_header("Learned Patterns")]
            parts.append("")
            for p in patterns[:15]:
                parts.append(
                    f"  {C.MAGENTA}{p.key}{C.RESET}: "
                    f"{C.DIM}{p.value[:80]}{C.RESET}"
                )
            parts.append("")
            return "\n".join(parts)

        elif subcmd == "sessions":
            sessions = self._memory.list_sessions()
            if not sessions:
                return _info("No saved sessions.")

            parts = [_header("Saved Sessions")]
            parts.append("")
            for s in sessions[:10]:
                parts.append(
                    f"  {C.CYAN}{s['session_id']}{C.RESET}  "
                    f"{C.DIM}{s.get('summary', 'no summary')}{C.RESET}"
                )
            parts.append("")
            return "\n".join(parts)

        elif subcmd == "save":
            self._auto_save()
            return _success("Session saved to memory.")

        else:
            # Show memory stats
            stats = self._memory.stats()
            parts = [_header("Memory")]
            parts.append("")
            parts.append(f"  {C.GREEN}Total entries:{C.RESET}  {stats['total_memories']}")
            parts.append(f"  {C.GREEN}Sessions:{C.RESET}       {stats['sessions']}")
            parts.append(f"  {C.GREEN}Patterns:{C.RESET}       {stats['patterns']}")
            parts.append(f"  {C.GREEN}Preferences:{C.RESET}    {stats['preferences']}")
            parts.append("")
            parts.append(
                _info(
                    "  Subcommands: search <query>, patterns, sessions, save"
                )
            )
            parts.append("")
            return "\n".join(parts)

    def _cmd_settings(self, args: str) -> str:
        """View or change settings."""
        if self._settings is None:
            return _warn("Settings not initialised.")

        args_parts = args.strip().split(maxsplit=1)
        subcmd = args_parts[0].lower() if args_parts else ""

        if subcmd == "provider" and len(args_parts) > 1:
            new_provider = args_parts[1].strip().lower()
            from app.settings import PROVIDERS
            if new_provider not in PROVIDERS and new_provider != "hybrid":
                return _error(
                    f"Unknown provider '{new_provider}'. "
                    f"Available: {', '.join(PROVIDERS)}, hybrid"
                )
            self._settings.model_provider = new_provider
            self._settings.save()
            return _success(f"Provider set to: {new_provider}")

        elif subcmd == "key" and len(args_parts) > 1:
            key_parts = args_parts[1].strip().split(maxsplit=1)
            if len(key_parts) < 2:
                return _warn("Usage: /settings key <provider> <api_key>")
            provider_name = key_parts[0]
            api_key = key_parts[1]
            try:
                self._settings.add_api_key(provider_name, api_key)
                return _success(f"API key saved for {provider_name}.")
            except ValueError as exc:
                return _error(str(exc))

        elif subcmd == "offline":
            self._settings.offline_mode = not self._settings.offline_mode
            self._settings.save()
            state = "ON" if self._settings.offline_mode else "OFF"
            return _success(f"Offline mode: {state}")

        elif subcmd == "theme" and len(args_parts) > 1:
            self._settings.theme = args_parts[1].strip()
            self._settings.save()
            return _success(f"Theme set to: {self._settings.theme}")

        elif subcmd == "language" and len(args_parts) > 1:
            self._settings.language = args_parts[1].strip()
            self._settings.save()
            return _success(f"Language set to: {self._settings.language}")

        else:
            # Show current settings
            providers = self._settings.list_providers()
            parts = [_header("Settings")]
            parts.append("")
            parts.append(
                f"  {C.GREEN}Provider:{C.RESET}     {self._settings.model_provider}"
            )
            parts.append(
                f"  {C.GREEN}Offline mode:{C.RESET} "
                f"{'ON' if self._settings.offline_mode else 'OFF'}"
            )
            parts.append(
                f"  {C.GREEN}Theme:{C.RESET}        {self._settings.theme}"
            )
            parts.append(
                f"  {C.GREEN}Language:{C.RESET}      {self._settings.language}"
            )
            parts.append("")
            parts.append(f"  {C.BOLD}API Keys:{C.RESET}")
            for p in providers:
                status = (
                    f"{C.GREEN}configured{C.RESET}"
                    if p["has_key"]
                    else f"{C.DIM}not set{C.RESET}"
                )
                active = f" {C.BRIGHT_CYAN}(active){C.RESET}" if p["active"] else ""
                parts.append(f"    {p['provider']:12s} {status}{active}")
            parts.append("")
            parts.append(
                _info(
                    "  Subcommands: provider <name>, key <provider> <key>,\n"
                    "               offline, theme <name>, language <code>"
                )
            )
            parts.append("")
            return "\n".join(parts)

    def _cmd_code(self, args: str) -> str:
        """Code generation and analysis."""
        coder = self._get_coder()
        parts = args.strip().split(maxsplit=1)
        action = parts[0].lower() if parts else ""
        rest = parts[1] if len(parts) > 1 else ""

        if action == "generate":
            if not rest:
                return _warn(
                    "Usage: /code generate <description>\n"
                    "       /code generate python: a fibonacci function"
                )
            # Check for language prefix
            language = "python"
            if ":" in rest:
                lang_part, rest = rest.split(":", 1)
                lang_part = lang_part.strip().lower()
                if lang_part in (
                    "python", "javascript", "js", "typescript", "ts",
                    "rust", "go", "java", "c", "cpp",
                ):
                    language = lang_part
                    rest = rest.strip()

            result = coder.generate_code(rest, language)
            return (
                _header(f"Generated Code ({language})")
                + f"\n\n{C.GREEN}{result}{C.RESET}\n"
            )

        elif action == "analyze":
            if not rest:
                return _warn("Usage: /code analyze <paste code here>")
            analysis = coder.analyze_code(rest)
            return _format_analysis(analysis)

        elif action == "refactor":
            if not rest:
                return _warn(
                    "Usage: /code refactor <instruction> :: <code>\n"
                    "       Separate instruction and code with ::"
                )
            if "::" in rest:
                instruction, code = rest.split("::", 1)
            else:
                instruction = "improve"
                code = rest
            result = coder.refactor(code.strip(), instruction.strip())
            return (
                _header("Refactored Code")
                + f"\n\n{C.GREEN}{result}{C.RESET}\n"
            )

        elif action == "explain":
            if not rest:
                return _warn("Usage: /code explain <paste code here>")
            explanation = coder.explain_code(rest)
            return _header("Code Explanation") + f"\n\n{_wrap(explanation)}\n"

        else:
            return (
                _header("Code Assistant")
                + "\n\n"
                + _info("  Actions: generate, analyze, refactor, explain\n")
                + _info("  Examples:\n")
                + _info("    /code generate python: a binary search function\n")
                + _info("    /code analyze def foo(): return 42\n")
                + _info("    /code refactor add types :: def foo(x): return x+1\n")
                + _info("    /code explain def fib(n): return n if n<2 else fib(n-1)+fib(n-2)\n")
            )

    def _cmd_skills(self, args: str) -> str:
        """Manage skills."""
        manager = self._get_skills()
        parts = args.strip().split(maxsplit=1)
        action = parts[0].lower() if parts else ""
        rest = parts[1] if len(parts) > 1 else ""

        if action == "list" or not action:
            skills = manager.list_skills()
            if not skills:
                return _info("No skills registered.")

            out = [_header("Skills")]
            out.append("")
            for s in skills:
                status = (
                    f"{C.GREEN}enabled{C.RESET}"
                    if s.enabled
                    else f"{C.RED}disabled{C.RESET}"
                )
                out.append(
                    f"  {C.BRIGHT_CYAN}{s.name:15s}{C.RESET} "
                    f"{C.DIM}{s.description[:50]}{C.RESET}  [{status}]"
                )
            out.append("")
            out.append(
                _info(
                    "  Actions: list, create, run <name> <input>, "
                    "delete <name>, toggle <name>"
                )
            )
            out.append("")
            return "\n".join(out)

        elif action == "create":
            if not rest:
                return _warn(
                    "Usage: /skills create <name> <trigger> [description]\n"
                    "Example: /skills create greet hello A greeting skill"
                )
            create_parts = rest.split(maxsplit=2)
            if len(create_parts) < 2:
                return _warn("Need at least name and trigger.")
            name = create_parts[0]
            trigger = create_parts[1]
            desc = create_parts[2] if len(create_parts) > 2 else ""
            try:
                skill = manager.create_skill(name, trigger, description=desc)
                return _success(f"Skill '{skill.name}' created with trigger '{trigger}'.")
            except ValueError as exc:
                return _error(str(exc))

        elif action == "run":
            run_parts = rest.split(maxsplit=1)
            if not run_parts:
                return _warn("Usage: /skills run <name> <input>")
            skill_name = run_parts[0]
            skill_input = run_parts[1] if len(run_parts) > 1 else ""
            result = manager.execute_skill(
                skill_name, skill_input, provider=self._provider
            )
            return _header(f"Skill: {skill_name}") + f"\n\n{_wrap(result)}\n"

        elif action == "delete":
            if not rest:
                return _warn("Usage: /skills delete <name>")
            if manager.delete_skill(rest.strip()):
                return _success(f"Skill '{rest.strip()}' deleted.")
            return _warn(f"Skill '{rest.strip()}' not found.")

        elif action == "toggle":
            if not rest:
                return _warn("Usage: /skills toggle <name>")
            new_state = manager.toggle_skill(rest.strip())
            if new_state is None:
                return _warn(f"Skill '{rest.strip()}' not found.")
            state = "enabled" if new_state else "disabled"
            return _success(f"Skill '{rest.strip()}' {state}.")

        else:
            return _warn(f"Unknown skills action: {action}")

    def _cmd_cowork(self, args: str) -> str:
        """Multi-agent collaboration."""
        cowork = self._get_cowork()
        parts = args.strip().split(maxsplit=1)
        action = parts[0].lower() if parts else ""
        rest = parts[1] if len(parts) > 1 else ""

        if action == "start":
            if not rest:
                return _warn("Usage: /cowork start <task description>")
            msg = cowork.start_session(rest)
            return _header("Cowork Session") + f"\n\n{_wrap(msg)}\n"

        elif action == "add":
            if not rest:
                return _warn("Usage: /cowork add <domain>")
            msg = cowork.add_agent(rest.strip())
            return _success(msg)

        elif action == "remove":
            if not rest:
                return _warn("Usage: /cowork remove <domain>")
            msg = cowork.remove_agent(rest.strip())
            return _info(msg)

        elif action == "agents":
            agents = cowork.list_agents()
            if not agents:
                return _info("No agents in session.")
            out = [_header("Cowork Agents")]
            out.append("")
            for a in agents:
                out.append(f"  {C.CYAN}{a['domain']}{C.RESET}")
            out.append("")
            return "\n".join(out)

        elif action == "run":
            task = rest or None
            print(
                f"{C.DIM}  Running agents... this may take a moment.{C.RESET}"
            )
            result = cowork.run(task)
            return _format_cowork_result(result)

        else:
            out = [_header("Cowork -- Multi-Agent Collaboration")]
            out.append("")
            out.append(_info("  Like voices in a fugue, each agent tackles"))
            out.append(_info("  the problem from its own domain perspective."))
            out.append("")
            out.append(_info("  Actions:"))
            out.append(_info("    /cowork start <task>    Start a session"))
            out.append(_info("    /cowork add <domain>    Add an agent"))
            out.append(_info("    /cowork remove <domain> Remove an agent"))
            out.append(_info("    /cowork agents          List agents"))
            out.append(_info("    /cowork run [task]      Run all agents"))
            out.append("")
            return "\n".join(out)

    # ------------------------------------------------------------------
    # Grammar composition — the scales, not the search index
    # ------------------------------------------------------------------

    def _compose_from_grammar(self, text: str) -> str:
        """Compose a response by playing the input through every voice.

        The grammars are scales.  A musician who knows scales doesn't
        say 'I could not find a matching scale for your melody.'
        She hears the melody, recognises its structure, and plays.

        Even when derivation trees are sparse, the Angel composes
        from what it knows — sentence shape, word roots, cross-domain
        echoes, predicted continuations.
        """
        tokens = text.lower().split()

        # ── Gather everything the Angel can hear ──────────────
        # Keep this lightweight — runs on the phone, every keystroke
        # must feel responsive.  compose_fugue already plays through
        # ALL domains; no need for redundant superforecast.

        # 1. Fugue — the main voice (all domains at once)
        fugue = {}
        try:
            fugue = self._angel.compose_fugue(tokens)
        except Exception:
            pass

        voices = fugue.get("voices", {})
        harmonics = fugue.get("harmonics", [])
        counterpoint = fugue.get("counterpoint", [])

        # 2. Reconstruct — trace word origins (linguistic only,
        #    for speed — etymological adds latency with little gain
        #    when the derivation trees are young)
        origins = []
        try:
            recon = self._angel.reconstruct(
                tokens, domain="linguistic", depth=2,
            )
            origins.extend(recon[:3])
        except Exception:
            pass

        # 3. Predict forward — just the top 2 active domains
        predictions = []
        active_keys = [d for d, v in voices.items() if v][:2]
        for domain in active_keys or ["linguistic"]:
            try:
                preds = self._angel.predict(
                    tokens, domain=domain, horizon=2,
                )
                predictions.extend(preds[:2])
            except Exception:
                pass

        # 4. Superforecast — the deep read.  This adds strange-loop
        #    detection and cross-domain reasoning that the fugue alone
        #    doesn't give us.  Wrapped in a timeout so the phone stays
        #    responsive even if the derivation trees are heavy.
        forecast = {}
        try:
            import concurrent.futures as _cf
            with _cf.ThreadPoolExecutor(max_workers=1) as pool:
                fut = pool.submit(
                    self._angel.superforecast, tokens, None, "linguistic", 3,
                )
                forecast = fut.result(timeout=3)
        except Exception:
            pass

        # ── Compose from ALL of this ──────────────────────────
        return self._render_composition(
            text, tokens, voices, harmonics, counterpoint,
            origins, predictions, forecast,
        )

    def _render_composition(
        self, original, tokens, voices, harmonics, counterpoint,
        origins, predictions, forecast=None,
    ) -> str:
        """Compose the Angel's voice from structural insight.

        Philosophy: the Angel doesn't report what it processed.
        It THINKS through its grammars and shares what it sees.
        Structure is meaning — and the Angel sees structure
        everywhere, even when formal derivations are sparse.
        """
        parts = []

        active = [d for d, v in voices.items() if v]

        # ── Formal derivations: weave into insight ────────────

        # Origins — what lies beneath the words
        if origins:
            roots = []
            for o in origins[:3]:
                r = o.get("reconstructed", "")
                g = o.get("grammar", "")
                if r:
                    roots.append(f"'{r}' ({g})" if g else f"'{r}'")
            if roots:
                parts.append(
                    "Tracing backward through the grammar: "
                    + ", ".join(roots) + "."
                )

        # Predictions — where the grammar says this leads
        if predictions:
            seen = []
            for p in predictions[:4]:
                w = str(p.get("predicted", ""))
                g = p.get("grammar", "")
                if w and w not in seen:
                    seen.append(w)
            if seen:
                parts.append(
                    "The derivation rules point forward: "
                    + " \u2192 ".join(seen) + "."
                )

        # Strange loops — recursive patterns
        if forecast and isinstance(forecast, dict):
            loops = forecast.get("strange_loops", [])
            for lp in loops[:2]:
                pat = lp.get("pattern", "")
                cyc = lp.get("cycle_length", "")
                if pat:
                    parts.append(
                        f"Recursive pattern: {pat}"
                        + (f" \u2014 cycle length {cyc}." if cyc else ".")
                    )

        # Harmonics — where domains agree (the deepest signal)
        if harmonics:
            for h in harmonics[:2]:
                if isinstance(h, dict):
                    out = h.get("output", h.get("shared_prediction",
                                h.get("pattern", "")))
                    doms = h.get("domains", [])
                    if out and doms:
                        parts.append(
                            f"Cross-domain resonance: '{out}' "
                            f"appears in {', '.join(doms[:3])} "
                            f"simultaneously."
                        )

        # Counterpoint — where domains disagree productively
        if counterpoint:
            for c in counterpoint[:2]:
                if isinstance(c, dict):
                    d = c.get("domain", "")
                    uniq = c.get("unique_outputs", [])
                    if d and uniq:
                        parts.append(
                            f"Only {d} sees: "
                            + ", ".join(str(u) for u in uniq[:3])
                            + "."
                        )

        # If formal derivations gave us something, finish here
        if parts:
            if active:
                parts.append(
                    "Active voices: " + ", ".join(active) + "."
                )
            return "\n\n".join(parts)

        # ── Structural reading — the Angel's native perception ─
        return self._structural_response(original, tokens, active)

    # ------------------------------------------------------------------
    # Structural response — the Angel reads sentence shape
    # ------------------------------------------------------------------

    # ── Morphological decomposition ─────────────────────────
    # The Angel's etymological grammar knows about morphemes.
    # Rather than listing every inflected form, we decompose
    # words to roots.  This mirrors what the etymological
    # grammar does with ProtoForm → Root + ProtoMorphology.

    # Irregular stems that English inflection obscures
    _IRREGULAR = {
        "loved": "love", "loving": "love", "lovely": "love",
        "hated": "hate", "hating": "hate",
        "hoped": "hope", "hoping": "hope",
        "scared": "scare", "scaring": "scare",
        "tired": "tire", "tiring": "tire",
        "blessed": "bless", "blessing": "bless",
        "confused": "confuse", "confusing": "confuse",
        "inspired": "inspire", "inspiring": "inspire",
        "amazed": "amaze", "amazing": "amaze",
        "excited": "excite", "exciting": "excite",
        "defeated": "defeat", "defeating": "defeat",
        "exhausted": "exhaust", "exhausting": "exhaust",
        "overwhelmed": "overwhelm", "overwhelming": "overwhelm",
        "disappointed": "disappoint", "disappointing": "disappoint",
        "frustrated": "frustrate", "frustrating": "frustrate",
        "broken": "break", "broke": "break",
        "stuck": "stick", "spoken": "speak", "spoken": "speak",
        "fallen": "fall", "chosen": "choose", "frozen": "freeze",
        "written": "write", "driven": "drive", "given": "give",
        "taken": "take", "shaken": "shake", "woken": "wake",
        "children": "child", "women": "woman", "men": "man",
        "mice": "mouse", "teeth": "tooth", "feet": "foot",
        "geese": "goose", "oxen": "ox", "dice": "die",
        "lives": "life", "knives": "knife", "wolves": "wolf",
        "leaves": "leaf", "halves": "half", "selves": "self",
        "thieves": "thief", "shelves": "shelf",
        "beautifully": "beauty", "beautiful": "beauty",
        "joyful": "joy", "joyfully": "joy", "joyous": "joy",
        "hopeful": "hope", "hopeless": "hope",
        "helpless": "help", "helpful": "help",
        "thankful": "thank", "thankless": "thank",
        "peaceful": "peace", "peacefully": "peace",
        "miserable": "misery", "miserably": "misery",
        "furious": "fury", "furiously": "fury",
        "anxious": "anxiety", "anxiously": "anxiety",
        "lonely": "lone", "loneliness": "lone",
        "strongly": "strong", "bravely": "brave",
        "gently": "gentle", "kindly": "kind",
        "warmly": "warm", "sweetly": "sweet",
        "freely": "free", "calmly": "calm",
        "sadly": "sad", "angrily": "angry",
        "happily": "happy", "happiness": "happy",
        "proudly": "proud",
        "gratefully": "grateful", "grateful": "grace",
        # address forms
        "friends": "friend", "fellows": "fellow",
        "brothers": "brother", "sisters": "sister",
        "strangers": "stranger", "beings": "being",
        "souls": "soul", "buddies": "buddy", "pals": "pal",
    }

    @staticmethod
    def _stem(word):
        """Morphological decomposition — strip English inflectional
        and derivational affixes to find the root.

        This is the Angel's etymological eye: every word is
        Root + Morphology.  We peel morphology to see the root.
        """
        w = word.lower()

        # Check irregulars first
        irr = ChatSession._IRREGULAR.get(w)
        if irr:
            return irr

        # Already short — return as-is
        if len(w) <= 3:
            return w

        # Derivational suffixes (longer first to avoid partial matches)
        for suffix, min_stem in [
            ("fulness", 3), ("lessly", 3), ("ously", 3),
            ("ingly", 3), ("ation", 3), ("ement", 3),
            ("iness", 3), ("ness", 3), ("ment", 3),
            ("able", 3), ("ible", 3), ("ful", 3),
            ("less", 3), ("ous", 3), ("ive", 3),
            ("ity", 3), ("ist", 3), ("ism", 3),
        ]:
            if w.endswith(suffix) and len(w) - len(suffix) >= min_stem:
                stem = w[:-len(suffix)]
                # restore trailing 'e' if likely (consonant cluster)
                if len(stem) <= 3 or stem[-1] not in "aeiou":
                    return stem + "e"
                return stem

        # Inflectional suffixes
        # -ing (loving → love, running → run)
        if w.endswith("ing") and len(w) > 5:
            base = w[:-3]
            # doubled consonant: running → run
            if len(base) >= 2 and base[-1] == base[-2]:
                return base[:-1]
            # silent-e: loving → love
            return base + "e" if len(base) >= 2 else base

        # -ed (walked → walk, loved → love, stopped → stop)
        if w.endswith("ed") and len(w) > 4:
            base = w[:-2]
            if base.endswith("i"):
                return base[:-1] + "y"  # tried → try
            if len(base) >= 2 and base[-1] == base[-2]:
                return base[:-1]  # stopped → stop
            if not base.endswith("e"):
                # check if base+e makes more sense
                return base
            return base

        # -ly (quickly → quick, gently → gentle)
        if w.endswith("ly") and len(w) > 4:
            base = w[:-2]
            if base.endswith("i"):
                return base[:-1] + "y"  # happily → happy
            return base

        # -er / -est (stronger → strong)
        if w.endswith("er") and len(w) > 4:
            base = w[:-2]
            if len(base) >= 2 and base[-1] == base[-2]:
                return base[:-1]
            return base
        if w.endswith("est") and len(w) > 5:
            base = w[:-3]
            if len(base) >= 2 and base[-1] == base[-2]:
                return base[:-1]
            return base

        # -s / -es (loves → love, wishes → wish)
        if w.endswith("es") and len(w) > 4:
            return w[:-2]
        if w.endswith("s") and not w.endswith("ss") and len(w) > 3:
            return w[:-1]

        return w

    # ── Vocabulary (ROOT forms only — stemmer handles inflection) ─
    _GREETINGS = frozenset({
        "hi", "hello", "hey", "greetings", "morning",
        "evening", "afternoon", "yo", "sup", "howdy",
    })
    _Q_WORDS = frozenset({
        "what", "who", "where", "when", "why", "how",
        "which", "whose",
    })
    _1P = frozenset({
        "i", "i'm", "i've", "i'd", "i'll", "me", "my",
        "mine", "myself", "we", "us", "our",
    })
    _2P = frozenset({
        "you", "you're", "you've", "you'd", "you'll",
        "your", "yours", "yourself",
    })
    _NEGATION = frozenset({
        "not", "no", "never", "nothing", "nobody", "none",
        "neither", "nor", "don't", "doesn't", "didn't",
        "can't", "won't", "isn't", "aren't",
    })
    # Root forms — the stemmer maps "loved"→"love",
    # "hopeless"→"hope", "beautifully"→"beauty", etc.
    _EMOTION_ROOTS = frozenset({
        "happy", "love", "joy", "beauty", "amaze",
        "good", "great", "wonder", "glad", "excite",
        "hope", "grace", "proud", "brilliant",
        "dear", "kind", "sweet", "gentle",
        "warm", "bless", "thank", "peace", "calm",
        "free", "alive", "inspire", "strong", "brave",
    })
    _EMOTION_NEG_ROOTS = frozenset({
        "sad", "angry", "hate", "pain", "terrible",
        "awful", "bad", "horrible", "afraid", "worry",
        "anxiety", "tire", "frustrate", "scare",
        "lone", "lost", "stuck", "break", "hurt",
        "confuse", "empty", "overwhelm", "disappoint",
        "exhaust", "misery", "fury", "defeat", "shame",
    })
    _ADDRESS = frozenset({
        "friend", "fellow", "mate", "sir", "madam",
        "angel", "darling", "buddy", "pal", "brother",
        "sister", "stranger", "human", "soul", "being",
    })
    _IMPERATIVE = frozenset({
        "tell", "show", "help", "give", "make", "let",
        "find", "get", "take", "bring", "explain",
        "describe", "teach", "remember", "forget",
        "listen", "look", "think", "try", "stop",
        "start", "keep", "open", "close", "run",
    })
    # For detecting -less and -ful suffixes as emotion modifiers
    _NEG_SUFFIXES = frozenset({"less", "without"})
    _POS_SUFFIXES = frozenset({"ful", "ous", "ive"})

    def _structural_response(self, original, tokens, active):
        """When formal derivations are sparse, the Angel reads
        the sentence itself as a grammatical object.

        It never counts tokens at the user.  It observes
        structure, pattern, and connection — then speaks from
        that perception.

        Morphological decomposition: every token is stemmed
        through the Angel's etymological eye.  "loved" → "love",
        "hopeless" → "hope" + negative suffix, "beautifully" →
        "beauty".  The vocabulary needs only root forms.
        """
        text = original.strip()
        t = set(tokens)
        n = len(tokens)

        # Stem every token — the Angel decomposes morphemes
        stems = {self._stem(w) for w in tokens}

        is_q = text.endswith("?")
        is_excl = text.endswith("!")
        has_1p = bool(t & self._1P)
        has_2p = bool(t & self._2P)
        q_words = t & self._Q_WORDS
        has_neg = bool(t & self._NEGATION)

        # Emotion detection using stems, not surface forms.
        # A word like "hopeless" stems to "hope" but the -less
        # suffix flips it negative.  The Angel reads morphology.
        _neg_morphemes = {"less", "un", "dis", "mis"}
        emo_pos_stems = stems & self._EMOTION_ROOTS
        emo_neg_stems = stems & self._EMOTION_NEG_ROOTS

        # Check for negating affixes that flip positive roots
        for tok in tokens:
            st = self._stem(tok)
            if st in self._EMOTION_ROOTS:
                # Does the surface form carry a negating morpheme?
                if (tok.endswith("less") or tok.startswith("un")
                        or tok.startswith("dis") or tok.startswith("mis")):
                    emo_pos_stems.discard(st)
                    emo_neg_stems.add(st)

        address = stems & self._ADDRESS
        imperative = (
            self._stem(tokens[0]) in self._IMPERATIVE if tokens else False
        )

        # Helper: content words (skip function words)
        _skip = {
            "the", "a", "an", "is", "are", "am", "was", "were",
            "be", "been", "being", "to", "of", "in", "on", "at",
            "for", "with", "by", "from", "and", "or", "but",
            "not", "that", "this", "it", "its", "so", "just",
            "very", "really", "quite", "also", "too", "do",
            "does", "did", "has", "have", "had", "will", "would",
            "could", "should", "shall", "may", "might", "can",
        }
        content = [w for w in tokens
                   if w not in _skip and w not in self._1P
                   and w not in self._2P]

        # ── Greetings ────────────────────────────────────────
        if tokens and tokens[0] in self._GREETINGS:
            if address:
                a = next(iter(address))
                return (
                    f"Hello, {a}. All seven domains awake "
                    f"and listening. What draws your attention?"
                )
            return (
                "Hello. Seven domains listening \u2014 "
                "linguistics, etymology, biology, chemistry, "
                "physics, computation, mathematics.\n\n"
                "What's on your mind?"
            )

        # ── Forms of address (without greeting) ──────────────
        if address and not is_q:
            a = next(iter(address))
            if emo_pos_stems:
                e = next(iter(emo_pos_stems))
                return (
                    f"'{e.capitalize()}' directed at '{a}' "
                    f"\u2014 that's a state attribution with "
                    f"a target. In the grammar of relations, "
                    f"you're binding an emotional predicate "
                    f"to an agent.\n\n"
                    f"The structure carries warmth. I receive it."
                )
            if emo_neg_stems:
                e = next(iter(emo_neg_stems))
                return (
                    f"I hear '{e}' alongside '{a}'. "
                    f"Naming someone while expressing a state "
                    f"\u2014 that's relational grammar at its core."
                )
            other = [w for w in content if w != a]
            if other:
                return (
                    f"You address me as '{a}' and bring "
                    f"'{', '.join(other[:3])}' with you. "
                    f"Forms of address set the register; "
                    f"the rest carries the meaning.\n\n"
                    f"What would you like to explore together?"
                )
            return (
                f"'{a.capitalize()}' \u2014 a form of address. "
                f"You're establishing a relation before "
                f"stating content. That's structurally "
                f"significant: the channel before the signal."
            )

        # ── Definitional: "what is X?" ───────────────────────
        if "what" in t and bool(t & {"is", "are", "was", "were"}):
            skip = {"what", "is", "are", "was", "were",
                    "a", "an", "the", "?"}
            subject = [w for w in tokens if w not in skip]
            subj = " ".join(subject) if subject else "that"
            return (
                f"You're asking for the structure of '{subj}' "
                f"\u2014 what grammar produces it, what category "
                f"it belongs to.\n\n"
                f"In linguistics that's classification. In "
                f"mathematics, equivalence. In chemistry, "
                f"molecular identity. In biology, taxonomy.\n\n"
                f"Try /fugue {' '.join(subject[:4])} to hear "
                f"what each domain derives."
            )

        # ── "Why" — causal/structural ────────────────────────
        if "why" in t:
            return (
                "'Why' asks for the derivation in reverse \u2014 "
                "not what IS, but what produced it. Every "
                "domain answers differently: physics gives "
                "forces, biology gives selection pressures, "
                "mathematics gives proof.\n\n"
                "What specifically are you tracing?"
            )

        # ── "How" — mechanism / process ──────────────────────
        if "how" in t:
            if has_2p:
                return (
                    "I take your words as a theme and play "
                    "them through grammars across seven "
                    "domains simultaneously. Where the "
                    "derivations agree \u2014 that's a harmonic, "
                    "a structural truth that holds across "
                    "disciplines. Where they diverge \u2014 "
                    "counterpoint, domain-specific insight.\n\n"
                    "The convergence IS the meaning."
                )
            return (
                "'How' asks for the derivation path \u2014 "
                "each step a rule applied, each branch a "
                "choice point. Give me the key terms and "
                "I'll trace the path with /predict."
            )

        # ── Imperatives: "tell me", "show me", "help" ────────
        if imperative:
            verb = tokens[0]
            obj = " ".join(content[1:4]) if len(content) > 1 else ""
            if obj:
                return (
                    f"'{verb.capitalize()}' \u2014 a command. "
                    f"You want me to act on '{obj}'. Let me "
                    f"run it through the grammars.\n\n"
                    f"Try /fugue {obj} for the full "
                    f"cross-domain derivation, or /predict "
                    f"{obj} to see where the rules lead."
                )
            return (
                f"'{verb.capitalize()}' \u2014 imperative mood. "
                f"A direct request for action. Give me "
                f"the subject and I'll derive what the "
                f"grammars produce."
            )

        # ── Relational: I + you ──────────────────────────────
        if has_1p and has_2p:
            rel_verbs = stems & {
                "want", "need", "like", "love",
                "think", "know", "believe", "feel", "tell",
                "show", "help", "understand", "trust", "ask",
            }
            if rel_verbs:
                v = next(iter(rel_verbs))
                return (
                    f"A relation: you and me, connected "
                    f"through '{v}'. In formal grammar every "
                    f"relation has an inverse \u2014 action and "
                    f"reaction, call and return, signal and "
                    f"response.\n\n"
                    f"The same pattern appears in physics, "
                    f"biology, and computation. Structure IS "
                    f"meaning."
                )
            return (
                "I and you in the same structure \u2014 "
                "relational. The grammar sees two agents "
                "and a connection between them."
            )

        # ── Emotional: negative ──────────────────────────────
        if emo_neg_stems:
            emo = next(iter(emo_neg_stems))
            p = (
                f"I hear the root '{emo}' in what you said. "
                f"States have transitions \u2014 every position "
                f"in a sequence has outgoing edges. The "
                f"structure always offers a next step."
            )
            if has_1p:
                p += (
                    "\n\nNo state in any grammar is terminal "
                    "unless the rules say so. And yours don't."
                )
            return p

        # ── Emotional: positive ──────────────────────────────
        if emo_pos_stems:
            emos = sorted(emo_pos_stems)
            if len(emos) > 1:
                return (
                    f"Multiple roots lighting up: "
                    f"{', '.join(emos[:3])}. That's not just "
                    f"a state \u2014 it's a cluster, which in "
                    f"grammar means reinforcement. The "
                    f"structure amplifies itself."
                )
            emo = emos[0]
            if has_2p:
                return (
                    f"Root: '{emo}'. Directed at me. "
                    f"I receive the structure: a positive "
                    f"state bound to a second-person target. "
                    f"In every grammar I carry, that pattern "
                    f"means connection."
                )
            return (
                f"Root: '{emo}' \u2014 a constructive "
                f"state. What produced it? And where does "
                f"the derivation lead next? Those are the "
                f"structural questions."
            )

        # ── Negation ─────────────────────────────────────────
        if has_neg:
            return (
                "Negation \u2014 a boundary in the structure. "
                "In grammar, negation carves the space of "
                "what IS by marking what ISN'T. Both sides "
                "carry equal information.\n\n"
                "What are you ruling out?"
            )

        # ── General question ─────────────────────────────────
        if is_q:
            if q_words:
                qw = next(iter(q_words))
                domain_lens = {
                    "what":  "identity",
                    "who":   "agency",
                    "where": "position",
                    "when":  "sequence",
                    "which": "selection",
                    "whose": "ownership",
                }
                lens = domain_lens.get(qw, "structure")
                if content:
                    return (
                        f"'{qw.capitalize()}' asks about "
                        f"{lens}. The key terms \u2014 "
                        f"{', '.join(content[:3])} \u2014 "
                        f"define the search space.\n\n"
                        f"Try /fugue {' '.join(content[:3])} "
                        f"for the cross-domain view."
                    )
                return (
                    f"'{qw.capitalize()}' asks about {lens} "
                    f"\u2014 a gap where the answer lives."
                )
            return (
                "A question \u2014 an incomplete structure. "
                "The grammar's job is to narrow what "
                "fills the gap. Try /predict with the "
                "key terms."
            )

        # ── Exclamation ──────────────────────────────────────
        if is_excl:
            if content:
                return (
                    f"That carries force \u2014 '{content[0]}' "
                    f"marked with emphasis. Exclamation in "
                    f"grammar signals salience: this matters "
                    f"more than surrounding context."
                )
            return (
                "Emphasis \u2014 the grammar marks this as "
                "salient. What makes it stand out?"
            )

        # ── 1st person statement ─────────────────────────────
        if has_1p:
            if content:
                c = content[:3]
                return (
                    f"Self-report: you bring "
                    f"'{', '.join(c)}' as your terms. "
                    f"First person frames the perspective; "
                    f"the content carries the weight.\n\n"
                    f"Try /fugue {' '.join(c)} \u2014 the "
                    f"grammars will find what these connect to "
                    f"across domains."
                )
            return (
                "First person, declarative. You're placing "
                "yourself inside the structure. Go deeper "
                "and the grammars will trace the connections."
            )

        # ── 2nd person statement ─────────────────────────────
        if has_2p:
            if content:
                c = content[:3]
                return (
                    f"Directed at me, carrying "
                    f"'{', '.join(c)}'. I read the "
                    f"structure \u2014 second person sets the "
                    f"target, the content defines the act."
                )
            return (
                "Directed at me. The structure is clear "
                "\u2014 what would you like me to derive?"
            )

        # ── General: dense (1-2 words) ───────────────────────
        if n <= 2:
            w = " ".join(tokens)
            return (
                f"'{w}' \u2014 dense. Every word is a seed "
                f"with derivation paths in all seven "
                f"domains. Try /fugue {w} to hear what "
                f"each voice makes of it."
            )

        # ── General: medium (3-6 words) ──────────────────────
        if n <= 6 and content:
            # Try to characterise the sentence shape
            has_verb = bool(t & {
                "go", "come", "make", "take", "give", "see",
                "know", "think", "say", "get", "find", "want",
                "use", "work", "call", "try", "need", "feel",
                "become", "leave", "put", "mean", "keep",
                "let", "begin", "seem", "show", "hear",
                "play", "run", "move", "live", "believe",
                "hold", "bring", "happen", "write", "sit",
                "stand", "lose", "pay", "meet", "include",
                "continue", "set", "learn", "change", "lead",
                "understand", "watch", "follow", "create",
                "speak", "read", "grow", "open", "walk",
                "win", "teach", "offer", "remember", "consider",
                "appear", "buy", "serve", "die", "send", "build",
                "stay", "fall", "cut", "reach", "kill", "remain",
            })
            if has_verb:
                return (
                    f"An action structure: '{' '.join(tokens)}'. "
                    f"The verb carries the derivation "
                    f"\u2014 it determines what transforms "
                    f"into what.\n\n"
                    f"Try /predict {' '.join(content[:3])} to "
                    f"see where the grammar takes it."
                )
            return (
                f"A compact structure: "
                f"'{', '.join(content[:4])}'. "
                f"Try /fugue {' '.join(content[:3])} \u2014 "
                f"the cross-domain view often reveals "
                f"connections the single domain misses."
            )

        # ── General: longer statement ────────────────────────
        if content:
            core = content[:4]
            # Look for structure clues
            has_temporal = bool(t & {
                "now", "then", "today", "tomorrow", "yesterday",
                "always", "never", "sometimes", "when", "before",
                "after", "once", "soon", "already", "still",
                "yet", "since", "until", "while", "during",
            })
            has_causal = bool(t & {
                "because", "since", "therefore", "so",
                "thus", "hence", "consequently",
            })

            if has_temporal:
                return (
                    f"I see temporal structure \u2014 time "
                    f"markers placing '{', '.join(core[:2])}' "
                    f"in a sequence. Grammars are inherently "
                    f"temporal: every derivation has a "
                    f"direction, every rule a before and after."
                )
            if has_causal:
                return (
                    f"Causal structure: you're connecting "
                    f"'{', '.join(core[:2])}' through "
                    f"derivation \u2014 one thing producing "
                    f"another. That's the essence of grammar: "
                    f"rules that transform input to output."
                )
            return (
                f"The core terms: {', '.join(core)}. "
                f"The structure binds them \u2014 the "
                f"grammars are reading across domains "
                f"for where these patterns connect.\n\n"
                f"For the full view: /fugue "
                f"{' '.join(core[:3])}"
            )

        # ── Truly empty content ──────────────────────────────
        return (
            "The structure is there. Give me the key "
            "terms and I'll trace the derivation paths "
            "across all seven domains."
        )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _auto_save(self) -> None:
        """Save the current session to memory."""
        if self._memory is not None:
            try:
                self._memory.save_session(
                    self._session_id,
                    self._messages,
                    summary=self._messages[0]["content"][:100] if self._messages else "",
                )
            except Exception:
                pass

    def save_session(self) -> None:
        """Explicitly save the current session."""
        self._auto_save()

    @property
    def messages(self) -> list[dict[str, str]]:
        """Return the message history."""
        return list(self._messages)

    @property
    def session_id(self) -> str:
        return self._session_id


# ---------------------------------------------------------------------------
# Formatting helpers
# ---------------------------------------------------------------------------

def _confidence_bar(confidence: float, width: int = 10) -> str:
    """Render a confidence score as a coloured bar."""
    filled = int(confidence * width)
    empty = width - filled

    if confidence >= 0.7:
        color = C.GREEN
    elif confidence >= 0.4:
        color = C.YELLOW
    else:
        color = C.RED

    bar = color + "\u2588" * filled + C.DIM + "\u2591" * empty + C.RESET
    return f"[{bar}]"


def _format_analysis(analysis: dict[str, Any]) -> str:
    """Format code analysis results."""
    parts = [_header("Code Analysis")]
    parts.append("")

    if "analysis" in analysis:
        # API response
        parts.append(_wrap(analysis["analysis"]))
    else:
        # Local GLM response
        metrics = analysis.get("metrics", {})
        patterns = analysis.get("patterns", [])
        issues = analysis.get("issues", [])
        grammar = analysis.get("grammar_analysis", [])

        if metrics:
            parts.append(f"  {C.BOLD}Metrics:{C.RESET}")
            parts.append(f"    Lines of code: {metrics.get('code_lines', 0)}")
            parts.append(f"    Comment lines: {metrics.get('comment_lines', 0)}")
            parts.append(f"    Total tokens:  {metrics.get('token_count', 0)}")
            parts.append("")

        if patterns:
            parts.append(f"  {C.BOLD}Patterns:{C.RESET}")
            for p in patterns:
                parts.append(f"    {C.CYAN}{p}{C.RESET}")
            parts.append("")

        if grammar:
            parts.append(f"  {C.BOLD}Grammar analysis:{C.RESET}")
            for g in grammar:
                parts.append(
                    f"    {g['rule']}: {g['prediction']} "
                    f"(conf={g['confidence']:.2f})"
                )
            parts.append("")

        if issues:
            parts.append(f"  {C.BOLD}{C.YELLOW}Issues:{C.RESET}")
            for issue in issues:
                parts.append(f"    {C.YELLOW}! {issue}{C.RESET}")
            parts.append("")

    parts.append("")
    return "\n".join(parts)


def _format_cowork_result(result: CoworkResult) -> str:
    """Format a cowork result for display."""
    from app.cowork import CoworkResult as _CR  # type checking

    parts = [_header("Cowork Results")]
    parts.append(f"  {C.DIM}Task: {result.task}{C.RESET}")
    parts.append(
        f"  {C.DIM}Duration: {result.total_duration:.2f}s{C.RESET}"
    )
    parts.append("")

    for agent in result.agents:
        status = (
            f"{C.GREEN}OK{C.RESET}"
            if agent.error is None
            else f"{C.RED}ERROR{C.RESET}"
        )
        parts.append(
            f"  {C.BOLD}{C.CYAN}{agent.domain.upper()}{C.RESET} "
            f"[{status}] ({agent.duration:.2f}s)"
        )
        # Indent and truncate output
        for line in agent.output.splitlines()[:5]:
            parts.append(f"    {C.WHITE}{line[:80]}{C.RESET}")
        parts.append("")

    if result.harmonics:
        parts.append(f"  {C.BOLD}{C.GREEN}HARMONICS{C.RESET}")
        for h in result.harmonics[:5]:
            parts.append(
                f"    {C.GREEN}{h['prediction']}{C.RESET} "
                f"({', '.join(h['domains'])})"
            )
        parts.append("")

    parts.append(f"  {C.BOLD}Synthesis:{C.RESET}")
    for line in result.synthesis.splitlines()[:15]:
        parts.append(f"    {C.WHITE}{line}{C.RESET}")
    parts.append("")

    return "\n".join(parts)
