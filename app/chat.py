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

        if self._provider is not None:
            # Build conversation context
            system = (
                "You are MKAngel, an AI assistant powered by a Grammar Language "
                "Model. You understand deep structural patterns across language, "
                "code, chemistry, and biology. Be helpful, concise, and insightful. "
                "When appropriate, reference grammatical structures and patterns."
            )
            # Include recent history for context
            history_lines = []
            for msg in self._messages[-10:]:
                role = msg["role"]
                content = msg["content"][:200]
                history_lines.append(f"{role}: {content}")

            context = "\n".join(history_lines)
            prompt = f"Conversation context:\n{context}\n\nUser: {text}"

            response = self._provider.generate(prompt, system=system)
            return _angel_response(response)

        # Local GLM path
        if self._angel is not None:
            tokens = text.lower().split()
            try:
                predictions = self._angel.predict(
                    tokens, domain="linguistic", horizon=5
                )
                if predictions:
                    parts = [_header("Angel")]
                    parts.append("")
                    predicted = [
                        str(p.get("predicted", "?")) for p in predictions[:5]
                    ]
                    parts.append(
                        _wrap(f"Grammatical derivation: {' '.join(predicted)}")
                    )
                    parts.append("")

                    if matching:
                        skill_names = [s.name for s in matching]
                        parts.append(
                            _info(f"  Matching skills: {', '.join(skill_names)}")
                        )
                        parts.append(
                            _info(f"  Use /skills run <name> to execute.")
                        )
                        parts.append("")

                    parts.append(
                        _info(
                            "  Tip: Use /predict, /forecast, or /fugue for "
                            "deeper analysis."
                        )
                    )
                    return "\n".join(parts)
            except Exception:
                pass

        # Absolute fallback
        parts = [_header("Angel")]
        parts.append("")
        parts.append(
            _wrap(
                "I received your message. The local Grammar Language Model "
                "processes input through grammatical derivation rules. "
                "For richer conversations, configure an API provider with "
                "/settings."
            )
        )
        if matching:
            skill_names = [s.name for s in matching]
            parts.append("")
            parts.append(
                _info(f"  Matching skills: {', '.join(skill_names)}")
            )
        parts.append("")
        parts.append(_info("  Type /help for available commands."))
        return "\n".join(parts)

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
