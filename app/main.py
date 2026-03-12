"""
MKAngel -- Main entry point.

Launches the interactive Grammar Language Model assistant.
Works on Termux (Android), desktop Linux/macOS/Windows.
"""

from __future__ import annotations

import os
import sys
import time
import shutil


# ---------------------------------------------------------------------------
# ANSI colours
# ---------------------------------------------------------------------------

class C:
    RESET   = "\033[0m"
    BOLD    = "\033[1m"
    DIM     = "\033[2m"
    ITALIC  = "\033[3m"

    BLACK   = "\033[30m"
    RED     = "\033[31m"
    GREEN   = "\033[32m"
    YELLOW  = "\033[33m"
    BLUE    = "\033[34m"
    MAGENTA = "\033[35m"
    CYAN    = "\033[36m"
    WHITE   = "\033[37m"

    BRIGHT_RED     = "\033[91m"
    BRIGHT_GREEN   = "\033[92m"
    BRIGHT_YELLOW  = "\033[93m"
    BRIGHT_BLUE    = "\033[94m"
    BRIGHT_MAGENTA = "\033[95m"
    BRIGHT_CYAN    = "\033[96m"
    BRIGHT_WHITE   = "\033[97m"

    BG_BLACK = "\033[40m"


# ---------------------------------------------------------------------------
# The Banner -- BE NOT AFRAID
# ---------------------------------------------------------------------------

_BANNER = r"""
{m}  ██████╗  ███████╗    ███╗   ██╗  ██████╗  ████████╗
  ██╔══██╗ ██╔════╝    ████╗  ██║ ██╔═══██╗ ╚══██╔══╝
  ██████╔╝ █████╗      ██╔██╗ ██║ ██║   ██║    ██║
  ██╔══██╗ ██╔══╝      ██║╚██╗██║ ██║   ██║    ██║
  ██████╔╝ ███████╗    ██║ ╚████║ ╚██████╔╝    ██║
  ╚═════╝  ╚══════╝    ╚═╝  ╚═══╝  ╚═════╝     ╚═╝{r}

{c}     █████╗  ███████╗ ██████╗   █████╗  ██╗ ██████╗
    ██╔══██╗ ██╔════╝ ██╔══██╗ ██╔══██╗ ██║ ██╔══██╗
    ███████║ █████╗   ██████╔╝ ███████║ ██║ ██║  ██║
    ██╔══██║ ██╔══╝   ██╔══██╗ ██╔══██║ ██║ ██║  ██║
    ██║  ██║ ██║      ██║  ██║ ██║  ██║ ██║ ██████╔╝
    ╚═╝  ╚═╝ ╚═╝      ╚═╝  ╚═╝ ╚═╝  ╚═╝ ╚═╝ ╚═════╝{r}
"""

_WINGS = r"""
{d}                     .     .
                    / \   / \
                   /   \ /   \
                  /     X     \
                 /     / \     \
                /     /   \     \
               /     /     \     \{r}
{y}            --+-----+-+---+-+-----+--{r}
{d}               \     \     /     /
                \     \   /     /
                 \     \ /     /
                  \     X     /
                   \   / \   /
                    \ /   \ /
                     '     '{r}
"""

_SUBTITLE = (
    "{d}         Grammar Language Model  v0.1.0{r}\n"
    "{d}      learns scales -- plays masterpieces{r}\n"
)


def _print_banner() -> None:
    """Display the BE NOT AFRAID startup banner."""
    width = shutil.get_terminal_size((80, 24)).columns

    # Format with colours
    banner = _BANNER.format(
        m=C.BRIGHT_MAGENTA + C.BOLD,
        c=C.BRIGHT_CYAN + C.BOLD,
        r=C.RESET,
    )
    wings = _WINGS.format(
        d=C.DIM + C.WHITE,
        y=C.BRIGHT_YELLOW + C.BOLD,
        r=C.RESET,
    )
    subtitle = _SUBTITLE.format(
        d=C.DIM + C.WHITE,
        r=C.RESET,
    )

    print()
    print(banner)
    print(wings)
    print(subtitle)
    print()


# ---------------------------------------------------------------------------
# System status display
# ---------------------------------------------------------------------------

def _print_status(angel, settings) -> None:
    """Print system status after initialisation."""
    try:
        info = angel.introspect()
    except Exception:
        info = {}

    width = shutil.get_terminal_size((80, 24)).columns
    line = C.DIM + "\u2500" * min(width - 4, 60) + C.RESET

    print(f"  {line}")
    print()
    print(f"  {C.GREEN}{C.BOLD}Angel awakened.{C.RESET}")
    print()

    items = [
        ("Domains",       ", ".join(info.get("domains_loaded", []))),
        ("Grammars",      str(info.get("total_grammars", 0))),
        ("Rules",         str(info.get("total_rules", 0))),
        ("Productions",   str(info.get("total_productions", 0))),
        ("Strange loops", str(info.get("strange_loops_detected", 0))),
        ("Substrates",    ", ".join(info.get("substrates_loaded", []))),
        ("Model params",  f"{info.get('model_params', 0):,}"),
    ]
    max_label = max(len(label) for label, _ in items)
    for label, value in items:
        padding = " " * (max_label - len(label))
        print(f"  {C.CYAN}{label}:{C.RESET}{padding}  {value}")

    # Provider info
    provider_name = settings.model_provider
    offline = settings.offline_mode
    mode = "offline" if offline else f"online ({provider_name})"
    print(f"  {C.CYAN}Mode:{C.RESET}{'':>{max_label - 4}}  {mode}")
    print()
    print(f"  {line}")
    print()
    print(
        f"  {C.DIM}The scales are learned. Ready for masterpieces.{C.RESET}"
    )
    print(
        f"  {C.DIM}Type {C.BRIGHT_CYAN}/help{C.DIM} for commands, "
        f"or just start talking.{C.RESET}"
    )
    print()


# ---------------------------------------------------------------------------
# Input prompt
# ---------------------------------------------------------------------------

def _prompt() -> str:
    """The interactive input prompt."""
    return f"{C.BRIGHT_CYAN}{C.BOLD}\u276f {C.RESET}"


# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------

def main() -> None:
    """MKAngel main entry point.

    Displays the banner, initialises the system, and enters the
    interactive chat loop.
    """
    # Clear screen
    print("\033[2J\033[H", end="")

    # Show the banner
    _print_banner()

    # Initialise settings
    from app.settings import Settings
    settings = Settings.load()

    # Initialise the Angel
    print(f"  {C.DIM}Awakening the Angel...{C.RESET}", end="", flush=True)
    start = time.time()

    from glm.angel import Angel
    angel = Angel()
    try:
        angel.awaken()
    except Exception as exc:
        print(f"\r  {C.YELLOW}Angel initialisation warning: {exc}{C.RESET}")
        print(f"  {C.DIM}Continuing with limited functionality.{C.RESET}")

    elapsed = time.time() - start
    print(f"\r  {C.GREEN}Angel awakened in {elapsed:.1f}s.{' ' * 20}{C.RESET}")
    print()

    # Show status
    _print_status(angel, settings)

    # Initialise memory
    from app.memory import Memory
    memory = Memory()

    # Initialise provider (share the angel instance)
    from app.providers import get_provider
    provider = get_provider(settings, angel=angel)

    # Create chat session
    from app.chat import ChatSession
    session = ChatSession(
        angel=angel,
        memory=memory,
        settings=settings,
        provider=provider,
    )

    # Interactive loop
    try:
        while True:
            try:
                user_input = input(_prompt())
            except EOFError:
                break

            text = user_input.strip()
            if not text:
                continue

            # Process input
            response = session.process_input(text)

            # Check for exit signal
            if response == "__EXIT__":
                session.save_session()
                print()
                print(
                    f"  {C.DIM}Session saved. "
                    f"The Angel rests.{C.RESET}"
                )
                print(
                    f"  {C.BRIGHT_MAGENTA}{C.BOLD}"
                    f"Be not afraid.{C.RESET}"
                )
                print()
                break

            # Display response
            if response:
                print(response)

    except KeyboardInterrupt:
        # Graceful exit on Ctrl+C
        session.save_session()
        print()
        print(f"\n  {C.DIM}Session saved. The Angel rests.{C.RESET}")
        print(f"  {C.BRIGHT_MAGENTA}{C.BOLD}Be not afraid.{C.RESET}")
        print()

    finally:
        memory.close()


if __name__ == "__main__":
    main()
