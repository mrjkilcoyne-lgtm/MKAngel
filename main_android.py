"""
MKAngel -- Android (Kivy) entry point.

Wraps the Grammar Language Model in a touch-friendly mobile UI.
Run via buildozer: buildozer android debug deploy run
"""

from __future__ import annotations

import threading
import time

from kivy.app import App
from kivy.clock import Clock
from kivy.core.window import Window
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.button import Button
from kivy.uix.label import Label
from kivy.uix.scrollview import ScrollView
from kivy.uix.textinput import TextInput
from kivy.utils import get_color_from_hex

# Colours
BG = get_color_from_hex("#0d0d0d")
FG = get_color_from_hex("#e0e0e0")
ACCENT = get_color_from_hex("#bb86fc")
CYAN = get_color_from_hex("#00bcd4")
DIM = get_color_from_hex("#666666")
GREEN = get_color_from_hex("#4caf50")

Window.clearcolor = BG

BANNER = (
    "[color=bb86fc][b]BE NOT AFRAID[/b][/color]\n"
    "[color=00bcd4]Grammar Language Model v0.2.0[/color]\n"
    "[color=666666]learns scales -- plays masterpieces[/color]\n"
)


class ChatLog(ScrollView):
    """Scrollable chat history."""

    def __init__(self, **kw):
        super().__init__(**kw)
        self.label = Label(
            text=BANNER,
            markup=True,
            font_size="14sp",
            color=FG,
            size_hint_y=None,
            halign="left",
            valign="top",
            padding=(16, 8),
        )
        self.label.bind(texture_size=self.label.setter("size"))
        self.label.bind(size=lambda *_: setattr(
            self.label, "text_size", (self.label.width, None)
        ))
        self.add_widget(self.label)

    def append(self, text: str) -> None:
        self.label.text += "\n" + text
        Clock.schedule_once(lambda _: self._scroll_bottom(), 0.1)

    def _scroll_bottom(self):
        self.scroll_y = 0


class MKAngelApp(App):
    """MKAngel Android application."""

    title = "MKAngel"

    def build(self):
        self.angel = None
        self.session = None
        self._ready = False

        root = BoxLayout(orientation="vertical", padding=8, spacing=4)

        # Chat log
        self.chat = ChatLog(size_hint=(1, 1))
        root.add_widget(self.chat)

        # Input row
        input_row = BoxLayout(size_hint_y=None, height="48dp", spacing=4)

        self.text_input = TextInput(
            hint_text="Talk to the Angel...",
            multiline=False,
            font_size="16sp",
            background_color=get_color_from_hex("#1a1a1a"),
            foreground_color=FG,
            cursor_color=ACCENT,
            padding=(12, 12),
            size_hint_x=0.8,
        )
        self.text_input.bind(on_text_validate=self._on_send)
        input_row.add_widget(self.text_input)

        send_btn = Button(
            text=">",
            font_size="20sp",
            size_hint_x=0.2,
            background_color=ACCENT,
            color=(1, 1, 1, 1),
            bold=True,
        )
        send_btn.bind(on_press=self._on_send)
        input_row.add_widget(send_btn)

        root.add_widget(input_row)

        # Boot angel in background
        threading.Thread(target=self._boot_angel, daemon=True).start()

        return root

    def _boot_angel(self):
        """Initialise the Angel off the main thread."""
        Clock.schedule_once(
            lambda _: self.chat.append(
                "[color=666666]Awakening the Angel...[/color]"
            )
        )
        start = time.time()

        try:
            from glm.angel import Angel
            self.angel = Angel()
            self.angel.awaken()
        except Exception as exc:
            Clock.schedule_once(
                lambda _: self.chat.append(
                    f"[color=ffcc00]Warning: {exc}[/color]"
                )
            )

        elapsed = time.time() - start

        # Init session
        try:
            from app.settings import Settings
            from app.memory import Memory
            from app.providers import get_provider
            from app.chat import ChatSession

            settings = Settings.load()
            memory = Memory()
            provider = get_provider(settings)
            self.session = ChatSession(
                angel=self.angel,
                memory=memory,
                settings=settings,
                provider=provider,
            )
        except Exception as exc:
            Clock.schedule_once(
                lambda _: self.chat.append(
                    f"[color=ffcc00]Session init: {exc}[/color]"
                )
            )

        self._ready = True

        # Show status
        try:
            info = self.angel.introspect() if self.angel else {}
        except Exception:
            info = {}

        status_lines = [
            f"[color=4caf50][b]Angel awakened in {elapsed:.1f}s[/b][/color]",
            "",
            f"[color=00bcd4]Domains:[/color]  {', '.join(info.get('domains_loaded', []))}",
            f"[color=00bcd4]Grammars:[/color] {info.get('total_grammars', 0)}",
            f"[color=00bcd4]Rules:[/color]    {info.get('total_rules', 0)}",
            f"[color=00bcd4]Loops:[/color]    {info.get('strange_loops_detected', 0)}",
            f"[color=00bcd4]Params:[/color]   {info.get('model_params', 0):,}",
            "",
            "[color=666666]Type /help for commands, or just start talking.[/color]",
        ]
        Clock.schedule_once(
            lambda _: self.chat.append("\n".join(status_lines))
        )

    def _on_send(self, *_args):
        text = self.text_input.text.strip()
        if not text:
            return
        self.text_input.text = ""

        # Show user message
        self.chat.append(f"[color=bb86fc][b]> {text}[/b][/color]")

        if not self._ready:
            self.chat.append("[color=666666]Still awakening...[/color]")
            return

        # Process in background
        threading.Thread(
            target=self._process, args=(text,), daemon=True
        ).start()

    def _process(self, text: str):
        try:
            if self.session:
                response = self.session.process_input(text)
                if response == "__EXIT__":
                    self.session.save_session()
                    Clock.schedule_once(
                        lambda _: self.chat.append(
                            "[color=bb86fc][b]Be not afraid.[/b][/color]"
                        )
                    )
                    return
                if response:
                    # Strip ANSI codes for Kivy display
                    import re
                    clean = re.sub(r"\033\[[0-9;]*m", "", str(response))
                    Clock.schedule_once(
                        lambda _, r=clean: self.chat.append(r)
                    )
            else:
                Clock.schedule_once(
                    lambda _: self.chat.append(
                        "[color=ffcc00]Session not available.[/color]"
                    )
                )
        except Exception as exc:
            Clock.schedule_once(
                lambda _, e=str(exc): self.chat.append(
                    f"[color=ff4444]{e}[/color]"
                )
            )


if __name__ == "__main__":
    MKAngelApp().run()
