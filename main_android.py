"""
MKAngel -- Android (Kivy) entry point.

Premium dark-mode chat interface for the Grammar Language Model.
OLED-optimised, chat-bubble layout inspired by LINE / WeChat / iMessage.

Run via buildozer:  buildozer android debug deploy run
"""

from __future__ import annotations

import re
import threading
import time

from kivy.app import App
from kivy.clock import Clock
from kivy.core.window import Window
from kivy.graphics import Color, RoundedRectangle, Rectangle
from kivy.metrics import dp, sp
from kivy.uix.anchorlayout import AnchorLayout
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.button import Button
from kivy.uix.label import Label
from kivy.uix.scrollview import ScrollView
from kivy.uix.textinput import TextInput
from kivy.uix.widget import Widget
from kivy.utils import get_color_from_hex


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  Theme — OLED-first dark palette
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

_hex = get_color_from_hex

BG            = _hex("#000000")          # true OLED black
SURFACE       = _hex("#0e0e14")          # message bubble
SURFACE_HEAD  = _hex("#0c0c12")          # header bar
SURFACE_INPUT = _hex("#16161e")          # input field fill
ACCENT        = _hex("#bb86fc")          # primary purple
ACCENT_TINT   = [0.733, 0.525, 0.988, 0.10]
TEAL          = _hex("#03dac6")          # info accent
TEXT          = _hex("#f0f0f2")          # primary text
TEXT_SEC      = _hex("#9e9ea8")          # secondary text
TEXT_DIM      = _hex("#555560")          # dimmed / hint
GREEN         = _hex("#66bb6a")          # success
AMBER         = _hex("#ffab40")          # warning
RED           = _hex("#ef5350")          # error
SEP_COLOR     = [1, 1, 1, 0.04]         # hairline separator

Window.clearcolor = BG


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  Hairline separator
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class _Sep(Widget):
    def __init__(self, **kw):
        kw.setdefault("size_hint_y", None)
        kw.setdefault("height", 1)
        super().__init__(**kw)
        with self.canvas:
            Color(*SEP_COLOR)
            self._r = Rectangle(pos=self.pos, size=self.size)
        self.bind(pos=self._upd, size=self._upd)

    def _upd(self, *_):
        self._r.pos = self.pos
        self._r.size = self.size


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  Chat message bubble
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class _Bubble(BoxLayout):
    """Single message with rounded-rectangle background."""

    _STYLES = {
        "user":    (ACCENT_TINT, TEXT,  [dp(18), dp(18), dp(4),  dp(18)]),
        "angel":   (list(SURFACE), TEXT, [dp(4),  dp(18), dp(18), dp(18)]),
        "system":  ([0, 0, 0, 0], TEXT_DIM, [dp(8)] * 4),
        "success": ([0.40, 0.73, 0.42, 0.10], GREEN, [dp(14)] * 4),
        "error":   ([0.94, 0.33, 0.31, 0.10], AMBER, [dp(14)] * 4),
    }

    def __init__(self, text: str, kind: str = "angel", **kw):
        kw.setdefault("size_hint_y", None)
        super().__init__(orientation="vertical", **kw)

        bg, fg, radius = self._STYLES.get(kind, self._STYLES["angel"])
        halign = "center" if kind == "system" else "left"
        pad = dp(14)

        self._label = Label(
            text=text, markup=True, font_size=sp(14),
            color=fg, size_hint_y=None,
            halign=halign, valign="top",
            padding=(pad, pad),
        )
        self._label.bind(texture_size=self._resize)
        self._label.bind(width=lambda *_: setattr(
            self._label, "text_size", (self._label.width - 2 * pad, None)
        ))

        with self.canvas.before:
            Color(*bg)
            self._bg = RoundedRectangle(
                pos=self.pos, size=self.size, radius=radius,
            )
        self.bind(pos=self._canvas_upd, size=self._canvas_upd)
        self.add_widget(self._label)

    def _resize(self, *_):
        th = self._label.texture_size[1]
        if th > 0:
            self._label.height = th + dp(4)
            self.height = self._label.height + dp(4)

    def _canvas_upd(self, *_):
        self._bg.pos = self.pos
        self._bg.size = self.size


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  Scrollable chat view
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class ChatView(ScrollView):
    def __init__(self, **kw):
        super().__init__(**kw)
        self._box = BoxLayout(
            orientation="vertical", size_hint_y=None,
            spacing=dp(10), padding=[dp(10), dp(12)],
        )
        self._box.bind(minimum_height=self._box.setter("height"))
        self.add_widget(self._box)

    # ── public API ───────────────────────────────────────────────────────
    def add(self, text: str, kind: str = "angel") -> None:
        bubble = _Bubble(text, kind=kind)

        # Alignment: user → right, angel/error → left, system → centre
        if kind == "system":
            bubble.size_hint_x = 0.92
            row = BoxLayout(size_hint_y=None, height=dp(36))
            row.add_widget(Widget(size_hint_x=0.04))
            row.add_widget(bubble)
            row.add_widget(Widget(size_hint_x=0.04))
        elif kind == "user":
            bubble.size_hint_x = 0.78
            row = BoxLayout(size_hint_y=None, height=dp(36))
            row.add_widget(Widget(size_hint_x=0.22))
            row.add_widget(bubble)
        else:
            bubble.size_hint_x = 0.82
            row = BoxLayout(size_hint_y=None, height=dp(36))
            row.add_widget(bubble)
            row.add_widget(Widget(size_hint_x=0.18))

        bubble.bind(height=lambda _, v: setattr(row, "height", v))
        self._box.add_widget(row)
        Clock.schedule_once(lambda _: setattr(self, "scroll_y", 0), 0.12)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  Header
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class _Header(BoxLayout):
    def __init__(self, **kw):
        kw.setdefault("size_hint_y", None)
        kw.setdefault("height", dp(68))
        kw.setdefault("padding", [dp(16), dp(10), dp(16), dp(4)])
        super().__init__(orientation="vertical", **kw)

        with self.canvas.before:
            Color(*SURFACE_HEAD)
            self._bg = Rectangle(pos=self.pos, size=self.size)
        self.bind(pos=self._upd, size=self._upd)

        t = Label(
            text="[b]\u2727  M K A N G E L  \u2727[/b]", markup=True,
            font_size=sp(20), color=ACCENT,
            size_hint_y=0.58, halign="center", valign="bottom",
        )
        t.bind(size=lambda *_: setattr(t, "text_size", (t.width, None)))

        s = Label(
            text="Grammar Language Model", font_size=sp(11),
            color=TEXT_DIM, size_hint_y=0.42,
            halign="center", valign="top",
        )
        s.bind(size=lambda *_: setattr(s, "text_size", (s.width, None)))

        self.add_widget(t)
        self.add_widget(s)

    def _upd(self, *_):
        self._bg.pos = self.pos
        self._bg.size = self.size


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  Typing indicator
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class _Thinking(Label):
    def __init__(self, **kw):
        kw.setdefault("size_hint_y", None)
        kw.setdefault("height", dp(24))
        kw.setdefault("font_size", sp(12))
        kw.setdefault("color", TEXT_DIM)
        kw.setdefault("markup", True)
        kw.setdefault("halign", "left")
        kw.setdefault("valign", "center")
        super().__init__(**kw)
        self.text = ""
        self._n = 0
        self._ev = None
        self.padding = (dp(22), 0)
        self.bind(size=lambda *_: setattr(
            self, "text_size", (self.width, self.height)
        ))

    def start(self):
        self._n = 0
        if self._ev:
            self._ev.cancel()
        self._ev = Clock.schedule_interval(self._tick, 0.35)

    def stop(self):
        if self._ev:
            self._ev.cancel()
            self._ev = None
        self.text = ""

    def _tick(self, _):
        self._n = (self._n % 3) + 1
        dots = "\u25cf" * self._n + "\u25cb" * (3 - self._n)
        self.text = f"[color=555560]  {dots}[/color]"


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  Input bar
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class _InputBar(BoxLayout):
    def __init__(self, on_send=None, **kw):
        kw.setdefault("size_hint_y", None)
        kw.setdefault("height", dp(58))
        kw.setdefault("spacing", dp(8))
        kw.setdefault("padding", [dp(10), dp(8)])
        super().__init__(**kw)
        self._cb = on_send

        with self.canvas.before:
            Color(*SURFACE_HEAD)
            self._bg = Rectangle(pos=self.pos, size=self.size)
        self.bind(pos=self._upd, size=self._upd)

        self.inp = TextInput(
            hint_text="Talk to the Angel\u2026",
            multiline=False, font_size=sp(15),
            background_color=SURFACE_INPUT,
            foreground_color=TEXT,
            hint_text_color=TEXT_DIM,
            cursor_color=ACCENT,
            padding=(dp(16), dp(12)),
            size_hint_x=0.82,
        )
        self.inp.bind(on_text_validate=self._fire)
        self.add_widget(self.inp)

        btn = Button(
            text="\u25b6", font_size=sp(18), bold=True,
            size_hint_x=0.18,
            background_color=ACCENT, color=[1, 1, 1, 1],
        )
        btn.bind(on_press=self._fire)
        self.add_widget(btn)

    def _fire(self, *_):
        t = self.inp.text.strip()
        if t and self._cb:
            self._cb(t)
        self.inp.text = ""

    def _upd(self, *_):
        self._bg.pos = self.pos
        self._bg.size = self.size


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  Application
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class MKAngelApp(App):
    """MKAngel Android application."""

    title = "MKAngel"

    def build(self):
        Window.softinput_mode = "below_target"

        self.angel = None
        self.session = None
        self._ready = False

        root = BoxLayout(orientation="vertical")

        self.header = _Header()
        root.add_widget(self.header)
        root.add_widget(_Sep())

        self.chat = ChatView(size_hint=(1, 1))
        root.add_widget(self.chat)

        self.thinking = _Thinking()
        root.add_widget(self.thinking)

        root.add_widget(_Sep())
        self.bar = _InputBar(on_send=self._on_send)
        root.add_widget(self.bar)

        # Welcome
        self.chat.add(
            "[color=bb86fc][b]BE NOT AFRAID[/b][/color]",
            kind="system",
        )
        self.chat.add(
            "[color=555560]Awakening the Angel\u2026[/color]",
            kind="system",
        )

        threading.Thread(target=self._boot, daemon=True).start()
        return root

    # ── background boot ──────────────────────────────────────────────────
    def _boot(self):
        t0 = time.time()

        # 1. GLM
        try:
            from glm.angel import Angel
            self.angel = Angel()
            self.angel.awaken()
        except Exception as exc:
            m = str(exc)
            Clock.schedule_once(
                lambda _, m=m: self.chat.add(
                    f"[color=ffab40]\u26a0 GLM: {m}[/color]", kind="error",
                )
            )

        elapsed = time.time() - t0

        # 2. Session
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
            m = str(exc)
            Clock.schedule_once(
                lambda _, m=m: self.chat.add(
                    f"[color=ffab40]\u26a0 Session: {m}[/color]", kind="error",
                )
            )

        self._ready = True

        # 3. Status report
        try:
            info = self.angel.introspect() if self.angel else {}
        except Exception:
            info = {}

        domains = ", ".join(info.get("domains_loaded", [])) or "\u2014"
        g  = info.get("total_grammars", 0)
        ru = info.get("total_rules", 0)
        lo = info.get("strange_loops_detected", 0)
        pa = info.get("model_params", 0)

        status = (
            f"[color=66bb6a][b]\u2726 Angel awakened[/b]  {elapsed:.1f}s[/color]\n\n"
            f"[color=03dac6]Domains[/color]   {domains}\n"
            f"[color=03dac6]Grammars[/color]  {g}    "
            f"[color=03dac6]Rules[/color]  {ru}\n"
            f"[color=03dac6]Loops[/color]     {lo}    "
            f"[color=03dac6]Params[/color] {pa:,}\n\n"
            f"[color=555560]Type [b]/help[/b] or just talk.[/color]"
        )
        Clock.schedule_once(
            lambda _: self.chat.add(status, kind="success")
        )

    # ── send handler (main thread) ───────────────────────────────────────
    def _on_send(self, text: str):
        self.chat.add(text, kind="user")

        if not self._ready:
            self.chat.add(
                "[color=555560]Still awakening\u2026[/color]", kind="system",
            )
            return

        self.thinking.start()
        threading.Thread(
            target=self._process, args=(text,), daemon=True,
        ).start()

    # ── process input (background thread) ────────────────────────────────
    def _process(self, text: str):
        try:
            if self.session:
                resp = self.session.process_input(text)

                if resp == "__EXIT__":
                    self.session.save_session()
                    def _bye(_):
                        self.thinking.stop()
                        self.chat.add(
                            "[color=bb86fc][b]Be not afraid.[/b][/color]",
                            kind="system",
                        )
                    Clock.schedule_once(_bye)
                    return

                if resp:
                    clean = re.sub(r"\033\[[0-9;]*m", "", str(resp))
                    def _show(_, r=clean):
                        self.thinking.stop()
                        self.chat.add(r, kind="angel")
                    Clock.schedule_once(_show)
                else:
                    Clock.schedule_once(lambda _: self.thinking.stop())
            else:
                def _no_sess(_):
                    self.thinking.stop()
                    self.chat.add(
                        "[color=ffab40]Session not available.[/color]",
                        kind="error",
                    )
                Clock.schedule_once(_no_sess)

        except Exception as exc:
            err = str(exc)
            def _err(_, e=err):
                self.thinking.stop()
                self.chat.add(f"[color=ef5350]{e}[/color]", kind="error")
            Clock.schedule_once(_err)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

if __name__ == "__main__":
    MKAngelApp().run()
