"""
MKAngel -- Android (Kivy) entry point.

Gesture-driven panel system for the Grammar Language Model.
Eight panels accessible via swipe, tap, long-press, and pinch gestures.

Run via buildozer:  buildozer android debug deploy run
"""

from __future__ import annotations

import re
import threading
import time

from kivy.animation import Animation
from kivy.app import App
from kivy.clock import Clock
from kivy.core.window import Window
from kivy.graphics import Color, RoundedRectangle, Rectangle
from kivy.metrics import dp, sp
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.button import Button
from kivy.uix.floatlayout import FloatLayout
from kivy.uix.label import Label
from kivy.uix.scrollview import ScrollView
from kivy.uix.textinput import TextInput
from kivy.uix.widget import Widget

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  Vestment tokens — design system
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

try:
    from app.vestments import get_vestment, vestment_to_css, ALL_VESTMENTS
except ImportError:
    # Minimal fallback if vestments module unavailable
    ALL_VESTMENTS = {}
    def get_vestment(name=None):
        from kivy.utils import get_color_from_hex as _h
        _t = lambda c, a=1.0: {"kivy": list(_h(c))[:3] + [a], "css": c}
        return {
            "name": "Celestial Dark", "id": "celestial_dark",
            "bg": _t("#000000"), "surface": _t("#0e0e14"),
            "surface_head": _t("#0c0c12"), "surface_input": _t("#16161e"),
            "accent": _t("#D4AF37"), "accent_purple": _t("#bb86fc"),
            "teal": _t("#03dac6"), "text": _t("#f0f0f2"),
            "text_sec": _t("#9e9ea8"), "text_dim": _t("#555560"),
            "success": _t("#66bb6a"), "warning": _t("#ffab40"),
            "error": _t("#ef5350"), "separator": _t("#ffffff", 0.04),
            "bubble_user": _t("#bb86fc", 0.10),
            "bubble_angel": _t("#0e0e14"),
        }
    def vestment_to_css(v): return ""

try:
    from app.gestures import GestureDetector, GestureAction
except ImportError:
    # Stub if gestures module unavailable
    class GestureAction:
        CHAT = "chat"; VOICE = "voice"; HOSTS = "hosts"
        WINGS = "wings"; SKILLS = "skills"; ASPECTS = "aspects"
        VESTMENTS = "vestments"; DOCUMENTS = "documents"
    class GestureDetector(Widget):
        def __init__(self, callback=None, **kw):
            super().__init__(**kw)
        def set_callback(self, fn): pass

try:
    from app.documents import DocumentManager
except ImportError:
    DocumentManager = None

# Load active vestment
V = get_vestment()

Window.clearcolor = V["bg"]["kivy"]


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  Helper — colour shorthand
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def _c(token_name: str) -> list:
    """Pull a kivy rgba list from the active vestment."""
    return V[token_name]["kivy"]

def _css(token_name: str) -> str:
    """Pull the CSS hex from the active vestment."""
    return V[token_name]["css"]


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  Hairline separator
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class _Sep(Widget):
    def __init__(self, **kw):
        kw.setdefault("size_hint_y", None)
        kw.setdefault("height", 1)
        super().__init__(**kw)
        with self.canvas:
            Color(*_c("separator"))
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

    def __init__(self, text: str, kind: str = "angel", **kw):
        kw.setdefault("size_hint_y", None)
        super().__init__(orientation="vertical", **kw)

        styles = {
            "user":    (_c("bubble_user"),  _c("text"),     [dp(18), dp(18), dp(4),  dp(18)]),
            "angel":   (_c("bubble_angel"), _c("text"),     [dp(4),  dp(18), dp(18), dp(18)]),
            "system":  ([0, 0, 0, 0],      _c("text_dim"), [dp(8)] * 4),
            "success": (_c("success")[:3] + [0.10], _c("success"), [dp(14)] * 4),
            "error":   (_c("error")[:3] + [0.10],  _c("warning"), [dp(14)] * 4),
        }

        bg, fg, radius = styles.get(kind, styles["angel"])
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

    def add(self, text: str, kind: str = "angel") -> None:
        bubble = _Bubble(text, kind=kind)

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
            Color(*_c("surface_head"))
            self._bg = Rectangle(pos=self.pos, size=self.size)
        self.bind(pos=self._upd, size=self._upd)

        t = Label(
            text="[b]\u2727  M K A N G E L  \u2727[/b]", markup=True,
            font_size=sp(20), color=_c("accent"),
            size_hint_y=0.58, halign="center", valign="bottom",
        )
        t.bind(size=lambda *_: setattr(t, "text_size", (t.width, None)))

        s = Label(
            text="Grammar Language Model", font_size=sp(11),
            color=_c("text_dim"), size_hint_y=0.42,
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
        kw.setdefault("color", _c("text_dim"))
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
        dim_hex = _css("text_dim").lstrip("#")
        self.text = f"[color={dim_hex}]  {dots}[/color]"


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
            Color(*_c("surface_head"))
            self._bg = Rectangle(pos=self.pos, size=self.size)
        self.bind(pos=self._upd, size=self._upd)

        self.inp = TextInput(
            hint_text="Talk to the Angel\u2026",
            multiline=False, font_size=sp(15),
            background_color=_c("surface_input"),
            foreground_color=_c("text"),
            hint_text_color=_c("text_dim"),
            cursor_color=_c("accent"),
            padding=(dp(16), dp(12)),
            size_hint_x=0.82,
        )
        self.inp.bind(on_text_validate=self._fire)
        self.add_widget(self.inp)

        btn = Button(
            text="\u25b6", font_size=sp(18), bold=True,
            size_hint_x=0.18,
            background_color=_c("accent"), color=[1, 1, 1, 1],
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
#  Panel base — shared background + layout for secondary panels
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class _PanelBase(BoxLayout):
    """Base for non-chat panels. Full-screen with vestment background."""

    def __init__(self, **kw):
        kw.setdefault("orientation", "vertical")
        kw.setdefault("opacity", 0)
        kw.setdefault("padding", [dp(20), dp(16)])
        kw.setdefault("spacing", dp(12))
        super().__init__(**kw)
        self.disabled = True

        with self.canvas.before:
            Color(*_c("bg"))
            self._bg = Rectangle(pos=self.pos, size=self.size)
        self.bind(pos=self._upd_bg, size=self._upd_bg)

    def _upd_bg(self, *_):
        self._bg.pos = self.pos
        self._bg.size = self.size

    def _make_title(self, text: str) -> Label:
        accent_hex = _css("accent").lstrip("#")
        lbl = Label(
            text=f"[b][color={accent_hex}]{text}[/color][/b]",
            markup=True, font_size=sp(22), color=_c("text"),
            size_hint_y=None, height=dp(48),
            halign="center", valign="middle",
        )
        lbl.bind(size=lambda *_: setattr(lbl, "text_size", lbl.size))
        return lbl

    def _make_back_btn(self, callback) -> Button:
        return Button(
            text="\u2190 Back to Chat", font_size=sp(14),
            size_hint=(1, None), height=dp(44),
            background_color=_c("surface"),
            color=_c("accent"),
            on_press=callback,
        )


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  Voice panel — swipe up
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class _VoicePanel(_PanelBase):
    def __init__(self, go_back=None, **kw):
        super().__init__(**kw)

        self.add_widget(self._make_title("\u2727 Voice"))
        self.add_widget(Widget(size_hint_y=0.2))

        # Mic icon placeholder
        mic = Label(
            text="\U0001F399", font_size=sp(72),
            size_hint_y=None, height=dp(100),
            halign="center", valign="middle",
        )
        mic.bind(size=lambda *_: setattr(mic, "text_size", mic.size))
        self.add_widget(mic)

        hint = Label(
            text="Swipe up to speak", font_size=sp(16),
            color=_c("text_sec"), size_hint_y=None, height=dp(40),
            halign="center", valign="middle",
        )
        hint.bind(size=lambda *_: setattr(hint, "text_size", hint.size))
        self.add_widget(hint)

        sub = Label(
            text="Voice mode coming in Phase 2",
            font_size=sp(12), color=_c("text_dim"),
            size_hint_y=None, height=dp(30),
            halign="center", valign="middle",
        )
        sub.bind(size=lambda *_: setattr(sub, "text_size", sub.size))
        self.add_widget(sub)

        self.add_widget(Widget(size_hint_y=1))

        if go_back:
            self.add_widget(self._make_back_btn(go_back))


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  Host panel — swipe right
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class _HostPanel(_PanelBase):
    def __init__(self, go_back=None, **kw):
        super().__init__(**kw)

        self.add_widget(self._make_title("\u2727 Hosts"))
        self.add_widget(Widget(size_hint_y=0.1))

        self._status = Label(
            text="No hosts deployed", font_size=sp(14),
            color=_c("text_sec"), size_hint_y=None, height=dp(40),
            halign="center", valign="middle",
        )
        self._status.bind(size=lambda *_: setattr(
            self._status, "text_size", self._status.size
        ))
        self.add_widget(self._status)

        deploy_btn = Button(
            text="Deploy Host", font_size=sp(16),
            size_hint=(0.6, None), height=dp(48),
            pos_hint={"center_x": 0.5},
            background_color=_c("accent"), color=[1, 1, 1, 1],
        )
        deploy_btn.bind(on_press=self._deploy)
        self.add_widget(deploy_btn)

        self.add_widget(Widget(size_hint_y=1))

        if go_back:
            self.add_widget(self._make_back_btn(go_back))

    def _deploy(self, *_):
        self._status.text = "Host deployment coming in Phase 2"
        self._status.color = _c("warning")


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  Skills overlay — long press
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class _SkillsOverlay(_PanelBase):
    """Semi-transparent floating overlay with skill list."""

    def __init__(self, go_back=None, **kw):
        super().__init__(**kw)

        # Override bg for semi-transparency
        self.canvas.before.clear()
        with self.canvas.before:
            Color(*(_c("bg")[:3] + [0.85]))
            self._bg = Rectangle(pos=self.pos, size=self.size)
        self.bind(pos=self._upd_bg, size=self._upd_bg)

        self.add_widget(self._make_title("\u2727 Skills"))

        _skills = [
            ("/help", "Show available commands"),
            ("/code", "Generate or edit code"),
            ("/doc", "Create a document"),
            ("/search", "Search the web"),
            ("/voice", "Toggle voice mode"),
            ("/clear", "Clear conversation"),
        ]

        for cmd, desc in _skills:
            row = BoxLayout(
                size_hint_y=None, height=dp(40),
                spacing=dp(8), padding=[dp(8), 0],
            )
            accent_hex = _css("accent").lstrip("#")
            cmd_lbl = Label(
                text=f"[b][color={accent_hex}]{cmd}[/color][/b]",
                markup=True, font_size=sp(14),
                size_hint_x=0.3, halign="right", valign="middle",
            )
            cmd_lbl.bind(size=lambda w, *_: setattr(w, "text_size", w.size))
            desc_lbl = Label(
                text=desc, font_size=sp(13),
                color=_c("text_sec"),
                size_hint_x=0.7, halign="left", valign="middle",
            )
            desc_lbl.bind(size=lambda w, *_: setattr(w, "text_size", w.size))
            row.add_widget(cmd_lbl)
            row.add_widget(desc_lbl)
            self.add_widget(row)

        self.add_widget(Widget(size_hint_y=1))

        if go_back:
            self.add_widget(self._make_back_btn(go_back))


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  Aspect switcher — swipe down
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class _AspectSwitcher(_PanelBase):
    """Four aspect cards — the Angel's modes of being."""

    _ASPECTS = [
        ("Awakening",       "Initial boot state",              "\u2726"),
        ("Companion",       "Personal assistant, warm & kind",  "\u2665"),
        ("Command Centre",  "Power mode, tools & hosts",        "\u2318"),
        ("Oracle",          "Deep analysis & foresight",        "\u2609"),
    ]

    def __init__(self, on_select=None, go_back=None, **kw):
        super().__init__(**kw)
        self._on_select = on_select

        self.add_widget(self._make_title("\u2727 Aspects"))

        for name, desc, icon in self._ASPECTS:
            card = BoxLayout(
                size_hint_y=None, height=dp(64),
                spacing=dp(12), padding=[dp(12), dp(6)],
            )
            with card.canvas.before:
                Color(*_c("surface"))
                card._bg = RoundedRectangle(
                    pos=card.pos, size=card.size, radius=[dp(12)],
                )
            card.bind(
                pos=lambda w, *_: setattr(w._bg, "pos", w.pos),
                size=lambda w, *_: setattr(w._bg, "size", w.size),
            )

            icon_lbl = Label(
                text=icon, font_size=sp(28),
                size_hint_x=0.15, halign="center", valign="middle",
            )
            icon_lbl.bind(size=lambda w, *_: setattr(w, "text_size", w.size))

            info = BoxLayout(orientation="vertical", size_hint_x=0.85)
            n_lbl = Label(
                text=f"[b]{name}[/b]", markup=True,
                font_size=sp(15), color=_c("text"),
                size_hint_y=0.55, halign="left", valign="bottom",
            )
            n_lbl.bind(size=lambda w, *_: setattr(w, "text_size", w.size))
            d_lbl = Label(
                text=desc, font_size=sp(12),
                color=_c("text_dim"),
                size_hint_y=0.45, halign="left", valign="top",
            )
            d_lbl.bind(size=lambda w, *_: setattr(w, "text_size", w.size))
            info.add_widget(n_lbl)
            info.add_widget(d_lbl)

            card.add_widget(icon_lbl)
            card.add_widget(info)
            self.add_widget(card)

        self.add_widget(Widget(size_hint_y=1))

        if go_back:
            self.add_widget(self._make_back_btn(go_back))


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  Vestment switcher — double tap
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class _VestmentSwitcher(_PanelBase):
    """Four mini-preview cards for each vestment."""

    _VESTMENTS = [
        ("celestial_dark",  "Celestial Dark",  "OLED black + gold"),
        ("ethereal_light",  "Ethereal Light",  "Warm white + silver"),
        ("living_gradient", "Living Gradient",  "Aurora animation"),
        ("minimal_power",   "Minimal Power",   "Monochrome typography"),
    ]

    def __init__(self, on_select=None, go_back=None, **kw):
        super().__init__(**kw)
        self._on_select = on_select

        self.add_widget(self._make_title("\u2727 Vestments"))

        for vid, name, desc in self._VESTMENTS:
            try:
                vest = ALL_VESTMENTS.get(vid, V)
                preview_bg = vest["surface"]["kivy"]
                preview_accent = vest["accent"]["kivy"]
            except Exception:
                preview_bg = _c("surface")
                preview_accent = _c("accent")

            card = BoxLayout(
                size_hint_y=None, height=dp(56),
                spacing=dp(10), padding=[dp(12), dp(4)],
            )
            with card.canvas.before:
                Color(*preview_bg)
                card._bg = RoundedRectangle(
                    pos=card.pos, size=card.size, radius=[dp(10)],
                )
            card.bind(
                pos=lambda w, *_: setattr(w._bg, "pos", w.pos),
                size=lambda w, *_: setattr(w._bg, "size", w.size),
            )

            swatch = Widget(size_hint_x=0.08)
            with swatch.canvas:
                Color(*preview_accent)
                swatch._dot = RoundedRectangle(
                    pos=swatch.pos, size=(dp(20), dp(20)),
                    radius=[dp(10)],
                )
            swatch.bind(
                pos=lambda w, *_: setattr(w._dot, "pos", (
                    w.x + w.width / 2 - dp(10),
                    w.y + w.height / 2 - dp(10),
                )),
            )

            info = BoxLayout(orientation="vertical", size_hint_x=0.92)
            n_lbl = Label(
                text=f"[b]{name}[/b]", markup=True,
                font_size=sp(14), color=preview_accent,
                size_hint_y=0.55, halign="left", valign="bottom",
            )
            n_lbl.bind(size=lambda w, *_: setattr(w, "text_size", w.size))
            d_lbl = Label(
                text=desc, font_size=sp(11),
                color=_c("text_dim"),
                size_hint_y=0.45, halign="left", valign="top",
            )
            d_lbl.bind(size=lambda w, *_: setattr(w, "text_size", w.size))
            info.add_widget(n_lbl)
            info.add_widget(d_lbl)

            card.add_widget(swatch)
            card.add_widget(info)
            self.add_widget(card)

        self.add_widget(Widget(size_hint_y=1))

        if go_back:
            self.add_widget(self._make_back_btn(go_back))


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  Document panel — pinch
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class _DocumentPanel(_PanelBase):
    """Placeholder for the Quill.js WebView editor."""

    def __init__(self, go_back=None, **kw):
        super().__init__(**kw)

        self.add_widget(self._make_title("\u2727 Documents"))
        self.add_widget(Widget(size_hint_y=0.15))

        icon = Label(
            text="\U0001F4C4", font_size=sp(64),
            size_hint_y=None, height=dp(80),
            halign="center", valign="middle",
        )
        icon.bind(size=lambda *_: setattr(icon, "text_size", icon.size))
        self.add_widget(icon)

        info = Label(
            text="Document Editor", font_size=sp(18),
            color=_c("text"), size_hint_y=None, height=dp(36),
            halign="center", valign="middle",
        )
        info.bind(size=lambda *_: setattr(info, "text_size", info.size))
        self.add_widget(info)

        sub = Label(
            text="Pinch to open \u2014 WebView loads on Android device",
            font_size=sp(12), color=_c("text_dim"),
            size_hint_y=None, height=dp(30),
            halign="center", valign="middle",
        )
        sub.bind(size=lambda *_: setattr(sub, "text_size", sub.size))
        self.add_widget(sub)

        self.add_widget(Widget(size_hint_y=1))

        if go_back:
            self.add_widget(self._make_back_btn(go_back))


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  Wings panel — swipe left
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class _WingsPanel(_PanelBase):
    """Placeholder for Phase 2 extensions system."""

    def __init__(self, go_back=None, **kw):
        super().__init__(**kw)

        self.add_widget(self._make_title("\u2727 Wings"))
        self.add_widget(Widget(size_hint_y=0.2))

        icon = Label(
            text="\U0001F54A", font_size=sp(64),
            size_hint_y=None, height=dp(80),
            halign="center", valign="middle",
        )
        icon.bind(size=lambda *_: setattr(icon, "text_size", icon.size))
        self.add_widget(icon)

        info = Label(
            text="Extensions & integrations", font_size=sp(16),
            color=_c("text_sec"), size_hint_y=None, height=dp(36),
            halign="center", valign="middle",
        )
        info.bind(size=lambda *_: setattr(info, "text_size", info.size))
        self.add_widget(info)

        sub = Label(
            text="Wings unfold in Phase 2",
            font_size=sp(12), color=_c("text_dim"),
            size_hint_y=None, height=dp(30),
            halign="center", valign="middle",
        )
        sub.bind(size=lambda *_: setattr(sub, "text_size", sub.size))
        self.add_widget(sub)

        self.add_widget(Widget(size_hint_y=1))

        if go_back:
            self.add_widget(self._make_back_btn(go_back))


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  Chat panel — wraps header + chat view + thinking + input bar
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class _ChatPanel(BoxLayout):
    """The default panel: header, scrollable chat, thinking indicator, input."""

    def __init__(self, on_send=None, **kw):
        kw.setdefault("orientation", "vertical")
        super().__init__(**kw)

        with self.canvas.before:
            Color(*_c("bg"))
            self._bg = Rectangle(pos=self.pos, size=self.size)
        self.bind(pos=self._upd_bg, size=self._upd_bg)

        self.header = _Header()
        self.add_widget(self.header)
        self.add_widget(_Sep())

        self.chat = ChatView(size_hint=(1, 1))
        self.add_widget(self.chat)

        self.thinking = _Thinking()
        self.add_widget(self.thinking)

        self.add_widget(_Sep())
        self.bar = _InputBar(on_send=on_send)
        self.add_widget(self.bar)

    def _upd_bg(self, *_):
        self._bg.pos = self.pos
        self._bg.size = self.size


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  Application — the Angel's body
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class MKAngelApp(App):
    """MKAngel Android application — gesture-driven panel system."""

    title = "MKAngel"

    def build(self):
        Window.softinput_mode = "below_target"

        self.angel = None
        self.session = None
        self._ready = False
        self._current_panel = GestureAction.CHAT

        # ── Root: FloatLayout to stack panels + gesture overlay ───────
        root = FloatLayout()

        # ── Build panels ─────────────────────────────────────────────
        go_back = lambda *_: self._show_panel(GestureAction.CHAT)

        self._chat_panel = _ChatPanel(on_send=self._on_send)
        self._chat_panel.size_hint = (1, 1)

        self._voice_panel = _VoicePanel(go_back=go_back)
        self._voice_panel.size_hint = (1, 1)

        self._host_panel = _HostPanel(go_back=go_back)
        self._host_panel.size_hint = (1, 1)

        self._skills_overlay = _SkillsOverlay(go_back=go_back)
        self._skills_overlay.size_hint = (1, 1)

        self._aspect_switcher = _AspectSwitcher(go_back=go_back)
        self._aspect_switcher.size_hint = (1, 1)

        self._vestment_switcher = _VestmentSwitcher(go_back=go_back)
        self._vestment_switcher.size_hint = (1, 1)

        self._document_panel = _DocumentPanel(go_back=go_back)
        self._document_panel.size_hint = (1, 1)

        self._wings_panel = _WingsPanel(go_back=go_back)
        self._wings_panel.size_hint = (1, 1)

        # Panel registry — maps gesture action to panel widget
        self._panels = {
            GestureAction.CHAT:       self._chat_panel,
            GestureAction.VOICE:      self._voice_panel,
            GestureAction.HOSTS:      self._host_panel,
            GestureAction.SKILLS:     self._skills_overlay,
            GestureAction.ASPECTS:    self._aspect_switcher,
            GestureAction.VESTMENTS:  self._vestment_switcher,
            GestureAction.DOCUMENTS:  self._document_panel,
            GestureAction.WINGS:      self._wings_panel,
        }

        # Add panels to root (chat first = bottom layer, visible by default)
        for name, panel in self._panels.items():
            root.add_widget(panel)

        # Chat starts visible
        self._chat_panel.opacity = 1
        self._chat_panel.disabled = False

        # ── Gesture detector (topmost transparent layer) ─────────────
        self.gesture = GestureDetector(callback=self._on_gesture)
        self.gesture.size_hint = (1, 1)
        root.add_widget(self.gesture)

        # ── Convenience aliases ──────────────────────────────────────
        self.chat = self._chat_panel.chat
        self.thinking = self._chat_panel.thinking

        # ── Welcome — Companion aspect greeting ──────────────────────
        accent_hex = _css("accent").lstrip("#")
        self.chat.add(
            f"[color={accent_hex}][b]BE NOT AFRAID[/b][/color]",
            kind="system",
        )
        self.chat.add(
            f"[color={_css('text_dim').lstrip('#')}]Awakening the Angel\u2026[/color]",
            kind="system",
        )

        threading.Thread(target=self._boot, daemon=True).start()
        return root

    # ── gesture callback ─────────────────────────────────────────────
    def _on_gesture(self, action: str):
        self._show_panel(action)

    # ── panel switching with fade animation ──────────────────────────
    def _show_panel(self, name: str):
        if name == self._current_panel:
            return
        if name not in self._panels:
            return

        for pname, panel in self._panels.items():
            if pname == name:
                panel.opacity = 0
                panel.disabled = False
                Animation(opacity=1, d=0.2).start(panel)
            else:
                Animation(opacity=0, d=0.2).start(panel)
                panel.disabled = True

        self._current_panel = name

    # ── background boot ──────────────────────────────────────────────
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
                    f"[color={_css('warning').lstrip('#')}]"
                    f"\u26a0 GLM: {m}[/color]", kind="error",
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
                    f"[color={_css('warning').lstrip('#')}]"
                    f"\u26a0 Session: {m}[/color]", kind="error",
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

        success_hex = _css("success").lstrip("#")
        teal_hex    = _css("teal").lstrip("#")
        dim_hex     = _css("text_dim").lstrip("#")

        status = (
            f"[color={success_hex}][b]\u2726 Angel awakened[/b]  {elapsed:.1f}s[/color]\n\n"
            f"[color={teal_hex}]Domains[/color]   {domains}\n"
            f"[color={teal_hex}]Grammars[/color]  {g}    "
            f"[color={teal_hex}]Rules[/color]  {ru}\n"
            f"[color={teal_hex}]Loops[/color]     {lo}    "
            f"[color={teal_hex}]Params[/color] {pa:,}\n\n"
            f"[color={dim_hex}]What's on your mind?[/color]"
        )
        Clock.schedule_once(
            lambda _: self.chat.add(status, kind="success")
        )

    # ── send handler (main thread) ───────────────────────────────────
    def _on_send(self, text: str):
        # If not on chat panel, switch to it
        if self._current_panel != GestureAction.CHAT:
            self._show_panel(GestureAction.CHAT)

        self.chat.add(text, kind="user")

        if not self._ready:
            dim_hex = _css("text_dim").lstrip("#")
            self.chat.add(
                f"[color={dim_hex}]Still awakening\u2026[/color]", kind="system",
            )
            return

        self.thinking.start()
        threading.Thread(
            target=self._process, args=(text,), daemon=True,
        ).start()

    # ── process input (background thread) ────────────────────────────
    def _process(self, text: str):
        try:
            if self.session:
                resp = self.session.process_input(text)

                if resp == "__EXIT__":
                    self.session.save_session()
                    accent_hex = _css("accent_purple").lstrip("#")
                    def _bye(_):
                        self.thinking.stop()
                        self.chat.add(
                            f"[color={accent_hex}][b]Be not afraid.[/b][/color]",
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
                warn_hex = _css("warning").lstrip("#")
                def _no_sess(_):
                    self.thinking.stop()
                    self.chat.add(
                        f"[color={warn_hex}]Session not available.[/color]",
                        kind="error",
                    )
                Clock.schedule_once(_no_sess)

        except Exception as exc:
            err = str(exc)
            err_hex = _css("error").lstrip("#")
            def _err(_, e=err):
                self.thinking.stop()
                self.chat.add(f"[color={err_hex}]{e}[/color]", kind="error")
            Clock.schedule_once(_err)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

if __name__ == "__main__":
    MKAngelApp().run()
