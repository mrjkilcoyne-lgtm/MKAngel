"""
MKAngel -- Android (Kivy) entry point.

Clean chat interface for the Grammar Language Model.
Settings accessible via the cog icon. Android back button supported.

Run via buildozer:  buildozer android debug deploy run
"""

from __future__ import annotations

import os
import re
import sys
import threading
import time

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  Ensure app source directory is on sys.path (critical for Android)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
_app_dir = os.path.dirname(os.path.abspath(__file__))
if _app_dir not in sys.path:
    sys.path.insert(0, _app_dir)

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
    from app.vestments import get_vestment
except ImportError:
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

V = get_vestment()
Window.clearcolor = V["bg"]["kivy"]


def _c(token: str) -> list:
    """Pull a kivy rgba list from the active vestment."""
    return V[token]["kivy"]


def _css(token: str) -> str:
    """Pull the CSS hex from the active vestment."""
    return V[token]["css"]


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
    """Single message with rounded-rectangle background and subtle glow."""

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

        # Glow only on angel and success bubbles
        self._has_glow = kind in ("angel", "success")
        if kind == "angel":
            glow_c = _c("accent")[:3] + [0.07]
        elif kind == "success":
            glow_c = _c("success")[:3] + [0.07]
        else:
            glow_c = [0, 0, 0, 0]
        self._glow_radius = [r + dp(3) for r in radius]

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
            if self._has_glow:
                Color(*glow_c)
                self._glow = RoundedRectangle(
                    pos=(self.x - dp(2), self.y - dp(2)),
                    size=(self.width + dp(4), self.height + dp(4)),
                    radius=self._glow_radius,
                )
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
        if self._has_glow:
            self._glow.pos = (self.x - dp(2), self.y - dp(2))
            self._glow.size = (self.width + dp(4), self.height + dp(4))
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
#  Header with settings cog
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class _Header(BoxLayout):
    def __init__(self, on_settings=None, **kw):
        kw.setdefault("size_hint_y", None)
        kw.setdefault("height", dp(68))
        kw.setdefault("padding", [dp(16), dp(10), dp(8), dp(4)])
        super().__init__(orientation="vertical", **kw)

        with self.canvas.before:
            Color(*_c("surface_head"))
            self._bg = Rectangle(pos=self.pos, size=self.size)
            Color(*(_c("accent")[:3] + [0.15]))
            self._accent_line = Rectangle(
                pos=(self.x, self.y), size=(self.width, dp(1)),
            )
        self.bind(pos=self._upd, size=self._upd)

        # Top row: spacer + title + cog
        top = BoxLayout(size_hint_y=0.58)
        top.add_widget(Widget(size_hint_x=None, width=dp(40)))

        title = Label(
            text="[b]\u2727  M K A N G E L  \u2727[/b]", markup=True,
            font_size=sp(20), color=_c("accent"),
            halign="center", valign="bottom",
        )
        title.bind(size=lambda *_: setattr(title, "text_size", (title.width, None)))
        top.add_widget(title)

        cog = Button(
            text="\u2699", font_size=sp(22),
            size_hint=(None, 1), width=dp(40),
            background_color=[0, 0, 0, 0], color=_c("text_dim"),
        )
        if on_settings:
            cog.bind(on_press=lambda *_: on_settings())
        top.add_widget(cog)

        self.add_widget(top)

        sub = Label(
            text="Grammar Language Model", font_size=sp(11),
            color=_c("text_dim"), size_hint_y=0.42,
            halign="center", valign="top",
        )
        sub.bind(size=lambda *_: setattr(sub, "text_size", (sub.width, None)))
        self.add_widget(sub)

    def _upd(self, *_):
        self._bg.pos = self.pos
        self._bg.size = self.size
        self._accent_line.pos = (self.x, self.y)
        self._accent_line.size = (self.width, dp(1))


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
            Color(*(_c("accent")[:3] + [0.08]))
            self._top_glow = Rectangle(
                pos=(self.x, self.y + self.height - dp(1)),
                size=(self.width, dp(1)),
            )
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
        self._top_glow.pos = (self.x, self.y + self.height - dp(1))
        self._top_glow.size = (self.width, dp(1))
        self._bg.pos = self.pos
        self._bg.size = self.size


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  Settings panel — live data, no placeholders
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class _SettingsPanel(BoxLayout):
    """Shows GLM status, provider, and memory stats — things that work."""

    def __init__(self, on_back=None, **kw):
        kw.setdefault("orientation", "vertical")
        kw.setdefault("opacity", 0)
        kw.setdefault("padding", [dp(16), dp(12)])
        kw.setdefault("spacing", dp(8))
        super().__init__(**kw)
        self.disabled = True

        with self.canvas.before:
            Color(*_c("bg"))
            self._bg = Rectangle(pos=self.pos, size=self.size)
        self.bind(pos=self._upd_bg, size=self._upd_bg)

        # ── Header row ───────────────────────────────────────
        hdr = BoxLayout(size_hint_y=None, height=dp(56), padding=[0, dp(8)])

        back = Button(
            text="\u2190 Back", font_size=sp(14),
            size_hint=(None, 1), width=dp(80),
            background_color=[0, 0, 0, 0], color=_c("accent"),
        )
        if on_back:
            back.bind(on_press=lambda *_: on_back())
        hdr.add_widget(back)

        accent_hex = _css("accent").lstrip("#")
        title = Label(
            text=f"[b][color={accent_hex}]\u2727 Settings[/color][/b]",
            markup=True, font_size=sp(18),
            halign="center", valign="middle",
        )
        title.bind(size=lambda *_: setattr(title, "text_size", title.size))
        hdr.add_widget(title)

        # Spacer to balance the back button
        hdr.add_widget(Widget(size_hint_x=None, width=dp(80)))

        self.add_widget(hdr)
        self.add_widget(_Sep())

        # ── Scrollable content ───────────────────────────────
        sv = ScrollView(size_hint=(1, 1))
        self._content = BoxLayout(
            orientation="vertical", size_hint_y=None,
            spacing=dp(12), padding=[dp(4), dp(8)],
        )
        self._content.bind(minimum_height=self._content.setter("height"))
        sv.add_widget(self._content)
        self.add_widget(sv)

        # Initial placeholder
        self._placeholder = Label(
            text="Loading\u2026", markup=True, font_size=sp(13),
            color=_c("text_sec"), size_hint_y=None, height=dp(40),
            halign="center", valign="middle",
        )
        self._placeholder.bind(size=lambda *_: setattr(
            self._placeholder, "text_size", self._placeholder.size
        ))
        self._content.add_widget(self._placeholder)

    def update_info(self, angel_info, settings_info, memory_info):
        """Refresh settings display with live data."""
        self._content.clear_widgets()

        accent = _css("accent").lstrip("#")
        teal = _css("teal").lstrip("#")
        dim = _css("text_dim").lstrip("#")
        text_hex = _css("text").lstrip("#")

        # ── GLM card ─────────────────────────────────────────
        domains = angel_info.get("domains_loaded", [])
        g = angel_info.get("total_grammars", 0)
        ru = angel_info.get("total_rules", 0)
        lo = angel_info.get("strange_loops_detected", 0)
        pa = angel_info.get("model_params", 0)

        glm = (
            f"[color={accent}][b]\u2727 Grammar Language Model[/b][/color]\n\n"
            f"[color={teal}]Domains[/color]   [color={text_hex}]{len(domains)}[/color]\n"
            f"[color={teal}]Grammars[/color]  [color={text_hex}]{g}[/color]    "
            f"[color={teal}]Rules[/color]  [color={text_hex}]{ru}[/color]\n"
            f"[color={teal}]Loops[/color]     [color={text_hex}]{lo}[/color]    "
            f"[color={teal}]Params[/color] [color={text_hex}]{pa:,}[/color]\n\n"
            f"[color={dim}]{', '.join(domains) if domains else 'None loaded'}[/color]"
        )
        self._content.add_widget(self._make_card(glm))

        # ── Provider card ────────────────────────────────────
        provider = settings_info.get("provider", "local")
        offline = settings_info.get("offline", True)
        mode = "offline" if offline else "online"

        prov = (
            f"[color={accent}][b]\u2727 Provider[/b][/color]\n\n"
            f"[color={teal}]Active[/color]   [color={text_hex}]{provider}[/color]\n"
            f"[color={teal}]Mode[/color]     [color={text_hex}]{mode}[/color]"
        )
        self._content.add_widget(self._make_card(prov))

        # ── Memory card ──────────────────────────────────────
        ms = memory_info.get("sessions", 0)
        mp = memory_info.get("patterns", 0)
        mpr = memory_info.get("preferences", 0)

        mem = (
            f"[color={accent}][b]\u2727 Memory[/b][/color]\n\n"
            f"[color={teal}]Sessions[/color]  [color={text_hex}]{ms}[/color]    "
            f"[color={teal}]Patterns[/color]  [color={text_hex}]{mp}[/color]    "
            f"[color={teal}]Prefs[/color]  [color={text_hex}]{mpr}[/color]"
        )
        self._content.add_widget(self._make_card(mem))

        # ── Version ──────────────────────────────────────────
        ver = (
            f"\n[color={dim}]MKAngel v0.2.0[/color]\n"
            f"[color={dim}]Celestial Dark vestment[/color]\n"
        )
        ver_lbl = Label(
            text=ver, markup=True, font_size=sp(12),
            color=_c("text_dim"), size_hint_y=None, height=dp(50),
            halign="center", valign="middle",
        )
        ver_lbl.bind(size=lambda *_: setattr(ver_lbl, "text_size", ver_lbl.size))
        self._content.add_widget(ver_lbl)

    def _make_card(self, markup_text):
        """Rounded card with surface background."""
        card = BoxLayout(
            orientation="vertical", size_hint_y=None,
            padding=[dp(16), dp(12)],
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

        lbl = Label(
            text=markup_text, markup=True, font_size=sp(13),
            color=_c("text"), size_hint_y=None,
            halign="left", valign="top",
        )
        lbl.bind(texture_size=lambda w, ts: setattr(w, "height", ts[1] + dp(8)))
        lbl.bind(width=lambda w, *_: setattr(w, "text_size", (w.width, None)))
        lbl.bind(height=lambda w, h: setattr(card, "height", h + dp(24)))
        card.add_widget(lbl)
        return card

    def _upd_bg(self, *_):
        self._bg.pos = self.pos
        self._bg.size = self.size


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  Chat panel — header + scrollable chat + thinking + input
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class _ChatPanel(BoxLayout):
    def __init__(self, on_send=None, on_settings=None, **kw):
        kw.setdefault("orientation", "vertical")
        super().__init__(**kw)

        with self.canvas.before:
            Color(*_c("bg"))
            self._bg = Rectangle(pos=self.pos, size=self.size)
        self.bind(pos=self._upd_bg, size=self._upd_bg)

        self.header = _Header(on_settings=on_settings)
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
#  Application
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class MKAngelApp(App):
    title = "MKAngel"

    def build(self):
        Window.softinput_mode = "below_target"
        Window.bind(on_keyboard=self._on_key)

        self.angel = None
        self.session = None
        self._settings_obj = None
        self._memory_obj = None
        self._ready = False
        self._showing_settings = False

        # ── Two-panel layout: chat (default) + settings (hidden) ──
        root = FloatLayout()

        self._chat_panel = _ChatPanel(
            on_send=self._on_send,
            on_settings=self._show_settings,
        )
        self._chat_panel.size_hint = (1, 1)

        self._settings_panel = _SettingsPanel(on_back=self._show_chat)
        self._settings_panel.size_hint = (1, 1)

        root.add_widget(self._chat_panel)
        root.add_widget(self._settings_panel)

        # Convenience aliases
        self.chat = self._chat_panel.chat
        self.thinking = self._chat_panel.thinking

        # ── Welcome ──────────────────────────────────────────
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

    # ── Android back / ESC ────────────────────────────────────
    def _on_key(self, window, key, *args):
        if key == 27:  # Back on Android, ESC on desktop
            if self._showing_settings:
                self._show_chat()
                return True  # consumed — stay in app
            return False  # let system handle (exit)
        return False

    # ── Panel switching ───────────────────────────────────────
    def _show_settings(self):
        if self._showing_settings:
            return
        self._showing_settings = True
        self._settings_panel.disabled = False
        self._settings_panel.opacity = 1
        self._chat_panel.opacity = 0
        self._chat_panel.disabled = True
        self._refresh_settings()

    def _show_chat(self):
        if not self._showing_settings:
            return
        self._showing_settings = False
        self._chat_panel.disabled = False
        self._chat_panel.opacity = 1
        self._settings_panel.opacity = 0
        self._settings_panel.disabled = True

    def _refresh_settings(self):
        angel_info = {}
        if self.angel:
            try:
                angel_info = self.angel.introspect()
            except Exception:
                pass

        settings_info = {}
        if self._settings_obj:
            settings_info = {
                "provider": self._settings_obj.model_provider,
                "offline": self._settings_obj.offline_mode,
            }

        memory_info = {}
        if self._memory_obj:
            try:
                memory_info = self._memory_obj.stats()
            except Exception:
                pass

        self._settings_panel.update_info(angel_info, settings_info, memory_info)

    # ── Background boot ───────────────────────────────────────
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

            self._settings_obj = Settings.load()
            self._memory_obj = Memory()
            provider = get_provider(self._settings_obj)
            self.session = ChatSession(
                angel=self.angel,
                memory=self._memory_obj,
                settings=self._settings_obj,
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

    # ── Send handler (main thread) ────────────────────────────
    def _on_send(self, text: str):
        if self._showing_settings:
            self._show_chat()

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

    # ── Process input (background thread) ─────────────────────
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
