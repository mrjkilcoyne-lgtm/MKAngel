"""
MKAngel -- Android (Kivy) entry point.

Clean chat interface for the Grammar Language Model.
Settings accessible via the cog icon. Android back button supported.

Run via buildozer:  buildozer android debug deploy run
"""

from __future__ import annotations

import logging
import os
import re
import sys
import threading
import time

log = logging.getLogger(__name__)

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
            text="[b]~  M K A N G E L  ~[/b]", markup=True,
            font_size=sp(20), color=_c("accent"),
            halign="center", valign="bottom",
        )
        title.bind(size=lambda *_: setattr(title, "text_size", (title.width, None)))
        top.add_widget(title)

        cog = Button(
            text="@", font_size=sp(22),
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
            input_type="text",
            keyboard_suggestions=True,
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
            text=">", font_size=sp(18), bold=True,
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

    # Kivy's disabled property only propagates to children that exist
    # at the time it's set.  Children added later in __init__ miss it,
    # so the ScrollView eats touches even when the panel is invisible.
    # Explicit passthrough fixes this once and for all.
    def on_touch_down(self, touch):
        if self.opacity == 0:
            return False
        return super().on_touch_down(touch)

    def on_touch_move(self, touch):
        if self.opacity == 0:
            return False
        return super().on_touch_move(touch)

    def on_touch_up(self, touch):
        if self.opacity == 0:
            return False
        return super().on_touch_up(touch)

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
            text="< Back", font_size=sp(14),
            size_hint=(None, 1), width=dp(80),
            background_color=[0, 0, 0, 0], color=_c("accent"),
        )
        if on_back:
            back.bind(on_press=lambda *_: on_back())
        hdr.add_widget(back)

        accent_hex = _css("accent").lstrip("#")
        title = Label(
            text=f"[b][color={accent_hex}]~ Settings[/color][/b]",
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

    def update_info(self, angel_info, settings_info, memory_info,
                    conductor_info=None):
        """Refresh settings display with live data."""
        self._content.clear_widgets()

        accent = _css("accent").lstrip("#")
        teal = _css("teal").lstrip("#")
        dim = _css("text_dim").lstrip("#")
        text_hex = _css("text").lstrip("#")
        success = _css("success").lstrip("#")

        # ── GLM card ─────────────────────────────────────────
        domains = angel_info.get("domains_loaded", [])
        g = angel_info.get("total_grammars", 0)
        ru = angel_info.get("total_rules", 0)
        lo = angel_info.get("strange_loops_detected", 0)
        pa = angel_info.get("model_params", 0)

        glm = (
            f"[color={accent}][b]~ Grammar Language Model[/b][/color]\n\n"
            f"[color={teal}]Domains[/color]   [color={text_hex}]{len(domains)}[/color]\n"
            f"[color={teal}]Grammars[/color]  [color={text_hex}]{g}[/color]    "
            f"[color={teal}]Rules[/color]  [color={text_hex}]{ru}[/color]\n"
            f"[color={teal}]Loops[/color]     [color={text_hex}]{lo}[/color]    "
            f"[color={teal}]Params[/color] [color={text_hex}]{pa:,}[/color]\n\n"
            f"[color={dim}]{', '.join(domains) if domains else 'None loaded'}[/color]"
        )
        self._content.add_widget(self._make_card(glm))

        # ── Orchestra card (conductor subsystems) ────────────
        if conductor_info:
            subs = conductor_info.get("subsystems", {})
            active = sum(1 for v in subs.values() if v == "active")
            total = len(subs)
            active_names = [k for k, v in subs.items() if v == "active"]
            inactive_names = [k for k, v in subs.items() if v != "active"]

            orch = (
                f"[color={accent}][b]~ Orchestra[/b][/color]\n\n"
                f"[color={teal}]Voices[/color]   "
                f"[color={success}]{active}[/color]"
                f"[color={text_hex}]/{total}[/color]\n\n"
                f"[color={teal}]Active[/color]\n"
                f"[color={text_hex}]{', '.join(active_names)}[/color]"
            )
            if inactive_names:
                orch += (
                    f"\n\n[color={teal}]Inactive[/color]\n"
                    f"[color={dim}]{', '.join(inactive_names)}[/color]"
                )
            self._content.add_widget(self._make_card(orch))

            # Language
            lang = conductor_info.get("language", "en")
            provider_name = conductor_info.get("provider_name", "local")
            prov_card = (
                f"[color={accent}][b]~ Provider[/b][/color]\n\n"
                f"[color={teal}]Active[/color]     [color={text_hex}]{provider_name}[/color]\n"
                f"[color={teal}]Language[/color]   [color={text_hex}]{lang}[/color]"
            )
            self._content.add_widget(self._make_card(prov_card))
        else:
            # ── Provider card (fallback without conductor) ────
            provider = settings_info.get("provider", "local")
            offline = settings_info.get("offline", True)
            mode = "offline" if offline else "online"
            prov = (
                f"[color={accent}][b]~ Provider[/b][/color]\n\n"
                f"[color={teal}]Active[/color]   [color={text_hex}]{provider}[/color]\n"
                f"[color={teal}]Mode[/color]     [color={text_hex}]{mode}[/color]"
            )
            self._content.add_widget(self._make_card(prov))

        # ── Memory card ──────────────────────────────────────
        ms = memory_info.get("sessions", 0)
        mp = memory_info.get("patterns", 0)
        mpr = memory_info.get("preferences", 0)

        mem = (
            f"[color={accent}][b]~ Memory[/b][/color]\n\n"
            f"[color={teal}]Sessions[/color]  [color={text_hex}]{ms}[/color]    "
            f"[color={teal}]Patterns[/color]  [color={text_hex}]{mp}[/color]    "
            f"[color={teal}]Prefs[/color]  [color={text_hex}]{mpr}[/color]"
        )
        self._content.add_widget(self._make_card(mem))

        # ── Commands card ────────────────────────────────────
        if conductor_info:
            cmds = (
                f"[color={accent}][b]~ Commands[/b][/color]\n\n"
                f"[color={teal}]/health[/color]    [color={dim}]system health check[/color]\n"
                f"[color={teal}]/growth[/color]    [color={dim}]learning summary[/color]\n"
                f"[color={teal}]/diagnose[/color]  [color={dim}]diagnose a symptom[/color]\n"
                f"[color={teal}]/consent[/color]   [color={dim}]privacy consent[/color]\n"
                f"[color={teal}]/privacy[/color]   [color={dim}]privacy notice[/color]\n"
                f"[color={teal}]/export[/color]    [color={dim}]export your data[/color]\n"
                f"[color={teal}]/forget[/color]    [color={dim}]delete your data[/color]\n"
                f"[color={teal}]/language[/color]  [color={dim}]set language[/color]"
            )
            self._content.add_widget(self._make_card(cmds))

        # ── Version ──────────────────────────────────────────
        ver = (
            f"\n[color={dim}]MKAngel v0.3.0[/color]\n"
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
        self._conductor = None
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

    # ── Dream service ───────────────────────────────────────────

    def _on_state_change(self, old_state: str, new_state: str):
        """Called when ChatSession transitions sleep state."""
        log.info("Angel state: %s -> %s", old_state, new_state)
        if new_state == "DROWSY":
            # Transition to SLEEPING and run the dream cycle
            if self.session:
                self.session._transition_state("SLEEPING")
            self._run_dream_fallback()

    def _run_dream_fallback(self):
        """Run dream cycle in a background thread."""

        def _dream():
            try:
                from app.dream_service import run_dream_cycle

                voice = None
                try:
                    from glm.voice import Voice
                    voice = Voice()
                except Exception:
                    pass

                result = run_dream_cycle(
                    angel=self.angel,
                    memory=self._memory_obj,
                    voice=voice,
                    trigger_type="self",
                )
                if result["success"]:
                    n = len(result["artifacts"])
                    log.info("Dream cycle complete: %d artifacts", n)

                    # Notify on main thread
                    dim_hex = _css("text_dim").lstrip("#")
                    accent_hex = _css("accent").lstrip("#")

                    def _notify(_):
                        if n > 0:
                            self.chat.add(
                                f"[color={accent_hex}][b]zzz...[/b][/color] "
                                f"[color={dim_hex}]dreamed {n} "
                                f"artifact{'s' if n != 1 else ''}[/color]",
                                kind="system",
                            )
                    Clock.schedule_once(_notify)
                else:
                    log.warning("Dream cycle failed: %s", result.get("error"))

                # Transition to WAKING
                if self.session:
                    self.session._transition_state("WAKING")

            except Exception as exc:
                log.error("Dream fallback failed: %s", exc)
                if self.session:
                    self.session._transition_state("AWAKE")

        threading.Thread(target=_dream, daemon=True).start()

    def _check_wake_greeting(self):
        """Check for unseen dreams and show wake greeting."""
        if self.session is None:
            return
        try:
            greeting = self.session.check_wake_greeting()
            if greeting:
                accent_hex = _css("accent").lstrip("#")

                def _show(_):
                    self.chat.add(
                        f"[color={accent_hex}]{greeting}[/color]",
                        kind="angel",
                    )
                Clock.schedule_once(_show)
        except Exception as exc:
            log.warning("Wake greeting failed: %s", exc)

    # ── App lifecycle ────────────────────────────────────────────

    def on_resume(self):
        """Called when the app returns to foreground on Android."""
        self._check_wake_greeting()

    def on_stop(self):
        """Called when the app is closing -- graceful shutdown."""
        if self._conductor:
            try:
                self._conductor.shutdown()
            except Exception:
                pass
        elif self.session:
            try:
                self.session.save_session()
            except Exception:
                pass
        if not self._conductor and self._memory_obj:
            try:
                self._memory_obj.close()
            except Exception:
                pass

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

        conductor_info = None
        if self._conductor:
            try:
                conductor_info = self._conductor.get_status()
            except Exception:
                pass

        self._settings_panel.update_info(
            angel_info, settings_info, memory_info, conductor_info
        )

    # ── Background boot ───────────────────────────────────────
    def _boot(self):
        t0 = time.time()
        provider = None

        # ── 1. Try conductor-first boot (owns all subsystems) ──
        try:
            from app.conductor import AngelConductor
            self._conductor = AngelConductor()
            self._conductor.awaken()
            # Use conductor's subsystems (avoid double-init)
            self.angel = self._conductor.angel
            self._settings_obj = self._conductor.settings
            self._memory_obj = self._conductor.memory
            provider = self._conductor.provider
        except Exception as exc:
            self._conductor = None
            log.warning("Conductor boot failed (%s), falling back", exc)

        # ── 2. Fallback: manual boot if conductor failed ──
        if self._conductor is None:
            try:
                from glm.angel import Angel
                self.angel = Angel()
                self.angel.awaken()
            except Exception as exc:
                m = str(exc)
                Clock.schedule_once(
                    lambda _, m=m: self.chat.add(
                        f"[color={_css('warning').lstrip('#')}]"
                        f"GLM: {m}[/color]", kind="error",
                    )
                )

            try:
                from app.settings import Settings
                from app.memory import Memory
                from app.providers import get_provider as _gp
                self._settings_obj = Settings.load()
                self._memory_obj = Memory()
                provider = _gp(self._settings_obj)
            except Exception as exc:
                m = str(exc)
                Clock.schedule_once(
                    lambda _, m=m: self.chat.add(
                        f"[color={_css('warning').lstrip('#')}]"
                        f"Boot: {m}[/color]", kind="error",
                    )
                )

        elapsed = time.time() - t0

        # ── 3. ChatSession (history, /exit, /help, /predict) ──
        try:
            from app.chat import ChatSession
            self.session = ChatSession(
                angel=self.angel,
                memory=self._memory_obj,
                settings=self._settings_obj,
                provider=provider,
            )
            self.session.set_on_state_change(self._on_state_change)
        except Exception as exc:
            m = str(exc)
            Clock.schedule_once(
                lambda _, m=m: self.chat.add(
                    f"[color={_css('warning').lstrip('#')}]"
                    f"Session: {m}[/color]", kind="error",
                )
            )

        self._ready = True

        # ── 4. Status report ──
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
        accent_hex  = _css("accent").lstrip("#")

        # Conductor subsystem summary
        conductor_line = ""
        if self._conductor:
            try:
                cst = self._conductor.get_status()
                subs = cst.get("subsystems", {})
                active = sum(1 for v in subs.values() if v == "active")
                total = len(subs)
                active_names = [k for k, v in subs.items() if v == "active"]
                conductor_line = (
                    f"\n[color={accent_hex}][b]Orchestra[/b][/color]  "
                    f"[color={teal_hex}]{active}/{total}[/color] voices\n"
                    f"[color={dim_hex}]{', '.join(active_names)}[/color]"
                )
            except Exception:
                conductor_line = (
                    f"\n[color={accent_hex}][b]Orchestra[/b][/color]  active"
                )

        status = (
            f"[color={success_hex}][b]Angel awakened[/b]  {elapsed:.1f}s[/color]\n\n"
            f"[color={teal_hex}]Domains[/color]   {domains}\n"
            f"[color={teal_hex}]Grammars[/color]  {g}    "
            f"[color={teal_hex}]Rules[/color]  {ru}\n"
            f"[color={teal_hex}]Loops[/color]     {lo}    "
            f"[color={teal_hex}]Params[/color] {pa:,}"
            f"{conductor_line}\n\n"
            f"[color={dim_hex}]What's on your mind?[/color]"
        )
        def _show_status(_):
            self.chat.add(status, kind="success")
            self._check_wake_greeting()

        Clock.schedule_once(_show_status)

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
            resp = None

            # 1. Try conductor commands first (/ commands)
            if text.startswith("/") and self._conductor:
                cmd_resp = self._conductor.handle_command(text)
                if cmd_resp is not None:
                    clean = re.sub(r"\033\[[0-9;]*m", "", str(cmd_resp))
                    def _show_cmd(_, r=clean):
                        self.thinking.stop()
                        self.chat.add(r, kind="angel")
                    Clock.schedule_once(_show_cmd)
                    return

            # 2. For non-/ text, use conductor's full pipeline
            #    (routing, senses, compliance, tongue formatting)
            if self._conductor and not text.startswith("/"):
                resp = self._conductor.process(text)

            # 3. Fall through to ChatSession (handles /exit, /help,
            #    /predict, and plain chat if no conductor)
            elif self.session:
                import concurrent.futures as _cf
                with _cf.ThreadPoolExecutor(max_workers=1) as pool:
                    fut = pool.submit(self.session.process_input, text)
                    try:
                        resp = fut.result(timeout=8)
                    except _cf.TimeoutError:
                        resp = (
                            "Still thinking... the grammar took "
                            "a moment. Try a shorter phrase, or "
                            "use /predict with a few key words."
                        )
            else:
                warn_hex = _css("warning").lstrip("#")
                def _no_sess(_):
                    self.thinking.stop()
                    self.chat.add(
                        f"[color={warn_hex}]Session not available.[/color]",
                        kind="error",
                    )
                Clock.schedule_once(_no_sess)
                return

            # 4. Handle exit signal
            if resp == "__EXIT__":
                if self._conductor:
                    try:
                        shutdown_msg = self._conductor.shutdown()
                    except Exception:
                        shutdown_msg = None
                if self.session:
                    self.session.save_session()
                accent_hex = _css("accent_purple").lstrip("#")
                dim_hex = _css("text_dim").lstrip("#")
                def _bye(_):
                    self.thinking.stop()
                    if self._conductor:
                        try:
                            sm = shutdown_msg
                            if sm:
                                self.chat.add(
                                    f"[color={dim_hex}]{sm}[/color]",
                                    kind="system",
                                )
                        except Exception:
                            pass
                    self.chat.add(
                        f"[color={accent_hex}][b]Be not afraid.[/b][/color]",
                        kind="system",
                    )
                Clock.schedule_once(_bye)
                return

            # 5. Display response
            if resp:
                clean = re.sub(r"\033\[[0-9;]*m", "", str(resp))
                def _show(_, r=clean):
                    self.thinking.stop()
                    self.chat.add(r, kind="angel")
                Clock.schedule_once(_show)
            else:
                Clock.schedule_once(lambda _: self.thinking.stop())

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
