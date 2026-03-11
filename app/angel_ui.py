"""
The Angel UI — MKAngel's universal interface.

One input. Any request. The Angel handles the rest.

The Angel is the TARDIS console: simple on the outside,
infinite on the inside. You ask, she flies.

Architecture:
- AngelInput: Universal text/voice input widget
- AngelResponse: Adaptive response display (text, code, cards, media)
- AngelOrb: The floating action indicator (glass orb, pulses when thinking)
- AngelSuggestions: Context-aware quick actions
- AngelScreen: The main Kivy screen combining all elements
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Callable
from enum import Enum, auto

# Import Kivy components
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.floatlayout import FloatLayout
from kivy.uix.label import Label
from kivy.uix.textinput import TextInput
from kivy.uix.scrollview import ScrollView
from kivy.uix.widget import Widget
from kivy.uix.screenmanager import Screen
from kivy.graphics import Color, RoundedRectangle, Line, Ellipse
from kivy.animation import Animation
from kivy.clock import Clock
from kivy.properties import (
    StringProperty, BooleanProperty, NumericProperty,
    ListProperty, ObjectProperty, DictProperty,
)
from kivy.metrics import dp, sp

from app.vestments import get_vestment


class ResponseKind(Enum):
    TEXT = auto()
    CODE = auto()
    CARD = auto()
    LIST = auto()
    ERROR = auto()
    THINKING = auto()
    SYSTEM = auto()


@dataclass
class AngelMessage:
    content: str
    kind: ResponseKind
    sender: str = "angel"  # "user" or "angel" or "system"
    metadata: dict[str, Any] = field(default_factory=dict)
    timestamp: float = field(default_factory=time.time)


class AngelOrb(Widget):
    """The glass orb — visual heartbeat of the Angel.

    Floats above the input. Pulses gently when idle.
    Spins and glows when thinking. Changes color by intent.
    Frosted glass aesthetic — translucent with blur effect.
    """
    # Properties
    is_thinking = BooleanProperty(False)
    orb_color = ListProperty([0.545, 0.361, 0.965, 0.6])  # lavender
    pulse_scale = NumericProperty(1.0)

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.size_hint = (None, None)
        self.size = (dp(48), dp(48))
        self.bind(pos=self._draw, size=self._draw)
        self.bind(is_thinking=self._on_thinking_change)
        self._pulse_anim = None
        Clock.schedule_once(lambda dt: self._draw(), 0)
        self._start_idle_pulse()

    def _draw(self, *args):
        self.canvas.clear()
        with self.canvas:
            # Outer glow
            Color(*self.orb_color[:3], self.orb_color[3] * 0.3)
            s = self.pulse_scale
            glow_size = (self.width * 1.4 * s, self.height * 1.4 * s)
            glow_pos = (
                self.center_x - glow_size[0] / 2,
                self.center_y - glow_size[1] / 2,
            )
            Ellipse(pos=glow_pos, size=glow_size)

            # Glass orb body
            Color(*self.orb_color[:3], self.orb_color[3] * 0.7)
            orb_size = (self.width * s, self.height * s)
            orb_pos = (
                self.center_x - orb_size[0] / 2,
                self.center_y - orb_size[1] / 2,
            )
            Ellipse(pos=orb_pos, size=orb_size)

            # Inner highlight (glass reflection)
            Color(1, 1, 1, 0.25)
            hi_size = (self.width * 0.5 * s, self.height * 0.35 * s)
            hi_pos = (
                self.center_x - hi_size[0] / 2,
                self.center_y + self.height * 0.08,
            )
            Ellipse(pos=hi_pos, size=hi_size)

    def _start_idle_pulse(self):
        if self._pulse_anim:
            self._pulse_anim.cancel(self)
        anim = (
            Animation(pulse_scale=1.08, duration=1.5, t='in_out_sine') +
            Animation(pulse_scale=1.0, duration=1.5, t='in_out_sine')
        )
        anim.repeat = True
        anim.bind(on_progress=lambda *a: self._draw())
        anim.start(self)
        self._pulse_anim = anim

    def _on_thinking_change(self, instance, value):
        if self._pulse_anim:
            self._pulse_anim.cancel(self)
        if value:
            # Fast pulse when thinking
            anim = (
                Animation(pulse_scale=1.15, duration=0.4, t='in_out_sine') +
                Animation(pulse_scale=0.95, duration=0.4, t='in_out_sine')
            )
            anim.repeat = True
            anim.bind(on_progress=lambda *a: self._draw())
            anim.start(self)
            self._pulse_anim = anim
        else:
            self._start_idle_pulse()

    def set_intent_color(self, intent: str):
        """Change orb color based on detected intent."""
        colors = {
            "CHAT": [0.545, 0.361, 0.965, 0.6],     # lavender
            "CODE": [0.024, 0.714, 0.831, 0.6],      # teal
            "SEARCH": [0.925, 0.286, 0.600, 0.6],    # rose
            "CREATE": [0.063, 0.725, 0.506, 0.6],    # green
            "ANALYZE": [0.545, 0.361, 0.965, 0.6],   # lavender
            "TRANSLATE": [0.961, 0.620, 0.043, 0.6],  # amber
            "PREDICT": [0.925, 0.286, 0.600, 0.6],    # rose
            "COMMAND": [0.118, 0.161, 0.231, 0.6],    # slate
        }
        self.orb_color = colors.get(intent, colors["CHAT"])
        self._draw()


class GlassBubble(BoxLayout):
    """A frosted glass message bubble."""

    def __init__(self, message: AngelMessage, vestment: dict, **kwargs):
        super().__init__(**kwargs)
        self.orientation = 'vertical'
        self.padding = [dp(16), dp(12)]
        self.spacing = dp(4)
        self.size_hint_y = None

        is_user = message.sender == "user"

        if is_user:
            bg_token = vestment.get("bubble_user", {})
            bg_rgba = bg_token.get("kivy", [0.545, 0.361, 0.965, 0.08])
        else:
            bg_token = vestment.get("bubble_angel", {})
            bg_rgba = bg_token.get("kivy", [1, 1, 1, 0.85])

        text_token = vestment.get("text", {})
        text_color = text_token.get("kivy", [0.118, 0.161, 0.231, 1])

        # Content label
        content = Label(
            text=message.content,
            color=text_color,
            font_size=sp(15),
            size_hint_y=None,
            halign='left' if not is_user else 'right',
            valign='top',
            markup=True,
            text_size=(None, None),
        )
        content.bind(
            texture_size=lambda inst, ts: setattr(inst, 'height', ts[1]),
            width=lambda inst, w: setattr(inst, 'text_size', (w, None)),
        )
        self.add_widget(content)
        self.bind(minimum_height=self.setter('height'))

        # Draw glass background
        with self.canvas.before:
            Color(*bg_rgba)
            self._bg_rect = RoundedRectangle(
                pos=self.pos, size=self.size, radius=[dp(16)]
            )
            # Glass border
            border_token = vestment.get("glass_border", {})
            border_rgba = border_token.get("kivy", [1, 1, 1, 0.18]) if isinstance(border_token, dict) else [1, 1, 1, 0.18]
            Color(*border_rgba)
            self._border = Line(
                rounded_rectangle=[*self.pos, *self.size, dp(16)],
                width=dp(0.5),
            )

        self.bind(pos=self._update_canvas, size=self._update_canvas)

    def _update_canvas(self, *args):
        self._bg_rect.pos = self.pos
        self._bg_rect.size = self.size
        self._border.rounded_rectangle = [*self.pos, *self.size, dp(16)]


class AngelInput(BoxLayout):
    """Universal input — text field with glass styling and send button."""

    text = StringProperty("")
    on_submit = ObjectProperty(None)

    def __init__(self, vestment: dict, **kwargs):
        super().__init__(**kwargs)
        self.orientation = 'horizontal'
        self.size_hint_y = None
        self.height = dp(56)
        self.padding = [dp(8), dp(8)]
        self.spacing = dp(8)
        self.vestment = vestment

        input_token = vestment.get("surface_input", {})
        input_bg = input_token.get("kivy", [0.96, 0.97, 0.98, 0.85])
        text_token = vestment.get("text", {})
        text_color = text_token.get("kivy", [0.118, 0.161, 0.231, 1])
        hint_token = vestment.get("text_dim", {})
        hint_color = hint_token.get("kivy", [0.58, 0.64, 0.72, 1])

        self._input = TextInput(
            hint_text="Ask me anything...",
            multiline=False,
            size_hint_x=1,
            font_size=sp(16),
            foreground_color=text_color,
            hint_text_color=hint_color,
            background_color=[0, 0, 0, 0],
            cursor_color=text_color,
            padding=[dp(16), dp(12)],
        )
        self._input.bind(on_text_validate=self._on_enter)
        self.add_widget(self._input)

        # Draw glass background
        with self.canvas.before:
            Color(*input_bg)
            self._bg = RoundedRectangle(
                pos=self.pos, size=self.size, radius=[dp(28)]
            )
        self.bind(pos=self._update_bg, size=self._update_bg)

    def _update_bg(self, *args):
        self._bg.pos = self.pos
        self._bg.size = self.size

    def _on_enter(self, instance):
        text = instance.text.strip()
        if text and self.on_submit:
            self.on_submit(text)
            instance.text = ""

    @property
    def input_text(self):
        return self._input.text


class AngelSuggestions(BoxLayout):
    """Quick-action suggestion chips — glass pills."""

    def __init__(self, vestment: dict, on_tap: Callable | None = None, **kwargs):
        super().__init__(**kwargs)
        self.orientation = 'horizontal'
        self.size_hint_y = None
        self.height = dp(44)
        self.spacing = dp(8)
        self.padding = [dp(16), dp(4)]
        self._on_tap = on_tap
        self._vestment = vestment

    def set_suggestions(self, suggestions: list[str]):
        self.clear_widgets()
        accent = self._vestment.get("accent", {}).get("kivy", [0.545, 0.361, 0.965, 1])
        for s in suggestions[:4]:
            chip = Label(
                text=s,
                size_hint=(None, 1),
                width=dp(len(s) * 8 + 24),
                color=accent,
                font_size=sp(13),
                halign='center',
                valign='middle',
            )
            chip.bind(size=chip.setter('text_size'))
            self.add_widget(chip)


class AngelScreen(Screen):
    """The main Angel screen — the TARDIS console.

    Layout:
    - Top: subtle app bar with orb
    - Middle: scrollable message area (glass bubbles)
    - Bottom: suggestion chips + glass input
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.name = "angel"
        self._vestment = get_vestment("angel_glass")
        self._messages: list[AngelMessage] = []
        self._router = None
        self._provider = None
        self._build_ui()

    def _build_ui(self):
        v = self._vestment
        bg_token = v.get("bg", {})
        bg_rgba = bg_token.get("kivy", [0.94, 0.96, 0.97, 1])

        root = FloatLayout()

        # Background
        with root.canvas.before:
            Color(*bg_rgba)
            self._bg_rect = RoundedRectangle(pos=root.pos, size=root.size)
        root.bind(
            pos=lambda i, p: setattr(self._bg_rect, 'pos', p),
            size=lambda i, s: setattr(self._bg_rect, 'size', s),
        )

        # Main column
        main = BoxLayout(
            orientation='vertical',
            size_hint=(1, 1),
            padding=[dp(0), dp(0)],
        )

        # Message scroll area
        self._scroll = ScrollView(
            size_hint=(1, 1),
            do_scroll_x=False,
            bar_width=dp(2),
        )
        self._message_box = BoxLayout(
            orientation='vertical',
            size_hint_y=None,
            spacing=dp(8),
            padding=[dp(16), dp(16)],
        )
        self._message_box.bind(minimum_height=self._message_box.setter('height'))
        self._scroll.add_widget(self._message_box)
        main.add_widget(self._scroll)

        # Suggestions
        self._suggestions = AngelSuggestions(
            vestment=v,
            on_tap=self._on_suggestion,
        )
        self._suggestions.set_suggestions([
            "What can you do?",
            "Write code",
            "Translate",
            "Predict",
        ])
        main.add_widget(self._suggestions)

        # Input area
        self._angel_input = AngelInput(vestment=v)
        self._angel_input.on_submit = self._on_user_input
        main.add_widget(self._angel_input)

        root.add_widget(main)

        # Floating orb
        self._orb = AngelOrb()
        self._orb.pos_hint = {'center_x': 0.5}
        self._orb.y = dp(70)
        root.add_widget(self._orb)

        self.add_widget(root)

        # Welcome message
        self._add_message(AngelMessage(
            content="Hello. I'm the Angel. Ask me anything.",
            kind=ResponseKind.TEXT,
            sender="angel",
        ))

    def _on_user_input(self, text: str):
        self._add_message(AngelMessage(
            content=text,
            kind=ResponseKind.TEXT,
            sender="user",
        ))
        self._orb.is_thinking = True
        # Process asynchronously
        Clock.schedule_once(lambda dt: self._process(text), 0.1)

    def _process(self, text: str):
        """Route through the GLM and respond."""
        try:
            # Import router lazily
            if self._router is None:
                try:
                    from glm.router import Router
                    self._router = Router()
                except ImportError:
                    self._router = None

            if self._router and self._provider:
                provider, enriched, route = self._router.route(
                    text, {"default": self._provider}
                )
                self._orb.set_intent_color(route.intent.name)
                response = provider.generate(enriched)
            elif self._provider:
                response = self._provider.generate(text)
            else:
                # Fallback: use local GLM
                from app.providers import LocalProvider
                local = LocalProvider()
                response = local.generate(text)

            self._add_message(AngelMessage(
                content=response,
                kind=ResponseKind.TEXT,
                sender="angel",
            ))
        except Exception as exc:
            self._add_message(AngelMessage(
                content=f"Error: {exc}",
                kind=ResponseKind.ERROR,
                sender="system",
            ))
        finally:
            self._orb.is_thinking = False

    def _on_suggestion(self, text: str):
        self._on_user_input(text)

    def _add_message(self, msg: AngelMessage):
        self._messages.append(msg)
        bubble = GlassBubble(msg, self._vestment)
        self._message_box.add_widget(bubble)
        # Scroll to bottom
        Clock.schedule_once(lambda dt: setattr(self._scroll, 'scroll_y', 0), 0.1)

    def set_provider(self, provider):
        self._provider = provider

    def set_router(self, router):
        self._router = router
