"""
Gesture recognition for the Angel's full wingspan.

8 gestures, each activating a different power:
  Tap           → Chat (default)
  Swipe Up      → Voice mode
  Swipe Right   → Hosts (parallel agents)
  Swipe Left    → Wings (extensions)
  Long Press    → Skills overlay
  Swipe Down    → Aspect switcher
  Double Tap    → Vestment switcher
  Pinch         → Document mode
"""

from __future__ import annotations

import time
from kivy.metrics import dp
from kivy.uix.widget import Widget
from kivy.clock import Clock


class GestureAction:
    CHAT = "chat"
    VOICE = "voice"
    HOSTS = "hosts"
    WINGS = "wings"
    SKILLS = "skills"
    ASPECTS = "aspects"
    VESTMENTS = "vestments"
    DOCUMENTS = "documents"


class GestureDetector(Widget):
    """
    Transparent overlay that detects gestures and dispatches actions.

    Place this as the top-level widget in the layout; it passes through
    all touch events it doesn't consume.
    """

    # Thresholds
    SWIPE_MIN_DIST = dp(80)        # min distance for a swipe
    SWIPE_MAX_TIME = 0.5           # max seconds for a swipe
    LONG_PRESS_TIME = 0.6          # seconds to trigger long press
    DOUBLE_TAP_TIME = 0.3          # max gap between taps
    EDGE_ZONE = dp(30)             # edge zone for directional swipes
    PINCH_MIN_DIST_CHANGE = dp(50) # min distance change for pinch

    def __init__(self, callback=None, **kwargs):
        super().__init__(**kwargs)
        self._callback = callback
        self._touch_start = None
        self._touch_start_time = 0
        self._long_press_event = None
        self._last_tap_time = 0
        self._touches = {}  # multi-touch tracking

    def set_callback(self, fn):
        self._callback = fn

    def _dispatch(self, action: str):
        if self._callback:
            self._callback(action)

    def on_touch_down(self, touch):
        if len(self._touches) == 0:
            self._touch_start = (touch.x, touch.y)
            self._touch_start_time = time.time()
            # Schedule long press
            self._long_press_event = Clock.schedule_once(
                self._on_long_press, self.LONG_PRESS_TIME
            )
        self._touches[touch.uid] = (touch.x, touch.y)
        return False  # pass through

    def on_touch_move(self, touch):
        if touch.uid in self._touches:
            self._touches[touch.uid] = (touch.x, touch.y)
            # Cancel long press on significant movement
            if self._touch_start:
                dx = abs(touch.x - self._touch_start[0])
                dy = abs(touch.y - self._touch_start[1])
                if dx > dp(15) or dy > dp(15):
                    self._cancel_long_press()
        return False

    def on_touch_up(self, touch):
        self._cancel_long_press()

        if touch.uid not in self._touches:
            return False

        del self._touches[touch.uid]

        if self._touch_start is None:
            return False

        elapsed = time.time() - self._touch_start_time
        dx = touch.x - self._touch_start[0]
        dy = touch.y - self._touch_start[1]
        dist = (dx**2 + dy**2) ** 0.5
        sx, sy = self._touch_start

        # ── Swipe detection ──
        if dist > self.SWIPE_MIN_DIST and elapsed < self.SWIPE_MAX_TIME:
            if abs(dy) > abs(dx):
                # Vertical swipe
                if dy > 0:
                    # Swipe down — from top edge → aspects
                    if sy > self.height * 0.8:
                        self._dispatch(GestureAction.ASPECTS)
                        return True
                else:
                    # Swipe up — from bottom edge → voice
                    if sy < self.height * 0.25:
                        self._dispatch(GestureAction.VOICE)
                        return True
            else:
                # Horizontal swipe
                if dx > 0:
                    # Swipe right — from left edge → hosts
                    if sx < self.EDGE_ZONE:
                        self._dispatch(GestureAction.HOSTS)
                        return True
                else:
                    # Swipe left — from right edge → wings
                    if sx > self.width - self.EDGE_ZONE:
                        self._dispatch(GestureAction.WINGS)
                        return True

            self._touch_start = None
            return False

        # ── Double tap detection ──
        now = time.time()
        if dist < dp(15) and elapsed < 0.3:
            if (now - self._last_tap_time) < self.DOUBLE_TAP_TIME:
                self._dispatch(GestureAction.VESTMENTS)
                self._last_tap_time = 0
                self._touch_start = None
                return True
            self._last_tap_time = now

        self._touch_start = None
        return False

    def _on_long_press(self, _dt):
        self._dispatch(GestureAction.SKILLS)
        self._long_press_event = None

    def _cancel_long_press(self):
        if self._long_press_event:
            self._long_press_event.cancel()
            self._long_press_event = None
