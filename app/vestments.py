"""
Vestment system — the Angel's visual identities.

Four vestments:
  - Celestial Dark: OLED black, gold, constellation particles
  - Ethereal Light: Warm white, silver, glassmorphism
  - Living Gradient: Aurora animation, breathing backgrounds
  - Minimal Power: Monochrome, typography-driven

Each vestment is a dict of design tokens consumed by both
Kivy widgets (as rgba tuples) and WebView CSS (as hex strings).
"""

from __future__ import annotations
from kivy.utils import get_color_from_hex

_hex = get_color_from_hex


def _token(hex_color: str, alpha: float = 1.0) -> dict:
    """Create a token with both Kivy rgba and CSS hex."""
    rgba = list(_hex(hex_color))
    rgba[3] = alpha
    return {"kivy": rgba, "css": hex_color, "alpha": alpha}


CELESTIAL_DARK = {
    "name": "Celestial Dark",
    "id": "celestial_dark",
    "bg":            _token("#000000"),
    "surface":       _token("#0e0e14"),
    "surface_head":  _token("#0c0c12"),
    "surface_input": _token("#16161e"),
    "accent":        _token("#D4AF37"),       # gold
    "accent_purple": _token("#bb86fc"),
    "teal":          _token("#03dac6"),
    "text":          _token("#f0f0f2"),
    "text_sec":      _token("#9e9ea8"),
    "text_dim":      _token("#555560"),
    "success":       _token("#66bb6a"),
    "warning":       _token("#ffab40"),
    "error":         _token("#ef5350"),
    "separator":     _token("#ffffff", 0.04),
    "bubble_user":   _token("#bb86fc", 0.10),
    "bubble_angel":  _token("#0e0e14"),
    "particle":      _token("#D4AF37", 0.3),
}

ETHEREAL_LIGHT = {
    "name": "Ethereal Light",
    "id": "ethereal_light",
    "bg":            _token("#FAFAFA"),
    "surface":       _token("#FFFFFF", 0.85),
    "surface_head":  _token("#F5F5F5"),
    "surface_input": _token("#EEEEEE"),
    "accent":        _token("#C0C0C0"),
    "accent_purple": _token("#7C4DFF"),
    "teal":          _token("#00897B"),
    "text":          _token("#333333"),
    "text_sec":      _token("#666666"),
    "text_dim":      _token("#999999"),
    "success":       _token("#4CAF50"),
    "warning":       _token("#FF9800"),
    "error":         _token("#F44336"),
    "separator":     _token("#000000", 0.06),
    "bubble_user":   _token("#E8EAF6"),
    "bubble_angel":  _token("#FFFFFF"),
    "particle":      _token("#C0C0C0", 0.2),
}

LIVING_GRADIENT = {
    "name": "Living Gradient",
    "id": "living_gradient",
    "bg":            _token("#0D001A"),
    "surface":       _token("#1A0033", 0.8),
    "surface_head":  _token("#0D001A", 0.9),
    "surface_input": _token("#2D004D"),
    "accent":        _token("#00FFFF"),
    "accent_purple": _token("#FF00FF"),
    "teal":          _token("#00FFAA"),
    "text":          _token("#FFFFFF"),
    "text_sec":      _token("#CCCCFF"),
    "text_dim":      _token("#8888AA"),
    "success":       _token("#00FF88"),
    "warning":       _token("#FFAA00"),
    "error":         _token("#FF4444"),
    "separator":     _token("#FFFFFF", 0.08),
    "bubble_user":   _token("#FF00FF", 0.15),
    "bubble_angel":  _token("#1A0033", 0.8),
    "particle":      _token("#00FFFF", 0.4),
}

MINIMAL_POWER = {
    "name": "Minimal Power",
    "id": "minimal_power",
    "bg":            _token("#FCFCFC"),
    "surface":       _token("#FFFFFF"),
    "surface_head":  _token("#FAFAFA"),
    "surface_input": _token("#F5F5F5"),
    "accent":        _token("#1A237E"),
    "accent_purple": _token("#1A237E"),
    "teal":          _token("#1A237E"),
    "text":          _token("#212121"),
    "text_sec":      _token("#757575"),
    "text_dim":      _token("#BDBDBD"),
    "success":       _token("#1B5E20"),
    "warning":       _token("#E65100"),
    "error":         _token("#B71C1C"),
    "separator":     _token("#000000", 0.08),
    "bubble_user":   _token("#E8EAF6"),
    "bubble_angel":  _token("#FFFFFF"),
    "particle":      _token("#1A237E", 0.1),
}

ANGEL_GLASS = {
    "name": "Angel Glass",
    "id": "angel_glass",
    "bg":            _token("#F0F4F8"),
    "surface":       _token("#FFFFFF", 0.7),
    "surface_head":  _token("#E8EEF4"),
    "surface_input": _token("#F5F8FB", 0.85),
    "accent":        _token("#8B5CF6"),       # soft lavender
    "accent_sec":    _token("#EC4899"),       # rose
    "teal":          _token("#06B6D4"),       # soft mint
    "text":          _token("#1E293B"),       # dark slate
    "text_sec":      _token("#64748B"),       # medium slate
    "text_dim":      _token("#94A3B8"),       # light slate
    "success":       _token("#10B981"),       # soft green
    "warning":       _token("#F59E0B"),       # warm amber
    "error":         _token("#EF4444"),       # soft rose
    "separator":     _token("#000000", 0.06),
    "bubble_user":   _token("#8B5CF6", 0.08),
    "bubble_angel":  _token("#FFFFFF", 0.85),
    "particle":      _token("#8B5CF6", 0.2),
    # Glassmorphism extras
    "glass_blur":    20,
    "glass_border":  _token("#FFFFFF", 0.18),
    "glass_shadow":  _token("#000000", 0.05),
}

ALL_VESTMENTS = {
    "celestial_dark": CELESTIAL_DARK,
    "ethereal_light": ETHEREAL_LIGHT,
    "living_gradient": LIVING_GRADIENT,
    "minimal_power": MINIMAL_POWER,
    "angel_glass": ANGEL_GLASS,
}

DEFAULT_VESTMENT = "angel_glass"


def get_vestment(name: str | None = None) -> dict:
    """Return a vestment token dict by id."""
    return ALL_VESTMENTS.get(name or DEFAULT_VESTMENT, CELESTIAL_DARK)


def vestment_to_css(vestment: dict) -> str:
    """Export vestment tokens as CSS custom properties for WebView."""
    lines = [":root {"]
    for key, val in vestment.items():
        if isinstance(val, dict) and "css" in val:
            lines.append(f"  --{key.replace('_', '-')}: {val['css']};")
            if val.get("alpha", 1.0) < 1.0:
                lines.append(f"  --{key.replace('_', '-')}-alpha: {val['alpha']};")
    lines.append("}")
    return "\n".join(lines)
