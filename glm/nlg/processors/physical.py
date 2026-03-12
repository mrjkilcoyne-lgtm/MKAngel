"""Physical domain processor — built-in physics constants and formulas.

No external API — uses NIST CODATA 2018 constants and fundamental formulas.
Instant, offline, zero latency.
"""

from __future__ import annotations

import re
from typing import Dict, Optional, Tuple

from glm.core.mnemo_substrate import MnemoSequence
from . import DomainProcessor

# NIST CODATA 2018 constants + fundamental formulas
_CONSTANTS: Dict[str, Tuple[str, str, str]] = {
    # keyword -> (symbol, value, unit)
    "speed of light": ("c", "299792458", "m/s"),
    "light speed": ("c", "299792458", "m/s"),
    "gravity": ("g", "9.80665", "m/s\u00b2"),
    "gravitational constant": ("G", "6.674e-11", "N\u22c5m\u00b2/kg\u00b2"),
    "planck": ("h", "6.626e-34", "J\u22c5s"),
    "boltzmann": ("k_B", "1.381e-23", "J/K"),
    "avogadro": ("N_A", "6.022e23", "mol\u207b\u00b9"),
    "electron mass": ("m_e", "9.109e-31", "kg"),
    "proton mass": ("m_p", "1.673e-27", "kg"),
    "elementary charge": ("e", "1.602e-19", "C"),
    "charge": ("e", "1.602e-19", "C"),
    "force": ("F", "F = ma", "N"),
    "energy": ("E", "E = mc\u00b2", "J"),
    "momentum": ("p", "p = mv", "kg\u22c5m/s"),
    "acceleration": ("a", "a = F/m", "m/s\u00b2"),
    "velocity": ("v", "v = d/t", "m/s"),
    "wavelength": ("\u03bb", "\u03bb = c/f", "m"),
    "frequency": ("f", "f = c/\u03bb", "Hz"),
    "pressure": ("P", "P = F/A", "Pa"),
    "temperature": ("T", "T", "K"),
    "power": ("P", "P = W/t", "W"),
    "resistance": ("R", "V = IR", "\u03a9"),
    "voltage": ("V", "V = IR", "V"),
    "current": ("I", "I = V/R", "A"),
    "entropy": ("S", "S = k_B ln W", "J/K"),
    "magnetic field": ("B", "B", "T"),
}


def _match_constant(text: str) -> Tuple[Optional[str], Optional[str], Optional[str], Optional[str]]:
    lower = text.lower()
    for keyword, (symbol, value, unit) in _CONSTANTS.items():
        if keyword in lower:
            return symbol, value, unit, keyword
    return None, None, None, None


class PhysicalProcessor(DomainProcessor):
    domain = "physical"

    def process(self, text: str, mnemo_seq: MnemoSequence) -> Dict[str, str]:
        symbol, value, unit, concept = _match_constant(text)
        if symbol:
            return {
                "symbol": symbol,
                "value": value,
                "unit": unit,
                "quantity": concept,
                "concept": concept,
                "result": f"{symbol} = {value} {unit}",
                "description": f"{concept}: {symbol} = {value} {unit}",
                "law": f"definition of {concept}",
                "consequence": f"{symbol} = {value} {unit}",
            }

        return {"description": f"physical analysis: {text[:60]}"}
