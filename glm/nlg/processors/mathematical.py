"""Mathematical domain processor — Newton API integration.

API: https://newton.vercel.app/api/v2/{operation}/{expression}
Free, no auth required.

Operations: simplify, factor, derive, integrate, zeroes, tangent, area,
            cos, sin, tan, arccos, arcsin, arctan, abs, log
"""

from __future__ import annotations

import re
import urllib.parse
from typing import Dict

from glm.core.mnemo_substrate import MnemoSequence
from . import DomainProcessor
from ._http import fetch_json

_NEWTON_BASE = "https://newton.vercel.app/api/v2"

_OPERATION_MAP = {
    "simplify": "simplify",
    "solve": "zeroes",
    "factor": "factor",
    "derive": "derive",
    "differentiate": "derive",
    "integrate": "integrate",
    "integral": "integrate",
    "tangent": "tangent",
    "area": "area",
    "cosine": "cos",
    "sine": "sin",
}


def _extract_expression(text: str) -> str:
    """Pull a math expression out of natural language."""
    # Look for "x + 2 = 5" style patterns
    m = re.search(r'([0-9x\s\+\-\*\/\^\(\)\=\.]+)', text)
    if m:
        expr = m.group(1).strip()
        if len(expr) > 1:
            return expr
    # Fallback: tokens with math chars
    words = text.split()
    math_tokens = [w for w in words if any(c in w for c in "0123456789x+-*/^()=")]
    return " ".join(math_tokens) if math_tokens else text


def _detect_operation(text: str) -> str:
    lower = text.lower()
    for keyword, op in _OPERATION_MAP.items():
        if keyword in lower:
            return op
    return "simplify"


class MathProcessor(DomainProcessor):
    domain = "mathematical"

    def process(self, text: str, mnemo_seq: MnemoSequence) -> Dict[str, str]:
        expr = _extract_expression(text)
        operation = _detect_operation(text)
        encoded = urllib.parse.quote(expr, safe="")
        url = f"{_NEWTON_BASE}/{operation}/{encoded}"

        data = fetch_json(url)
        if data and "result" in data:
            return {
                "result": str(data["result"]),
                "operation": data.get("operation", operation),
                "expression": data.get("expression", expr),
                "operand": data.get("expression", expr),
                "method": operation,
                "description": f"{operation} of {expr} yields {data['result']}",
            }

        return {
            "expression": expr,
            "operand": expr,
            "operation": operation,
            "method": operation,
        }
