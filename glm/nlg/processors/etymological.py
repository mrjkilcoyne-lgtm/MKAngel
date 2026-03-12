"""Etymological domain processor — Datamuse API for word data.

API: https://api.datamuse.com/words?sp={word}&md=d&max=1
Free, no auth required.

Returns definitions and related word data. Combined with built-in
knowledge of common etymological roots.
"""

from __future__ import annotations

import re
from typing import Dict

from glm.core.mnemo_substrate import MnemoSequence
from . import DomainProcessor
from ._http import fetch_json

_DATAMUSE_BASE = "https://api.datamuse.com"

# Common etymological root languages
_ROOT_LANGUAGES: Dict[str, str] = {
    "latin": "Latin",
    "greek": "Greek",
    "french": "French",
    "german": "German",
    "old english": "Old English",
    "anglo": "Anglo-Saxon",
    "norse": "Old Norse",
    "sanskrit": "Sanskrit",
    "arabic": "Arabic",
    "spanish": "Spanish",
    "italian": "Italian",
    "portuguese": "Portuguese",
    "dutch": "Dutch",
    "celtic": "Celtic",
    "hebrew": "Hebrew",
    "persian": "Persian",
}

_STOPWORDS = frozenset({
    "the", "a", "an", "from", "root", "word", "origin", "of", "is", "in",
    "comes", "derives", "means", "meaning", "derived",
})


def _extract_word(text: str) -> str:
    words = re.findall(r'[a-zA-Z]+', text.lower())
    content = [w for w in words if w not in _STOPWORDS and len(w) > 2]
    return content[-1] if content else "word"


def _detect_root_language(text: str) -> str:
    lower = text.lower()
    for keyword, lang in _ROOT_LANGUAGES.items():
        if keyword in lower:
            return lang
    return ""


class EtymologicalProcessor(DomainProcessor):
    domain = "etymological"

    def process(self, text: str, mnemo_seq: MnemoSequence) -> Dict[str, str]:
        word = _extract_word(text)
        root_lang = _detect_root_language(text)

        url = f"{_DATAMUSE_BASE}/words?sp={word}&md=d&max=1"
        data = fetch_json(url)

        slots: Dict[str, str] = {"word": word}
        if root_lang:
            slots["root_language"] = root_lang
            slots["origin"] = root_lang
            slots["language"] = root_lang
            slots["source_language"] = root_lang

        if isinstance(data, list) and data:
            entry = data[0]
            defs = entry.get("defs", [])
            if defs:
                parts = defs[0].split("\t", 1)
                if len(parts) == 2:
                    slots["definition"] = parts[1]
                    slots["meaning"] = parts[1]
                    slots["part_of_speech"] = parts[0]

        # Build derived slots
        slots.setdefault("origin", "unknown origin")
        slots.setdefault("root", word)
        slots.setdefault("ancestor", f"earlier form of '{word}'")
        slots.setdefault("meaning", f"the concept of {word}")
        slots["result"] = slots.get("definition", f"etymology of {word}")
        slots["description"] = (
            f"'{word}' derives from {root_lang}"
            if root_lang else f"etymological analysis of '{word}'"
        )

        return slots
