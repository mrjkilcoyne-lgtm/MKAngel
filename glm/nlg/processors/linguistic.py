"""Linguistic domain processor — Free Dictionary API integration.

API: https://api.dictionaryapi.dev/api/v2/entries/en/{word}
Free, no auth required.

Returns definitions, phonetics, parts of speech, and example usage.
"""

from __future__ import annotations

import re
from typing import Dict

from glm.core.mnemo_substrate import MnemoSequence
from . import DomainProcessor
from ._http import fetch_json

_DICT_BASE = "https://api.dictionaryapi.dev/api/v2/entries/en"

_STOPWORDS = frozenset({
    "the", "a", "an", "is", "was", "are", "were", "be", "been", "being",
    "in", "on", "at", "to", "for", "of", "with", "by", "from", "as",
    "this", "that", "it", "its", "and", "or", "but", "not", "no",
    "has", "have", "had", "do", "does", "did", "will", "would",
    "can", "could", "shall", "should", "may", "might",
})


def _extract_word(text: str) -> str:
    """Find the most interesting word for dictionary lookup."""
    words = re.findall(r'[a-zA-Z]+', text.lower())
    content = [w for w in words if w not in _STOPWORDS and len(w) > 2]
    if content:
        return max(content, key=len)
    return words[0] if words else "word"


class LinguisticProcessor(DomainProcessor):
    domain = "linguistic"

    def process(self, text: str, mnemo_seq: MnemoSequence) -> Dict[str, str]:
        word = _extract_word(text)
        url = f"{_DICT_BASE}/{word}"
        data = fetch_json(url)

        if isinstance(data, list) and data:
            entry = data[0]
            meanings = entry.get("meanings", [])
            definition = ""
            part_of_speech = ""

            if meanings:
                m = meanings[0]
                part_of_speech = m.get("partOfSpeech", "")
                defs = m.get("definitions", [])
                if defs:
                    definition = defs[0].get("definition", "")

            phonetic = entry.get("phonetic", "")
            analysis = definition or f"used as a {part_of_speech}" if part_of_speech else ""

            return {
                "word": word,
                "definition": definition,
                "analysis": analysis,
                "role": part_of_speech or "content word",
                "part_of_speech": part_of_speech,
                "phonetic": phonetic,
                "description": (
                    f"{word} ({part_of_speech}): {definition}"
                    if definition else f"analysis of '{word}'"
                ),
                "result": definition or f"linguistic properties of {word}",
            }

        return {
            "word": word,
            "analysis": f"appears in the context of '{text[:40]}'",
            "role": "content word",
            "description": f"analysis of '{word}'",
        }
