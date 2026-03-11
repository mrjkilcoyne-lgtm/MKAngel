"""
Tongue — the Angel's voice in every language.

The GLM thinks in grammar — derivations, productions, strange loops.
The Tongue translates that internal representation into natural language
that humans understand, in whatever tongue they speak.

The wiring is "back to front": internally the Angel derives forward
through grammar rules, but the Tongue presents results backward —
from conclusion to explanation — because that's how humans prefer
to receive information.

Architecture:
  - Tongue: the main translator from grammar-space to human-space
  - OutputFormatter: formats responses for different contexts
  - LanguageDetector: identifies the user's language from their input
  - ResponseTemplate: templates for different response types
"""

from __future__ import annotations

import re
import time
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any


# ---------------------------------------------------------------------------
# Language detection (lightweight, no external deps)
# ---------------------------------------------------------------------------

class Language(Enum):
    """Supported output languages."""
    ENGLISH = "en"
    SPANISH = "es"
    FRENCH = "fr"
    GERMAN = "de"
    ITALIAN = "it"
    PORTUGUESE = "pt"
    DUTCH = "nl"
    RUSSIAN = "ru"
    JAPANESE = "ja"
    CHINESE = "zh"
    KOREAN = "ko"
    ARABIC = "ar"
    HINDI = "hi"
    UNKNOWN = "unknown"


# Language detection markers — common words/patterns per language
_LANG_MARKERS: dict[str, list[str]] = {
    "es": ["hola", "como", "estas", "gracias", "por favor", "que", "donde", "cuando",
           "tengo", "quiero", "puedo", "necesito", "ayuda", "bueno", "muy"],
    "fr": ["bonjour", "merci", "comment", "oui", "non", "je suis", "s'il vous",
           "pourquoi", "quand", "bien", "tres", "avec", "dans", "faire"],
    "de": ["hallo", "danke", "bitte", "wie", "warum", "ich bin", "kannst",
           "nicht", "auch", "aber", "oder", "und", "haben", "machen"],
    "it": ["ciao", "grazie", "come", "perche", "quando", "sono", "voglio",
           "posso", "molto", "bene", "anche", "questo", "quella"],
    "pt": ["ola", "obrigado", "como", "porque", "quando", "tenho", "quero",
           "posso", "muito", "bem", "tambem", "este", "essa"],
    "nl": ["hallo", "dank", "hoe", "waarom", "wanneer", "ik ben", "kan",
           "niet", "ook", "maar", "en", "hebben", "goed"],
    "ja": ["desu", "masu", "nani", "doko", "itsu", "watashi", "anata",
           "konnichiwa", "arigato", "hai", "iie", "sumimasen"],
    "zh": ["ni hao", "xie xie", "shi", "bu shi", "wo", "ni", "ta",
           "zhe", "na", "hen", "dui"],
    "ko": ["annyeong", "kamsahamnida", "ne", "aniyo", "mwo", "eodi",
           "eonje", "wae", "joayo"],
    "ar": ["marhaba", "shukran", "naam", "la", "kayf", "mata", "ayna",
           "limatha", "ana", "anta"],
    "hi": ["namaste", "dhanyavaad", "kya", "kaise", "kahan", "kab",
           "kyun", "main", "aap", "hai", "haan", "nahi"],
}


def detect_language(text: str) -> Language:
    """Detect language from text using keyword frequency.

    Simple but effective for determining output language.
    Falls back to ENGLISH if uncertain.
    """
    lower = text.lower()
    words = set(re.findall(r'\b\w+\b', lower))

    scores: dict[str, int] = {}
    for lang_code, markers in _LANG_MARKERS.items():
        score = sum(1 for m in markers if m in lower or m in words)
        if score > 0:
            scores[lang_code] = score

    if not scores:
        return Language.ENGLISH

    best = max(scores, key=scores.get)
    # Only switch if there's reasonable confidence (2+ markers)
    if scores[best] >= 2:
        for lang in Language:
            if lang.value == best:
                return lang

    return Language.ENGLISH


# ---------------------------------------------------------------------------
# Output formatting
# ---------------------------------------------------------------------------

class OutputFormat(Enum):
    """How to format the response."""
    PLAIN = auto()        # Plain text
    MARKDOWN = auto()     # Markdown formatted
    CHAT_BUBBLE = auto()  # For Kivy chat UI
    TERMINAL = auto()     # ANSI terminal colours
    HTML = auto()         # HTML for WebView
    JSON = auto()         # Structured JSON


@dataclass
class FormattedResponse:
    """A response formatted for display."""
    content: str
    format: OutputFormat
    language: Language
    metadata: dict[str, Any] = field(default_factory=dict)


class OutputFormatter:
    """Formats responses for different display contexts."""

    @staticmethod
    def format(
        content: str,
        fmt: OutputFormat = OutputFormat.PLAIN,
        language: Language = Language.ENGLISH,
    ) -> FormattedResponse:
        """Format content for the specified output context."""
        if fmt == OutputFormat.MARKDOWN:
            formatted = OutputFormatter._to_markdown(content)
        elif fmt == OutputFormat.TERMINAL:
            formatted = OutputFormatter._to_terminal(content)
        elif fmt == OutputFormat.HTML:
            formatted = OutputFormatter._to_html(content)
        elif fmt == OutputFormat.JSON:
            formatted = OutputFormatter._to_json(content)
        elif fmt == OutputFormat.CHAT_BUBBLE:
            formatted = content  # Raw text for Kivy Label
        else:
            formatted = content

        return FormattedResponse(
            content=formatted,
            format=fmt,
            language=language,
        )

    @staticmethod
    def _to_markdown(content: str) -> str:
        """Wrap content in markdown formatting."""
        # Detect code blocks and wrap them
        lines = content.split("\n")
        result = []
        in_code = False
        for line in lines:
            if line.strip().startswith(("def ", "class ", "import ", "from ")):
                if not in_code:
                    result.append("```python")
                    in_code = True
            elif in_code and not line.strip() and not line.startswith(" "):
                result.append("```")
                in_code = False
            result.append(line)
        if in_code:
            result.append("```")
        return "\n".join(result)

    @staticmethod
    def _to_terminal(content: str) -> str:
        """Add ANSI colour codes for terminal display."""
        # Highlight key patterns
        content = re.sub(
            r'(\[.*?\])',
            r'\033[36m\1\033[0m',  # cyan for brackets
            content,
        )
        content = re.sub(
            r'(confidence:?\s*\d+\.?\d*)',
            r'\033[33m\1\033[0m',  # yellow for confidence
            content,
        )
        content = re.sub(
            r'(error|Error|ERROR)',
            r'\033[31m\1\033[0m',  # red for errors
            content,
        )
        return content

    @staticmethod
    def _to_html(content: str) -> str:
        """Convert to basic HTML."""
        import html
        escaped = html.escape(content)
        paragraphs = escaped.split("\n\n")
        html_parts = [f"<p>{p.replace(chr(10), '<br>')}</p>" for p in paragraphs]
        return "\n".join(html_parts)

    @staticmethod
    def _to_json(content: str) -> str:
        """Wrap content in a JSON structure."""
        import json
        return json.dumps({
            "content": content,
            "timestamp": time.time(),
            "type": "response",
        }, indent=2)


# ---------------------------------------------------------------------------
# Response templates
# ---------------------------------------------------------------------------

_TEMPLATES: dict[str, dict[str, str]] = {
    "greeting": {
        "en": "Hello. I'm the Angel. How can I help?",
        "es": "Hola. Soy el Angel. Como puedo ayudarte?",
        "fr": "Bonjour. Je suis l'Ange. Comment puis-je aider?",
        "de": "Hallo. Ich bin der Engel. Wie kann ich helfen?",
        "it": "Ciao. Sono l'Angelo. Come posso aiutarti?",
        "pt": "Ola. Sou o Anjo. Como posso ajudar?",
        "ja": "Konnichiwa. Watashi wa Angel desu. Nani ka otetsudai shimashou ka?",
        "zh": "Ni hao. Wo shi Angel. Wo neng bang ni shenme?",
    },
    "thinking": {
        "en": "Let me think about that...",
        "es": "Dejame pensar en eso...",
        "fr": "Laissez-moi y reflechir...",
        "de": "Lass mich daruber nachdenken...",
        "it": "Lasciami pensare...",
        "pt": "Deixe-me pensar nisso...",
        "ja": "Kangaete mimasu...",
        "zh": "Rang wo xiang xiang...",
    },
    "error": {
        "en": "Something went wrong: {error}",
        "es": "Algo salio mal: {error}",
        "fr": "Quelque chose s'est mal passe: {error}",
        "de": "Etwas ist schiefgelaufen: {error}",
        "it": "Qualcosa e andato storto: {error}",
        "pt": "Algo deu errado: {error}",
        "ja": "Mondai ga hassei shimashita: {error}",
        "zh": "Chu xian le wenti: {error}",
    },
    "no_result": {
        "en": "I couldn't find an answer for that. Could you rephrase?",
        "es": "No pude encontrar una respuesta. Podrias reformularlo?",
        "fr": "Je n'ai pas trouve de reponse. Pourriez-vous reformuler?",
        "de": "Ich konnte keine Antwort finden. Konntest du es umformulieren?",
        "it": "Non ho trovato una risposta. Potresti riformulare?",
        "pt": "Nao encontrei resposta. Poderia reformular?",
        "ja": "Kotae ga mitsukarimasen deshita. Iikae shite itadakemasu ka?",
        "zh": "Wo zhao bu dao da an. Ni neng chong xin biao shu ma?",
    },
    "capabilities": {
        "en": "I can help with: chat, code, search, translation, predictions, math, documents, and more. What would you like to do?",
        "es": "Puedo ayudar con: chat, codigo, busqueda, traduccion, predicciones, matematicas, documentos y mas. Que te gustaria hacer?",
        "fr": "Je peux aider avec: chat, code, recherche, traduction, predictions, mathematiques, documents et plus. Que souhaitez-vous faire?",
        "de": "Ich kann helfen mit: Chat, Code, Suche, Ubersetzung, Vorhersagen, Mathematik, Dokumente und mehr. Was mochtest du tun?",
    },
}


def get_template(key: str, language: Language, **kwargs: str) -> str:
    """Get a localized template string."""
    templates = _TEMPLATES.get(key, {})
    lang_code = language.value if language != Language.UNKNOWN else "en"
    template = templates.get(lang_code, templates.get("en", ""))
    if kwargs and template:
        try:
            template = template.format(**kwargs)
        except KeyError:
            pass
    return template


# ---------------------------------------------------------------------------
# The Tongue — main translator
# ---------------------------------------------------------------------------

class Tongue:
    """The Angel's voice — translates grammar-space to human-space.

    The internal logic runs derivations forward through grammar rules.
    The Tongue presents results in reverse order: conclusion first,
    reasoning chain after. This is the "back to front" wiring —
    computers derive forward, humans prefer to read backward
    from the answer.

    The Tongue also handles language: it detects the user's language
    from their input and responds in the same language.
    """

    def __init__(self, default_language: Language = Language.ENGLISH):
        self._language = default_language
        self._format = OutputFormat.PLAIN
        self._formatter = OutputFormatter()

    @property
    def language(self) -> Language:
        return self._language

    @language.setter
    def language(self, lang: Language) -> None:
        self._language = lang

    @property
    def output_format(self) -> OutputFormat:
        return self._format

    @output_format.setter
    def output_format(self, fmt: OutputFormat) -> None:
        self._format = fmt

    def detect_and_set(self, user_input: str) -> Language:
        """Detect language from user input and set it as current."""
        detected = detect_language(user_input)
        if detected != Language.UNKNOWN:
            self._language = detected
        return self._language

    def speak(self, content: str) -> FormattedResponse:
        """Format content for output in current language and format."""
        return self._formatter.format(content, self._format, self._language)

    def translate_grammar_output(
        self,
        grammar_result: dict[str, Any],
    ) -> str:
        """Translate a GLM grammar output to human-readable text.

        The grammar result is typically a prediction or reconstruction
        dict from the Angel. The Tongue inverts the derivation chain:
        presents the conclusion first, then the reasoning.

        This is the "back to front" wiring: the derivation ran
        forward (A -> B -> C), but we present it as:
        "C (because B, which follows from A)"
        """
        # Extract key components
        predictions = grammar_result.get("predictions", [])
        reasoning = grammar_result.get("reasoning", [])
        confidence = grammar_result.get("overall_confidence", 0.0)
        domain = grammar_result.get("domain", "general")
        loops = grammar_result.get("strange_loops", [])

        parts = []

        # Conclusion first (back to front)
        if predictions:
            top = predictions[0]
            predicted = top.get("predicted", "unknown")
            parts.append(f"Prediction: {predicted}")
            parts.append(f"  Domain: {domain}")
            parts.append(f"  Confidence: {confidence:.0%}")

        # Strange loops (the recursive patterns)
        if loops:
            parts.append(f"  Recursive patterns detected: {len(loops)}")

        # Reasoning chain (the forward derivation, presented as explanation)
        if reasoning:
            parts.append("")
            parts.append("Reasoning:")
            for step in reasoning:
                parts.append(f"  - {step}")

        # Cross-domain harmonics
        harmonics = grammar_result.get("cross_domain_harmonics", [])
        if harmonics:
            domains = list({h.get("domain", "?") for h in harmonics})
            parts.append(f"  Cross-domain agreement: {', '.join(domains)}")

        return "\n".join(parts) if parts else "No derivation available."

    def format_tool_result(self, tool_output: str, tool_name: str) -> str:
        """Format a tool result for display."""
        if not tool_output:
            return get_template("no_result", self._language)
        return tool_output

    def greet(self) -> str:
        """Return a greeting in the current language."""
        return get_template("greeting", self._language)

    def thinking(self) -> str:
        """Return a 'thinking' message in the current language."""
        return get_template("thinking", self._language)

    def error(self, error_msg: str) -> str:
        """Return an error message in the current language."""
        return get_template("error", self._language, error=error_msg)

    def describe_capabilities(self) -> str:
        """Describe what the Angel can do, in the current language."""
        return get_template("capabilities", self._language)
