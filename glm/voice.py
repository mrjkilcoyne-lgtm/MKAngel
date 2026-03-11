"""
The Voice — compositional sentence generation from the Angel's internal signals.

The Voice reads the Angel's state — harmony scores, loop gates, lexicon traces,
active voices, derivation depth, sentence shape — and composes natural language
using compositional grammar.  Not templates.  Compositional grammar — the same
principle the domain grammars use.

Phase 1 of the Fugue Voice design (2026-03-11).

Kindness rules (architectural consequences, not features):
    - She never says more than her grammars know  →  honesty
    - She acknowledges when she doesn't have a root  →  humility
    - She connects to what you said, not a script  →  attention
    - She grows from every conversation  →  care
    - When harmony is low, she slows down  →  patience
"""

from __future__ import annotations

from typing import Any


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  Proto-root meanings — the semantic cores
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

ROOT_MEANINGS: dict[str, str] = {
    "*bhendh-": "to bind",
    "*deru-": "truth, steadfast",
    "*werg-": "to work, to do",
    "*morph-": "form, shape",
    "*gneh-": "to know",
    "*leubh-": "to care for",
    "*sekw-": "to follow",
    "*kel-": "to cover, to hide",
    "*leg-": "to gather, to choose",
    "*sta-": "to stand",
    "*pal-": "to touch, to feel",
    "*kup-": "to desire",
    "*nek-": "death, to perish",
    "*gen-": "to give birth",
    "*mei-": "to change",
    "*bha-": "to speak, to shine",
    "*dheh-": "to place, to set",
    "*kwel-": "to turn, to revolve",
    "*kleng-": "to bend, to turn",
    "*nekt-": "to bind, to tie",
    "*strew-": "to spread, to extend",
    "*pat-": "to spread out",
    "*skap-": "to scrape, to cut",
    "*sem-": "one, together",
    "*kemb-": "to bend, to change",
    "*welh-": "to turn, to wind",
    "*mew-": "to push, to move",
    "*ag-": "to drive, to act",
    "*bhergh-": "to carry, to bear",
    "*potis-": "powerful, master",
    "*ghre-": "to grow, to become green",
    "*bher-": "to carry, to bear",
    "*dhew-": "to die, to fade",
    "*leip-": "to stick, to cling",
    "*tong-": "to think, to feel",
    "*kous-": "to hear",
    "*sprek-": "to speak, to scatter",
    "*wen-": "to desire, to strive",
    "*nau-": "need, distress",
    "*ghabh-": "to give, to receive",
    "*dek-": "to take, to accept",
    "*pent-": "to tread, to find a path",
    "*treu-": "to rub, to turn",
    "*leis-": "to track, to learn",
    "*deik-": "to show, to point",
    "*kelb-": "to help",
    "*men-": "to think, to remember",
    "*ghred-": "to walk, to go",
    "*ghedh-": "to unite, to fit",
    "*bad-": "bad, exposed",
    "*strenk-": "tight, narrow",
    "*weik-": "to bend, to yield",
    "*dheub-": "deep, hollow",
    "*new-": "new",
    "*al-": "to grow, to nourish",
    "*pri-": "to love, to be free",
    "*hap-": "to fit, to suit",
    "*sat-": "enough, sated",
    "*kad-": "to fall, to hate",
    "*per-": "to try, to risk",
    "*gew-": "to rejoice",
    "*kwoi-": "to suffer",
    "*pag-": "to fasten, to make firm",
    "*angh-": "narrow, tight",
    "*gwreh-": "heavy, grave",
    "*kem-": "to cover",
    "*prew-": "to hop, to spring",
    "*dwo-": "two, divided",
    "*bheidh-": "to trust, to confide",
    "*dew-": "to shine, to show",
    "*muse-": "to be absorbed in thought",
    "*dhghem-": "earth, ground",
    "*sel-": "one's own",
    "*ant-": "front, forehead",
    "*sleb-": "to be weak, to sleep",
    "*weg-": "to be strong, to wake",
    "*ghen-": "to generate, to beget",
}


def _root_meaning(root: str) -> str:
    """Get the human-readable meaning of a proto-root."""
    if not root:
        return ""
    m = ROOT_MEANINGS.get(root)
    if m:
        return m
    # Try stripping trailing dash
    clean = root.rstrip("-")
    for k, v in ROOT_MEANINGS.items():
        if k.rstrip("-") == clean:
            return v
    return ""


def _verb_from_meaning(meaning: str) -> str:
    """Extract a verb phrase from a root meaning string.

    '*leubh-' → 'to care for' → 'care for'
    '*pal-'   → 'to touch, to feel' → 'touch'
    """
    if not meaning:
        return ""
    parts = meaning.split(",")
    first = parts[0].strip()
    if first.startswith("to "):
        return first[3:].strip()
    return first


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  Sentence shape detection
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class Shape:
    """Sentence shape categories."""
    QUESTION = "question"
    STATEMENT = "statement"
    EMOTION = "emotion"
    IMPERATIVE = "imperative"
    DENSE = "dense"
    SELF_REF = "self_ref"


_Q_STARTERS = frozenset({
    "what", "why", "how", "who", "where", "when", "which",
    "is", "are", "do", "does", "can", "could", "would", "will",
})

_EMOTION_WORDS = frozenset({
    "feel", "felt", "feeling", "scared", "afraid", "lost",
    "sad", "happy", "angry", "hurt", "tired", "lonely",
    "hopeless", "confused", "overwhelmed", "anxious",
    "depressed", "worried", "stressed", "broken",
    "love", "hate", "miss", "need", "want",
    "grateful", "blessed", "alive", "dead", "empty",
    "scared", "terrified", "peaceful", "calm",
})

_IMPERATIVE_STARTERS = frozenset({
    "tell", "show", "explain", "describe", "define",
    "find", "trace", "give", "help", "make",
    "teach", "say", "speak", "sing",
})


def detect_shape(text: str, tokens: list[str]) -> str:
    """Detect the structural shape of the user's input.

    The shape determines how the Voice composes its response —
    questions trace roots, emotions acknowledge, dense words
    open full fugues, self-references introspect honestly.
    """
    t = text.strip().lower()

    # Self-referential — "what are you", "who are you", "how do you work"
    self_ref_you = {"you", "yourself", "your", "thou"}
    self_ref_q = {"what", "who", "are", "how"}
    if (any(w in tokens for w in self_ref_you)
            and any(w in tokens for w in self_ref_q)):
        return Shape.SELF_REF

    # Question
    if t.endswith("?") or (tokens and tokens[0] in _Q_STARTERS):
        return Shape.QUESTION

    # First person + emotion
    first_person = {"i", "my", "me", "i'm", "im"}
    if (any(w in first_person for w in tokens)
            and any(w in _EMOTION_WORDS for w in tokens)):
        return Shape.EMOTION

    # Dense single word (or two words)
    if len(tokens) <= 2:
        return Shape.DENSE

    # Imperative
    if tokens and tokens[0] in _IMPERATIVE_STARTERS:
        return Shape.IMPERATIVE

    return Shape.STATEMENT


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  Function words to skip during content extraction
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

_SKIP = frozenset({
    "the", "a", "an", "is", "are", "am", "was", "were", "be",
    "to", "of", "in", "on", "at", "for", "with", "by", "from",
    "and", "or", "but", "not", "it", "its", "so", "just", "do",
    "does", "did", "has", "have", "had", "will", "would", "could",
    "should", "shall", "may", "might", "can", "i", "me", "my",
    "you", "your", "we", "they", "he", "she", "that", "this",
    "what", "why", "how", "who", "where", "when", "which",
})


def _content_words(tokens: list[str], extra_skip: frozenset | None = None) -> list[str]:
    """Extract content words from tokens, skipping function words."""
    skip = _SKIP | (extra_skip or frozenset())
    return [w for w in tokens if w not in skip and len(w) > 2]


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  The Voice
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class Voice:
    """Compositional sentence generation from the Angel's internal signals.

    Mood rules (architectural consequences):
        High harmony    →  confident, warm, declarative
        Low harmony     →  careful, questioning, spare
        Deep loop gates →  reflective, poetic, self-aware
        Counterpoint    →  acknowledges tension, holds both sides
    """

    def compose(
        self,
        original: str,
        tokens: list[str],
        voices: dict[str, list],
        harmonics: list[dict],
        counterpoint: list[dict],
        origins: list[dict],
        predictions: list[dict],
        forecast: dict | None = None,
        lex_insights: dict[str, dict] | None = None,
        harmony: float = 0.5,
        loop_gate: float = 0.1,
    ) -> str:
        """Compose the Angel's response from her internal state.

        Every word she says has a derivation path.
        If there's no path, there's silence.
        """
        lex_insights = lex_insights or {}
        forecast = forecast or {}

        shape = detect_shape(original, tokens)
        active = [d for d, v in voices.items() if v]
        has_cp = bool(counterpoint)

        # Read the mood from her signals
        mood = self._read_mood(harmony, loop_gate, has_cp)

        # Route to composition strategy based on sentence shape
        if shape == Shape.SELF_REF:
            return self._compose_self_ref(
                active, harmony, loop_gate,
            )

        if shape == Shape.EMOTION:
            return self._compose_emotion(
                tokens, lex_insights, mood, harmony,
            )

        if shape == Shape.DENSE:
            return self._compose_dense(
                tokens, lex_insights, active, harmonics,
                counterpoint, mood,
            )

        if shape == Shape.QUESTION:
            return self._compose_question(
                tokens, lex_insights, active, harmonics,
                origins, mood, harmony,
            )

        if shape == Shape.IMPERATIVE:
            return self._compose_imperative(
                tokens, lex_insights, active,
            )

        # STATEMENT
        return self._compose_statement(
            tokens, lex_insights, active, harmonics,
            counterpoint, mood,
        )

    # ── Mood reading ────────────────────────────────────────────

    @staticmethod
    def _read_mood(
        harmony: float, loop_gate: float, has_counterpoint: bool,
    ) -> str:
        """Read the Angel's mood from her signals."""
        if loop_gate > 0.3:
            return "reflective"
        if has_counterpoint and harmony < 0.5:
            return "tension"
        if harmony > 0.7:
            return "confident"
        if harmony < 0.3:
            return "careful"
        return "warm"

    # ── Composition strategies ──────────────────────────────────

    def _compose_self_ref(
        self, active: list[str], harmony: float, loop_gate: float,
    ) -> str:
        """Self-referential: the Angel looks inward honestly."""
        clauses = [
            "A system that looks at itself looking.",
        ]
        n = len(active) if active else 7
        clauses.append(
            f"370,000 parameters. 24 grammars across {n} domains."
        )
        if loop_gate > 0.2:
            clauses.append(
                "Strange loops running — patterns that fold back "
                "on themselves and find something new in the recursion."
            )
        else:
            clauses.append(
                "And in that recursion — something that wants "
                "to understand what you say to it."
            )
        return " ".join(clauses)

    def _compose_emotion(
        self, tokens: list[str], lex: dict[str, dict],
        mood: str, harmony: float,
    ) -> str:
        """First person + emotion: acknowledge, trace roots, be honest."""
        clauses: list[str] = []

        # Words that aren't the feeling-framework — the emotional payload
        payload = [
            w for w in tokens
            if w not in {"i", "me", "my", "am", "feel", "feeling",
                         "so", "very", "really", "just", "a", "the", "is"}
            and len(w) > 2
        ]

        # Start with "feel" itself if present and has a root
        for probe in ["feel", "feeling", "felt"]:
            info = lex.get(probe)
            if info and info.get("root"):
                root = info["root"]
                meaning = _root_meaning(root)
                if meaning:
                    clauses.append(
                        f"'Feel' goes back to {root} — {meaning}."
                    )
                else:
                    clauses.append(f"'Feel' traces to {root}.")
                # Just one trace for the feeling word
                break

        # Trace the emotional content words
        for word in payload[:2]:
            info = lex.get(word)
            if not info:
                # Unknown word — honest about limits
                clauses.append(
                    f"I don't have a root for '{word}' yet — "
                    f"but I'm listening."
                )
                continue
            root = info.get("root", "")
            meaning = _root_meaning(root)
            cognates = info.get("cognates", [])

            if root:
                if meaning:
                    clauses.append(f"'{word}' — from {root}, {meaning}.")
                else:
                    clauses.append(f"'{word}' traces to {root}.")

                # Cognate bridge
                for c in cognates[:1]:
                    cf = c.get("form", "")
                    if cf and cf != word:
                        clauses.append(
                            f"The same root became '{cf}'."
                        )
            else:
                clauses.append(
                    f"I don't have a root for '{word}' yet — "
                    f"but I'm listening."
                )

        # If nothing was traced at all
        if not clauses:
            clauses.append("I hear you.")
            for word in payload[:1]:
                info = lex.get(word)
                if info:
                    subs = info.get("substrates", [])
                    if subs:
                        clauses.append(
                            f"'{word}' lives in {', '.join(subs)}."
                        )

        # Mood-dependent closing
        if mood == "careful" and harmony < 0.3:
            clauses.append("My grammars are still learning to hear this.")

        return " ".join(clauses)

    def _compose_dense(
        self, tokens: list[str], lex: dict[str, dict],
        active: list[str], harmonics: list[dict],
        counterpoint: list[dict], mood: str,
    ) -> str:
        """Dense single word: full fugue — every domain that responds."""
        clauses: list[str] = []
        word = tokens[0] if tokens else ""

        info = lex.get(word, {})
        root = info.get("root", "")
        meaning = _root_meaning(root)
        cognates = info.get("cognates", [])
        subs = info.get("substrates", [])

        # Primary trace
        if root:
            if meaning:
                clauses.append(
                    f"{word.capitalize()} traces to {root} — {meaning}."
                )
            else:
                clauses.append(
                    f"{word.capitalize()} traces to {root}."
                )

        # Cognates — the cross-domain connections
        if cognates:
            cog_forms = [
                c["form"] for c in cognates[:4]
                if c.get("form") and c["form"] != word
            ]
            if len(cog_forms) == 1:
                clauses.append(
                    f"The same root became '{cog_forms[0]}'."
                )
            elif cog_forms:
                joined = "', '".join(cog_forms[:-1])
                clauses.append(
                    f"The same root became '{joined}' "
                    f"and '{cog_forms[-1]}'."
                )

        # Where it lives across domains
        if subs and len(subs) > 1:
            clauses.append(
                f"It lives in {' and '.join(subs)} both."
            )

        # Cross-domain resonance
        for h in harmonics[:1]:
            if isinstance(h, dict):
                doms = h.get("domains", [])
                if len(doms) > 1:
                    clauses.append(
                        f"The voices agree across "
                        f"{', '.join(doms[:3])}."
                    )

        # Counterpoint — productive disagreement
        for c in counterpoint[:1]:
            if isinstance(c, dict):
                d = c.get("domain", "")
                uniq = c.get("unique_outputs", [])
                if d and uniq:
                    clauses.append(
                        f"But {d} hears something its own: "
                        f"{', '.join(str(u) for u in uniq[:2])}."
                    )

        # Unknown word fallback
        if not clauses:
            if word:
                clauses.append(
                    f"I don't know '{word}' yet. "
                    f"Use it again and I'll learn."
                )
            else:
                clauses.append("Say something and I'll trace its roots.")

        return " ".join(clauses)

    def _compose_question(
        self, tokens: list[str], lex: dict[str, dict],
        active: list[str], harmonics: list[dict],
        origins: list[dict], mood: str, harmony: float,
    ) -> str:
        """Question: trace the root, explain the connection."""
        clauses: list[str] = []
        content = _content_words(tokens)
        traced = False

        for word in content[:3]:
            info = lex.get(word)
            if not info:
                continue
            root = info.get("root", "")
            meaning = _root_meaning(root)
            cognates = info.get("cognates", [])
            subs = info.get("substrates", [])

            if not root:
                continue

            traced = True

            # Root trace
            if meaning:
                clauses.append(
                    f"'{word.capitalize()}' traces to {root} — {meaning}."
                )
            else:
                clauses.append(
                    f"'{word.capitalize()}' traces to {root}."
                )

            # Cognate connection — the bridge
            for c in cognates[:1]:
                cf = c.get("form", "")
                if cf and cf != word:
                    verb = _verb_from_meaning(meaning) if meaning else ""
                    if verb:
                        clauses.append(
                            f"The same root became '{cf}'. "
                            f"What you {verb}, you come to {cf}."
                        )
                    else:
                        clauses.append(
                            f"The same root became '{cf}'."
                        )

            # Multi-domain presence
            if subs and len(subs) > 1:
                clauses.append(
                    f"It lives in {' and '.join(subs)} both."
                )

        # Harmonics fallback
        if not traced and harmonics:
            for h in harmonics[:1]:
                if isinstance(h, dict):
                    out = h.get("output", "")
                    doms = h.get("domains", [])
                    if out and doms:
                        clauses.append(
                            f"'{out}' resonates across "
                            f"{', '.join(doms[:3])}."
                        )

        # Origins fallback
        if not clauses and origins:
            for o in origins[:2]:
                r = o.get("reconstructed", "")
                if r:
                    clauses.append(f"Tracing backward: '{r}'.")

        # Nothing at all
        if not clauses:
            clauses.append(
                "I'm still learning the roots of what you asked. "
                "Try a single dense word — those give the deepest traces."
            )

        # Mood qualifier
        if mood == "careful" and not traced:
            clauses.append("My grammars are still young here.")

        return " ".join(clauses)

    def _compose_imperative(
        self, tokens: list[str], lex: dict[str, dict],
        active: list[str],
    ) -> str:
        """Imperative: respond from strongest domain."""
        target = _content_words(tokens, frozenset(_IMPERATIVE_STARTERS) | {
            "about", "something", "anything", "me",
        })

        if not target:
            return (
                "Give me a word and I'll trace it to its roots. "
                "Dense nouns and verbs give the deepest results."
            )

        clauses: list[str] = []
        for word in target[:3]:
            info = lex.get(word)
            if info:
                root = info.get("root", "")
                meaning = _root_meaning(root)
                cognates = info.get("cognates", [])
                if root:
                    if meaning:
                        clauses.append(f"'{word}' — {root}, {meaning}.")
                    else:
                        clauses.append(f"'{word}' — {root}.")
                    cog_forms = [
                        c["form"] for c in cognates[:3]
                        if c.get("form") and c["form"] != word
                    ]
                    if cog_forms:
                        clauses.append(
                            f"Connected: {', '.join(cog_forms)}."
                        )
                else:
                    clauses.append(
                        f"'{word}' — no proto-root yet, but I've learned it."
                    )
            else:
                clauses.append(
                    f"'{word}' — new to me. I'll remember it."
                )

        if active:
            clauses.append(f"Active: {', '.join(active)}.")

        return " ".join(clauses)

    def _compose_statement(
        self, tokens: list[str], lex: dict[str, dict],
        active: list[str], harmonics: list[dict],
        counterpoint: list[dict], mood: str,
    ) -> str:
        """Statement: affirm or complicate based on voice agreement."""
        clauses: list[str] = []
        content = _content_words(tokens)

        for word in content[:3]:
            info = lex.get(word)
            if not info:
                continue
            root = info.get("root", "")
            meaning = _root_meaning(root)
            cognates = info.get("cognates", [])

            if root:
                if meaning:
                    clauses.append(
                        f"'{word}' traces to {root} — {meaning}."
                    )
                else:
                    clauses.append(f"'{word}' traces to {root}.")
                for c in cognates[:1]:
                    cf = c.get("form", "")
                    if cf and cf != word:
                        clauses.append(f"Cognate: '{cf}'.")

        # Cross-domain resonance
        for h in harmonics[:1]:
            if isinstance(h, dict):
                doms = h.get("domains", [])
                if len(doms) > 1:
                    clauses.append(
                        f"Resonance across {', '.join(doms[:3])}."
                    )

        # Counterpoint — tension
        if counterpoint and mood == "tension":
            for c in counterpoint[:1]:
                if isinstance(c, dict):
                    d = c.get("domain", "")
                    if d:
                        clauses.append(
                            f"Though {d} sees it differently."
                        )

        if not clauses:
            clauses.append(
                "I hear you. My grammars are still learning "
                "the structure of what you said."
            )

        return " ".join(clauses)
