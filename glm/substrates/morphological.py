"""
Morphological Substrate — the grammar of word formation.

Morphemes are the smallest meaning-bearing units: roots, prefixes, suffixes,
infixes, circumfixes.  The rules that combine them — affixation, compounding,
derivation, inflection, reduplication — form a grammar that is strikingly
parallel to molecular bonding (a root has "valence slots" that affixes fill)
and to syntax (morpheme order is constituency in miniature).

Strange loops: derivational morphology creates words that describe
morphological processes themselves ("un-do-able", "re-de-fine").  A
language's morphology is described *in* the language, using the very
morphological system being described.

Fugues: agglutinative languages (Turkish, Finnish, Swahili) build words by
stacking morphemes in strict order — a fugue of prefixes and suffixes,
each voice entering at its appointed position.

Isomorphisms:
- Morpheme ↔ functional group in chemistry (both modify a core structure).
- Inflectional paradigm ↔ conjugation table ↔ type class in Haskell.
- Compounding ↔ molecular concatenation ↔ string concatenation in code.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Dict, List, Optional, Set, Tuple

from ..core.substrate import Sequence, Substrate, Symbol, TransformationRule


# ---------------------------------------------------------------------------
# Morpheme types
# ---------------------------------------------------------------------------

class MorphemeType(Enum):
    ROOT = auto()
    PREFIX = auto()
    SUFFIX = auto()
    INFIX = auto()
    CIRCUMFIX = auto()


class GrammaticalFunction(Enum):
    """Major grammatical functions a morpheme can encode."""
    LEXICAL = auto()         # core meaning (root)
    DERIVATIONAL = auto()    # changes part of speech or meaning
    INFLECTIONAL = auto()    # marks tense, number, case, agreement
    FUNCTIONAL = auto()      # grammatical glue (determiners, complementisers)


# ---------------------------------------------------------------------------
# Morpheme — Symbol subclass
# ---------------------------------------------------------------------------

@dataclass
class Morpheme(Symbol):
    """A morpheme — the minimal meaning-bearing unit.

    Attributes beyond ``Symbol``:
    - morpheme_type: ROOT, PREFIX, SUFFIX, INFIX, CIRCUMFIX
    - meaning: semantic gloss
    - grammatical_function: LEXICAL, DERIVATIONAL, INFLECTIONAL, FUNCTIONAL
    - allomorphs: variant surface forms conditioned by context
    - selectional_restrictions: constraints on what this morpheme can
      attach to (e.g., prefix 'un-' selects adjectives/verbs)
    """

    morpheme_type: MorphemeType = MorphemeType.ROOT
    meaning: str = ""
    grammatical_function: GrammaticalFunction = GrammaticalFunction.LEXICAL
    allomorphs: List[str] = field(default_factory=list)
    selectional_restrictions: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.domain == "generic":
            self.domain = "morphological"
        # Roots have high valence (can take many affixes)
        # Affixes have lower valence
        if self.valence == 1:
            if self.morpheme_type == MorphemeType.ROOT:
                self.valence = 4
            elif self.morpheme_type == MorphemeType.CIRCUMFIX:
                self.valence = 2
            else:
                self.valence = 1

    def __hash__(self) -> int:
        return hash((self.form, self.domain, self.morpheme_type.name))

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Morpheme):
            return Symbol.__eq__(self, other)
        return (
            self.form == other.form
            and self.domain == other.domain
            and self.morpheme_type == other.morpheme_type
        )

    @property
    def is_root(self) -> bool:
        return self.morpheme_type == MorphemeType.ROOT

    @property
    def is_affix(self) -> bool:
        return self.morpheme_type != MorphemeType.ROOT

    @property
    def is_bound(self) -> bool:
        """Bound morphemes cannot stand alone (most affixes)."""
        return self.is_affix

    def can_attach_to(self, other: "Morpheme") -> bool:
        """Check selectional restrictions for attachment.

        A prefix wants to attach to a root (or another prefix's result);
        a suffix wants to follow a root.  Selectional restrictions can
        filter by part_of_speech, morpheme_type, etc.
        """
        if not self.selectional_restrictions:
            return True
        for key, value in self.selectional_restrictions.items():
            if key == "morpheme_type":
                if isinstance(value, (list, set)):
                    if other.morpheme_type not in value:
                        return False
                elif other.morpheme_type != value:
                    return False
            elif key == "part_of_speech":
                if isinstance(value, (list, set)):
                    if other.features.get("part_of_speech") not in value:
                        return False
                elif other.features.get("part_of_speech") != value:
                    return False
            else:
                if other.features.get(key) != value:
                    return False
        return True


# ---------------------------------------------------------------------------
# MorphologicalSubstrate
# ---------------------------------------------------------------------------

class MorphologicalSubstrate(Substrate):
    """Substrate for word formation — morphemes and the rules that combine them.

    Understands affixation, compounding, derivation, inflection, and
    reduplication.  Can decompose words into morphemes and predict/generate
    new word forms.
    """

    def __init__(self, name: str = "morphological") -> None:
        super().__init__(name, domain="morphological")
        self._morpheme_map: Dict[str, Morpheme] = {}
        self._decomposition_cache: Dict[str, List[List[Morpheme]]] = {}
        self._paradigms: Dict[str, Dict[str, List[Morpheme]]] = {}
        self._build_default_inventory()

    def _build_default_inventory(self) -> None:
        """Populate with English-centric morphemes for demonstration."""
        defaults: List[Tuple[str, MorphemeType, str, GrammaticalFunction, Dict[str, Any]]] = [
            # Prefixes
            ("un-", MorphemeType.PREFIX, "not / reverse", GrammaticalFunction.DERIVATIONAL,
             {"part_of_speech": {"adjective", "verb"}}),
            ("re-", MorphemeType.PREFIX, "again", GrammaticalFunction.DERIVATIONAL,
             {"part_of_speech": {"verb"}}),
            ("pre-", MorphemeType.PREFIX, "before", GrammaticalFunction.DERIVATIONAL,
             {"part_of_speech": {"verb", "noun"}}),
            ("dis-", MorphemeType.PREFIX, "not / opposite", GrammaticalFunction.DERIVATIONAL,
             {"part_of_speech": {"verb", "adjective"}}),
            ("mis-", MorphemeType.PREFIX, "wrongly", GrammaticalFunction.DERIVATIONAL,
             {"part_of_speech": {"verb"}}),
            ("over-", MorphemeType.PREFIX, "excessively", GrammaticalFunction.DERIVATIONAL,
             {"part_of_speech": {"verb"}}),

            # Suffixes — derivational
            ("-ness", MorphemeType.SUFFIX, "state of being", GrammaticalFunction.DERIVATIONAL,
             {"part_of_speech": {"adjective"}}),
            ("-able", MorphemeType.SUFFIX, "capable of", GrammaticalFunction.DERIVATIONAL,
             {"part_of_speech": {"verb"}}),
            ("-er", MorphemeType.SUFFIX, "agent / one who", GrammaticalFunction.DERIVATIONAL,
             {"part_of_speech": {"verb"}}),
            ("-tion", MorphemeType.SUFFIX, "act or result", GrammaticalFunction.DERIVATIONAL,
             {"part_of_speech": {"verb"}}),
            ("-ly", MorphemeType.SUFFIX, "in a manner", GrammaticalFunction.DERIVATIONAL,
             {"part_of_speech": {"adjective"}}),
            ("-ful", MorphemeType.SUFFIX, "full of", GrammaticalFunction.DERIVATIONAL,
             {"part_of_speech": {"noun"}}),
            ("-less", MorphemeType.SUFFIX, "without", GrammaticalFunction.DERIVATIONAL,
             {"part_of_speech": {"noun"}}),
            ("-ment", MorphemeType.SUFFIX, "result of action", GrammaticalFunction.DERIVATIONAL,
             {"part_of_speech": {"verb"}}),
            ("-ize", MorphemeType.SUFFIX, "to make", GrammaticalFunction.DERIVATIONAL,
             {"part_of_speech": {"adjective", "noun"}}),

            # Suffixes — inflectional
            ("-s", MorphemeType.SUFFIX, "plural / 3sg present", GrammaticalFunction.INFLECTIONAL, {}),
            ("-ed", MorphemeType.SUFFIX, "past tense", GrammaticalFunction.INFLECTIONAL,
             {"part_of_speech": {"verb"}}),
            ("-ing", MorphemeType.SUFFIX, "progressive", GrammaticalFunction.INFLECTIONAL,
             {"part_of_speech": {"verb"}}),
            ("-est", MorphemeType.SUFFIX, "superlative", GrammaticalFunction.INFLECTIONAL,
             {"part_of_speech": {"adjective"}}),
        ]

        for form, mtype, meaning, gfunc, restrictions in defaults:
            m = Morpheme(
                form=form,
                features={"category": mtype.name.lower()},
                domain="morphological",
                morpheme_type=mtype,
                meaning=meaning,
                grammatical_function=gfunc,
                selectional_restrictions=restrictions,
            )
            self._morpheme_map[form] = m
            self.add_symbol(m)

    def add_morpheme(self, morpheme: Morpheme) -> None:
        """Add a morpheme to the inventory."""
        morpheme.domain = self.domain
        self._morpheme_map[morpheme.form] = morpheme
        self.add_symbol(morpheme)

    def add_root(
        self,
        form: str,
        meaning: str = "",
        part_of_speech: str = "noun",
        **extra_features: Any,
    ) -> Morpheme:
        """Convenience: add a root morpheme."""
        features = {"part_of_speech": part_of_speech, "category": "root"}
        features.update(extra_features)
        m = Morpheme(
            form=form,
            features=features,
            domain="morphological",
            morpheme_type=MorphemeType.ROOT,
            meaning=meaning,
            grammatical_function=GrammaticalFunction.LEXICAL,
        )
        self.add_morpheme(m)
        return m

    # -- encode / decode ----------------------------------------------------

    def encode(self, raw_input: str) -> Sequence:
        """Decompose a word into a Sequence of Morphemes.

        Uses a greedy longest-match strategy, trying prefixes first,
        then root, then suffixes.
        """
        morphemes = self.decompose(raw_input)
        if morphemes:
            return Sequence(morphemes)
        # Fallback: treat the whole word as an unknown root
        unknown = Morpheme(
            form=raw_input,
            features={"part_of_speech": "unknown", "category": "root"},
            domain="morphological",
            morpheme_type=MorphemeType.ROOT,
            meaning="?",
        )
        return Sequence([unknown])

    def decode(self, sequence: Sequence) -> str:
        """Concatenate morphemes back into a surface form.

        Strips affix delimiters (hyphens) at junctions.
        """
        parts: List[str] = []
        for sym in sequence:
            form = sym.form
            # Strip leading hyphen from suffixes when joining
            if parts and form.startswith("-"):
                form = form[1:]
            # Strip trailing hyphen from prefixes when joining
            if form.endswith("-") and sym is not sequence[-1]:
                form = form[:-1]
            parts.append(form)
        return "".join(parts)

    # -- decomposition ------------------------------------------------------

    def decompose(self, word: str) -> List[Morpheme]:
        """Decompose a word into morphemes using recursive matching.

        Tries to find the longest known prefix, then recursively
        decomposes the remainder.  Also tries suffix stripping.
        Returns the decomposition with the most known morphemes.
        """
        cached = self._decomposition_cache.get(word)
        if cached:
            return cached[0] if cached else []

        candidates = self._decompose_recursive(word)
        if candidates:
            # Prefer decomposition with most known (non-unknown) morphemes
            best = max(
                candidates,
                key=lambda c: (
                    sum(1 for m in c if m.meaning != "?"),
                    -len(c),
                ),
            )
            self._decomposition_cache[word] = [best]
            return best
        return []

    def _decompose_recursive(
        self, word: str, depth: int = 0, max_depth: int = 10
    ) -> List[List[Morpheme]]:
        if depth > max_depth or not word:
            return []

        results: List[List[Morpheme]] = []

        # Try known roots first (exact match)
        for form, morpheme in self._morpheme_map.items():
            if morpheme.is_root and form == word:
                results.append([morpheme])

        # Try prefixes
        prefix_forms = sorted(
            (f for f, m in self._morpheme_map.items()
             if m.morpheme_type == MorphemeType.PREFIX),
            key=len, reverse=True,
        )
        for pf in prefix_forms:
            clean = pf.rstrip("-")
            if word.startswith(clean) and len(clean) < len(word):
                rest = word[len(clean):]
                sub_results = self._decompose_recursive(rest, depth + 1)
                for sub in sub_results:
                    results.append([self._morpheme_map[pf]] + sub)

        # Try suffixes
        suffix_forms = sorted(
            (f for f, m in self._morpheme_map.items()
             if m.morpheme_type == MorphemeType.SUFFIX),
            key=len, reverse=True,
        )
        for sf in suffix_forms:
            clean = sf.lstrip("-")
            if word.endswith(clean) and len(clean) < len(word):
                rest = word[: len(word) - len(clean)]
                sub_results = self._decompose_recursive(rest, depth + 1)
                for sub in sub_results:
                    results.append(sub + [self._morpheme_map[sf]])

        # If nothing else matched, treat the whole thing as unknown root
        if not results and word:
            unknown = Morpheme(
                form=word,
                features={"part_of_speech": "unknown", "category": "root"},
                domain="morphological",
                morpheme_type=MorphemeType.ROOT,
                meaning="?",
            )
            results.append([unknown])

        return results

    # -- word generation ----------------------------------------------------

    def generate_word(self, root: Morpheme, affixes: List[Morpheme]) -> Sequence:
        """Build a word by attaching affixes to a root.

        Prefixes are placed before the root, suffixes after.
        Selectional restrictions are checked.
        """
        prefixes = [a for a in affixes if a.morpheme_type == MorphemeType.PREFIX]
        suffixes = [a for a in affixes if a.morpheme_type == MorphemeType.SUFFIX]

        morphemes: List[Morpheme] = []

        for p in prefixes:
            if not p.can_attach_to(root):
                continue
            morphemes.append(p)

        morphemes.append(root)

        for s in suffixes:
            if not s.can_attach_to(root):
                continue
            morphemes.append(s)

        return Sequence(morphemes)  # type: ignore[arg-type]

    def inflect(
        self,
        root: Morpheme,
        features: Dict[str, str],
    ) -> Sequence:
        """Generate an inflected form of a root based on feature specifications.

        E.g., inflect(walk, {tense: 'past'}) -> walked
              inflect(walk, {tense: 'progressive'}) -> walking
        """
        affixes: List[Morpheme] = []
        feature_to_suffix = {
            "past": "-ed",
            "progressive": "-ing",
            "plural": "-s",
            "3sg": "-s",
            "superlative": "-est",
        }

        for feat_name, feat_value in features.items():
            suffix_form = feature_to_suffix.get(feat_value)
            if suffix_form and suffix_form in self._morpheme_map:
                affixes.append(self._morpheme_map[suffix_form])

        return self.generate_word(root, affixes)

    # -- compounding --------------------------------------------------------

    def compound(self, *roots: Morpheme) -> Sequence:
        """Create a compound word from multiple roots.

        Compounding is the morphological analog of molecular bonding:
        two independent units fuse into a new unit with emergent meaning.
        """
        return Sequence(list(roots))  # type: ignore[arg-type]

    # -- reduplication ------------------------------------------------------

    def reduplicate(
        self,
        sequence: Sequence,
        mode: str = "full",
    ) -> Sequence:
        """Apply reduplication — copying part or all of a form.

        Modes:
        - 'full': duplicate the entire sequence (Indonesian-style plurality)
        - 'partial_onset': copy the first consonant + vowel
        - 'partial_rhyme': copy the nucleus + coda of the last syllable
        """
        if mode == "full":
            return sequence + sequence
        elif mode == "partial_onset":
            # Copy first two symbols (approximate CV)
            prefix = sequence[:2] if len(sequence) >= 2 else sequence[:1]
            return prefix + sequence
        elif mode == "partial_rhyme":
            suffix = sequence[-2:] if len(sequence) >= 2 else sequence[-1:]
            return sequence + suffix
        else:
            return sequence

    # -- paradigm management ------------------------------------------------

    def add_paradigm(
        self,
        name: str,
        forms: Dict[str, List[Morpheme]],
    ) -> None:
        """Register an inflectional paradigm.

        *forms* maps feature-value labels (e.g. '1sg.present') to the
        list of morphemes that realise that cell.
        """
        self._paradigms[name] = forms

    def get_paradigm_form(
        self,
        paradigm_name: str,
        cell: str,
    ) -> Optional[Sequence]:
        """Look up a specific cell of a paradigm."""
        paradigm = self._paradigms.get(paradigm_name)
        if paradigm and cell in paradigm:
            return Sequence(paradigm[cell])  # type: ignore[arg-type]
        return None

    # -- morphological complexity -------------------------------------------

    def complexity(self, sequence: Sequence) -> Dict[str, int]:
        """Compute morphological complexity metrics for a word.

        Returns counts of roots, derivational affixes, inflectional affixes,
        and total morphemes.
        """
        roots = sum(
            1 for s in sequence
            if isinstance(s, Morpheme) and s.is_root
        )
        derivational = sum(
            1 for s in sequence
            if isinstance(s, Morpheme)
            and s.grammatical_function == GrammaticalFunction.DERIVATIONAL
        )
        inflectional = sum(
            1 for s in sequence
            if isinstance(s, Morpheme)
            and s.grammatical_function == GrammaticalFunction.INFLECTIONAL
        )
        return {
            "roots": roots,
            "derivational": derivational,
            "inflectional": inflectional,
            "total": len(sequence),
        }
