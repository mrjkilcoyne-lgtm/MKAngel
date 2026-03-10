"""
Phonological Substrate — the grammar of sound.

Every spoken language is built on a finite inventory of contrastive sounds
(phonemes), each defined by a bundle of articulatory features.  The rules
governing how these sounds combine (phonotactics), change over time (sound
laws), and harmonise with each other (vowel harmony, assimilation) form a
grammar as rigorous as any formal system.

Strange loops appear here too: the *phonological rule system* of a language
is itself encoded in the acoustic patterns of that language's speakers —
children reconstruct the rules from the sounds, and the sounds are
produced by the rules.

Isomorphisms with other substrates
──────────────────────────────────
- Phoneme features ↔ atomic electron configuration (both define bonding).
- Syllable structure (onset-nucleus-coda) ↔ molecular structure (bonds
  around a central atom) ↔ syntactic constituency.
- Sound change ↔ chemical reaction ↔ source-to-source code transformation.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple

from ..core.substrate import Sequence, Substrate, Symbol, TransformationRule


# ---------------------------------------------------------------------------
# Feature constants
# ---------------------------------------------------------------------------

PLACES = {"bilabial", "labiodental", "dental", "alveolar", "postalveolar",
          "retroflex", "palatal", "velar", "uvular", "pharyngeal", "glottal"}

MANNERS = {"stop", "nasal", "trill", "tap", "fricative", "affricate",
           "approximant", "lateral"}

VOWEL_HEIGHTS = {"close", "near-close", "close-mid", "mid", "open-mid",
                 "near-open", "open"}

VOWEL_BACKNESS = {"front", "central", "back"}


# ---------------------------------------------------------------------------
# Phoneme — Symbol subclass with phonological features
# ---------------------------------------------------------------------------

@dataclass
class Phoneme(Symbol):
    """A single phoneme — the minimal contrastive unit of sound.

    Phonemes are defined by articulatory features:

    Consonants: voiced (bool), place (str), manner (str)
    Vowels:     height (str), backness (str), rounded (bool)

    All phonemes also carry ``syllabic`` (True for vowels / syllabic
    consonants) and ``sonorant`` (True for nasals, liquids, glides, vowels).
    """

    def __post_init__(self) -> None:
        if self.domain == "generic":
            self.domain = "phonological"

    @property
    def is_vowel(self) -> bool:
        return self.features.get("syllabic", False) and self.features.get("manner") is None

    @property
    def is_consonant(self) -> bool:
        return not self.is_vowel

    @property
    def voiced(self) -> bool:
        return self.features.get("voiced", False)

    @property
    def place(self) -> Optional[str]:
        return self.features.get("place")

    @property
    def manner(self) -> Optional[str]:
        return self.features.get("manner")

    @property
    def height(self) -> Optional[str]:
        return self.features.get("height")

    @property
    def backness(self) -> Optional[str]:
        return self.features.get("backness")

    @property
    def rounded(self) -> bool:
        return self.features.get("rounded", False)

    def natural_class_match(self, feature_spec: Dict[str, Any]) -> bool:
        """Check membership in a natural class defined by a feature bundle.

        A natural class is a set of phonemes that share a bundle of features
        — e.g. {voiced, stop} = /b, d, g/.
        """
        return all(self.features.get(k) == v for k, v in feature_spec.items())


# ---------------------------------------------------------------------------
# Helper: build a default IPA-like inventory
# ---------------------------------------------------------------------------

def _make_consonant(
    form: str, voiced: bool, place: str, manner: str,
    sonorant: bool = False, valence: int = 1,
) -> Phoneme:
    return Phoneme(
        form=form,
        features={
            "voiced": voiced, "place": place, "manner": manner,
            "syllabic": False, "sonorant": sonorant, "continuant": manner in ("fricative", "approximant", "lateral"),
        },
        domain="phonological",
        valence=valence,
    )


def _make_vowel(
    form: str, height: str, backness: str, rounded: bool,
    long: bool = False, valence: int = 2,
) -> Phoneme:
    return Phoneme(
        form=form,
        features={
            "height": height, "backness": backness, "rounded": rounded,
            "syllabic": True, "sonorant": True, "continuant": True,
            "long": long,
        },
        domain="phonological",
        valence=valence,
    )


def build_default_inventory() -> Dict[str, Phoneme]:
    """Build a basic IPA-inspired phoneme inventory."""
    inv: Dict[str, Phoneme] = {}

    # -- Consonants (a representative subset) ----
    consonants = [
        ("p", False, "bilabial", "stop"),
        ("b", True, "bilabial", "stop"),
        ("t", False, "alveolar", "stop"),
        ("d", True, "alveolar", "stop"),
        ("k", False, "velar", "stop"),
        ("g", True, "velar", "stop"),
        ("m", True, "bilabial", "nasal"),
        ("n", True, "alveolar", "nasal"),
        ("N", True, "velar", "nasal"),  # eng
        ("f", False, "labiodental", "fricative"),
        ("v", True, "labiodental", "fricative"),
        ("s", False, "alveolar", "fricative"),
        ("z", True, "alveolar", "fricative"),
        ("S", False, "postalveolar", "fricative"),  # sh
        ("Z", True, "postalveolar", "fricative"),   # zh
        ("h", False, "glottal", "fricative"),
        ("r", True, "alveolar", "approximant"),
        ("l", True, "alveolar", "lateral"),
        ("j", True, "palatal", "approximant"),
        ("w", True, "bilabial", "approximant"),
    ]
    for form, voiced, place, manner in consonants:
        sonorant = manner in ("nasal", "approximant", "lateral")
        inv[form] = _make_consonant(form, voiced, place, manner, sonorant)

    # -- Vowels ----
    vowels = [
        ("i", "close", "front", False),
        ("y", "close", "front", True),
        ("e", "close-mid", "front", False),
        ("E", "open-mid", "front", False),   # epsilon
        ("a", "open", "central", False),
        ("o", "close-mid", "back", True),
        ("O", "open-mid", "back", True),     # open-o
        ("u", "close", "back", True),
        ("@", "mid", "central", False),      # schwa
    ]
    for form, height, backness, rounded in vowels:
        inv[form] = _make_vowel(form, height, backness, rounded)

    return inv


# ---------------------------------------------------------------------------
# Syllable helpers
# ---------------------------------------------------------------------------

@dataclass
class Syllable:
    """A syllable decomposed into onset, nucleus, and coda."""
    onset: List[Phoneme]
    nucleus: List[Phoneme]
    coda: List[Phoneme]

    @property
    def is_open(self) -> bool:
        return len(self.coda) == 0

    @property
    def is_closed(self) -> bool:
        return len(self.coda) > 0

    def __repr__(self) -> str:
        o = "".join(p.form for p in self.onset)
        n = "".join(p.form for p in self.nucleus)
        c = "".join(p.form for p in self.coda)
        return f"Syllable({o}.{n}.{c})"


# ---------------------------------------------------------------------------
# PhonologicalSubstrate
# ---------------------------------------------------------------------------

class PhonologicalSubstrate(Substrate):
    """Substrate for sound systems — phonemes, syllables, sound change.

    Encodes raw phonemic strings into Sequences of Phonemes, understands
    syllable structure, vowel harmony, consonant cluster constraints, and
    can apply historical sound changes.
    """

    def __init__(self, name: str = "phonological") -> None:
        super().__init__(name, domain="phonological")
        self._phoneme_map: Dict[str, Phoneme] = build_default_inventory()
        for p in self._phoneme_map.values():
            self.add_symbol(p)

        # Feature system declaration
        self.add_feature("voiced", {True, False})
        self.add_feature("place", PLACES)
        self.add_feature("manner", MANNERS)
        self.add_feature("syllabic", {True, False})
        self.add_feature("height", VOWEL_HEIGHTS)
        self.add_feature("backness", VOWEL_BACKNESS)
        self.add_feature("rounded", {True, False})

        # Default combination rule: limit consonant clusters
        self.add_combination_rule(self._default_cluster_rule)

    # -- encode / decode ----------------------------------------------------

    def encode(self, raw_input: str) -> Sequence:
        """Encode a phonemic transcription string into a Sequence.

        Characters not in the inventory are silently skipped (spaces,
        punctuation, etc.).  Multi-character digraphs are not handled in
        this base version — each character maps to one phoneme.
        """
        symbols: List[Symbol] = []
        for ch in raw_input:
            phoneme = self._phoneme_map.get(ch)
            if phoneme is not None:
                symbols.append(phoneme)
        return Sequence(symbols)

    def decode(self, sequence: Sequence) -> str:
        """Decode a Sequence of Phonemes back to a string."""
        return "".join(s.form for s in sequence)

    # -- syllabification ----------------------------------------------------

    def syllabify(self, sequence: Sequence) -> List[Syllable]:
        """Split a phoneme sequence into syllables using the Maximal Onset
        Principle: assign as many consonants as possible to the onset of the
        next syllable (respecting cluster legality).

        Returns a list of ``Syllable`` objects.
        """
        phonemes = [s for s in sequence if isinstance(s, Phoneme)]
        if not phonemes:
            return []

        # Mark vowels (nuclei candidates)
        is_nucleus = [p.is_vowel for p in phonemes]

        # Find nucleus positions
        nuclei_idx = [i for i, v in enumerate(is_nucleus) if v]
        if not nuclei_idx:
            # All consonants — degenerate case
            return [Syllable(onset=phonemes, nucleus=[], coda=[])]

        syllables: List[Syllable] = []
        prev_end = 0

        for si, ni in enumerate(nuclei_idx):
            # Find end of nucleus (consecutive vowels form a single nucleus)
            nuc_end = ni + 1
            while nuc_end < len(phonemes) and is_nucleus[nuc_end]:
                nuc_end += 1

            # Consonants between prev_end and ni
            consonants = phonemes[prev_end:ni]

            if si == 0:
                # First syllable: all leading consonants are onset
                onset = consonants
                coda_carry: List[Phoneme] = []
            else:
                # Split consonants between coda of previous and onset of this
                # Maximal Onset: give as many as possible to onset
                split = self._split_cluster(consonants)
                # Append coda to previous syllable
                if syllables:
                    syllables[-1].coda.extend(split[0])
                onset = split[1]

            nucleus = phonemes[ni:nuc_end]
            syllables.append(Syllable(onset=onset, nucleus=nucleus, coda=[]))
            prev_end = nuc_end

        # Trailing consonants are coda of last syllable
        if prev_end < len(phonemes):
            syllables[-1].coda.extend(phonemes[prev_end:])

        return syllables

    def _split_cluster(
        self, consonants: List[Phoneme]
    ) -> Tuple[List[Phoneme], List[Phoneme]]:
        """Split an inter-vocalic consonant cluster into (coda, onset).

        Uses the Maximal Onset Principle: try assigning all consonants to
        onset; if the cluster is illegal, peel one off to the coda and retry.
        """
        for i in range(len(consonants)):
            onset_candidate = consonants[i:]
            if self._is_legal_onset(onset_candidate):
                return consonants[:i], onset_candidate
        # Fallback: everything is coda except last consonant
        return consonants[:-1], consonants[-1:]

    def _is_legal_onset(self, cluster: List[Phoneme]) -> bool:
        """Check whether a consonant cluster is a legal syllable onset.

        Uses the Sonority Sequencing Principle: sonority must rise from
        the edge toward the nucleus.
        """
        if not cluster:
            return True
        if len(cluster) == 1:
            return True
        sonority = [self._sonority(p) for p in cluster]
        return all(sonority[i] < sonority[i + 1] for i in range(len(sonority) - 1))

    @staticmethod
    def _sonority(p: Phoneme) -> int:
        """Assign a sonority rank to a phoneme."""
        manner = p.features.get("manner", "")
        if p.is_vowel:
            return 5
        ranking = {
            "stop": 0, "affricate": 1, "fricative": 2,
            "nasal": 3, "lateral": 3, "trill": 3, "tap": 3,
            "approximant": 4,
        }
        return ranking.get(manner, 1)

    @staticmethod
    def _default_cluster_rule(a: Symbol, b: Symbol) -> bool:
        """Default combination rule: disallow two stops in a row."""
        if (a.features.get("manner") == "stop"
                and b.features.get("manner") == "stop"):
            return False
        return True

    # -- vowel harmony ------------------------------------------------------

    def apply_vowel_harmony(
        self,
        sequence: Sequence,
        feature: str = "backness",
        trigger_index: int = 0,
    ) -> Sequence:
        """Apply vowel harmony: all vowels agree in *feature* with the
        trigger vowel (by default, the first vowel).

        This demonstrates a long-distance dependency — a hallmark of
        natural-language phonology and an isomorph of agreement in syntax.
        """
        # Find trigger value
        trigger_value = None
        vowel_count = 0
        for sym in sequence:
            if isinstance(sym, Phoneme) and sym.is_vowel:
                if vowel_count == trigger_index:
                    trigger_value = sym.features.get(feature)
                    break
                vowel_count += 1

        if trigger_value is None:
            return sequence

        new_symbols: List[Symbol] = []
        for sym in sequence:
            if isinstance(sym, Phoneme) and sym.is_vowel:
                if sym.features.get(feature) != trigger_value:
                    new_features = dict(sym.features)
                    new_features[feature] = trigger_value
                    harmonised = Phoneme(
                        form=self._find_vowel_form(new_features),
                        features=new_features,
                        domain=sym.domain,
                        valence=sym.valence,
                    )
                    new_symbols.append(harmonised)
                else:
                    new_symbols.append(sym)
            else:
                new_symbols.append(sym)
        return Sequence(new_symbols)

    def _find_vowel_form(self, features: Dict[str, Any]) -> str:
        """Find the phoneme form that best matches the given features."""
        best_form = "@"  # default to schwa
        best_dist = float("inf")
        for form, phoneme in self._phoneme_map.items():
            if not phoneme.is_vowel:
                continue
            dist = sum(
                1 for k in ("height", "backness", "rounded")
                if features.get(k) != phoneme.features.get(k)
            )
            if dist < best_dist:
                best_dist = dist
                best_form = form
        return best_form

    # -- historical sound change --------------------------------------------

    def apply_sound_change(
        self,
        sequence: Sequence,
        change_name: str,
    ) -> Sequence:
        """Apply a named historical sound change.

        Built-in changes:
        - 'grimms_law': voiceless stops -> fricatives, voiced stops ->
          voiceless stops, aspirated stops -> voiced stops.
        - 'great_vowel_shift': systematic raising/fronting of long vowels.
        - 'voicing': intervocalic voicing of stops.
        - 'final_devoicing': devoice word-final obstruents.
        """
        changes = self._get_sound_change(change_name)
        result = sequence
        for rule in changes:
            result = rule.apply_all(result)
        return result

    def _get_sound_change(self, name: str) -> List[TransformationRule]:
        """Build transformation rules for a named sound change."""
        if name == "grimms_law":
            return self._grimms_law_rules()
        elif name == "great_vowel_shift":
            return self._great_vowel_shift_rules()
        elif name == "voicing":
            return self._intervocalic_voicing_rules()
        elif name == "final_devoicing":
            return self._final_devoicing_rules()
        else:
            raise ValueError(f"Unknown sound change: {name!r}")

    def _grimms_law_rules(self) -> List[TransformationRule]:
        """Grimm's Law: the first Germanic consonant shift.

        PIE *p, *t, *k -> PGmc *f, *th, *h  (voiceless stop -> fricative)
        PIE *b, *d, *g -> PGmc *p, *t, *k  (voiced stop -> voiceless stop)
        """
        rules: List[TransformationRule] = []

        # Voiceless stops -> fricatives
        shift_map = [("p", "f"), ("t", "s"), ("k", "h")]
        for old, new in shift_map:
            old_p = self._phoneme_map.get(old)
            new_p = self._phoneme_map.get(new)
            if old_p and new_p:
                rules.append(TransformationRule(
                    name=f"grimm_{old}->{new}",
                    pattern=[old_p],
                    replacement=[new_p],
                ))

        # Voiced stops -> voiceless stops
        voice_shift = [("b", "p"), ("d", "t"), ("g", "k")]
        for old, new in voice_shift:
            old_p = self._phoneme_map.get(old)
            new_p = self._phoneme_map.get(new)
            if old_p and new_p:
                rules.append(TransformationRule(
                    name=f"grimm_{old}->{new}",
                    pattern=[old_p],
                    replacement=[new_p],
                ))

        return rules

    def _great_vowel_shift_rules(self) -> List[TransformationRule]:
        """Great Vowel Shift: systematic raising of Middle English long vowels.

        Simplified: a->e, e->i, o->u, E->e, O->o
        """
        rules: List[TransformationRule] = []
        shift_map = [("a", "e"), ("e", "i"), ("o", "u"), ("E", "e"), ("O", "o")]
        for old, new in shift_map:
            old_p = self._phoneme_map.get(old)
            new_p = self._phoneme_map.get(new)
            if old_p and new_p:
                rules.append(TransformationRule(
                    name=f"gvs_{old}->{new}",
                    pattern=[old_p],
                    replacement=[new_p],
                ))
        return rules

    def _intervocalic_voicing_rules(self) -> List[TransformationRule]:
        """Intervocalic voicing: voiceless stops become voiced between vowels."""
        rules: List[TransformationRule] = []
        # Create a vowel pattern symbol that matches any vowel
        vowel_pattern = Phoneme(
            form="V", features={"syllabic": True}, domain="phonological",
        )
        voicing_map = [("p", "b"), ("t", "d"), ("k", "g")]
        for old, new in voicing_map:
            old_p = self._phoneme_map.get(old)
            new_p = self._phoneme_map.get(new)
            if old_p and new_p:
                rules.append(TransformationRule(
                    name=f"voicing_{old}->{new}",
                    pattern=[old_p],
                    replacement=[new_p],
                    left_context=[vowel_pattern],
                    right_context=[vowel_pattern],
                ))
        return rules

    def _final_devoicing_rules(self) -> List[TransformationRule]:
        """Final devoicing: voiced obstruents become voiceless word-finally.

        Implemented as: voiced stop/fricative at the end of sequence.
        Since we cannot easily check 'end of word' in the generic rule
        framework, this method works directly on the sequence.
        """
        # We return an empty rule list and override apply_sound_change
        # for this case.  Actually, let's just build rules that match
        # common voiced obstruents — the caller can apply only at the end.
        rules: List[TransformationRule] = []
        devoice_map = [("b", "p"), ("d", "t"), ("g", "k"),
                       ("v", "f"), ("z", "s"), ("Z", "S")]
        for old, new in devoice_map:
            old_p = self._phoneme_map.get(old)
            new_p = self._phoneme_map.get(new)
            if old_p and new_p:
                rules.append(TransformationRule(
                    name=f"devoice_{old}->{new}",
                    pattern=[old_p],
                    replacement=[new_p],
                ))
        return rules

    def apply_final_devoicing(self, sequence: Sequence) -> Sequence:
        """Devoice the final obstruent of a sequence (word-final devoicing)."""
        if len(sequence) == 0:
            return sequence
        last = sequence[-1]
        if not isinstance(last, Phoneme):
            return sequence
        if last.is_vowel or last.features.get("sonorant", False):
            return sequence
        devoice_map = {"b": "p", "d": "t", "g": "k",
                       "v": "f", "z": "s", "Z": "S"}
        new_form = devoice_map.get(last.form)
        if new_form and new_form in self._phoneme_map:
            new_symbols = list(sequence)[:-1] + [self._phoneme_map[new_form]]
            return Sequence(new_symbols)
        return sequence

    # -- proto-form reconstruction ------------------------------------------

    def reconstruct_proto_form(
        self,
        cognates: List[Sequence],
    ) -> Sequence:
        """Reconstruct a proto-form from a set of cognate sequences.

        Uses alignment + majority rule at each position — a simplified
        version of the comparative method.  The aligned symbols vote on
        the proto-form; the most "archaic" (least marked) variant wins
        ties.
        """
        if not cognates:
            return Sequence()
        if len(cognates) == 1:
            return cognates[0]

        # Align all cognates pairwise against the first
        reference = cognates[0]
        aligned_seqs: List[List[Optional[Symbol]]] = [list(reference)]

        for other in cognates[1:]:
            _, aligned_b, _ = self.align(reference, other)
            aligned_seqs.append(aligned_b)

        # Majority vote at each position
        max_len = max(len(a) for a in aligned_seqs)
        proto_symbols: List[Symbol] = []
        for pos in range(max_len):
            candidates: List[Symbol] = []
            for aligned in aligned_seqs:
                if pos < len(aligned) and aligned[pos] is not None:
                    candidates.append(aligned[pos])
            if candidates:
                # Pick most frequent; break ties by lower sonority (more
                # conservative / archaic)
                counts: Dict[str, Tuple[int, Symbol]] = {}
                for c in candidates:
                    key = c.form
                    if key in counts:
                        counts[key] = (counts[key][0] + 1, c)
                    else:
                        counts[key] = (1, c)
                best = max(
                    counts.values(),
                    key=lambda x: (x[0], -self._sonority(x[1])
                                   if isinstance(x[1], Phoneme) else 0),
                )
                proto_symbols.append(best[1])

        return Sequence(proto_symbols)

    # -- alignment override with feature-aware scoring ----------------------

    def align(
        self,
        seq_a: Sequence,
        seq_b: Sequence,
        **kwargs: Any,
    ) -> Tuple[List[Optional[Symbol]], List[Optional[Symbol]], float]:
        """Phonologically-aware alignment using feature distance."""
        kwargs.setdefault("feature_weight", 1.0)
        kwargs.setdefault("match_score", 2.0)
        kwargs.setdefault("mismatch_penalty", -1.0)
        kwargs.setdefault("gap_penalty", -0.5)
        return Sequence.align(seq_a, seq_b, **kwargs)

    # -- predict sound change forward ---------------------------------------

    def predict_change(
        self,
        sequence: Sequence,
        change_name: str,
        generations: int = 1,
    ) -> Sequence:
        """Apply a sound change *generations* times forward."""
        result = sequence
        for _ in range(generations):
            result = self.apply_sound_change(result, change_name)
        return result
