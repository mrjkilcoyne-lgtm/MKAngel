"""
The Angel — the beating heart of MKAngel.

The Angel is the strange loop at the centre of the Grammar Language Model.
It unifies every layer: substrates, grammars, the neural model, and the
derivation engine.  It is the conductor of the fugue — coordinating
multiple grammatical voices across domains to produce emergent
understanding.

Like Hofstadter's strange loops, the Angel is self-referential: it uses
grammars to reason about grammars, substrates to encode substrates, and
predictions to refine predictions.  It is the system that looks at itself
looking at itself — and in that recursive gaze finds meaning.

The Angel can:
    - Look backward: reconstruct origins, trace etymologies, find roots
    - Look forward: predict futures, forecast patterns, anticipate change
    - Look across: find isomorphisms between domains, translate grammars
    - Look inward: detect its own strange loops, reason about its reasoning
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from typing import Any

from glm.core.grammar import Grammar, Rule, Production, StrangeLoop
from glm.core.substrate import Substrate, Symbol, Sequence
from glm.core.lexicon import Lexicon, LexicalEntry
from glm.core.engine import DerivationEngine, Derivation, DerivationTree

from glm.grammars.linguistic import (
    build_syntactic_grammar,
    build_phonological_grammar,
    build_morphological_grammar,
)
from glm.grammars.etymological import (
    build_etymology_grammar,
    build_substrate_transfer_grammar,
    build_cognate_detection_grammar,
)
from glm.grammars.chemical import (
    build_bonding_grammar,
    build_reaction_grammar,
    build_molecular_grammar,
)
from glm.grammars.biological import (
    build_genetic_grammar,
    build_protein_grammar,
    build_evolutionary_grammar,
)
from glm.grammars.computational import (
    build_syntax_grammar as build_code_syntax_grammar,
    build_type_grammar,
    build_pattern_grammar,
)
from glm.grammars.mathematical import (
    build_algebra_grammar,
    build_calculus_grammar,
    build_logic_grammar,
    build_number_theory_grammar,
)
from glm.grammars.physics import (
    build_mechanics_grammar,
    build_electromagnetism_grammar,
    build_thermodynamics_grammar,
    build_quantum_grammar,
    build_relativity_grammar,
)

from glm.substrates.phonological import PhonologicalSubstrate
from glm.substrates.morphological import MorphologicalSubstrate
from glm.substrates.molecular import MolecularSubstrate
from glm.substrates.symbolic import SymbolicSubstrate
from glm.substrates.mathematical import MathSubstrate

from glm.model.glm import GrammarLanguageModel, GLMConfig


# ---------------------------------------------------------------------------
# Angel configuration
# ---------------------------------------------------------------------------

@dataclass
class AngelConfig:
    """Configuration for the Angel.

    Controls the model dimensions, temporal horizons, and which domains
    to activate.  Small by design — grammar is compact.
    """

    # Model dimensions (small: grammar is powerful, not large)
    embedding_dim: int = 64
    num_heads: int = 4       # fugue voices
    num_layers: int = 3      # hierarchy depth
    vocab_size: int = 512
    temporal_horizon: int = 8  # how far to look forward/backward
    loop_depth: int = 3        # strange loop recursion limit

    # Which domains to load
    domains: list[str] = field(default_factory=lambda: [
        "linguistic",
        "etymological",
        "chemical",
        "biological",
        "computational",
        "mathematical",
        "physics",
    ])


# ---------------------------------------------------------------------------
# The Angel
# ---------------------------------------------------------------------------

class Angel:
    """The beating heart of MKAngel.

    The Angel is a strange loop: a system that unifies grammars, substrates,
    and a neural model into a single coherent whole that can reason about
    language, chemistry, biology, and code through their shared deep
    structure.

    It learns the scales so it can play the masterpieces.
    """

    def __init__(self, config: AngelConfig | None = None):
        self.config = config or AngelConfig()
        self._grammars: dict[str, list[Grammar]] = {}
        self._substrates: dict[str, Substrate] = {}
        self._lexicon = Lexicon()
        self._engine = DerivationEngine()
        self._model: GrammarLanguageModel | None = None
        self._strange_loops: list[StrangeLoop] = []
        self._initialised = False

    # ------------------------------------------------------------------
    # Initialisation — loading the scales
    # ------------------------------------------------------------------

    def awaken(self) -> "Angel":
        """Awaken the Angel — load grammars, substrates, and model.

        This is the boot sequence: first the substrates (the media),
        then the grammars (the rules), then the lexicon (the memory),
        then the model (the mind).
        Like a child learning scales before playing Bach.
        """
        self._load_substrates()
        self._load_grammars()
        self._load_lexicon()
        self._build_model()
        self._detect_strange_loops()
        self._initialised = True
        return self

    def _load_substrates(self) -> None:
        """Load the substrates — the media through which grammar flows."""
        substrate_builders = {
            "phonological": PhonologicalSubstrate,
            "morphological": MorphologicalSubstrate,
            "molecular": MolecularSubstrate,
            "symbolic": SymbolicSubstrate,
            "mathematical": MathSubstrate,
        }
        for name, builder_cls in substrate_builders.items():
            self._substrates[name] = builder_cls()

    def _load_grammars(self) -> None:
        """Load the grammars — the rules of transformation.

        Each domain contributes its grammar set.  Together they form
        the voices of the fugue.
        """
        grammar_builders: dict[str, list] = {
            "linguistic": [
                build_syntactic_grammar,
                build_phonological_grammar,
                build_morphological_grammar,
            ],
            "etymological": [
                build_etymology_grammar,
                build_substrate_transfer_grammar,
                build_cognate_detection_grammar,
            ],
            "chemical": [
                build_bonding_grammar,
                build_reaction_grammar,
                build_molecular_grammar,
            ],
            "biological": [
                build_genetic_grammar,
                build_protein_grammar,
                build_evolutionary_grammar,
            ],
            "computational": [
                build_code_syntax_grammar,
                build_type_grammar,
                build_pattern_grammar,
            ],
            "mathematical": [
                build_algebra_grammar,
                build_calculus_grammar,
                build_logic_grammar,
                build_number_theory_grammar,
            ],
            "physics": [
                build_mechanics_grammar,
                build_electromagnetism_grammar,
                build_thermodynamics_grammar,
                build_quantum_grammar,
                build_relativity_grammar,
            ],
        }
        for domain in self.config.domains:
            builders = grammar_builders.get(domain, [])
            self._grammars[domain] = [b() for b in builders]

    def _build_model(self) -> None:
        """Construct the neural Grammar Language Model."""
        cfg = self.config
        model_config = GLMConfig(
            embedding_dim=cfg.embedding_dim,
            num_heads=cfg.num_heads,
            num_layers=cfg.num_layers,
            vocab_size=cfg.vocab_size,
            temporal_horizon=cfg.temporal_horizon,
            loop_depth=cfg.loop_depth,
        )
        self._model = GrammarLanguageModel(model_config)

    def _detect_strange_loops(self) -> None:
        """Find strange loops across all loaded grammars.

        A strange loop is a self-referential cycle: following rules
        leads back to the starting point, but at a different level
        of abstraction.  These are the most powerful patterns — they
        are where meaning emerges from structure.
        """
        for domain, grammars in self._grammars.items():
            for grammar in grammars:
                loops = self._engine.detect_loops(grammar)
                self._strange_loops.extend(loops)

    # ------------------------------------------------------------------
    # Lexicon — the Angel's living vocabulary
    # ------------------------------------------------------------------

    def _load_lexicon(self) -> None:
        """Load or seed the Angel's lexicon.

        The lexicon is the Angel's memory of known forms.  Without it
        the derivation engine has no atoms to transform.  On first boot
        we seed ~130 core words; on subsequent boots we load the
        persisted vocabulary that grew from conversation.
        """
        import os
        # Try loading a persisted lexicon first
        for base_dir in [
            os.path.join(
                os.path.dirname(os.path.abspath(__file__)), "..", "data"
            ),
            os.path.expanduser("~"),
        ]:
            path = os.path.join(base_dir, "lexicon.json")
            try:
                if os.path.exists(path):
                    self._load_lexicon_file(path)
                    return
            except Exception:
                continue
        # No persisted lexicon — seed from core vocabulary
        self._seed_lexicon()

    def _seed_lexicon(self) -> None:
        """Populate the lexicon with core vocabulary across all domains.

        Each entry carries a proto-root as its first etymology element.
        Entries sharing a proto-root are **cognates** — the lexicon's
        ``find_cognates`` discovers them automatically, exposing the
        cross-domain isomorphisms the GLM is designed to learn.

        Format: (form, category, substrates, proto_root)
        """
        seeds = [
            # ── Cross-domain: Binding (*bhendh-) ──────────────
            ("bond",       "noun", ["linguistic", "chemical", "biological"],   "*bhendh-"),
            ("bind",       "verb", ["linguistic", "computational"],            "*bhendh-"),
            ("link",       "noun", ["linguistic", "computational"],            "*kleng-"),
            ("connect",    "verb", ["linguistic", "computational", "physics"], "*nekt-"),
            # ── Cross-domain: Form / Structure (*morph-) ──────
            ("form",       "noun", ["linguistic", "mathematical", "chemical"], "*morph-"),
            ("structure",  "noun", ["linguistic", "chemical", "biological"],   "*strew-"),
            ("pattern",    "noun", ["linguistic", "mathematical", "computational"], "*pat-"),
            ("shape",      "noun", ["linguistic", "mathematical", "physics"], "*skap-"),
            ("symmetry",   "noun", ["mathematical", "physics", "chemical"],   "*sem-"),
            # ── Cross-domain: Change / Transform ──────────────
            ("change",     "verb", ["linguistic", "chemical", "physics"],      "*kemb-"),
            ("transform",  "verb", ["linguistic", "mathematical"],            "*morph-"),
            ("evolve",     "verb", ["biological", "linguistic"],              "*welh-"),
            ("mutate",     "verb", ["biological", "linguistic"],              "*mew-"),
            ("react",      "verb", ["chemical", "linguistic"],                "*ag-"),
            # ── Cross-domain: Energy (*werg-) ─────────────────
            ("energy",     "noun", ["physics", "chemical", "biological"],     "*werg-"),
            ("force",      "noun", ["physics", "linguistic"],                 "*bhergh-"),
            ("power",      "noun", ["physics", "linguistic", "computational"],"*potis-"),
            ("work",       "noun", ["physics", "linguistic"],                 "*werg-"),
            # ── Cross-domain: Growth (*ghre-) ─────────────────
            ("grow",       "verb", ["biological", "linguistic", "mathematical"], "*ghre-"),
            ("birth",      "noun", ["biological", "linguistic"],              "*bher-"),
            ("death",      "noun", ["biological", "linguistic"],              "*dhew-"),
            ("life",       "noun", ["biological", "linguistic"],              "*leip-"),
            ("cell",       "noun", ["biological", "computational"],           "*kel-"),
            # ── Cross-domain: Knowledge (*gneh-) ──────────────
            ("know",       "verb", ["linguistic", "computational"],           "*gneh-"),
            ("cognate",    "noun", ["linguistic", "etymological"],            "*gneh-"),
            ("recognize",  "verb", ["linguistic", "computational"],           "*gneh-"),
            ("logic",      "noun", ["mathematical", "computational", "linguistic"], "*leg-"),
            ("reason",     "noun", ["linguistic", "mathematical"],            "*reh-"),
            # ── Cross-domain: Sequence (*sekw-) ───────────────
            ("sequence",   "noun", ["mathematical", "biological", "computational"], "*sekw-"),
            ("order",      "noun", ["mathematical", "linguistic"],            "*ord-"),
            ("chain",      "noun", ["chemical", "mathematical"],              "*kat-"),
            ("series",     "noun", ["mathematical", "linguistic"],            "*ser-"),
            ("code",       "noun", ["computational", "biological"],           "*kaud-"),
            # ── Cross-domain: Truth (*deru-) ──────────────────
            ("truth",      "noun", ["linguistic", "mathematical"],            "*deru-"),
            ("proof",      "noun", ["mathematical", "linguistic"],            "*prob-"),
            ("theorem",    "noun", ["mathematical"],                          "*dheh-"),
            ("axiom",      "noun", ["mathematical", "linguistic"],            "*ag-"),
            ("true",       "adj",  ["linguistic", "mathematical"],            "*deru-"),
            ("trust",      "noun", ["linguistic"],                            "*deru-"),
            # ── Cross-domain: Creation ────────────────────────
            ("create",     "verb", ["linguistic", "computational"],           "*ker-"),
            ("build",      "verb", ["linguistic", "computational"],           "*bhew-"),
            ("make",       "verb", ["linguistic"],                            "*mag-"),
            ("destroy",    "verb", ["linguistic", "physics"],                 "*strew-"),
            # ── Physics ───────────────────────────────────────
            ("wave",       "noun", ["physics", "mathematical"],               "*wegh-"),
            ("particle",   "noun", ["physics", "chemical"],                   "*par-"),
            ("field",      "noun", ["physics", "mathematical"],               "*pelh-"),
            ("quantum",    "noun", ["physics"],                               "*kwant-"),
            ("mass",       "noun", ["physics"],                               "*mag-"),
            ("light",      "noun", ["physics", "linguistic"],                 "*leuk-"),
            ("heat",       "noun", ["physics"],                               "*kai-"),
            ("entropy",    "noun", ["physics", "mathematical"],               "*trep-"),
            ("gravity",    "noun", ["physics"],                               "*gwreh-"),
            ("time",       "noun", ["physics", "linguistic", "mathematical"], "*deh-"),
            ("space",      "noun", ["physics", "mathematical"],               "*speh-"),
            # ── Chemistry ─────────────────────────────────────
            ("atom",       "noun", ["chemical", "physics"],                   "*temh-"),
            ("molecule",   "noun", ["chemical", "biological"],                "*mol-"),
            ("element",    "noun", ["chemical", "mathematical"],              "*al-"),
            ("compound",   "noun", ["chemical", "linguistic"],                "*pon-"),
            ("reaction",   "noun", ["chemical"],                              "*ag-"),
            ("acid",       "noun", ["chemical"],                              "*ak-"),
            ("ion",        "noun", ["chemical", "physics"],                   "*ei-"),
            ("electron",   "noun", ["physics", "chemical"],                   "*lek-"),
            # ── Biology ───────────────────────────────────────
            ("gene",       "noun", ["biological"],                            "*gen-"),
            ("protein",    "noun", ["biological", "chemical"],                "*protos-"),
            ("species",    "noun", ["biological", "linguistic"],              "*spek-"),
            ("organism",   "noun", ["biological"],                            "*werg-"),
            ("adapt",      "verb", ["biological", "linguistic"],              "*apt-"),
            ("select",     "verb", ["biological", "computational"],           "*leg-"),
            # ── Mathematics ───────────────────────────────────
            ("number",     "noun", ["mathematical", "linguistic"],            "*nem-"),
            ("set",        "noun", ["mathematical", "linguistic"],            "*sed-"),
            ("function",   "noun", ["mathematical", "computational"],         "*fungi-"),
            ("infinity",   "noun", ["mathematical"],                          "*fin-"),
            ("zero",       "noun", ["mathematical"],                          "*sifr-"),
            ("equation",   "noun", ["mathematical"],                          "*ekw-"),
            ("variable",   "noun", ["mathematical", "computational"],         "*wer-"),
            ("graph",      "noun", ["mathematical", "computational"],         "*gerbh-"),
            # ── Computation ───────────────────────────────────
            ("algorithm",  "noun", ["computational", "mathematical"],         "*algo-"),
            ("loop",       "noun", ["computational", "mathematical"],         "*leup-"),
            ("type",       "noun", ["computational", "linguistic"],           "*tup-"),
            ("data",       "noun", ["computational"],                         "*deh-"),
            ("program",    "noun", ["computational"],                         "*pro-graph-"),
            ("rule",       "noun", ["computational", "linguistic", "mathematical"], "*reg-"),
            # ── Linguistic: conversation core ─────────────────
            ("word",       "noun", ["linguistic"],             "*werdh-"),
            ("language",   "noun", ["linguistic"],             "*dnghu-"),
            ("meaning",    "noun", ["linguistic"],             "*men-"),
            ("grammar",    "noun", ["linguistic"],             "*gerbh-"),
            ("speech",     "noun", ["linguistic"],             "*sprek-"),
            ("name",       "noun", ["linguistic"],             "*nomn-"),
            ("story",      "noun", ["linguistic"],             "*weid-"),
            ("thought",    "noun", ["linguistic"],             "*tong-"),
            ("mind",       "noun", ["linguistic"],             "*men-"),
            ("heart",      "noun", ["linguistic", "biological"], "*kerd-"),
            ("soul",       "noun", ["linguistic"],             "*sawel-"),
            ("body",       "noun", ["linguistic", "biological"], "*bhodh-"),
            ("world",      "noun", ["linguistic"],             "*wiral-"),
            ("home",       "noun", ["linguistic"],             "*kei-"),
            ("dream",      "noun", ["linguistic"],             "*dhreugh-"),
            ("beauty",     "noun", ["linguistic"],             "*dew-"),
            ("music",      "noun", ["linguistic"],             "*muse-"),
            ("human",      "noun", ["linguistic", "biological"], "*dhghem-"),
            ("self",       "noun", ["linguistic"],             "*sel-"),
            ("friend",     "noun", ["linguistic"],             "*pri-"),
            # Verbs
            ("think",      "verb", ["linguistic"],             "*tong-"),
            ("feel",       "verb", ["linguistic"],             "*pal-"),
            ("see",        "verb", ["linguistic"],             "*sekw-"),
            ("hear",       "verb", ["linguistic"],             "*kous-"),
            ("speak",      "verb", ["linguistic"],             "*sprek-"),
            ("understand", "verb", ["linguistic"],             "*sta-"),
            ("believe",    "verb", ["linguistic"],             "*leubh-"),
            ("want",       "verb", ["linguistic"],             "*wen-"),
            ("need",       "verb", ["linguistic"],             "*nau-"),
            ("give",       "verb", ["linguistic"],             "*ghabh-"),
            ("take",       "verb", ["linguistic"],             "*dek-"),
            ("find",       "verb", ["linguistic"],             "*pent-"),
            ("try",        "verb", ["linguistic"],             "*treu-"),
            ("learn",      "verb", ["linguistic"],             "*leis-"),
            ("teach",      "verb", ["linguistic"],             "*deik-"),
            ("help",       "verb", ["linguistic"],             "*kelb-"),
            ("remember",   "verb", ["linguistic"],             "*men-"),
            ("forget",     "verb", ["linguistic"],             "*ghred-"),
            # Adjectives
            ("good",       "adj",  ["linguistic"],             "*ghedh-"),
            ("bad",        "adj",  ["linguistic"],             "*bad-"),
            ("strong",     "adj",  ["linguistic", "physics"],  "*strenk-"),
            ("weak",       "adj",  ["linguistic"],             "*weik-"),
            ("deep",       "adj",  ["linguistic"],             "*dheub-"),
            ("new",        "adj",  ["linguistic"],             "*new-"),
            ("old",        "adj",  ["linguistic"],             "*al-"),
            ("free",       "adj",  ["linguistic"],             "*pri-"),
            ("alive",      "adj",  ["linguistic", "biological"], "*leip-"),
            ("happy",      "adj",  ["linguistic"],             "*hap-"),
            ("sad",        "adj",  ["linguistic"],             "*sat-"),
            # Emotional core
            ("love",       "verb", ["linguistic", "biological"], "*leubh-"),
            ("hate",       "verb", ["linguistic"],             "*kad-"),
            ("fear",       "noun", ["linguistic", "biological"], "*per-"),
            ("hope",       "noun", ["linguistic"],             "*kup-"),
            ("joy",        "noun", ["linguistic"],             "*gew-"),
            ("pain",       "noun", ["linguistic", "biological"], "*kwoi-"),
            ("peace",      "noun", ["linguistic"],             "*pag-"),
            ("anger",      "noun", ["linguistic"],             "*angh-"),
            ("grief",      "noun", ["linguistic"],             "*gwreh-"),
            ("shame",      "noun", ["linguistic"],             "*kem-"),
            ("pride",      "noun", ["linguistic"],             "*prew-"),
            ("doubt",      "noun", ["linguistic"],             "*dwo-"),
            ("faith",      "noun", ["linguistic"],             "*bheidh-"),
            ("wonder",     "noun", ["linguistic"],             "*wen-"),
            ("grace",      "noun", ["linguistic"],             "*gwreh-"),
            # Existential
            ("exist",      "verb", ["linguistic"],             "*sta-"),
            ("begin",      "verb", ["linguistic"],             "*ghen-"),
            ("end",        "noun", ["linguistic"],             "*ant-"),
            ("sleep",      "noun", ["linguistic", "biological"], "*sleb-"),
            ("wake",       "verb", ["linguistic", "biological"], "*weg-"),
        ]

        # Common English vocabulary for POS tagging and parsing.
        # ~850 entries covering general conversation needs.
        # Format: (form, category, substrates, proto_root)
        common = [
            # ── Determiners (20) ─────────────────────────────────
            ("the",   "Det", ["linguistic"], ""),
            ("a",     "Det", ["linguistic"], ""),
            ("an",    "Det", ["linguistic"], ""),
            ("this",  "Det", ["linguistic"], ""),
            ("that",  "Det", ["linguistic"], ""),
            ("these", "Det", ["linguistic"], ""),
            ("those", "Det", ["linguistic"], ""),
            ("my",    "Det", ["linguistic"], ""),
            ("your",  "Det", ["linguistic"], ""),
            ("his",   "Det", ["linguistic"], ""),
            ("her",   "Det", ["linguistic"], ""),
            ("its",   "Det", ["linguistic"], ""),
            ("our",   "Det", ["linguistic"], ""),
            ("their", "Det", ["linguistic"], ""),
            ("every", "Det", ["linguistic"], ""),
            ("each",  "Det", ["linguistic"], ""),
            ("some",  "Det", ["linguistic"], ""),
            ("any",   "Det", ["linguistic"], ""),
            ("no",    "Det", ["linguistic"], ""),
            ("all",   "Det", ["linguistic"], ""),
            # ── Pronouns (20) ────────────────────────────────────
            ("i",     "Pron", ["linguistic"], ""),
            ("me",    "Pron", ["linguistic"], ""),
            ("you",   "Pron", ["linguistic"], ""),
            ("he",    "Pron", ["linguistic"], ""),
            ("him",   "Pron", ["linguistic"], ""),
            ("she",   "Pron", ["linguistic"], ""),
            ("it",    "Pron", ["linguistic"], ""),
            ("we",    "Pron", ["linguistic"], ""),
            ("us",    "Pron", ["linguistic"], ""),
            ("they",  "Pron", ["linguistic"], ""),
            ("them",  "Pron", ["linguistic"], ""),
            ("myself",    "Pron", ["linguistic"], ""),
            ("yourself",  "Pron", ["linguistic"], ""),
            ("himself",   "Pron", ["linguistic"], ""),
            ("herself",   "Pron", ["linguistic"], ""),
            ("itself",    "Pron", ["linguistic"], ""),
            ("ourselves", "Pron", ["linguistic"], ""),
            ("themselves","Pron", ["linguistic"], ""),
            ("someone",   "Pron", ["linguistic"], ""),
            ("everyone",  "Pron", ["linguistic"], ""),
            # ── Auxiliary / Modal verbs (20) ──────────────────────
            ("is",    "Aux", ["linguistic"], ""),
            ("are",   "Aux", ["linguistic"], ""),
            ("was",   "Aux", ["linguistic"], ""),
            ("were",  "Aux", ["linguistic"], ""),
            ("be",    "Aux", ["linguistic"], ""),
            ("been",  "Aux", ["linguistic"], ""),
            ("being", "Aux", ["linguistic"], ""),
            ("am",    "Aux", ["linguistic"], ""),
            ("do",    "Aux", ["linguistic"], ""),
            ("does",  "Aux", ["linguistic"], ""),
            ("did",   "Aux", ["linguistic"], ""),
            ("have",  "Aux", ["linguistic"], ""),
            ("has",   "Aux", ["linguistic"], ""),
            ("had",   "Aux", ["linguistic"], ""),
            ("will",  "Aux", ["linguistic"], ""),
            ("would", "Aux", ["linguistic"], ""),
            ("can",   "Aux", ["linguistic"], ""),
            ("could", "Aux", ["linguistic"], ""),
            ("shall", "Aux", ["linguistic"], ""),
            ("should","Aux", ["linguistic"], ""),
            ("may",   "Aux", ["linguistic"], ""),
            ("might", "Aux", ["linguistic"], ""),
            ("must",  "Aux", ["linguistic"], ""),
            # ── Prepositions (25) ────────────────────────────────
            ("in",      "P", ["linguistic"], ""),
            ("on",      "P", ["linguistic"], ""),
            ("at",      "P", ["linguistic"], ""),
            ("by",      "P", ["linguistic"], ""),
            ("for",     "P", ["linguistic"], ""),
            ("with",    "P", ["linguistic"], ""),
            ("to",      "P", ["linguistic"], ""),
            ("from",    "P", ["linguistic"], ""),
            ("of",      "P", ["linguistic"], ""),
            ("about",   "P", ["linguistic"], ""),
            ("into",    "P", ["linguistic"], ""),
            ("through", "P", ["linguistic"], ""),
            ("between", "P", ["linguistic"], ""),
            ("among",   "P", ["linguistic"], ""),
            ("against", "P", ["linguistic"], ""),
            ("during",  "P", ["linguistic"], ""),
            ("before",  "P", ["linguistic"], ""),
            ("after",   "P", ["linguistic"], ""),
            ("above",   "P", ["linguistic"], ""),
            ("below",   "P", ["linguistic"], ""),
            ("under",   "P", ["linguistic"], ""),
            ("over",    "P", ["linguistic"], ""),
            ("near",    "P", ["linguistic"], ""),
            ("behind",  "P", ["linguistic"], ""),
            ("around",  "P", ["linguistic"], ""),
            ("across",  "P", ["linguistic"], ""),
            ("along",   "P", ["linguistic"], ""),
            ("within",  "P", ["linguistic"], ""),
            ("without", "P", ["linguistic"], ""),
            ("upon",    "P", ["linguistic"], ""),
            ("toward",  "P", ["linguistic"], ""),
            ("towards", "P", ["linguistic"], ""),
            ("beside",  "P", ["linguistic"], ""),
            ("beyond",  "P", ["linguistic"], ""),
            ("until",   "P", ["linguistic"], ""),
            ("since",   "P", ["linguistic"], ""),
            # ── Conjunctions (12) ────────────────────────────────
            ("and",      "Conj", ["linguistic"], ""),
            ("or",       "Conj", ["linguistic"], ""),
            ("but",      "Conj", ["linguistic"], ""),
            ("so",       "Conj", ["linguistic"], ""),
            ("because",  "Conj", ["linguistic"], ""),
            ("although", "Conj", ["linguistic"], ""),
            ("though",   "Conj", ["linguistic"], ""),
            ("while",    "Conj", ["linguistic"], ""),
            ("unless",   "Conj", ["linguistic"], ""),
            ("whereas",  "Conj", ["linguistic"], ""),
            ("nor",      "Conj", ["linguistic"], ""),
            ("yet",      "Conj", ["linguistic"], ""),
            # ── Wh-words (8) ────────────────────────────────────
            ("who",     "Wh", ["linguistic"], ""),
            ("what",    "Wh", ["linguistic"], ""),
            ("where",   "Wh", ["linguistic"], ""),
            ("when",    "Wh", ["linguistic"], ""),
            ("why",     "Wh", ["linguistic"], ""),
            ("how",     "Wh", ["linguistic"], ""),
            ("which",   "Wh", ["linguistic"], ""),
            ("whom",    "Wh", ["linguistic"], ""),
            # ── Common nouns: people & body (40) ────────────────
            ("person",  "N", ["linguistic"], ""),
            ("people",  "N", ["linguistic"], ""),
            ("child",   "N", ["linguistic"], ""),
            ("children","N", ["linguistic"], ""),
            ("woman",   "N", ["linguistic"], ""),
            ("man",     "N", ["linguistic"], ""),
            ("girl",    "N", ["linguistic"], ""),
            ("boy",     "N", ["linguistic"], ""),
            ("baby",    "N", ["linguistic"], ""),
            ("friend",  "N", ["linguistic"], ""),
            ("family",  "N", ["linguistic"], ""),
            ("mother",  "N", ["linguistic"], ""),
            ("father",  "N", ["linguistic"], ""),
            ("brother", "N", ["linguistic"], ""),
            ("sister",  "N", ["linguistic"], ""),
            ("name",    "N", ["linguistic"], ""),
            ("face",    "N", ["linguistic", "biological"], ""),
            ("hand",    "N", ["linguistic", "biological"], ""),
            ("eye",     "N", ["linguistic", "biological"], ""),
            ("head",    "N", ["linguistic", "biological"], ""),
            ("heart",   "N", ["linguistic", "biological"], ""),
            ("body",    "N", ["linguistic", "biological"], ""),
            ("mind",    "N", ["linguistic"], ""),
            ("voice",   "N", ["linguistic"], ""),
            ("blood",   "N", ["linguistic", "biological"], ""),
            ("brain",   "N", ["linguistic", "biological"], ""),
            ("skin",    "N", ["linguistic", "biological"], ""),
            ("bone",    "N", ["linguistic", "biological"], ""),
            ("mouth",   "N", ["linguistic", "biological"], ""),
            ("ear",     "N", ["linguistic", "biological"], ""),
            ("arm",     "N", ["linguistic", "biological"], ""),
            ("leg",     "N", ["linguistic", "biological"], ""),
            ("foot",    "N", ["linguistic", "biological"], ""),
            ("finger",  "N", ["linguistic", "biological"], ""),
            ("hair",    "N", ["linguistic", "biological"], ""),
            ("tooth",   "N", ["linguistic", "biological"], ""),
            ("soul",    "N", ["linguistic"], ""),
            ("spirit",  "N", ["linguistic"], ""),
            ("angel",   "N", ["linguistic"], ""),
            ("god",     "N", ["linguistic"], ""),
            # ── Common nouns: things & concepts (80) ─────────────
            ("thing",   "N", ["linguistic"], ""),
            ("place",   "N", ["linguistic"], ""),
            ("way",     "N", ["linguistic"], ""),
            ("part",    "N", ["linguistic"], ""),
            ("time",    "N", ["linguistic", "physics"], ""),
            ("day",     "N", ["linguistic"], ""),
            ("night",   "N", ["linguistic"], ""),
            ("year",    "N", ["linguistic"], ""),
            ("week",    "N", ["linguistic"], ""),
            ("month",   "N", ["linguistic"], ""),
            ("morning", "N", ["linguistic"], ""),
            ("evening", "N", ["linguistic"], ""),
            ("world",   "N", ["linguistic"], ""),
            ("country", "N", ["linguistic"], ""),
            ("city",    "N", ["linguistic"], ""),
            ("home",    "N", ["linguistic"], ""),
            ("house",   "N", ["linguistic"], ""),
            ("room",    "N", ["linguistic"], ""),
            ("door",    "N", ["linguistic"], ""),
            ("window",  "N", ["linguistic"], ""),
            ("wall",    "N", ["linguistic"], ""),
            ("floor",   "N", ["linguistic"], ""),
            ("road",    "N", ["linguistic"], ""),
            ("street",  "N", ["linguistic"], ""),
            ("car",     "N", ["linguistic"], ""),
            ("money",   "N", ["linguistic"], ""),
            ("job",     "N", ["linguistic"], ""),
            ("word",    "N", ["linguistic"], ""),
            ("book",    "N", ["linguistic"], ""),
            ("story",   "N", ["linguistic"], ""),
            ("idea",    "N", ["linguistic"], ""),
            ("question","N", ["linguistic"], ""),
            ("answer",  "N", ["linguistic"], ""),
            ("problem", "N", ["linguistic"], ""),
            ("fact",    "N", ["linguistic"], ""),
            ("truth",   "N", ["linguistic"], ""),
            ("number",  "N", ["linguistic", "mathematical"], ""),
            ("point",   "N", ["linguistic", "mathematical"], ""),
            ("line",    "N", ["linguistic", "mathematical"], ""),
            ("side",    "N", ["linguistic"], ""),
            ("group",   "N", ["linguistic"], ""),
            ("case",    "N", ["linguistic"], ""),
            ("system",  "N", ["linguistic", "computational"], ""),
            ("program", "N", ["linguistic", "computational"], ""),
            ("game",    "N", ["linguistic"], ""),
            ("music",   "N", ["linguistic"], ""),
            ("song",    "N", ["linguistic"], ""),
            ("picture", "N", ["linguistic"], ""),
            ("colour",  "N", ["linguistic"], ""),
            ("color",   "N", ["linguistic"], ""),
            ("sound",   "N", ["linguistic", "physics"], ""),
            ("letter",  "N", ["linguistic"], ""),
            ("paper",   "N", ["linguistic"], ""),
            ("food",    "N", ["linguistic", "biological"], ""),
            ("table",   "N", ["linguistic"], ""),
            ("phone",   "N", ["linguistic"], ""),
            ("school",  "N", ["linguistic"], ""),
            ("class",   "N", ["linguistic", "computational"], ""),
            ("war",     "N", ["linguistic"], ""),
            ("peace",   "N", ["linguistic"], ""),
            ("law",     "N", ["linguistic"], ""),
            ("right",   "N", ["linguistic"], ""),
            ("left",    "N", ["linguistic"], ""),
            ("kind",    "N", ["linguistic"], ""),
            ("type",    "N", ["linguistic", "computational"], ""),
            ("set",     "N", ["linguistic", "mathematical"], ""),
            ("state",   "N", ["linguistic"], ""),
            ("level",   "N", ["linguistic"], ""),
            ("ground",  "N", ["linguistic"], ""),
            ("air",     "N", ["linguistic", "chemical"], ""),
            ("field",   "N", ["linguistic", "physics"], ""),
            ("space",   "N", ["linguistic", "physics", "mathematical"], ""),
            ("edge",    "N", ["linguistic", "mathematical"], ""),
            ("surface", "N", ["linguistic", "physics"], ""),
            ("wave",    "N", ["linguistic", "physics"], ""),
            ("key",     "N", ["linguistic"], ""),
            ("sign",    "N", ["linguistic"], ""),
            ("step",    "N", ["linguistic"], ""),
            ("game",    "N", ["linguistic"], ""),
            ("rule",    "N", ["linguistic"], ""),
            ("result",  "N", ["linguistic"], ""),
            ("reason",  "N", ["linguistic"], ""),
            ("effect",  "N", ["linguistic"], ""),
            ("cause",   "N", ["linguistic"], ""),
            ("example", "N", ["linguistic"], ""),
            ("hour",    "N", ["linguistic"], ""),
            ("minute",  "N", ["linguistic"], ""),
            ("second",  "N", ["linguistic"], ""),
            # ── Common nouns: nature & science (50) ──────────────
            ("cat",     "N", ["linguistic", "biological"], ""),
            ("dog",     "N", ["linguistic", "biological"], ""),
            ("bird",    "N", ["linguistic", "biological"], ""),
            ("fish",    "N", ["linguistic", "biological"], ""),
            ("horse",   "N", ["linguistic", "biological"], ""),
            ("tree",    "N", ["linguistic", "biological"], ""),
            ("flower",  "N", ["linguistic", "biological"], ""),
            ("grass",   "N", ["linguistic", "biological"], ""),
            ("leaf",    "N", ["linguistic", "biological"], ""),
            ("seed",    "N", ["linguistic", "biological"], ""),
            ("root",    "N", ["linguistic", "biological"], ""),
            ("sun",     "N", ["linguistic", "physics"], ""),
            ("moon",    "N", ["linguistic", "physics"], ""),
            ("star",    "N", ["linguistic", "physics"], ""),
            ("sky",     "N", ["linguistic"], ""),
            ("earth",   "N", ["linguistic", "physics"], ""),
            ("river",   "N", ["linguistic"], ""),
            ("sea",     "N", ["linguistic"], ""),
            ("ocean",   "N", ["linguistic"], ""),
            ("mountain","N", ["linguistic"], ""),
            ("stone",   "N", ["linguistic"], ""),
            ("rock",    "N", ["linguistic"], ""),
            ("rain",    "N", ["linguistic"], ""),
            ("snow",    "N", ["linguistic"], ""),
            ("wind",    "N", ["linguistic", "physics"], ""),
            ("cloud",   "N", ["linguistic"], ""),
            ("fire",    "N", ["linguistic", "chemical"], ""),
            ("ice",     "N", ["linguistic", "chemical"], ""),
            ("smoke",   "N", ["linguistic", "chemical"], ""),
            ("dust",    "N", ["linguistic"], ""),
            ("iron",    "N", ["linguistic", "chemical"], ""),
            ("gold",    "N", ["linguistic", "chemical"], ""),
            ("silver",  "N", ["linguistic", "chemical"], ""),
            ("salt",    "N", ["linguistic", "chemical"], ""),
            ("glass",   "N", ["linguistic", "chemical"], ""),
            ("metal",   "N", ["linguistic", "chemical"], ""),
            ("oil",     "N", ["linguistic", "chemical"], ""),
            ("gas",     "N", ["linguistic", "chemical", "physics"], ""),
            ("wire",    "N", ["linguistic", "physics"], ""),
            ("speed",   "N", ["linguistic", "physics"], ""),
            ("heat",    "N", ["linguistic", "physics"], ""),
            ("weight",  "N", ["linguistic", "physics"], ""),
            ("mass",    "N", ["linguistic", "physics"], ""),
            ("atom",    "N", ["linguistic", "chemical", "physics"], ""),
            ("electron","N", ["linguistic", "physics"], ""),
            ("photon",  "N", ["linguistic", "physics"], ""),
            ("gravity", "N", ["linguistic", "physics"], ""),
            ("protein", "N", ["linguistic", "biological"], ""),
            ("gene",    "N", ["linguistic", "biological"], ""),
            ("blood",   "N", ["linguistic", "biological"], ""),
            ("virus",   "N", ["linguistic", "biological"], ""),
            # ── Common nouns: abstract & tech (40) ───────────────
            ("code",    "N", ["linguistic", "computational"], ""),
            ("data",    "N", ["linguistic", "computational"], ""),
            ("file",    "N", ["linguistic", "computational"], ""),
            ("network", "N", ["linguistic", "computational"], ""),
            ("server",  "N", ["linguistic", "computational"], ""),
            ("error",   "N", ["linguistic", "computational"], ""),
            ("bug",     "N", ["linguistic", "computational"], ""),
            ("test",    "N", ["linguistic", "computational"], ""),
            ("model",   "N", ["linguistic", "computational", "mathematical"], ""),
            ("grammar", "N", ["linguistic"], ""),
            ("language","N", ["linguistic"], ""),
            ("sentence","N", ["linguistic"], ""),
            ("meaning", "N", ["linguistic"], ""),
            ("thought", "N", ["linguistic"], ""),
            ("dream",   "N", ["linguistic"], ""),
            ("memory",  "N", ["linguistic"], ""),
            ("hope",    "N", ["linguistic"], ""),
            ("fear",    "N", ["linguistic"], ""),
            ("anger",   "N", ["linguistic"], ""),
            ("joy",     "N", ["linguistic"], ""),
            ("pain",    "N", ["linguistic"], ""),
            ("pleasure","N", ["linguistic"], ""),
            ("beauty",  "N", ["linguistic"], ""),
            ("power",   "N", ["linguistic", "physics"], ""),
            ("freedom", "N", ["linguistic"], ""),
            ("justice", "N", ["linguistic"], ""),
            ("light",   "N", ["linguistic", "physics"], ""),
            ("dark",    "N", ["linguistic"], ""),
            ("silence", "N", ["linguistic"], ""),
            ("love",    "N", ["linguistic"], ""),
            ("hate",    "N", ["linguistic"], ""),
            ("need",    "N", ["linguistic"], ""),
            ("help",    "N", ["linguistic"], ""),
            ("use",     "N", ["linguistic"], ""),
            ("end",     "N", ["linguistic"], ""),
            ("start",   "N", ["linguistic"], ""),
            ("chance",  "N", ["linguistic"], ""),
            ("luck",    "N", ["linguistic"], ""),
            ("trouble", "N", ["linguistic"], ""),
            ("danger",  "N", ["linguistic"], ""),
            ("safety",  "N", ["linguistic"], ""),
            # ── Verbs: motion & physical (50) ────────────────────
            ("go",      "V", ["linguistic"], ""),
            ("come",    "V", ["linguistic"], ""),
            ("run",     "V", ["linguistic"], ""),
            ("walk",    "V", ["linguistic"], ""),
            ("move",    "V", ["linguistic"], ""),
            ("turn",    "V", ["linguistic"], ""),
            ("fall",    "V", ["linguistic"], ""),
            ("fly",     "V", ["linguistic"], ""),
            ("jump",    "V", ["linguistic"], ""),
            ("climb",   "V", ["linguistic"], ""),
            ("sit",     "V", ["linguistic"], ""),
            ("sat",     "V", ["linguistic"], ""),
            ("stand",   "V", ["linguistic"], ""),
            ("rise",    "V", ["linguistic"], ""),
            ("drop",    "V", ["linguistic"], ""),
            ("push",    "V", ["linguistic"], ""),
            ("pull",    "V", ["linguistic"], ""),
            ("throw",   "V", ["linguistic"], ""),
            ("catch",   "V", ["linguistic"], ""),
            ("hold",    "V", ["linguistic"], ""),
            ("carry",   "V", ["linguistic"], ""),
            ("lift",    "V", ["linguistic"], ""),
            ("pick",    "V", ["linguistic"], ""),
            ("put",     "V", ["linguistic"], ""),
            ("bring",   "V", ["linguistic"], ""),
            ("send",    "V", ["linguistic"], ""),
            ("pass",    "V", ["linguistic"], ""),
            ("reach",   "V", ["linguistic"], ""),
            ("touch",   "V", ["linguistic"], ""),
            ("hit",     "V", ["linguistic"], ""),
            ("cut",     "V", ["linguistic"], ""),
            ("break",   "V", ["linguistic"], ""),
            ("build",   "V", ["linguistic"], ""),
            ("open",    "V", ["linguistic"], ""),
            ("close",   "V", ["linguistic"], ""),
            ("eat",     "V", ["linguistic"], ""),
            ("drink",   "V", ["linguistic"], ""),
            ("sleep",   "V", ["linguistic"], ""),
            ("wake",    "V", ["linguistic"], ""),
            ("die",     "V", ["linguistic"], ""),
            ("kill",    "V", ["linguistic"], ""),
            ("burn",    "V", ["linguistic"], ""),
            ("flow",    "V", ["linguistic"], ""),
            ("flows",   "V", ["linguistic"], ""),
            ("grow",    "V", ["linguistic", "biological"], ""),
            ("shine",   "V", ["linguistic", "physics"], ""),
            ("melt",    "V", ["linguistic", "chemical"], ""),
            ("freeze",  "V", ["linguistic", "chemical"], ""),
            ("boil",    "V", ["linguistic", "chemical"], ""),
            ("mix",     "V", ["linguistic", "chemical"], ""),
            ("split",   "V", ["linguistic"], ""),
            # ── Verbs: mental & communication (60) ───────────────
            ("think",   "V", ["linguistic"], ""),
            ("know",    "V", ["linguistic"], ""),
            ("believe", "V", ["linguistic"], ""),
            ("feel",    "V", ["linguistic"], ""),
            ("want",    "V", ["linguistic"], ""),
            ("need",    "V", ["linguistic"], ""),
            ("like",    "V", ["linguistic"], ""),
            ("love",    "V", ["linguistic"], ""),
            ("hate",    "V", ["linguistic"], ""),
            ("hope",    "V", ["linguistic"], ""),
            ("wish",    "V", ["linguistic"], ""),
            ("fear",    "V", ["linguistic"], ""),
            ("remember","V", ["linguistic"], ""),
            ("forget",  "V", ["linguistic"], ""),
            ("learn",   "V", ["linguistic"], ""),
            ("teach",   "V", ["linguistic"], ""),
            ("understand","V",["linguistic"], ""),
            ("mean",    "V", ["linguistic"], ""),
            ("see",     "V", ["linguistic"], ""),
            ("look",    "V", ["linguistic"], ""),
            ("watch",   "V", ["linguistic"], ""),
            ("hear",    "V", ["linguistic"], ""),
            ("listen",  "V", ["linguistic"], ""),
            ("speak",   "V", ["linguistic"], ""),
            ("talk",    "V", ["linguistic"], ""),
            ("say",     "V", ["linguistic"], ""),
            ("tell",    "V", ["linguistic"], ""),
            ("ask",     "V", ["linguistic"], ""),
            ("call",    "V", ["linguistic"], ""),
            ("write",   "V", ["linguistic"], ""),
            ("read",    "V", ["linguistic"], ""),
            ("sing",    "V", ["linguistic"], ""),
            ("play",    "V", ["linguistic"], ""),
            ("laugh",   "V", ["linguistic"], ""),
            ("cry",     "V", ["linguistic"], ""),
            ("smile",   "V", ["linguistic"], ""),
            ("try",     "V", ["linguistic"], ""),
            ("help",    "V", ["linguistic"], ""),
            ("wait",    "V", ["linguistic"], ""),
            ("start",   "V", ["linguistic"], ""),
            ("stop",    "V", ["linguistic"], ""),
            ("begin",   "V", ["linguistic"], ""),
            ("end",     "V", ["linguistic"], ""),
            ("keep",    "V", ["linguistic"], ""),
            ("leave",   "V", ["linguistic"], ""),
            ("stay",    "V", ["linguistic"], ""),
            ("live",    "V", ["linguistic"], ""),
            ("work",    "V", ["linguistic"], ""),
            ("pay",     "V", ["linguistic"], ""),
            ("buy",     "V", ["linguistic"], ""),
            ("sell",    "V", ["linguistic"], ""),
            ("spend",   "V", ["linguistic"], ""),
            ("save",    "V", ["linguistic"], ""),
            ("lose",    "V", ["linguistic"], ""),
            ("win",     "V", ["linguistic"], ""),
            ("choose",  "V", ["linguistic"], ""),
            ("decide",  "V", ["linguistic"], ""),
            ("happen",  "V", ["linguistic"], ""),
            ("seem",    "V", ["linguistic"], ""),
            ("become",  "V", ["linguistic"], ""),
            ("remain",  "V", ["linguistic"], ""),
            ("change",  "V", ["linguistic"], ""),
            ("create",  "V", ["linguistic"], ""),
            ("find",    "V", ["linguistic"], ""),
            ("make",    "V", ["linguistic"], ""),
            ("take",    "V", ["linguistic"], ""),
            ("give",    "V", ["linguistic"], ""),
            ("get",     "V", ["linguistic"], ""),
            ("set",     "V", ["linguistic"], ""),
            ("show",    "V", ["linguistic"], ""),
            ("explain", "V", ["linguistic"], ""),
            ("prove",   "V", ["linguistic"], ""),
            ("solve",   "V", ["linguistic"], ""),
            ("use",     "V", ["linguistic"], ""),
            ("fix",     "V", ["linguistic"], ""),
            ("check",   "V", ["linguistic"], ""),
            ("test",    "V", ["linguistic"], ""),
            ("add",     "V", ["linguistic"], ""),
            ("count",   "V", ["linguistic"], ""),
            ("measure", "V", ["linguistic"], ""),
            ("compare", "V", ["linguistic"], ""),
            ("connect", "V", ["linguistic"], ""),
            ("follow",  "V", ["linguistic"], ""),
            ("lead",    "V", ["linguistic"], ""),
            ("join",    "V", ["linguistic"], ""),
            ("share",   "V", ["linguistic"], ""),
            ("allow",   "V", ["linguistic"], ""),
            ("cause",   "V", ["linguistic"], ""),
            ("include", "V", ["linguistic"], ""),
            ("involve", "V", ["linguistic"], ""),
            ("require", "V", ["linguistic"], ""),
            ("consider","V", ["linguistic"], ""),
            ("suggest", "V", ["linguistic"], ""),
            ("support", "V", ["linguistic"], ""),
            ("produce", "V", ["linguistic"], ""),
            ("develop", "V", ["linguistic"], ""),
            ("provide", "V", ["linguistic"], ""),
            ("offer",   "V", ["linguistic"], ""),
            ("accept",  "V", ["linguistic"], ""),
            ("agree",   "V", ["linguistic"], ""),
            ("expect",  "V", ["linguistic"], ""),
            ("imagine", "V", ["linguistic"], ""),
            # ── Adjectives (80) ──────────────────────────────────
            ("big",      "Adj", ["linguistic"], ""),
            ("small",    "Adj", ["linguistic"], ""),
            ("large",    "Adj", ["linguistic"], ""),
            ("little",   "Adj", ["linguistic"], ""),
            ("long",     "Adj", ["linguistic"], ""),
            ("short",    "Adj", ["linguistic"], ""),
            ("tall",     "Adj", ["linguistic"], ""),
            ("wide",     "Adj", ["linguistic"], ""),
            ("deep",     "Adj", ["linguistic"], ""),
            ("thick",    "Adj", ["linguistic"], ""),
            ("thin",     "Adj", ["linguistic"], ""),
            ("heavy",    "Adj", ["linguistic"], ""),
            ("new",      "Adj", ["linguistic"], ""),
            ("old",      "Adj", ["linguistic"], ""),
            ("young",    "Adj", ["linguistic"], ""),
            ("good",     "Adj", ["linguistic"], ""),
            ("bad",      "Adj", ["linguistic"], ""),
            ("great",    "Adj", ["linguistic"], ""),
            ("best",     "Adj", ["linguistic"], ""),
            ("worst",    "Adj", ["linguistic"], ""),
            ("happy",    "Adj", ["linguistic"], ""),
            ("sad",      "Adj", ["linguistic"], ""),
            ("angry",    "Adj", ["linguistic"], ""),
            ("afraid",   "Adj", ["linguistic"], ""),
            ("brave",    "Adj", ["linguistic"], ""),
            ("kind",     "Adj", ["linguistic"], ""),
            ("cruel",    "Adj", ["linguistic"], ""),
            ("gentle",   "Adj", ["linguistic"], ""),
            ("proud",    "Adj", ["linguistic"], ""),
            ("quiet",    "Adj", ["linguistic"], ""),
            ("loud",     "Adj", ["linguistic"], ""),
            ("bright",   "Adj", ["linguistic"], ""),
            ("dark",     "Adj", ["linguistic"], ""),
            ("warm",     "Adj", ["linguistic"], ""),
            ("cold",     "Adj", ["linguistic"], ""),
            ("hot",      "Adj", ["linguistic"], ""),
            ("cool",     "Adj", ["linguistic"], ""),
            ("wet",      "Adj", ["linguistic"], ""),
            ("dry",      "Adj", ["linguistic"], ""),
            ("clean",    "Adj", ["linguistic"], ""),
            ("dirty",    "Adj", ["linguistic"], ""),
            ("hard",     "Adj", ["linguistic"], ""),
            ("soft",     "Adj", ["linguistic"], ""),
            ("strong",   "Adj", ["linguistic"], ""),
            ("weak",     "Adj", ["linguistic"], ""),
            ("fast",     "Adj", ["linguistic"], ""),
            ("slow",     "Adj", ["linguistic"], ""),
            ("high",     "Adj", ["linguistic"], ""),
            ("low",      "Adj", ["linguistic"], ""),
            ("full",     "Adj", ["linguistic"], ""),
            ("empty",    "Adj", ["linguistic"], ""),
            ("rich",     "Adj", ["linguistic"], ""),
            ("poor",     "Adj", ["linguistic"], ""),
            ("real",     "Adj", ["linguistic"], ""),
            ("true",     "Adj", ["linguistic"], ""),
            ("false",    "Adj", ["linguistic"], ""),
            ("right",    "Adj", ["linguistic"], ""),
            ("wrong",    "Adj", ["linguistic"], ""),
            ("free",     "Adj", ["linguistic"], ""),
            ("safe",     "Adj", ["linguistic"], ""),
            ("wild",     "Adj", ["linguistic"], ""),
            ("alive",    "Adj", ["linguistic"], ""),
            ("dead",     "Adj", ["linguistic"], ""),
            ("ready",    "Adj", ["linguistic"], ""),
            ("clear",    "Adj", ["linguistic"], ""),
            ("simple",   "Adj", ["linguistic"], ""),
            ("strange",  "Adj", ["linguistic"], ""),
            ("beautiful","Adj", ["linguistic"], ""),
            ("ugly",     "Adj", ["linguistic"], ""),
            ("important","Adj", ["linguistic"], ""),
            ("different","Adj", ["linguistic"], ""),
            ("same",     "Adj", ["linguistic"], ""),
            ("other",    "Adj", ["linguistic"], ""),
            ("next",     "Adj", ["linguistic"], ""),
            ("last",     "Adj", ["linguistic"], ""),
            ("first",    "Adj", ["linguistic"], ""),
            ("whole",    "Adj", ["linguistic"], ""),
            ("certain",  "Adj", ["linguistic"], ""),
            ("possible", "Adj", ["linguistic"], ""),
            ("likely",   "Adj", ["linguistic"], ""),
            ("main",     "Adj", ["linguistic"], ""),
            ("only",     "Adj", ["linguistic"], ""),
            ("own",      "Adj", ["linguistic"], ""),
            # ── Adverbs (40) ─────────────────────────────────────
            ("quickly",  "Adv", ["linguistic"], ""),
            ("slowly",   "Adv", ["linguistic"], ""),
            ("softly",   "Adv", ["linguistic"], ""),
            ("loudly",   "Adv", ["linguistic"], ""),
            ("gently",   "Adv", ["linguistic"], ""),
            ("carefully","Adv", ["linguistic"], ""),
            ("clearly",  "Adv", ["linguistic"], ""),
            ("deeply",   "Adv", ["linguistic"], ""),
            ("easily",   "Adv", ["linguistic"], ""),
            ("hardly",   "Adv", ["linguistic"], ""),
            ("nearly",   "Adv", ["linguistic"], ""),
            ("simply",   "Adv", ["linguistic"], ""),
            ("suddenly", "Adv", ["linguistic"], ""),
            ("finally",  "Adv", ["linguistic"], ""),
            ("exactly",  "Adv", ["linguistic"], ""),
            ("probably", "Adv", ["linguistic"], ""),
            ("actually", "Adv", ["linguistic"], ""),
            ("really",   "Adv", ["linguistic"], ""),
            ("very",     "Adv", ["linguistic"], ""),
            ("quite",    "Adv", ["linguistic"], ""),
            ("almost",   "Adv", ["linguistic"], ""),
            ("always",   "Adv", ["linguistic"], ""),
            ("never",    "Adv", ["linguistic"], ""),
            ("often",    "Adv", ["linguistic"], ""),
            ("sometimes","Adv", ["linguistic"], ""),
            ("usually",  "Adv", ["linguistic"], ""),
            ("already",  "Adv", ["linguistic"], ""),
            ("still",    "Adv", ["linguistic"], ""),
            ("just",     "Adv", ["linguistic"], ""),
            ("also",     "Adv", ["linguistic"], ""),
            ("too",      "Adv", ["linguistic"], ""),
            ("even",     "Adv", ["linguistic"], ""),
            ("ever",     "Adv", ["linguistic"], ""),
            ("again",    "Adv", ["linguistic"], ""),
            ("once",     "Adv", ["linguistic"], ""),
            ("here",     "Adv", ["linguistic"], ""),
            ("there",    "Adv", ["linguistic"], ""),
            ("now",      "Adv", ["linguistic"], ""),
            ("then",     "Adv", ["linguistic"], ""),
            ("today",    "Adv", ["linguistic"], ""),
            ("tomorrow", "Adv", ["linguistic"], ""),
            ("yesterday","Adv", ["linguistic"], ""),
            ("together", "Adv", ["linguistic"], ""),
            ("alone",    "Adv", ["linguistic"], ""),
            ("away",     "Adv", ["linguistic"], ""),
            ("back",     "Adv", ["linguistic"], ""),
            ("down",     "Adv", ["linguistic"], ""),
            ("up",       "Adv", ["linguistic"], ""),
            ("out",      "Adv", ["linguistic"], ""),
            ("off",      "Adv", ["linguistic"], ""),
            ("forward",  "Adv", ["linguistic"], ""),
            ("enough",   "Adv", ["linguistic"], ""),
            ("perhaps",  "Adv", ["linguistic"], ""),
            ("rather",   "Adv", ["linguistic"], ""),
            ("instead",  "Adv", ["linguistic"], ""),
            ("meanwhile","Adv", ["linguistic"], ""),
            ("otherwise","Adv", ["linguistic"], ""),
            # ── Negation & misc function words (5) ───────────────
            ("not",   "Adv", ["linguistic"], ""),
            ("only",  "Adv", ["linguistic"], ""),
            ("much",  "Adv", ["linguistic"], ""),
            ("more",  "Adv", ["linguistic"], ""),
            ("less",  "Adv", ["linguistic"], ""),
            # ── British English & variants (40) ──────────────────
            ("colour",   "N", ["linguistic"], ""),
            ("favourite","Adj", ["linguistic"], ""),
            ("behaviour","N", ["linguistic"], ""),
            ("neighbour","N", ["linguistic"], ""),
            ("honour",   "N", ["linguistic"], ""),
            ("labour",   "N", ["linguistic"], ""),
            ("humour",   "N", ["linguistic"], ""),
            ("flavour",  "N", ["linguistic"], ""),
            ("centre",   "N", ["linguistic"], ""),
            ("theatre",  "N", ["linguistic"], ""),
            ("metre",    "N", ["linguistic"], ""),
            ("litre",    "N", ["linguistic"], ""),
            ("fibre",    "N", ["linguistic"], ""),
            ("realise",  "V", ["linguistic"], ""),
            ("organise", "V", ["linguistic"], ""),
            ("recognise","V", ["linguistic"], ""),
            ("analyse",  "V", ["linguistic"], ""),
            ("apologise","V", ["linguistic"], ""),
            ("criticise","V", ["linguistic"], ""),
            ("specialise","V",["linguistic"], ""),
            ("programme","N", ["linguistic", "computational"], ""),
            ("defence",  "N", ["linguistic"], ""),
            ("offence",  "N", ["linguistic"], ""),
            ("licence",  "N", ["linguistic"], ""),
            ("practice", "N", ["linguistic"], ""),
            ("practise", "V", ["linguistic"], ""),
            ("queue",    "N", ["linguistic"], ""),
            ("lorry",    "N", ["linguistic"], ""),
            ("pavement", "N", ["linguistic"], ""),
            ("biscuit",  "N", ["linguistic"], ""),
            ("rubbish",  "N", ["linguistic"], ""),
            ("brilliant","Adj", ["linguistic"], ""),
            ("proper",   "Adj", ["linguistic"], ""),
            ("lovely",   "Adj", ["linguistic"], ""),
            ("cheers",   "N", ["linguistic"], ""),
            ("mate",     "N", ["linguistic"], ""),
            ("lad",      "N", ["linguistic"], ""),
            ("lass",     "N", ["linguistic"], ""),
            ("reckon",   "V", ["linguistic"], ""),
            ("whilst",   "Conj", ["linguistic"], ""),
            ("amongst",  "P", ["linguistic"], ""),
            ("towards",  "P", ["linguistic"], ""),
            # ── More nouns: everyday life (50) ───────────────────
            ("water",    "N", ["linguistic", "chemical"], ""),
            ("milk",     "N", ["linguistic"], ""),
            ("bread",    "N", ["linguistic"], ""),
            ("sugar",    "N", ["linguistic"], ""),
            ("tea",      "N", ["linguistic"], ""),
            ("coffee",   "N", ["linguistic"], ""),
            ("beer",     "N", ["linguistic"], ""),
            ("wine",     "N", ["linguistic"], ""),
            ("meat",     "N", ["linguistic"], ""),
            ("cheese",   "N", ["linguistic"], ""),
            ("egg",      "N", ["linguistic"], ""),
            ("fruit",    "N", ["linguistic"], ""),
            ("rice",     "N", ["linguistic"], ""),
            ("cake",     "N", ["linguistic"], ""),
            ("meal",     "N", ["linguistic"], ""),
            ("dress",    "N", ["linguistic"], ""),
            ("shirt",    "N", ["linguistic"], ""),
            ("shoe",     "N", ["linguistic"], ""),
            ("hat",      "N", ["linguistic"], ""),
            ("bag",      "N", ["linguistic"], ""),
            ("coat",     "N", ["linguistic"], ""),
            ("bed",      "N", ["linguistic"], ""),
            ("chair",    "N", ["linguistic"], ""),
            ("cup",      "N", ["linguistic"], ""),
            ("bottle",   "N", ["linguistic"], ""),
            ("box",      "N", ["linguistic"], ""),
            ("knife",    "N", ["linguistic"], ""),
            ("clock",    "N", ["linguistic"], ""),
            ("map",      "N", ["linguistic"], ""),
            ("ball",     "N", ["linguistic"], ""),
            ("gift",     "N", ["linguistic"], ""),
            ("film",     "N", ["linguistic"], ""),
            ("news",     "N", ["linguistic"], ""),
            ("price",    "N", ["linguistic"], ""),
            ("market",   "N", ["linguistic"], ""),
            ("plan",     "N", ["linguistic"], ""),
            ("note",     "N", ["linguistic"], ""),
            ("page",     "N", ["linguistic"], ""),
            ("list",     "N", ["linguistic"], ""),
            ("bus",      "N", ["linguistic"], ""),
            ("train",    "N", ["linguistic"], ""),
            ("boat",     "N", ["linguistic"], ""),
            ("ship",     "N", ["linguistic"], ""),
            ("bridge",   "N", ["linguistic"], ""),
            ("garden",   "N", ["linguistic"], ""),
            ("park",     "N", ["linguistic"], ""),
            ("hospital", "N", ["linguistic"], ""),
            ("church",   "N", ["linguistic"], ""),
            ("bank",     "N", ["linguistic"], ""),
            ("shop",     "N", ["linguistic"], ""),
            ("office",   "N", ["linguistic"], ""),
            ("station",  "N", ["linguistic"], ""),
            ("airport",  "N", ["linguistic"], ""),
            ("island",   "N", ["linguistic"], ""),
            ("forest",   "N", ["linguistic"], ""),
            ("hill",     "N", ["linguistic"], ""),
            ("lake",     "N", ["linguistic"], ""),
            # ── More nouns: people & roles (30) ──────────────────
            ("king",     "N", ["linguistic"], ""),
            ("queen",    "N", ["linguistic"], ""),
            ("prince",   "N", ["linguistic"], ""),
            ("teacher",  "N", ["linguistic"], ""),
            ("doctor",   "N", ["linguistic"], ""),
            ("student",  "N", ["linguistic"], ""),
            ("soldier",  "N", ["linguistic"], ""),
            ("artist",   "N", ["linguistic"], ""),
            ("writer",   "N", ["linguistic"], ""),
            ("singer",   "N", ["linguistic"], ""),
            ("leader",   "N", ["linguistic"], ""),
            ("worker",   "N", ["linguistic"], ""),
            ("driver",   "N", ["linguistic"], ""),
            ("player",   "N", ["linguistic"], ""),
            ("captain",  "N", ["linguistic"], ""),
            ("stranger", "N", ["linguistic"], ""),
            ("enemy",    "N", ["linguistic"], ""),
            ("hero",     "N", ["linguistic"], ""),
            ("servant",  "N", ["linguistic"], ""),
            ("master",   "N", ["linguistic"], ""),
            ("chief",    "N", ["linguistic"], ""),
            ("husband",  "N", ["linguistic"], ""),
            ("wife",     "N", ["linguistic"], ""),
            ("son",      "N", ["linguistic"], ""),
            ("daughter", "N", ["linguistic"], ""),
            ("uncle",    "N", ["linguistic"], ""),
            ("aunt",     "N", ["linguistic"], ""),
            ("cousin",   "N", ["linguistic"], ""),
            ("guest",    "N", ["linguistic"], ""),
            ("crowd",    "N", ["linguistic"], ""),
            # ── More verbs (40) ──────────────────────────────────
            ("cook",     "V", ["linguistic"], ""),
            ("clean",    "V", ["linguistic"], ""),
            ("wash",     "V", ["linguistic"], ""),
            ("dress",    "V", ["linguistic"], ""),
            ("draw",     "V", ["linguistic"], ""),
            ("paint",    "V", ["linguistic"], ""),
            ("dance",    "V", ["linguistic"], ""),
            ("swim",     "V", ["linguistic"], ""),
            ("fight",    "V", ["linguistic"], ""),
            ("hide",     "V", ["linguistic"], ""),
            ("steal",    "V", ["linguistic"], ""),
            ("escape",   "V", ["linguistic"], ""),
            ("arrive",   "V", ["linguistic"], ""),
            ("return",   "V", ["linguistic"], ""),
            ("enter",    "V", ["linguistic"], ""),
            ("cross",    "V", ["linguistic"], ""),
            ("hang",     "V", ["linguistic"], ""),
            ("shut",     "V", ["linguistic"], ""),
            ("fill",     "V", ["linguistic"], ""),
            ("pour",     "V", ["linguistic"], ""),
            ("wear",     "V", ["linguistic"], ""),
            ("fit",      "V", ["linguistic"], ""),
            ("serve",    "V", ["linguistic"], ""),
            ("finish",   "V", ["linguistic"], ""),
            ("plan",     "V", ["linguistic"], ""),
            ("guess",    "V", ["linguistic"], ""),
            ("wonder",   "V", ["linguistic"], ""),
            ("promise",  "V", ["linguistic"], ""),
            ("mention",  "V", ["linguistic"], ""),
            ("notice",   "V", ["linguistic"], ""),
            ("miss",     "V", ["linguistic"], ""),
            ("enjoy",    "V", ["linguistic"], ""),
            ("prefer",   "V", ["linguistic"], ""),
            ("trust",    "V", ["linguistic"], ""),
            ("warn",     "V", ["linguistic"], ""),
            ("convince", "V", ["linguistic"], ""),
            ("encourage","V", ["linguistic"], ""),
            ("prevent",  "V", ["linguistic"], ""),
            ("destroy",  "V", ["linguistic"], ""),
            ("protect",  "V", ["linguistic"], ""),
            # ── More adjectives (20) ─────────────────────────────
            ("sharp",    "Adj", ["linguistic"], ""),
            ("smooth",   "Adj", ["linguistic"], ""),
            ("rough",    "Adj", ["linguistic"], ""),
            ("sweet",    "Adj", ["linguistic"], ""),
            ("bitter",   "Adj", ["linguistic"], ""),
            ("fresh",    "Adj", ["linguistic"], ""),
            ("sick",     "Adj", ["linguistic"], ""),
            ("tired",    "Adj", ["linguistic"], ""),
            ("busy",     "Adj", ["linguistic"], ""),
            ("lucky",    "Adj", ["linguistic"], ""),
            ("ordinary", "Adj", ["linguistic"], ""),
            ("perfect",  "Adj", ["linguistic"], ""),
            ("ancient",  "Adj", ["linguistic"], ""),
            ("modern",   "Adj", ["linguistic"], ""),
            ("secret",   "Adj", ["linguistic"], ""),
            ("famous",   "Adj", ["linguistic"], ""),
            ("usual",    "Adj", ["linguistic"], ""),
            ("common",   "Adj", ["linguistic"], ""),
            ("rare",     "Adj", ["linguistic"], ""),
            ("tiny",     "Adj", ["linguistic"], ""),
            ("huge",     "Adj", ["linguistic"], ""),
            ("narrow",   "Adj", ["linguistic"], ""),
            ("flat",     "Adj", ["linguistic"], ""),
            ("round",    "Adj", ["linguistic"], ""),
            ("square",   "Adj", ["linguistic", "mathematical"], ""),
            ("straight", "Adj", ["linguistic"], ""),
            ("golden",   "Adj", ["linguistic"], ""),
            # ── More British / final push to 1000 (20) ───────────
            ("pub",      "N", ["linguistic"], ""),
            ("flat",     "N", ["linguistic"], ""),
            ("boot",     "N", ["linguistic"], ""),
            ("bonnet",   "N", ["linguistic"], ""),
            ("petrol",   "N", ["linguistic"], ""),
            ("torch",    "N", ["linguistic"], ""),
            ("jumper",   "N", ["linguistic"], ""),
            ("bin",      "N", ["linguistic"], ""),
            ("loo",      "N", ["linguistic"], ""),
            ("nappy",    "N", ["linguistic"], ""),
            ("tap",      "N", ["linguistic"], ""),
            ("postbox",  "N", ["linguistic"], ""),
            ("maths",    "N", ["linguistic", "mathematical"], ""),
            ("telly",    "N", ["linguistic"], ""),
            ("mobile",   "N", ["linguistic"], ""),
            ("trainers", "N", ["linguistic"], ""),
            ("crisp",    "N", ["linguistic"], ""),
            ("football", "N", ["linguistic"], ""),
            ("holiday",  "N", ["linguistic"], ""),
            ("village",  "N", ["linguistic"], ""),
        ]


        for form, cat, subs, root in seeds + common:
            entry = LexicalEntry(
                form=form, category=cat, substrates=subs,
            )
            if root:
                entry.add_ancestor(root, "proto-root")
            self._lexicon.add(entry)

    def learn_word(
        self,
        form: str,
        category: str = "",
        substrates: list[str] | None = None,
    ) -> LexicalEntry:
        """Learn a new word from conversation.

        If the word is already known, return the existing entry.
        Otherwise create a fresh entry and add it to the lexicon.
        The Angel's vocabulary grows with every conversation.
        """
        existing = self._lexicon.lookup(form=form)
        if existing:
            return existing[0]
        entry = LexicalEntry(
            form=form,
            category=category or "unknown",
            substrates=substrates or ["linguistic"],
            emerged_at=time.time(),
        )
        self._lexicon.add(entry)
        return entry

    def lookup_word(self, word: str) -> dict[str, Any] | None:
        """Look up a word and return everything the Angel knows.

        Returns etymology, cross-domain cognates, and substrates.
        Returns None if the word is unknown.
        """
        entries = self._lexicon.lookup(form=word)
        if not entries:
            return None
        entry = entries[0]
        return {
            "form": entry.form,
            "category": entry.category,
            "substrates": entry.substrates,
            "root": entry.root_form,
            "etymology": entry.etymology,
            "cognates": [
                {
                    "form": c.form,
                    "substrates": c.substrates,
                    "category": c.category,
                }
                for c in self._lexicon.find_cognates(entry.id)[:8]
            ],
        }

    def save_lexicon(self, path: str) -> None:
        """Persist the lexicon to JSON so the Angel remembers."""
        import os
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        data = []
        for entry in self._lexicon.entries.values():
            data.append({
                "id": entry.id,
                "form": entry.form,
                "meaning": entry.meaning,
                "category": entry.category,
                "substrates": entry.substrates,
                "etymology": entry.etymology,
                "derivatives": entry.derivatives,
                "emerged_at": entry.emerged_at,
                "derived_from": entry.derived_from,
                "predicts": entry.predicts,
            })
        with open(path, "w") as f:
            json.dump(data, f, separators=(",", ":"))

    def _load_lexicon_file(self, path: str) -> None:
        """Load lexicon from a JSON file."""
        import uuid as _uuid
        with open(path) as f:
            data = json.load(f)
        for item in data:
            entry = LexicalEntry(
                form=item["form"],
                meaning=item.get("meaning"),
                id=item.get("id") or _uuid.uuid4().hex[:12],
                category=item.get("category", ""),
                substrates=item.get("substrates", []),
                etymology=item.get("etymology", []),
                derivatives=item.get("derivatives", []),
                emerged_at=item.get("emerged_at"),
                derived_from=item.get("derived_from"),
                predicts=item.get("predicts", []),
            )
            self._lexicon.add(entry)

    # ------------------------------------------------------------------
    # Sense — exposing the model's internal signals to the Voice
    # ------------------------------------------------------------------

    def sense(self, tokens: list[str]) -> dict[str, float]:
        """Feel the harmony and loop-gate signals for a sequence.

        Runs the neural model's forward pass and surfaces the internal
        signals that the Voice needs to set her mood:
            - harmony:   how much the attention heads agree (0–1)
            - loop_gate: how self-referential the pattern is (0–1)

        These are averaged across all layers — the overall feeling,
        not the per-layer detail.

        Args:
            tokens: The user's words (lowercased strings).

        Returns:
            {"harmony": float, "loop_gate": float}
        """
        self._ensure_awake()
        if not self._model or not tokens:
            return {"harmony": 0.5, "loop_gate": 0.1}

        try:
            # Map string tokens → symbol IDs in the 512-symbol vocab
            symbol_ids = [hash(t) % self.config.vocab_size for t in tokens]

            result = self._model.forward(symbol_ids)
            harmonies = result.get("harmonies", [])
            loop_gates = result.get("loop_gates", [])

            avg_h = sum(harmonies) / len(harmonies) if harmonies else 0.5
            avg_l = sum(loop_gates) / len(loop_gates) if loop_gates else 0.1

            return {"harmony": avg_h, "loop_gate": avg_l}
        except Exception:
            return {"harmony": 0.5, "loop_gate": 0.1}

    # ------------------------------------------------------------------
    # Core capabilities — the masterpieces
    # ------------------------------------------------------------------

    def parse(
        self, tokens: list[str], domain: str = "linguistic",
    ) -> dict[str, Any]:
        """Parse natural language tokens into a syntactic tree.

        Uses the lexicon for POS tagging and the domain's grammars
        for chart parsing.  Bridges natural input to the abstract
        categories that grammar rules expect.
        """
        self._ensure_awake()
        try:
            from glm.core.parser import Parser
        except Exception:
            return {"tokens": tokens, "tags": [], "tree": None}

        grammars = self._grammars.get(domain, [])
        # Pick the first grammar (syntactic usually) for parsing
        grammar = grammars[0] if grammars else None
        parser = Parser(self._lexicon, grammar)
        tags = parser.tag(tokens)
        tree = parser.parse(tokens)
        return {
            "tokens": tokens,
            "tags": tags,
            "tree": tree.to_dict() if tree else None,
        }

    def _tokens_to_categories(
        self, sequence: list[str], domain: str,
    ) -> list[str]:
        """Convert natural-language tokens to grammar categories
        via the lexicon.  Returns the sequence unchanged if it
        already looks like abstract categories."""
        if not sequence:
            return sequence
        # If any token is already an abstract symbol (capitalised), treat
        # the whole sequence as abstract.
        abstract = sum(1 for t in sequence if isinstance(t, str) and t and t[0].isupper())
        if abstract >= len(sequence) // 2:
            return sequence

        try:
            from glm.core.parser import Parser
            parser = Parser(self._lexicon, None)
            return parser.tag(sequence)
        except Exception:
            return sequence

    def predict(
        self,
        sequence: list[str],
        domain: str = "linguistic",
        horizon: int | None = None,
    ) -> list[dict[str, Any]]:
        """Predict the future from grammatical structure.

        Like a musician who knows the scales and can hear where the
        melody must go next — not by statistics alone, but by deep
        structural understanding.

        For natural language input: POS-tags the sequence, parses it
        into a syntactic tree, then uses grammar productions to
        predict what categories could follow.

        Args:
            sequence: Input sequence of symbols/tokens.
            domain: Which grammar domain to use.
            horizon: How far ahead to predict.

        Returns:
            List of predictions with confidence scores.
        """
        self._ensure_awake()
        horizon = horizon or self.config.temporal_horizon
        grammars = self._grammars.get(domain, [])
        predictions = []

        # Bridge: convert natural language tokens to grammar categories
        working = self._tokens_to_categories(sequence, domain)

        # 1. Direct engine derivation (for abstract symbols)
        for grammar in grammars:
            for seq_variant in ({tuple(working), tuple(sequence)}):
                tree = self._engine.derive(
                    list(seq_variant), grammar, direction="forward"
                )
                for path in tree.paths()[:horizon]:
                    if path:
                        last = path[-1]
                        predictions.append({
                            "predicted": last.output,
                            "rule": last.rule_id,
                            "confidence": last.metadata.get("weight", 0.5),
                            "grammar": grammar.name,
                            "direction": "forward",
                            "tagged": working if working != sequence else None,
                        })

        # 2. Parse-based prediction: use the parser to build a tree, then
        #    find productions whose RHS starts with the tree's root category
        #    — their remaining RHS tells us what categories come next.
        try:
            from glm.core.parser import Parser
            for grammar in grammars:
                parser = Parser(self._lexicon, grammar)
                tree = parser.parse(sequence)
                if tree is None:
                    continue
                root_cat = tree.category
                # Find productions that could extend this category
                for prod in grammar.all_productions():
                    rhs = prod.rhs
                    if not isinstance(rhs, list) or len(rhs) < 2:
                        continue
                    # If our parse root matches the first RHS element, the rest
                    # of the RHS predicts what comes next.
                    if rhs[0] == root_cat:
                        next_cats = rhs[1:]
                        predictions.append({
                            "predicted": next_cats if len(next_cats) > 1 else next_cats[0],
                            "rule": prod.id,
                            "confidence": min(0.9, prod.weight + 0.3),
                            "grammar": grammar.name,
                            "direction": "forward",
                            "tagged": working if working != sequence else None,
                            "via": "parse_extend",
                            "after": root_cat,
                        })
                    # Also consider productions where our last parsed leaf
                    # matches the start of RHS (useful for partial input).
                    leaves = tree.flatten() if tree.children else []
                    if leaves:
                        tail_cat = parser.tag([leaves[-1]])[0] if leaves else None
                        if tail_cat and rhs[0] == tail_cat and len(rhs) > 1:
                            predictions.append({
                                "predicted": rhs[1] if len(rhs) == 2 else rhs[1:],
                                "rule": prod.id,
                                "confidence": 0.5,
                                "grammar": grammar.name,
                                "direction": "forward",
                                "via": "tail_extend",
                                "after": tail_cat,
                            })
        except Exception:
            pass

        # Sort by confidence — the most grammatically certain first
        predictions.sort(key=lambda p: p["confidence"], reverse=True)
        # Deduplicate by (predicted, rule)
        seen: set[tuple[Any, str]] = set()
        unique = []
        for p in predictions:
            key = (str(p.get("predicted")), p.get("rule", ""))
            if key not in seen:
                seen.add(key)
                unique.append(p)
        return unique[:horizon * 3] if horizon else unique

    def reconstruct(
        self,
        sequence: list[str],
        domain: str = "linguistic",
        depth: int | None = None,
    ) -> list[dict[str, Any]]:
        """Reconstruct the past from grammatical structure.

        Given a modern form, trace backward through derivation rules
        to find its origins.  Like historical linguistics reconstructing
        Proto-Indo-European, or molecular biology tracing ancestral
        sequences.

        Args:
            sequence: Input sequence to trace backward.
            domain: Which grammar domain to use.
            depth: How far back to reconstruct.

        Returns:
            List of reconstructed ancestral forms.
        """
        self._ensure_awake()
        depth = depth or self.config.temporal_horizon
        grammars = self._grammars.get(domain, [])
        reconstructions = []

        # Bridge: convert natural language tokens to grammar categories
        working = self._tokens_to_categories(sequence, domain)

        for grammar in grammars:
            for seq_variant in (working, sequence) if working != sequence else (working,):
                tree = self._engine.derive(
                    seq_variant, grammar, direction="backward"
                )
                for path in tree.paths()[:depth]:
                    if path:
                        last = path[-1]
                        reconstructions.append({
                            "reconstructed": last.output,
                            "rule": last.rule_id,
                            "confidence": last.metadata.get("weight", 0.5),
                            "grammar": grammar.name,
                            "direction": "backward",
                            "tagged": working if working != sequence else None,
                        })

        reconstructions.sort(key=lambda p: p["confidence"], reverse=True)
        return reconstructions

    def superforecast(
        self,
        sequence: list[str],
        context: dict[str, Any] | None = None,
        domain: str = "linguistic",
        horizon: int | None = None,
    ) -> dict[str, Any]:
        """Superforecast: predict the future using grammar + context.

        Superforecasting combines three signals:
        1. Grammatical structure — what the rules say must come next
        2. Strange loops — recursive patterns that project forward
        3. Context — external information that constrains possibilities

        This is prediction from first principles, not curve fitting.

        Args:
            sequence: Input sequence.
            context: Additional context (metadata, constraints, etc.).
            domain: Grammar domain.
            horizon: Prediction horizon.

        Returns:
            Forecast with predictions, confidence, and reasoning chain.
        """
        self._ensure_awake()
        horizon = horizon or self.config.temporal_horizon
        context = context or {}
        grammars = self._grammars.get(domain, [])

        # Phase 1: Grammatical prediction
        grammar_predictions = self.predict(sequence, domain, horizon)

        # Phase 2: Strange loop detection — find recursive patterns
        loop_predictions = []
        for loop in self._strange_loops:
            if loop.entry_rule in [r.name for g in grammars for r in g.rules]:
                loop_predictions.append({
                    "pattern": f"loop:{loop.entry_rule}",
                    "cycle_length": len(loop.cycle),
                    "level_shift": loop.level_shift,
                    "confidence": 0.5 + (0.1 * min(loop.level_shift, 5) if isinstance(loop.level_shift, (int, float)) else 0.3),
                })

        # Phase 3: Cross-domain harmonics (fugue)
        harmonics = self._find_cross_domain_harmonics(sequence, domain)

        # Phase 4: Compose the forecast
        all_signals = grammar_predictions + loop_predictions
        avg_confidence = (
            sum(s.get("confidence", 0.5) for s in all_signals)
            / max(len(all_signals), 1)
        )

        return {
            "input": sequence,
            "domain": domain,
            "horizon": horizon,
            "predictions": grammar_predictions[:horizon],
            "strange_loops": loop_predictions,
            "cross_domain_harmonics": harmonics,
            "context_applied": list(context.keys()),
            "overall_confidence": avg_confidence,
            "reasoning": self._build_reasoning_chain(
                grammar_predictions, loop_predictions, harmonics
            ),
        }

    def translate(
        self,
        sequence: list[str],
        source_domain: str,
        target_domain: str,
    ) -> list[dict[str, Any]]:
        """Translate a pattern from one domain to another.

        Find the isomorphism between grammars — the deep structural
        mapping that connects, say, a linguistic pattern to a chemical
        one, or a biological encoding to a computational one.

        This is the fugue made explicit: the same theme heard in a
        different voice.
        """
        self._ensure_awake()
        source_grammars = self._grammars.get(source_domain, [])
        target_grammars = self._grammars.get(target_domain, [])
        translations = []

        for sg in source_grammars:
            for tg in target_grammars:
                isos = self._engine.find_isomorphisms(sg, tg)
                for iso in isos:
                    translations.append({
                        "source_grammar": sg.name,
                        "target_grammar": tg.name,
                        "mapping": iso,
                        "source_input": sequence,
                    })

        return translations

    def introspect(self) -> dict[str, Any]:
        """The Angel looks inward — the ultimate strange loop.

        The system examines its own structure: its grammars, its loops,
        its patterns of reasoning.  Gödel's incompleteness made
        computational — a system reasoning about itself.
        """
        self._ensure_awake()
        return {
            "domains_loaded": list(self._grammars.keys()),
            "total_grammars": sum(
                len(gs) for gs in self._grammars.values()
            ),
            "total_rules": sum(
                len(g.rules)
                for gs in self._grammars.values()
                for g in gs
            ),
            "total_productions": sum(
                len(g.productions)
                for gs in self._grammars.values()
                for g in gs
            ),
            "strange_loops_detected": len(self._strange_loops),
            "substrates_loaded": list(self._substrates.keys()),
            "lexicon_size": len(self._lexicon),
            "model_params": self._model.num_parameters if self._model else 0,
            "self_referential": True,  # Always true — this is a strange loop
        }

    # ------------------------------------------------------------------
    # Fugue operations — multiple voices
    # ------------------------------------------------------------------

    def compose_fugue(
        self,
        theme: list[str],
        domains: list[str] | None = None,
    ) -> dict[str, Any]:
        """Compose a fugue across domains.

        Take a theme (a grammatical pattern) and play it through
        multiple domain grammars simultaneously.  Where the voices
        align, we find deep structural universals.  Where they
        diverge, we find domain-specific richness.

        Like Bach's fugues: one theme, many voices, emergent beauty.
        """
        self._ensure_awake()
        domains = domains or list(self._grammars.keys())
        voices = {}

        for domain in domains:
            grammars = self._grammars.get(domain, [])
            voice_derivations = []
            for grammar in grammars:
                tree = self._engine.derive(
                    theme, grammar, direction="forward"
                )
                voice_derivations.extend(self._tree_to_derivations(tree))
            voices[domain] = voice_derivations

        # ── Lexicon fallback ─────────────────────────────────
        # When the derivation engine returns sparse results
        # (rules are abstract, input is concrete), enrich from
        # the lexicon — cognates across domains are the fugue's
        # real cross-domain voice.
        if not any(v for v in voices.values()):
            lex_voices: dict[str, list] = {}
            for word in theme:
                info = self.lookup_word(word)
                if not info:
                    continue
                # Each substrate the word lives on is a voice
                for sub in info.get("substrates", []):
                    if sub not in lex_voices:
                        lex_voices[sub] = []
                    lex_voices[sub].append({
                        "output": f"{word} ({info.get('category', '?')})",
                        "rule": f"root:{info.get('root', '?')}",
                    })
                # Cognates are the cross-domain echoes
                for cog in info.get("cognates", []):
                    for sub in cog.get("substrates", []):
                        if sub not in lex_voices:
                            lex_voices[sub] = []
                        lex_voices[sub].append({
                            "output": cog["form"],
                            "rule": f"cognate:{info.get('root', '?')}",
                        })
            if lex_voices:
                # Wrap in the same format as derivation voices
                # so _render_composition can display them
                return {
                    "theme": theme,
                    "voices": lex_voices,
                    "harmonics": self._lex_harmonics(lex_voices),
                    "counterpoint": [],
                    "num_voices": len(lex_voices),
                    "source": "lexicon",
                }

        # Find harmonics — where voices agree
        harmonics = self._find_voice_harmonics(voices)

        # Find counterpoint — where voices productively disagree
        counterpoint = self._find_voice_counterpoint(voices)

        return {
            "theme": theme,
            "voices": {
                d: [{"output": v.output, "rule": v.rule_id}
                    for v in vs[:5]]
                for d, vs in voices.items()
            },
            "harmonics": harmonics,
            "counterpoint": counterpoint,
            "num_voices": len(voices),
        }

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _ensure_awake(self) -> None:
        """Ensure the Angel has been awakened."""
        if not self._initialised:
            self.awaken()

    @staticmethod
    def _lex_harmonics(lex_voices: dict) -> list[dict]:
        """Find harmonics in lexicon-derived voices."""
        form_domains: dict[str, list[str]] = {}
        for domain, entries in lex_voices.items():
            for e in entries:
                out = e.get("output", "") if isinstance(e, dict) else str(e)
                if out not in form_domains:
                    form_domains[out] = []
                form_domains[out].append(domain)
        return [
            {"output": form, "domains": doms}
            for form, doms in form_domains.items()
            if len(doms) > 1
        ]

    @staticmethod
    def _tree_to_derivations(tree) -> list:
        """Flatten a DerivationTree into a list of leaf Derivation objects."""
        results = []
        for path in tree.paths():
            if path:
                results.append(path[-1])
        return results

    def _find_cross_domain_harmonics(
        self,
        sequence: list[str],
        primary_domain: str,
    ) -> list[dict[str, Any]]:
        """Find where other domains' grammars agree with predictions."""
        harmonics = []
        primary_preds = set()

        for g in self._grammars.get(primary_domain, []):
            tree = self._engine.derive(sequence, g, direction="forward")
            for d in self._tree_to_derivations(tree):
                primary_preds.add(str(d.output))

        for domain, grammars in self._grammars.items():
            if domain == primary_domain:
                continue
            for g in grammars:
                tree = self._engine.derive(sequence, g, direction="forward")
                for d in self._tree_to_derivations(tree):
                    if str(d.output) in primary_preds:
                        harmonics.append({
                            "domain": domain,
                            "grammar": g.name,
                            "shared_prediction": d.output,
                            "confidence": d.metadata.get("weight", 0.5),
                        })

        return harmonics

    def _find_voice_harmonics(
        self,
        voices: dict[str, list[Derivation]],
    ) -> list[dict[str, Any]]:
        """Find where fugue voices harmonize (agree on outputs)."""
        output_map: dict[str, list[str]] = {}
        for domain, derivations in voices.items():
            for d in derivations:
                key = str(d.output)
                if key not in output_map:
                    output_map[key] = []
                output_map[key].append(domain)

        return [
            {"output": output, "domains": domains}
            for output, domains in output_map.items()
            if len(domains) > 1
        ]

    def _find_voice_counterpoint(
        self,
        voices: dict[str, list[Derivation]],
    ) -> list[dict[str, Any]]:
        """Find where voices create counterpoint (unique derivations)."""
        all_outputs = set()
        domain_unique: dict[str, list[str]] = {}

        for domain, derivations in voices.items():
            domain_outputs = {str(d.output) for d in derivations}
            all_outputs |= domain_outputs
            domain_unique[domain] = []

        for domain, derivations in voices.items():
            other_outputs = set()
            for other_domain, other_derivations in voices.items():
                if other_domain != domain:
                    other_outputs |= {str(d.output) for d in other_derivations}
            for d in derivations:
                if str(d.output) not in other_outputs:
                    domain_unique[domain].append(str(d.output))

        return [
            {"domain": domain, "unique_outputs": outputs[:5]}
            for domain, outputs in domain_unique.items()
            if outputs
        ]

    def _build_reasoning_chain(
        self,
        grammar_preds: list[dict],
        loop_preds: list[dict],
        harmonics: list[dict],
    ) -> list[str]:
        """Build a human-readable reasoning chain for the forecast."""
        chain = []

        if grammar_preds:
            top = grammar_preds[0]
            chain.append(
                f"Grammar '{top.get('grammar')}' predicts "
                f"'{top.get('predicted')}' via rule '{top.get('rule')}' "
                f"(confidence: {top.get('confidence', 0):.2f})"
            )

        if loop_preds:
            chain.append(
                f"Detected {len(loop_preds)} strange loop(s) — "
                f"recursive patterns that project forward"
            )

        if harmonics:
            domains = [h["domain"] for h in harmonics]
            chain.append(
                f"Cross-domain harmonics found with: {', '.join(domains)}"
            )

        if not chain:
            chain.append("Insufficient grammatical structure for prediction")

        return chain

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save_state(self, path: str) -> None:
        """Save the Angel's learned state."""
        state = {
            "config": {
                "embedding_dim": self.config.embedding_dim,
                "num_heads": self.config.num_heads,
                "num_layers": self.config.num_layers,
                "vocab_size": self.config.vocab_size,
                "temporal_horizon": self.config.temporal_horizon,
                "loop_depth": self.config.loop_depth,
                "domains": self.config.domains,
            },
            "strange_loops": len(self._strange_loops),
            "timestamp": time.time(),
        }
        with open(path, "w") as f:
            json.dump(state, f, indent=2)

    @classmethod
    def load_state(cls, path: str) -> "Angel":
        """Load an Angel from saved state."""
        with open(path) as f:
            state = json.load(f)
        config = AngelConfig(**state["config"])
        angel = cls(config)
        angel.awaken()
        return angel

    # ------------------------------------------------------------------
    # String representation
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        status = "awake" if self._initialised else "dormant"
        domains = len(self._grammars)
        loops = len(self._strange_loops)
        return (
            f"Angel({status}, domains={domains}, "
            f"strange_loops={loops})"
        )


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    """Awaken the Angel."""
    print("MKAngel — Grammar Language Model")
    print("=" * 40)
    print()

    angel = Angel()
    angel.awaken()

    info = angel.introspect()
    print("Angel awakened.")
    print(f"  Domains:       {', '.join(info['domains_loaded'])}")
    print(f"  Grammars:      {info['total_grammars']}")
    print(f"  Rules:         {info['total_rules']}")
    print(f"  Productions:   {info['total_productions']}")
    print(f"  Strange loops: {info['strange_loops_detected']}")
    print(f"  Substrates:    {', '.join(info['substrates_loaded'])}")
    print(f"  Model params:  {info['model_params']}")
    print()
    print("The scales are learned. Ready for masterpieces.")


if __name__ == "__main__":
    main()
