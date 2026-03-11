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
        for form, cat, subs, root in seeds:
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

        for grammar in grammars:
            tree = self._engine.derive(
                sequence, grammar, direction="forward"
            )
            # Extract derivation paths from the tree
            for path in tree.paths()[:horizon]:
                if path:
                    last = path[-1]
                    predictions.append({
                        "predicted": last.output,
                        "rule": last.rule_id,
                        "confidence": last.metadata.get("weight", 0.5),
                        "grammar": grammar.name,
                        "direction": "forward",
                    })

        # Sort by confidence — the most grammatically certain first
        predictions.sort(key=lambda p: p["confidence"], reverse=True)
        return predictions

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

        for grammar in grammars:
            tree = self._engine.derive(
                sequence, grammar, direction="backward"
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
