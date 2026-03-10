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
        then the grammars (the rules), then the model (the mind).
        Like a child learning scales before playing Bach.
        """
        self._load_substrates()
        self._load_grammars()
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
