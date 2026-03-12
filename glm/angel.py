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
from glm.core.reasoning import ReasoningEngine, ReasoningChain, ReasoningStep

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
from glm.grammars.reasoning import build_reasoning_grammar

from glm.substrates.phonological import PhonologicalSubstrate
from glm.substrates.morphological import MorphologicalSubstrate
from glm.substrates.molecular import MolecularSubstrate
from glm.substrates.symbolic import SymbolicSubstrate
from glm.substrates.mathematical import MathSubstrate

from glm.model.glm import GrammarLanguageModel, GLMConfig
from glm.mnemo.language import encode as mnemo_encode, decode as mnemo_decode


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
        "reasoning",
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

    # Basic word → grammatical category map for bridging raw text
    # to formal grammar symbols.  Grows as the Angel learns.
    _DEFAULT_CATEGORIES: dict[str, str] = {
        # Determiners
        "the": "Det", "a": "Det", "an": "Det", "this": "Det", "that": "Det",
        "these": "Det", "those": "Det", "my": "Det", "your": "Det",
        "his": "Det", "her": "Det", "its": "Det", "our": "Det", "their": "Det",
        "some": "Det", "any": "Det", "every": "Det", "each": "Det",
        # Pronouns (mapped to NP since they fill NP slots)
        "i": "NP", "me": "NP", "you": "NP", "he": "NP", "she": "NP",
        "it": "NP", "we": "NP", "they": "NP", "them": "NP", "us": "NP",
        "who": "Wh", "what": "Wh", "which": "Wh", "where": "Wh",
        "when": "Wh", "why": "Wh", "how": "Wh",
        # Prepositions
        "in": "P", "on": "P", "at": "P", "to": "P", "for": "P",
        "with": "P", "by": "P", "from": "P", "of": "P", "about": "P",
        "into": "P", "through": "P", "between": "P", "after": "P",
        "before": "P", "during": "P", "without": "P", "under": "P",
        "over": "P", "above": "P", "below": "P", "up": "P", "down": "P",
        # Common verbs
        "is": "V", "are": "V", "was": "V", "were": "V", "be": "V",
        "been": "V", "being": "V", "am": "V",
        "have": "V", "has": "V", "had": "V", "having": "V",
        "do": "V", "does": "V", "did": "V",
        "will": "I", "would": "I", "could": "I", "should": "I",
        "might": "I", "must": "I", "shall": "I", "can": "I", "may": "I",
        "say": "V", "said": "V", "go": "V", "went": "V", "get": "V",
        "got": "V", "make": "V", "made": "V", "know": "V", "knew": "V",
        "think": "V", "thought": "V", "take": "V", "took": "V",
        "see": "V", "saw": "V", "come": "V", "came": "V",
        "want": "V", "look": "V", "give": "V", "use": "V",
        "find": "V", "tell": "V", "put": "V", "run": "V",
        "feel": "V", "try": "V", "leave": "V", "call": "V",
        "like": "V", "love": "V", "hate": "V", "need": "V",
        "keep": "V", "let": "V", "begin": "V", "show": "V",
        "hear": "V", "play": "V", "move": "V", "live": "V",
        "believe": "V", "happen": "V", "work": "V", "learn": "V",
        "understand": "V", "watch": "V", "follow": "V", "stop": "V",
        "create": "V", "speak": "V", "read": "V", "write": "V",
        "grow": "V", "open": "V", "walk": "V", "win": "V",
        "teach": "V", "build": "V", "lose": "V", "sing": "V",
        "looks": "V", "sits": "V", "stands": "V",
        # Adjectives
        "good": "Adj", "bad": "Adj", "big": "Adj", "small": "Adj",
        "great": "Adj", "old": "Adj", "new": "Adj", "long": "Adj",
        "high": "Adj", "little": "Adj", "own": "Adj", "other": "Adj",
        "right": "Adj", "large": "Adj", "young": "Adj", "important": "Adj",
        "different": "Adj", "early": "Adj", "real": "Adj", "hard": "Adj",
        "beautiful": "Adj", "strange": "Adj", "alive": "Adj", "much": "Adj",
        "difficult": "Adj", "backwards": "Adj", "human": "Adj",
        # Adverbs
        "not": "Adv", "very": "Adv", "also": "Adv", "often": "Adv",
        "just": "Adv", "now": "Adv", "then": "Adv", "here": "Adv",
        "there": "Adv", "always": "Adv", "never": "Adv", "still": "Adv",
        "already": "Adv", "together": "Adv", "everywhere": "Adv",
        "upwards": "Adv",
        # Complementisers / conjunctions
        "that": "C", "if": "C", "whether": "C",
        "and": "Conj", "but": "Conj", "or": "Conj", "so": "Conj",
        "because": "Conj", "although": "Conj", "while": "Conj",
        # Nouns (common)
        "system": "N", "time": "N", "people": "N", "way": "N",
        "day": "N", "man": "N", "woman": "N", "child": "N",
        "world": "N", "life": "N", "hand": "N", "part": "N",
        "place": "N", "thing": "N", "year": "N", "name": "N",
        "home": "N", "book": "N", "word": "N", "music": "N",
        "language": "N", "grammar": "N", "angel": "N", "loop": "N",
        "pattern": "N", "structure": "N", "meaning": "N", "rule": "N",
        "voice": "N", "fugue": "N", "scale": "N", "domain": "N",
        "prediction": "N", "future": "N", "past": "N", "history": "N",
        "sofa": "N", "chairs": "N", "pub": "N", "shops": "N",
        "property": "N", "personality": "N", "ones": "N",
        "parameters": "N", "grammars": "N", "recursion": "N",
        "lock": "N", "female": "N",
    }

    def __init__(self, config: AngelConfig | None = None):
        self.config = config or AngelConfig()
        self._grammars: dict[str, list[Grammar]] = {}
        self._substrates: dict[str, Substrate] = {}
        self._lexicon = Lexicon()
        self._engine = DerivationEngine()
        self._model: GrammarLanguageModel | None = None
        self._strange_loops: list[StrangeLoop] = []
        self._initialised = False
        self._word_categories: dict[str, str] = dict(self._DEFAULT_CATEGORIES)
        self._learned_words: dict[str, int] = {}  # word → encounter count
        self._reasoner: ReasoningEngine | None = None

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
        self._build_reasoner()
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
            "reasoning": [
                build_reasoning_grammar,
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

    def _build_reasoner(self) -> None:
        """Construct the reasoning engine — the chain-of-thought reasoner."""
        self._reasoner = ReasoningEngine(
            engine=self._engine,
            grammars=self._grammars,
            beam_width=5,
            max_depth=20,
            min_confidence=0.01,
        )

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
    # Text → grammar bridge
    # ------------------------------------------------------------------

    def categorize_tokens(self, tokens: list[str]) -> list[str]:
        """Map raw word tokens to grammatical categories.

        Unknown words are heuristically categorized based on morphology:
        - Words ending in -ly → Adv
        - Words ending in -ness/-ment/-tion/-sion → N
        - Words ending in -ful/-less/-ous/-ive/-able → Adj
        - Words ending in -ing/-ed/-es/-s (after consonant) → V
        - Capitalized words → N (proper noun)
        - Otherwise → N (default)

        Each encounter with an unknown word is recorded; the Angel
        learns from repeated exposure.
        """
        categories = []
        for token in tokens:
            word = token.lower().strip(".,!?;:'\"()-")
            if not word:
                continue
            cat = self._word_categories.get(word)
            if cat:
                categories.append(cat)
                continue
            # Heuristic categorization
            cat = self._guess_category(word)
            # Track encounters — learn on repetition
            self._learned_words[word] = self._learned_words.get(word, 0) + 1
            if self._learned_words[word] >= 2:
                # Seen twice — commit to lexicon
                self._word_categories[word] = cat
                self._lexicon.add(LexicalEntry(
                    form=word, category=cat, substrates=["linguistic"],
                ))
            categories.append(cat)
        return categories

    @staticmethod
    def _guess_category(word: str) -> str:
        """Heuristic POS guess for an unknown word."""
        if not word:
            return "N"
        if word.endswith("ly"):
            return "Adv"
        if word.endswith(("ness", "ment", "tion", "sion", "ity", "ence", "ance")):
            return "N"
        if word.endswith(("ful", "less", "ous", "ive", "able", "ible", "ical", "ent")):
            return "Adj"
        if word.endswith("ing"):
            return "V"
        if word.endswith("ed"):
            return "V"
        if word[0].isupper():
            return "N"
        return "N"

    def learn_word(self, word: str, category: str) -> None:
        """Explicitly teach the Angel a word's category."""
        word = word.lower().strip()
        if word:
            self._word_categories[word] = category
            self._lexicon.add(LexicalEntry(
                form=word, category=category, substrates=["linguistic"],
            ))

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

        # Try raw sequence first
        predictions = self._derive_predictions(sequence, grammars, horizon)

        # If raw tokens yielded nothing, try grammatical categories
        if not predictions and domain == "linguistic":
            categories = self.categorize_tokens(sequence)
            if categories:
                predictions = self._derive_predictions(
                    categories, grammars, horizon
                )
                # Try individual category symbols as single strings
                # (productions match on strings like "NP", "VP", etc.)
                if not predictions:
                    for cat in set(categories):
                        predictions.extend(
                            self._derive_single(cat, grammars, horizon)
                        )

        # Sort by confidence — the most grammatically certain first
        predictions.sort(key=lambda p: p["confidence"], reverse=True)
        return predictions

    def _derive_predictions(
        self,
        sequence: list[str],
        grammars: list,
        horizon: int,
    ) -> list[dict[str, Any]]:
        """Run derivation on a sequence across grammars."""
        predictions = []
        for grammar in grammars:
            tree = self._engine.derive(
                sequence, grammar, direction="forward"
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
                    })
        return predictions

    def _derive_single(
        self,
        symbol: str,
        grammars: list,
        horizon: int,
    ) -> list[dict[str, Any]]:
        """Derive from a single grammar symbol (e.g. 'NP', 'V')."""
        predictions = []
        for grammar in grammars:
            tree = self._engine.derive(
                symbol, grammar, direction="forward"
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
                    })
        return predictions

    def respond(self, text: str) -> dict[str, Any]:
        """Generate a structured response to natural language input.

        This is the Angel's main conversational entry point. It combines:
        1. MNEMO encoding (grounding — understanding intent through Mnemo)
        2. Token categorization (understanding the input's structure)
        3. Grammar-based prediction (what comes next)
        4. Morphological analysis (word-level patterns)
        5. Cross-domain insight (connections to other grammars)

        Returns a dict with keys:
            tokens, categories, predictions, analysis, loops_active,
            learned_words, mnemo, mnemo_decoded, response_text
        """
        self._ensure_awake()
        tokens = text.lower().split()
        categories = self.categorize_tokens(tokens)

        # MNEMO grounding — encode the input to understand its intent
        try:
            mnemo = mnemo_encode(text)
            mnemo_decoded = mnemo_decode(mnemo)
        except Exception:
            mnemo = "*a"
            mnemo_decoded = {"description": "universal analyze", "is_valid": True}

        # Grammar predictions
        predictions = self.predict(tokens, domain="linguistic", horizon=5)

        # If MNEMO detected a non-linguistic domain, try that domain too
        mnemo_domain = mnemo_decoded.get("operations", [{}])[0].get(
            "domain", "linguistic"
        ) if mnemo_decoded.get("operations") else "linguistic"
        if mnemo_domain != "linguistic" and mnemo_domain in self._grammars:
            domain_preds = self.predict(tokens, domain=mnemo_domain, horizon=3)
            predictions.extend(domain_preds)

        # Structural analysis
        cat_counts: dict[str, int] = {}
        for cat in categories:
            cat_counts[cat] = cat_counts.get(cat, 0) + 1

        # Find which words are new (unknown)
        new_words = []
        known_words = []
        for t in tokens:
            word = t.lower().strip(".,!?;:'\"()-")
            if not word:
                continue
            if word in self._word_categories:
                known_words.append(word)
            else:
                new_words.append(word)

        # Morphological insights
        morpho_insights = self._analyze_morphology(tokens)

        # Reasoning — the GLM's chain-of-thought pipeline
        reasoning_result = self.reason(text, max_depth=10)

        # Build a response
        info = self.introspect()

        return {
            "tokens": tokens,
            "categories": categories,
            "category_structure": cat_counts,
            "predictions": predictions[:5],
            "new_words": new_words,
            "known_words": known_words,
            "morphological": morpho_insights,
            "reasoning": {
                "chain_depth": reasoning_result["chain_depth"],
                "confidence": reasoning_result["confidence"],
                "domains_used": reasoning_result["domains_used"],
                "trace": reasoning_result["trace"][:10],  # Top 10 steps
                "answer": reasoning_result["answer"],
                "analogies": reasoning_result["analogies_used"],
            },
            "mnemo": mnemo,
            "mnemo_decoded": mnemo_decoded.get("description", ""),
            "mnemo_domain": mnemo_domain,
            "loops_active": len(self._strange_loops),
            "grammars_active": info["total_grammars"],
            "rules_active": info["total_rules"],
            "lexicon_size": len(self._word_categories),
        }

    def _analyze_morphology(self, tokens: list[str]) -> list[dict[str, str]]:
        """Analyze morphological structure of tokens."""
        insights = []
        morpho_grammars = self._grammars.get("linguistic", [])

        for token in tokens:
            word = token.lower().strip(".,!?;:'\"()-")
            if not word or len(word) < 3:
                continue

            # Try morphological derivation
            for grammar in morpho_grammars:
                if "morpholog" in grammar.name.lower():
                    tree = self._engine.derive(
                        word, grammar, direction="backward", max_steps=20
                    )
                    for path in tree.paths()[:3]:
                        if path and path[-1] is not None:
                            last = path[-1]
                            insights.append({
                                "word": word,
                                "analysis": str(last.output),
                                "rule": last.rule_id,
                            })
                            break

            # Simple suffix analysis
            for suffix, info in [
                ("ing", "progressive/gerund"),
                ("ed", "past tense/participle"),
                ("ly", "adverb derivation"),
                ("ness", "noun from adjective"),
                ("tion", "nominalisation"),
                ("ful", "adjective: having quality"),
                ("less", "adjective: without"),
                ("able", "adjective: capable of"),
                ("er", "comparative/agent"),
                ("est", "superlative"),
                ("ment", "nominalisation"),
                ("ous", "adjective: possessing"),
            ]:
                if word.endswith(suffix) and len(word) > len(suffix) + 1:
                    root = word[:-len(suffix)]
                    insights.append({
                        "word": word,
                        "analysis": f"{root} + -{suffix} ({info})",
                        "rule": f"morpho_{suffix}",
                    })
                    break

        return insights

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

    # ------------------------------------------------------------------
    # Reasoning pipeline — the GLM's superpower
    # ------------------------------------------------------------------

    def reason(
        self,
        question: str,
        *,
        domains: list[str] | None = None,
        goal: Any | None = None,
        max_depth: int = 15,
    ) -> dict[str, Any]:
        """Reason about a question via grammar derivation chains.

        This is the GLM's core advantage over LLMs: instead of
        predicting the next token by statistics, it derives conclusions
        from rules.  Each step is justified.  Each chain is verifiable.

        The pipeline:
        1. Parse input into symbolic form
        2. Identify relevant domains via MNEMO
        3. Chain grammar derivations via beam search
        4. When stuck, try analogical transfer
        5. Self-improve by learning from successful chains
        6. Return a verifiable reasoning trace

        Args:
            question: Natural language question or problem.
            domains:  Which domains to reason in (None = all).
            goal:     Optional target conclusion to reach.
            max_depth: Maximum reasoning chain depth.

        Returns:
            Dict with reasoning_chain, trace, answer, confidence,
            domains_used, analogies_used, stats.
        """
        self._ensure_awake()

        # Step 1: Parse to symbolic form
        tokens = question.lower().split()
        categories = self.categorize_tokens(tokens)

        # Step 2: Identify domains via MNEMO
        try:
            mnemo = mnemo_encode(question)
            mnemo_decoded = mnemo_decode(mnemo)
        except Exception:
            mnemo = "*a"
            mnemo_decoded = {"description": "universal analyze", "is_valid": True}

        mnemo_domain = mnemo_decoded.get("operations", [{}])[0].get(
            "domain", "reasoning"
        ) if mnemo_decoded.get("operations") else "reasoning"

        # Determine which domains to search
        if domains is None:
            domains = ["reasoning"]  # Always include reasoning
            if mnemo_domain != "reasoning" and mnemo_domain in self._grammars:
                domains.append(mnemo_domain)
            # Add linguistic for natural language questions
            if "linguistic" not in domains:
                domains.append("linguistic")

        # Step 3: Build symbolic start form
        # Try multiple representations to maximize rule firing
        start_forms = self._build_start_forms(question, tokens, categories)

        # Step 4: Run the reasoning engine on each start form
        best_chain = ReasoningChain(question=question)
        for start_form in start_forms:
            chain = self._reasoner.reason(
                question=question,
                start_form=start_form,
                goal=goal,
                domains=domains,
                max_depth=max_depth,
            )
            if chain.confidence > best_chain.confidence:
                best_chain = chain

        # Step 5: Build structured response
        return {
            "question": question,
            "reasoning_chain": [
                {
                    "step": i + 1,
                    "from": step.input,
                    "to": step.output,
                    "rule": step.rule_name,
                    "domain": step.domain,
                    "confidence": step.confidence,
                    "justification": step.justification,
                }
                for i, step in enumerate(best_chain.steps)
            ],
            "trace": best_chain.trace,
            "answer": best_chain.answer,
            "confidence": best_chain.confidence,
            "chain_depth": best_chain.depth,
            "domains_used": best_chain.domains_used,
            "analogies_used": best_chain.analogies_used,
            "is_complete": best_chain.is_complete,
            "mnemo": mnemo,
            "stats": self._reasoner.get_stats(),
        }

    def reason_bidirectional(
        self,
        question: str,
        start: Any,
        goal: Any,
        *,
        domains: list[str] | None = None,
    ) -> dict[str, Any]:
        """Reason forward from start AND backward from goal.

        Meet-in-the-middle search: O(2*b^(d/2)) instead of O(b^d).
        For grammar-constrained search, this can be 50,000x fewer
        states than unidirectional.
        """
        self._ensure_awake()
        domains = domains or list(self._grammars.keys())

        chain = self._reasoner.reason_bidirectional(
            question=question,
            start_form=start,
            goal_form=goal,
            domains=domains,
        )

        return {
            "question": question,
            "reasoning_chain": [
                {
                    "step": i + 1,
                    "from": step.input,
                    "to": step.output,
                    "rule": step.rule_name,
                    "domain": step.domain,
                    "confidence": step.confidence,
                    "justification": step.justification,
                }
                for i, step in enumerate(chain.steps)
            ],
            "trace": chain.trace,
            "answer": chain.answer,
            "confidence": chain.confidence,
            "chain_depth": chain.depth,
            "is_complete": chain.is_complete,
        }

    def reason_by_analogy(
        self,
        question: str,
        source_domain: str,
        target_domain: str,
    ) -> dict[str, Any]:
        """Solve a problem by analogical transfer between domains.

        The GLM's unique capability: if a chemistry problem has the
        same grammatical structure as a linguistics problem, solve
        it in linguistics and transfer the solution.
        """
        self._ensure_awake()

        tokens = question.lower().split()
        start_forms = self._build_start_forms(question, tokens,
                                               self.categorize_tokens(tokens))

        best_chain = ReasoningChain(question=question)
        for start_form in start_forms:
            # First try in source domain
            chain = self._reasoner.reason(
                question=question,
                start_form=start_form,
                domains=[source_domain, target_domain],
            )
            if chain.confidence > best_chain.confidence:
                best_chain = chain

        return {
            "question": question,
            "source_domain": source_domain,
            "target_domain": target_domain,
            "reasoning_chain": [
                {
                    "step": i + 1,
                    "from": step.input,
                    "to": step.output,
                    "rule": step.rule_name,
                    "domain": step.domain,
                    "justification": step.justification,
                }
                for i, step in enumerate(best_chain.steps)
            ],
            "trace": best_chain.trace,
            "answer": best_chain.answer,
            "confidence": best_chain.confidence,
            "analogies_used": best_chain.analogies_used,
        }

    def _build_start_forms(
        self,
        question: str,
        tokens: list[str],
        categories: list[str],
    ) -> list[Any]:
        """Build multiple symbolic representations for the reasoning engine.

        The more representations we try, the more rules can fire.
        Grammar-constrained search keeps this tractable.
        """
        forms: list[Any] = []

        # Raw text
        forms.append(question)

        # Token list
        if tokens:
            forms.append(tokens)

        # Category sequence
        if categories:
            forms.append(categories)

        # Individual categories (as strings for production matching)
        for cat in set(categories):
            forms.append(cat)

        # Pairs of tokens for relational rules
        for i in range(len(tokens) - 1):
            forms.append((tokens[i], tokens[i + 1]))

        # Detect structural patterns in the question
        q_lower = question.lower()
        if " implies " in q_lower:
            forms.append(q_lower)
        if " causes " in q_lower:
            forms.append(q_lower)
        if " is a " in q_lower:
            forms.append(q_lower)
        if " if " in q_lower and " then " in q_lower:
            # Convert if-then to implies
            parts = q_lower.split(" if ", 1)
            if " then " in parts[-1]:
                cond_parts = parts[-1].split(" then ", 1)
                forms.append(f"{cond_parts[0].strip()} implies {cond_parts[1].strip()}")
        if " and " in q_lower:
            # Split conjunctions for individual reasoning
            conjuncts = q_lower.split(" and ")
            for c in conjuncts:
                c = c.strip()
                if c:
                    forms.append(c)

        return forms

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
            "reasoning_stats": self._reasoner.get_stats() if self._reasoner else {},
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
