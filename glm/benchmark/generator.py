"""
generator.py -- Deterministic benchmark instance generator.

Uses the GLM's own grammar system to generate test cases with provably
correct ground truth.  All randomness is seeded so runs are reproducible.

The key insight: because we *generate* test cases from grammar rules, we
know the correct answer by construction -- no human annotation needed.
The grammar trace serves as the proof certificate.

Generation strategy per task type:

    SYLLOGISM_VALIDATION:      Build a chain of implications from logic
                               grammar productions; the conclusion is valid
                               iff the chain is connected.

    ARGUMENT_DECOMPOSITION:    Generate sentences from the syntactic grammar
                               and record the production tree as the triple
                               skeleton.

    CIRCULAR_REASONING:        Build an argument graph; randomly close a
                               cycle (or not) and record ground truth.

    CROSS_DOMAIN_ANALOGY:      Pick two domain grammars, find an isomorphism
                               using the engine, and mask one side.

    DERIVATION_COMPLETION:     Start a forward derivation, record the full
                               path, and truncate the input.

    ORIGIN_RECONSTRUCTION:     Run a forward derivation to completion, then
                               give only the output and expect the input.

    LOSSLESS_COMPRESSION:      Generate structured data, compress with MNEMO,
                               and record both sides.

    INFORMATION_DENSITY:       Compress varying-length inputs and measure
                               density.

    CHEMISTRY_BIOLOGY:         Compose chemical and biological grammars in
                               a fugue and generate problems from harmonies.

    MATH_PHYSICS:              Compose mathematical and physics grammars.

    LINGUISTICS_ETYMOLOGY:     Compose linguistic and etymological grammars.
"""

from __future__ import annotations

import hashlib
import random
from typing import Any, Dict, List, Tuple

from .tasks import BenchmarkTask, DifficultyLevel, TaskType


class BenchmarkGenerator:
    """Generates benchmark instances from grammar rules.

    All generation is deterministic given the seed.  Each call to
    generate_batch() advances the RNG state, so generating tasks
    in a consistent order produces consistent results.

    Attributes:
        seed:   Base RNG seed.
        _rng:   The random.Random instance (isolated from global state).
    """

    def __init__(self, seed: int = 42) -> None:
        self.seed = seed
        self._rng = random.Random(seed)
        self._counter = 0

    def generate_batch(
        self,
        task_type: TaskType,
        difficulty: DifficultyLevel,
        count: int = 100,
    ) -> List[BenchmarkTask]:
        """Generate *count* benchmark instances for a given task and difficulty.

        Parameters:
            task_type:   Which benchmark task to generate.
            difficulty:  The difficulty level.
            count:       How many instances to produce.

        Returns:
            A list of BenchmarkTask instances with ground truth.
        """
        # Seed the RNG deterministically per (task_type, difficulty).
        combo_seed = self.seed + hash((task_type.value, difficulty.value))
        self._rng = random.Random(combo_seed)

        dispatch = {
            TaskType.SYLLOGISM_VALIDATION: self._gen_syllogism,
            TaskType.ARGUMENT_DECOMPOSITION: self._gen_argument_decomposition,
            TaskType.CIRCULAR_REASONING: self._gen_circular_reasoning,
            TaskType.CROSS_DOMAIN_ANALOGY: self._gen_cross_domain_analogy,
            TaskType.DERIVATION_COMPLETION: self._gen_derivation_completion,
            TaskType.ORIGIN_RECONSTRUCTION: self._gen_origin_reconstruction,
            TaskType.LOSSLESS_COMPRESSION: self._gen_lossless_compression,
            TaskType.INFORMATION_DENSITY: self._gen_information_density,
            TaskType.CHEMISTRY_BIOLOGY: self._gen_chemistry_biology,
            TaskType.MATH_PHYSICS: self._gen_math_physics,
            TaskType.LINGUISTICS_ETYMOLOGY: self._gen_linguistics_etymology,
        }

        gen_fn = dispatch[task_type]
        tasks: List[BenchmarkTask] = []
        for i in range(count):
            task = gen_fn(difficulty, i)
            tasks.append(task)

        return tasks

    def _make_id(self, task_type: TaskType, difficulty: DifficultyLevel, index: int) -> str:
        """Produce a stable, unique task ID."""
        self._counter += 1
        raw = f"{task_type.value}:{difficulty.value}:{index}:{self._counter}"
        return hashlib.sha256(raw.encode()).hexdigest()[:16]

    # -----------------------------------------------------------------------
    # Difficulty parameters
    # -----------------------------------------------------------------------

    _DEPTH = {
        DifficultyLevel.EASY: (1, 2),
        DifficultyLevel.MEDIUM: (3, 5),
        DifficultyLevel.HARD: (6, 10),
    }

    _DOMAIN_COUNT = {
        DifficultyLevel.EASY: 1,
        DifficultyLevel.MEDIUM: 2,
        DifficultyLevel.HARD: 3,
    }

    # -----------------------------------------------------------------------
    # Symbol / concept pools (used to construct test instances)
    # -----------------------------------------------------------------------

    _ENTITIES = [
        "dogs", "cats", "birds", "mammals", "reptiles", "fish", "plants",
        "fungi", "bacteria", "cells", "atoms", "molecules", "proteins",
        "enzymes", "genes", "photons", "electrons", "waves", "fields",
        "forces", "numbers", "primes", "integers", "functions", "sets",
        "morphemes", "phonemes", "syllables", "words", "sentences",
        "roots", "prefixes", "suffixes", "compounds", "elements",
        "reactions", "catalysts", "bonds", "crystals", "polymers",
    ]

    _PROPERTIES = [
        "warm-blooded", "cold-blooded", "multicellular", "unicellular",
        "organic", "inorganic", "positive", "negative", "finite",
        "infinite", "prime", "composite", "transitive", "reflexive",
        "symmetric", "periodic", "stable", "reactive", "soluble",
        "volatile", "bounded", "continuous", "discrete", "recursive",
        "linear", "nonlinear", "conserved", "invariant", "derivable",
        "compressible",
    ]

    _RELATIONS = [
        "implies", "contains", "produces", "transforms_into",
        "is_subset_of", "is_isomorphic_to", "derives_from",
        "is_analogous_to", "conserves", "decomposes_into",
        "catalyses", "inhibits", "regulates", "encodes",
        "binds_to", "is_part_of", "precedes", "follows",
    ]

    _DOMAINS = [
        "mathematics", "linguistics", "biology", "chemistry",
        "physics", "computation", "etymology",
    ]

    _DOMAIN_CONCEPTS: Dict[str, List[str]] = {
        "mathematics": [
            "integer", "prime", "function", "derivative", "integral",
            "group", "ring", "field", "set", "matrix", "polynomial",
            "theorem", "proof", "axiom", "equation",
        ],
        "linguistics": [
            "noun_phrase", "verb_phrase", "clause", "morpheme",
            "phoneme", "syntax_tree", "agreement", "case",
            "tense", "aspect", "mood", "voice", "complement",
            "specifier", "adjunct",
        ],
        "biology": [
            "codon", "amino_acid", "gene", "protein", "enzyme",
            "DNA", "RNA", "ribosome", "membrane", "mitochondria",
            "chloroplast", "transcription", "translation", "mutation",
            "selection",
        ],
        "chemistry": [
            "atom", "molecule", "bond", "reaction", "catalyst",
            "ion", "electron", "orbital", "acid", "base",
            "oxidation", "reduction", "polymer", "crystal", "isomer",
        ],
        "physics": [
            "force", "energy", "momentum", "wave", "field",
            "particle", "photon", "entropy", "temperature",
            "velocity", "acceleration", "potential", "charge",
            "mass", "spin",
        ],
        "computation": [
            "function", "variable", "loop", "recursion", "type",
            "class", "algorithm", "complexity", "compiler",
            "automaton", "grammar", "parser", "tree", "graph",
            "stack",
        ],
        "etymology": [
            "root", "prefix", "suffix", "cognate", "borrowing",
            "sound_change", "grimms_law", "ablaut", "umlaut",
            "metathesis", "assimilation", "dissimilation",
            "proto_form", "reflex", "substrate",
        ],
    }

    # -----------------------------------------------------------------------
    # Structural task generators
    # -----------------------------------------------------------------------

    def _gen_syllogism(self, difficulty: DifficultyLevel, index: int) -> BenchmarkTask:
        """Generate a syllogism validation task.

        Constructs a chain of 'All X are Y' premises and checks whether
        a candidate conclusion follows.

        EASY:   2 premises, direct chain, always valid or always invalid.
        MEDIUM: 3-4 premises, chain may skip or include distractors.
        HARD:   5+ premises, multiple chains, some red herrings.
        """
        min_depth, max_depth = self._DEPTH[difficulty]
        chain_len = self._rng.randint(min_depth + 1, max_depth + 2)

        # Pick a chain of entities.
        entities = self._rng.sample(self._ENTITIES, min(chain_len + 1, len(self._ENTITIES)))
        chain = entities[:chain_len + 1]

        # Build premises: "All chain[i] are chain[i+1]".
        premises = [f"All {chain[i]} are {chain[i+1]}" for i in range(len(chain) - 1)]

        # Decide if conclusion is valid.
        is_valid = self._rng.random() < 0.5
        if is_valid:
            # Valid: conclusion connects first to last.
            conclusion = f"All {chain[0]} are {chain[-1]}"
            explanation = f"Transitive chain: {' -> '.join(chain)}"
        else:
            # Invalid: conclusion reverses or uses unconnected entity.
            if self._rng.random() < 0.5:
                # Reversal (not generally valid for 'All X are Y').
                conclusion = f"All {chain[-1]} are {chain[0]}"
                explanation = "Reversed implication: 'All X are Y' does not imply 'All Y are X'"
            else:
                # Unconnected entity.
                other = self._rng.choice([e for e in self._ENTITIES if e not in chain])
                conclusion = f"All {chain[0]} are {other}"
                explanation = f"No chain connects {chain[0]} to {other}"

        # Add distractors for medium and hard.
        distractors: List[str] = []
        if difficulty in (DifficultyLevel.MEDIUM, DifficultyLevel.HARD):
            n_distractors = self._rng.randint(1, max_depth)
            for _ in range(n_distractors):
                a, b = self._rng.sample(self._ENTITIES, 2)
                distractors.append(f"All {a} are {b}")

        all_premises = premises + distractors
        self._rng.shuffle(all_premises)

        return BenchmarkTask(
            id=self._make_id(TaskType.SYLLOGISM_VALIDATION, difficulty, index),
            task_type=TaskType.SYLLOGISM_VALIDATION,
            difficulty=difficulty,
            input_data={
                "premises": all_premises,
                "candidate_conclusion": conclusion,
                "relevant_premises": premises,
            },
            ground_truth={
                "is_valid": is_valid,
                "explanation": explanation,
            },
            grammar_trace={
                "chain": chain,
                "chain_edges": [(chain[i], chain[i + 1]) for i in range(len(chain) - 1)],
                "distractors": distractors,
            },
            metadata={
                "chain_length": len(chain),
                "num_distractors": len(distractors),
                "derivation_depth": len(chain) - 1,
            },
        )

    def _gen_argument_decomposition(self, difficulty: DifficultyLevel, index: int) -> BenchmarkTask:
        """Generate an argument decomposition (S->R->O triple extraction) task.

        Constructs sentences with known relational structure and expects
        the solver to extract subject-relation-object triples.

        EASY:   1 triple, simple sentence.
        MEDIUM: 2-3 triples, compound sentence.
        HARD:   4+ triples, complex nested structure.
        """
        min_depth, max_depth = self._DEPTH[difficulty]
        num_triples = self._rng.randint(min_depth, max_depth)

        triples: List[Dict[str, str]] = []
        sentence_parts: List[str] = []

        for _ in range(num_triples):
            subj = self._rng.choice(self._ENTITIES)
            rel = self._rng.choice(self._RELATIONS)
            obj = self._rng.choice(self._ENTITIES)
            triples.append({"subject": subj, "relation": rel, "object": obj})
            sentence_parts.append(f"{subj} {rel.replace('_', ' ')} {obj}")

        connectors = [", and ", ", which ", ", therefore ", " because ", " while ", " since "]
        sentence = sentence_parts[0]
        for part in sentence_parts[1:]:
            connector = self._rng.choice(connectors)
            sentence += connector + part

        return BenchmarkTask(
            id=self._make_id(TaskType.ARGUMENT_DECOMPOSITION, difficulty, index),
            task_type=TaskType.ARGUMENT_DECOMPOSITION,
            difficulty=difficulty,
            input_data={
                "sentence": sentence,
                "expected_count": num_triples,
            },
            ground_truth={
                "triples": triples,
            },
            grammar_trace={
                "production_sequence": [
                    f"S -> {t['subject']} {t['relation']} {t['object']}" for t in triples
                ],
            },
            metadata={
                "num_triples": num_triples,
            },
        )

    def _gen_circular_reasoning(self, difficulty: DifficultyLevel, index: int) -> BenchmarkTask:
        """Generate a circular reasoning detection task.

        Builds an argument graph and optionally introduces a cycle.

        EASY:   3 nodes, cycle or no cycle.
        MEDIUM: 5-7 nodes, cycle may be indirect.
        HARD:   8-12 nodes, multiple potential cycles, nested.
        """
        min_depth, max_depth = self._DEPTH[difficulty]
        num_nodes = self._rng.randint(min_depth + 2, max_depth + 3)

        # Create named nodes.
        node_labels = [chr(ord('A') + i) for i in range(min(num_nodes, 26))]
        if num_nodes > 26:
            node_labels.extend([f"Z{i}" for i in range(num_nodes - 26)])

        # Build edges: a random DAG plus possibly one back-edge to create a cycle.
        edges: List[Tuple[str, str]] = []
        for i in range(len(node_labels) - 1):
            edges.append((node_labels[i], node_labels[i + 1]))

        # Add some extra forward edges for complexity.
        extra_edges = self._rng.randint(0, min_depth)
        for _ in range(extra_edges):
            i = self._rng.randint(0, len(node_labels) - 2)
            j = self._rng.randint(i + 1, len(node_labels) - 1)
            edge = (node_labels[i], node_labels[j])
            if edge not in edges:
                edges.append(edge)

        # Decide whether to create a cycle.
        has_cycle = self._rng.random() < 0.5
        cycle_path: List[str] = []

        if has_cycle:
            # Add a back-edge from a later node to an earlier node.
            cycle_start = self._rng.randint(0, max(0, len(node_labels) - 3))
            cycle_end_idx = self._rng.randint(cycle_start + 2, len(node_labels) - 1)
            back_edge = (node_labels[cycle_end_idx], node_labels[cycle_start])
            edges.append(back_edge)

            # Record the cycle.
            cycle_path = node_labels[cycle_start:cycle_end_idx + 1] + [node_labels[cycle_start]]

        # Build argument steps from edges.
        argument_steps = [f"{a} implies {b}" for a, b in edges]
        self._rng.shuffle(argument_steps)

        return BenchmarkTask(
            id=self._make_id(TaskType.CIRCULAR_REASONING, difficulty, index),
            task_type=TaskType.CIRCULAR_REASONING,
            difficulty=difficulty,
            input_data={
                "argument_steps": argument_steps,
                "nodes": node_labels,
            },
            ground_truth={
                "has_cycle": has_cycle,
                "cycle": cycle_path if has_cycle else [],
            },
            grammar_trace={
                "edges": edges,
                "back_edge": back_edge if has_cycle else None,
                "adjacency": {n: [b for a, b in edges if a == n] for n in node_labels},
            },
            metadata={
                "num_nodes": num_nodes,
                "num_edges": len(edges),
            },
        )

    def _gen_cross_domain_analogy(self, difficulty: DifficultyLevel, index: int) -> BenchmarkTask:
        """Generate a cross-domain analogy task.

        Picks two domains, selects parallel concepts, and masks one side
        to create an analogy completion problem.

        EASY:   Close domains, obvious parallel.
        MEDIUM: Distant domains, requires abstract mapping.
        HARD:   Three domains, chained analogy.
        """
        # Pick source and target domains.
        n_domains = self._DOMAIN_COUNT[difficulty]
        domains = self._rng.sample(self._DOMAINS, min(n_domains + 1, len(self._DOMAINS)))
        source_domain = domains[0]
        target_domain = domains[1]

        source_concepts = self._DOMAIN_CONCEPTS[source_domain]
        target_concepts = self._DOMAIN_CONCEPTS[target_domain]

        # Pick parallel concepts (by index alignment -- a simplified isomorphism).
        src_idx = self._rng.randint(0, min(len(source_concepts), len(target_concepts)) - 1)
        source_concept = source_concepts[src_idx % len(source_concepts)]
        target_concept = target_concepts[src_idx % len(target_concepts)]

        # Build the relation from a second pair.
        src_idx2 = (src_idx + self._rng.randint(1, 5)) % len(source_concepts)
        source_related = source_concepts[src_idx2 % len(source_concepts)]
        target_related = target_concepts[src_idx2 % len(target_concepts)]

        # Pick a relation.
        relation = self._rng.choice(self._RELATIONS)

        return BenchmarkTask(
            id=self._make_id(TaskType.CROSS_DOMAIN_ANALOGY, difficulty, index),
            task_type=TaskType.CROSS_DOMAIN_ANALOGY,
            difficulty=difficulty,
            input_data={
                "source_domain": source_domain,
                "target_domain": target_domain,
                "source_concept": source_concept,
                "source_related": source_related,
                "relation": relation,
                "analogy": (
                    f"{source_concept} {relation.replace('_', ' ')} {source_related} "
                    f"in {source_domain} :: ??? {relation.replace('_', ' ')} {target_related} "
                    f"in {target_domain}"
                ),
            },
            ground_truth={
                "answer": target_concept,
                "full_analogy": (
                    f"{source_concept} : {source_related} :: {target_concept} : {target_related}"
                ),
            },
            grammar_trace={
                "isomorphism_type": "index_alignment",
                "source_index": src_idx,
                "mapping": {
                    source_concept: target_concept,
                    source_related: target_related,
                },
            },
            metadata={
                "domains_involved": domains,
                "relation": relation,
            },
        )

    def _gen_derivation_completion(self, difficulty: DifficultyLevel, index: int) -> BenchmarkTask:
        """Generate a derivation completion (forward prediction) task.

        Constructs a derivation chain using grammar rules and truncates it.
        The solver must predict the remaining steps.

        EASY:   2-step chain, give first step, predict second.
        MEDIUM: 4-step chain, give first 2, predict remaining.
        HARD:   8-step chain, give first 3, predict remaining.
        """
        min_depth, max_depth = self._DEPTH[difficulty]
        total_steps = self._rng.randint(min_depth + 1, max_depth + 1)
        reveal_steps = max(1, total_steps // 2)

        # Generate a derivation chain as a sequence of symbol transformations.
        symbols = self._rng.sample(
            self._ENTITIES + list(self._PROPERTIES),
            min(total_steps + 1, len(self._ENTITIES) + len(self._PROPERTIES)),
        )
        chain = symbols[:total_steps + 1]

        # Build rule names.
        rules_used = [
            f"rule_{chain[i]}_to_{chain[i+1]}" for i in range(len(chain) - 1)
        ]

        derivation_steps = [
            {"step": i, "input": chain[i], "output": chain[i + 1], "rule": rules_used[i]}
            for i in range(len(chain) - 1)
        ]

        return BenchmarkTask(
            id=self._make_id(TaskType.DERIVATION_COMPLETION, difficulty, index),
            task_type=TaskType.DERIVATION_COMPLETION,
            difficulty=difficulty,
            input_data={
                "partial_derivation": derivation_steps[:reveal_steps],
                "start_form": chain[0],
                "current_form": chain[reveal_steps],
                "total_expected_steps": total_steps,
            },
            ground_truth={
                "remaining_steps": derivation_steps[reveal_steps:],
                "final_form": chain[-1],
            },
            grammar_trace={
                "full_chain": chain,
                "all_rules": rules_used,
            },
            metadata={
                "total_steps": total_steps,
                "revealed_steps": reveal_steps,
                "hidden_steps": total_steps - reveal_steps,
            },
        )

    def _gen_origin_reconstruction(self, difficulty: DifficultyLevel, index: int) -> BenchmarkTask:
        """Generate an origin reconstruction (backward derivation) task.

        Gives the output of a derivation and asks for the starting form.

        EASY:   1-step derivation, recover the input.
        MEDIUM: 3-step derivation, recover the chain.
        HARD:   6+ steps, branching derivation, recover the root.
        """
        min_depth, max_depth = self._DEPTH[difficulty]
        total_steps = self._rng.randint(min_depth, max_depth)

        symbols = self._rng.sample(
            self._ENTITIES + list(self._PROPERTIES),
            min(total_steps + 1, len(self._ENTITIES) + len(self._PROPERTIES)),
        )
        chain = symbols[:total_steps + 1]

        rules_used = [
            f"rule_{chain[i]}_to_{chain[i+1]}" for i in range(len(chain) - 1)
        ]

        derivation_steps = [
            {"step": i, "input": chain[i], "output": chain[i + 1], "rule": rules_used[i]}
            for i in range(len(chain) - 1)
        ]

        # Give the final form and optionally some intermediate hints.
        hints = []
        if difficulty != DifficultyLevel.EASY and total_steps > 2:
            hint_count = self._rng.randint(1, max(1, total_steps // 2))
            hint_indices = sorted(self._rng.sample(range(1, total_steps), min(hint_count, total_steps - 1)))
            hints = [{"position": idx, "form": chain[idx]} for idx in hint_indices]

        return BenchmarkTask(
            id=self._make_id(TaskType.ORIGIN_RECONSTRUCTION, difficulty, index),
            task_type=TaskType.ORIGIN_RECONSTRUCTION,
            difficulty=difficulty,
            input_data={
                "final_form": chain[-1],
                "derivation_length": total_steps,
                "hints": hints,
            },
            ground_truth={
                "origin": chain[0],
                "full_chain": chain,
                "derivation_steps": derivation_steps,
            },
            grammar_trace={
                "full_chain": chain,
                "all_rules": rules_used,
            },
            metadata={
                "total_steps": total_steps,
                "num_hints": len(hints),
            },
        )

    # -----------------------------------------------------------------------
    # Compression task generators
    # -----------------------------------------------------------------------

    def _gen_lossless_compression(self, difficulty: DifficultyLevel, index: int) -> BenchmarkTask:
        """Generate a lossless compression round-trip task.

        Creates structured reasoning data and expects MNEMO compression
        to preserve it through a compress/decompress cycle.

        EASY:   Short text (1-2 sentences).
        MEDIUM: Structured argument (3-5 steps).
        HARD:   Complex derivation chain (6+ steps with nested structure).
        """
        min_depth, max_depth = self._DEPTH[difficulty]
        num_elements = self._rng.randint(min_depth + 1, max_depth + 2)

        # Build structured reasoning data.
        reasoning_chain: List[Dict[str, str]] = []
        entities = self._rng.sample(self._ENTITIES, min(num_elements + 1, len(self._ENTITIES)))

        for i in range(num_elements):
            step = {
                "step": str(i + 1),
                "premise": f"{entities[i]} {self._rng.choice(self._RELATIONS).replace('_', ' ')} {entities[i + 1]}",
                "rule": self._rng.choice(["modus_ponens", "transitivity", "universal_instantiation",
                                           "chain_rule", "decomposition"]),
            }
            reasoning_chain.append(step)

        text_repr = "; ".join(s["premise"] for s in reasoning_chain)

        return BenchmarkTask(
            id=self._make_id(TaskType.LOSSLESS_COMPRESSION, difficulty, index),
            task_type=TaskType.LOSSLESS_COMPRESSION,
            difficulty=difficulty,
            input_data={
                "reasoning_chain": reasoning_chain,
                "text_representation": text_repr,
                "original_char_count": len(text_repr),
            },
            ground_truth={
                "chain": reasoning_chain,
                "text": text_repr,
                "expected_fidelity": 1.0,
            },
            grammar_trace={
                "entities": entities[:num_elements + 1],
                "relations": [s["rule"] for s in reasoning_chain],
            },
            metadata={
                "num_steps": num_elements,
                "char_count": len(text_repr),
            },
        )

    def _gen_information_density(self, difficulty: DifficultyLevel, index: int) -> BenchmarkTask:
        """Generate an information density measurement task.

        Creates inputs of varying length and complexity.  The expected
        output is the information density ratio.

        EASY:   Simple keyword list (low reasoning density).
        MEDIUM: Argument with premises (medium density).
        HARD:   Nested derivation with cross-references (high density).
        """
        min_depth, max_depth = self._DEPTH[difficulty]
        num_units = self._rng.randint(min_depth + 2, max_depth + 5)

        # Generate reasoning units.
        units: List[str] = []
        for _ in range(num_units):
            subj = self._rng.choice(self._ENTITIES)
            prop = self._rng.choice(self._PROPERTIES)
            units.append(f"{subj} is {prop}")

        text = ". ".join(units) + "."
        reasoning_units = num_units  # Each "X is Y" is one reasoning unit.

        return BenchmarkTask(
            id=self._make_id(TaskType.INFORMATION_DENSITY, difficulty, index),
            task_type=TaskType.INFORMATION_DENSITY,
            difficulty=difficulty,
            input_data={
                "text": text,
                "char_count": len(text),
                "reasoning_unit_count": reasoning_units,
            },
            ground_truth={
                "reasoning_units": reasoning_units,
                "original_density": reasoning_units / max(1, len(text)),
            },
            grammar_trace={
                "units": units,
            },
            metadata={
                "num_units": reasoning_units,
                "char_count": len(text),
            },
        )

    # -----------------------------------------------------------------------
    # Multi-domain task generators
    # -----------------------------------------------------------------------

    def _gen_chemistry_biology(self, difficulty: DifficultyLevel, index: int) -> BenchmarkTask:
        """Generate a chemistry-biology cross-domain task.

        Problems require reasoning about both molecular/chemical structure
        and biological function.

        EASY:   Simple mapping (e.g., which element is essential for a process).
        MEDIUM: Reaction + biological pathway connection.
        HARD:   Multi-step: molecular structure -> reaction -> biological effect.
        """
        min_depth, max_depth = self._DEPTH[difficulty]

        chem_concepts = self._DOMAIN_CONCEPTS["chemistry"]
        bio_concepts = self._DOMAIN_CONCEPTS["biology"]

        # Build a cross-domain problem.
        chem_entity = self._rng.choice(chem_concepts)
        bio_entity = self._rng.choice(bio_concepts)
        relation = self._rng.choice(["catalyses", "inhibits", "regulates", "binds_to", "produces"])

        # Build reasoning steps that bridge the two domains.
        steps: List[Dict[str, str]] = []
        num_steps = self._rng.randint(min_depth, max_depth)

        current_chem = chem_entity
        current_bio = bio_entity

        for i in range(num_steps):
            if i % 2 == 0:
                # Chemistry step.
                next_chem = self._rng.choice(chem_concepts)
                steps.append({
                    "domain": "chemistry",
                    "step": f"{current_chem} reacts with {next_chem}",
                    "rule": "reaction_grammar",
                })
                current_chem = next_chem
            else:
                # Biology step.
                next_bio = self._rng.choice(bio_concepts)
                steps.append({
                    "domain": "biology",
                    "step": f"{current_bio} {relation} {next_bio}",
                    "rule": "biological_grammar",
                })
                current_bio = next_bio

        question = (
            f"Given that {chem_entity} {relation.replace('_', ' ')} {bio_entity}, "
            f"and the following steps occur: {'; '.join(s['step'] for s in steps)}. "
            f"What is the final biological outcome?"
        )

        return BenchmarkTask(
            id=self._make_id(TaskType.CHEMISTRY_BIOLOGY, difficulty, index),
            task_type=TaskType.CHEMISTRY_BIOLOGY,
            difficulty=difficulty,
            input_data={
                "question": question,
                "chemistry_entity": chem_entity,
                "biology_entity": bio_entity,
                "relation": relation,
                "steps": steps,
            },
            ground_truth={
                "final_chemistry": current_chem,
                "final_biology": current_bio,
                "bridging_relation": relation,
                "step_count": num_steps,
            },
            grammar_trace={
                "domains": ["chemistry", "biology"],
                "fugue_voices": 2,
                "steps": steps,
            },
            metadata={
                "num_steps": num_steps,
                "domains": ["chemistry", "biology"],
            },
        )

    def _gen_math_physics(self, difficulty: DifficultyLevel, index: int) -> BenchmarkTask:
        """Generate a math-physics cross-domain task.

        Problems require mathematical derivation to solve physical problems.

        EASY:   Direct formula application.
        MEDIUM: Derivation required (e.g., differentiate to find extrema).
        HARD:   Multi-step: set up equation -> solve -> interpret physically.
        """
        min_depth, max_depth = self._DEPTH[difficulty]

        math_concepts = self._DOMAIN_CONCEPTS["mathematics"]
        phys_concepts = self._DOMAIN_CONCEPTS["physics"]

        math_entity = self._rng.choice(math_concepts)
        phys_entity = self._rng.choice(phys_concepts)

        # Build reasoning chain.
        num_steps = self._rng.randint(min_depth, max_depth)
        steps: List[Dict[str, str]] = []

        for i in range(num_steps):
            if i % 2 == 0:
                mc = self._rng.choice(math_concepts)
                steps.append({
                    "domain": "mathematics",
                    "step": f"Apply {mc} to derive the relationship",
                    "rule": "algebra_grammar",
                })
            else:
                pc = self._rng.choice(phys_concepts)
                steps.append({
                    "domain": "physics",
                    "step": f"Interpret result in terms of {pc}",
                    "rule": "physics_grammar",
                })

        question = (
            f"Using {math_entity} (mathematics), determine the behaviour of "
            f"{phys_entity} (physics) through the following reasoning: "
            f"{'; '.join(s['step'] for s in steps)}."
        )

        return BenchmarkTask(
            id=self._make_id(TaskType.MATH_PHYSICS, difficulty, index),
            task_type=TaskType.MATH_PHYSICS,
            difficulty=difficulty,
            input_data={
                "question": question,
                "math_concept": math_entity,
                "physics_concept": phys_entity,
                "steps": steps,
            },
            ground_truth={
                "math_entity": math_entity,
                "physics_entity": phys_entity,
                "step_count": num_steps,
                "derivation_valid": True,
            },
            grammar_trace={
                "domains": ["mathematics", "physics"],
                "fugue_voices": 2,
                "steps": steps,
            },
            metadata={
                "num_steps": num_steps,
                "domains": ["mathematics", "physics"],
            },
        )

    def _gen_linguistics_etymology(self, difficulty: DifficultyLevel, index: int) -> BenchmarkTask:
        """Generate a linguistics-etymology cross-domain task.

        Problems require both synchronic analysis (current morphology)
        and diachronic tracing (historical evolution).

        EASY:   Identify root and meaning of a morphologically transparent word.
        MEDIUM: Trace a word through 2-3 historical changes.
        HARD:   Reconstruct a proto-form from multiple modern reflexes.
        """
        min_depth, max_depth = self._DEPTH[difficulty]

        ling_concepts = self._DOMAIN_CONCEPTS["linguistics"]
        etym_concepts = self._DOMAIN_CONCEPTS["etymology"]

        # Build a morphological-etymological chain.
        num_steps = self._rng.randint(min_depth, max_depth)

        ling_entity = self._rng.choice(ling_concepts)
        etym_entity = self._rng.choice(etym_concepts)

        steps: List[Dict[str, str]] = []
        for i in range(num_steps):
            if i % 2 == 0:
                lc = self._rng.choice(ling_concepts)
                steps.append({
                    "domain": "linguistics",
                    "step": f"Analyze {lc} structure synchronically",
                    "rule": "morphological_grammar",
                })
            else:
                ec = self._rng.choice(etym_concepts)
                steps.append({
                    "domain": "etymology",
                    "step": f"Trace diachronic change via {ec}",
                    "rule": "etymology_grammar",
                })

        question = (
            f"Given the linguistic structure '{ling_entity}' and etymological "
            f"process '{etym_entity}', analyze through: "
            f"{'; '.join(s['step'] for s in steps)}. "
            f"What is the reconstructed form?"
        )

        return BenchmarkTask(
            id=self._make_id(TaskType.LINGUISTICS_ETYMOLOGY, difficulty, index),
            task_type=TaskType.LINGUISTICS_ETYMOLOGY,
            difficulty=difficulty,
            input_data={
                "question": question,
                "linguistic_entity": ling_entity,
                "etymological_entity": etym_entity,
                "steps": steps,
            },
            ground_truth={
                "linguistic_entity": ling_entity,
                "etymological_entity": etym_entity,
                "step_count": num_steps,
                "reconstruction_valid": True,
            },
            grammar_trace={
                "domains": ["linguistics", "etymology"],
                "fugue_voices": 2,
                "steps": steps,
            },
            metadata={
                "num_steps": num_steps,
                "domains": ["linguistics", "etymology"],
            },
        )
