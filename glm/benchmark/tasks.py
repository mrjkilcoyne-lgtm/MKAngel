"""
tasks.py -- Benchmark task type definitions for the GLM reasoning harness.

Defines the taxonomy of structural reasoning, compression, and multi-domain
tasks that the benchmark suite tests.  Each task type maps directly to a
GLM capability: DAG analysis, skeleton extraction, strange loop detection,
isomorphism discovery, forward/backward derivation, MNEMO compression,
and fugue composition.

Example task instances
----------------------

Syllogism validation (EASY)::

    {
        "premises": ["All mammals are warm-blooded", "All dogs are mammals"],
        "candidate_conclusion": "All dogs are warm-blooded",
        "ground_truth": True,
        "structure": {"chain": ["dogs -> mammals", "mammals -> warm-blooded"]},
    }

Circular reasoning detection (MEDIUM)::

    {
        "argument_steps": [
            "A implies B",
            "B implies C",
            "C implies A",
        ],
        "ground_truth": {"has_cycle": True, "cycle": ["A", "B", "C", "A"]},
    }

Cross-domain analogy (HARD)::

    {
        "source_domain": "biology",
        "target_domain": "chemistry",
        "source_relation": "DNA transcription produces mRNA",
        "analogy_template": "??? produces reaction product",
        "ground_truth": "catalyst",
        "isomorphism_type": "production_relation",
    }
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional


# ---------------------------------------------------------------------------
# Difficulty levels
# ---------------------------------------------------------------------------

class DifficultyLevel(Enum):
    """Parametric difficulty for benchmark instances.

    EASY:   Shallow derivations (depth 1-2), single domain, no loops.
    MEDIUM: Moderate depth (3-5), may involve loops or 2 domains.
    HARD:   Deep derivations (6+), multi-domain, nested loops.
    """
    EASY = "easy"
    MEDIUM = "medium"
    HARD = "hard"


# ---------------------------------------------------------------------------
# Task types
# ---------------------------------------------------------------------------

class TaskType(Enum):
    """Enumeration of all benchmark task types.

    Structural reasoning tasks:
        SYLLOGISM_VALIDATION      Grammar DAG analysis on logical arguments.
        ARGUMENT_DECOMPOSITION    Extract S->R->O triples (skeleton extraction).
        CIRCULAR_REASONING        Detect cycles in argument structure.
        CROSS_DOMAIN_ANALOGY      Isomorphism discovery between domains.
        DERIVATION_COMPLETION     Forward derivation: predict next steps.
        ORIGIN_RECONSTRUCTION     Backward derivation: trace to origins.

    Compression tasks:
        LOSSLESS_COMPRESSION      MNEMO round-trip: compress, decompress, verify.
        INFORMATION_DENSITY       Bits of reasoning per character.

    Multi-domain tasks:
        CHEMISTRY_BIOLOGY         Problems spanning chemistry and biology.
        MATH_PHYSICS              Problems spanning math and physics.
        LINGUISTICS_ETYMOLOGY     Problems spanning linguistics and etymology.
    """

    # -- Structural reasoning --
    SYLLOGISM_VALIDATION = "syllogism_validation"
    ARGUMENT_DECOMPOSITION = "argument_decomposition"
    CIRCULAR_REASONING = "circular_reasoning"
    CROSS_DOMAIN_ANALOGY = "cross_domain_analogy"
    DERIVATION_COMPLETION = "derivation_completion"
    ORIGIN_RECONSTRUCTION = "origin_reconstruction"

    # -- Compression --
    LOSSLESS_COMPRESSION = "lossless_compression"
    INFORMATION_DENSITY = "information_density"

    # -- Multi-domain --
    CHEMISTRY_BIOLOGY = "chemistry_biology"
    MATH_PHYSICS = "math_physics"
    LINGUISTICS_ETYMOLOGY = "linguistics_etymology"


# Grouped lists for convenience.
STRUCTURAL_TASKS: List[TaskType] = [
    TaskType.SYLLOGISM_VALIDATION,
    TaskType.ARGUMENT_DECOMPOSITION,
    TaskType.CIRCULAR_REASONING,
    TaskType.CROSS_DOMAIN_ANALOGY,
    TaskType.DERIVATION_COMPLETION,
    TaskType.ORIGIN_RECONSTRUCTION,
]

COMPRESSION_TASKS: List[TaskType] = [
    TaskType.LOSSLESS_COMPRESSION,
    TaskType.INFORMATION_DENSITY,
]

MULTIDOMAIN_TASKS: List[TaskType] = [
    TaskType.CHEMISTRY_BIOLOGY,
    TaskType.MATH_PHYSICS,
    TaskType.LINGUISTICS_ETYMOLOGY,
]

ALL_TASK_TYPES: List[TaskType] = STRUCTURAL_TASKS + COMPRESSION_TASKS + MULTIDOMAIN_TASKS


# ---------------------------------------------------------------------------
# Task descriptor metadata
# ---------------------------------------------------------------------------

TASK_METADATA: Dict[TaskType, Dict[str, Any]] = {
    TaskType.SYLLOGISM_VALIDATION: {
        "name": "Syllogism Validation",
        "category": "structural",
        "glm_capability": "Grammar DAG analysis",
        "description": (
            "Given a set of premises and a candidate conclusion, determine "
            "whether the conclusion follows logically.  The GLM represents "
            "the argument as a directed acyclic graph of rule applications "
            "and checks reachability."
        ),
        "metrics": ["accuracy", "time_ms", "derivation_depth"],
    },
    TaskType.ARGUMENT_DECOMPOSITION: {
        "name": "Argument Decomposition",
        "category": "structural",
        "glm_capability": "Skeleton extraction (S->R->O triples)",
        "description": (
            "Extract subject-relation-object triples from natural language "
            "arguments.  The GLM uses its syntactic grammar to parse the "
            "sentence and its logic grammar to identify the relational "
            "skeleton."
        ),
        "metrics": ["f1_score", "precision", "recall", "time_ms"],
    },
    TaskType.CIRCULAR_REASONING: {
        "name": "Circular Reasoning Detection",
        "category": "structural",
        "glm_capability": "Strange loop detection",
        "description": (
            "Identify cycles in argument chains.  The GLM models the "
            "argument as a grammar and uses find_loops() to detect "
            "self-referential cycles -- the same mechanism that finds "
            "strange loops in any grammar."
        ),
        "metrics": ["accuracy", "cycle_precision", "cycle_recall", "time_ms"],
    },
    TaskType.CROSS_DOMAIN_ANALOGY: {
        "name": "Cross-Domain Analogy",
        "category": "structural",
        "glm_capability": "Isomorphism discovery",
        "description": (
            "Complete analogies across domains: 'X is to domain_A as ??? "
            "is to domain_B'.  The GLM uses find_isomorphisms() to discover "
            "structural mappings between grammars from different domains."
        ),
        "metrics": ["accuracy", "confidence", "time_ms"],
    },
    TaskType.DERIVATION_COMPLETION: {
        "name": "Derivation Completion",
        "category": "structural",
        "glm_capability": "Forward derivation",
        "description": (
            "Given a partial derivation chain, predict the next N steps.  "
            "The GLM applies grammar rules forward (derive/superforecast) "
            "to generate the continuation."
        ),
        "metrics": ["accuracy", "partial_match", "time_ms"],
    },
    TaskType.ORIGIN_RECONSTRUCTION: {
        "name": "Origin Reconstruction",
        "category": "structural",
        "glm_capability": "Backward derivation",
        "description": (
            "Given a derived output, trace backward to recover the "
            "originating form.  The GLM uses reconstruct() -- backward "
            "derivation through the grammar."
        ),
        "metrics": ["accuracy", "partial_match", "time_ms"],
    },
    TaskType.LOSSLESS_COMPRESSION: {
        "name": "Lossless Reasoning Compression",
        "category": "compression",
        "glm_capability": "MNEMO hyper-compression codec",
        "description": (
            "Compress an argument or derivation to MNEMO notation, then "
            "decompress and verify semantic equivalence.  Tests MNEMO's "
            "ability to losslessly encode reasoning chains."
        ),
        "metrics": ["round_trip_fidelity", "compression_ratio", "time_ms"],
    },
    TaskType.INFORMATION_DENSITY: {
        "name": "Information Density",
        "category": "compression",
        "glm_capability": "MNEMO compression efficiency",
        "description": (
            "Measure bits of reasoning content per character of compressed "
            "representation.  Compares MNEMO density to raw text density."
        ),
        "metrics": ["bits_per_char", "reasoning_units_per_token", "time_ms"],
    },
    TaskType.CHEMISTRY_BIOLOGY: {
        "name": "Chemistry-Biology Cross-Domain",
        "category": "multi_domain",
        "glm_capability": "Fugue composition (chemistry + biology)",
        "description": (
            "Problems requiring simultaneous chemical and biological "
            "reasoning: e.g., predict enzyme function from molecular "
            "structure, or determine how a chemical reaction affects a "
            "biological pathway."
        ),
        "metrics": ["accuracy", "domain_coverage", "time_ms"],
    },
    TaskType.MATH_PHYSICS: {
        "name": "Math-Physics Cross-Domain",
        "category": "multi_domain",
        "glm_capability": "Fugue composition (math + physics)",
        "description": (
            "Problems requiring mathematical derivation to solve physical "
            "problems: e.g., derive a conservation law from symmetry, "
            "or predict physical behaviour from differential equations."
        ),
        "metrics": ["accuracy", "derivation_correctness", "time_ms"],
    },
    TaskType.LINGUISTICS_ETYMOLOGY: {
        "name": "Linguistics-Etymology Cross-Domain",
        "category": "multi_domain",
        "glm_capability": "Fugue composition (linguistics + etymology)",
        "description": (
            "Problems requiring both synchronic linguistic analysis and "
            "diachronic etymological tracing: e.g., predict a word's "
            "meaning from its morphological and historical root structure."
        ),
        "metrics": ["accuracy", "morpheme_precision", "time_ms"],
    },
}


# ---------------------------------------------------------------------------
# BenchmarkTask — a single benchmark instance
# ---------------------------------------------------------------------------

@dataclass
class BenchmarkTask:
    """A single benchmark test instance.

    This is the unit of work: one question with a known answer, generated
    from grammar rules so the ground truth is provably correct.

    Attributes:
        id:             Unique identifier (task_type + difficulty + index).
        task_type:      Which benchmark task this instance belongs to.
        difficulty:     Difficulty level (easy / medium / hard).
        input_data:     The problem statement / input for the solver.
        ground_truth:   The known-correct answer.
        grammar_trace:  The grammar rules used to generate this instance
                        (the proof that ground_truth is correct).
        metadata:       Additional information (domain, depth, etc.).
    """

    id: str
    task_type: TaskType
    difficulty: DifficultyLevel
    input_data: Dict[str, Any]
    ground_truth: Any
    grammar_trace: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def category(self) -> str:
        """Return the high-level category (structural / compression / multi_domain)."""
        return TASK_METADATA[self.task_type]["category"]

    @property
    def glm_capability(self) -> str:
        """Return which GLM capability this task exercises."""
        return TASK_METADATA[self.task_type]["glm_capability"]

    def __repr__(self) -> str:
        return (
            f"BenchmarkTask(id={self.id!r}, "
            f"type={self.task_type.value}, "
            f"difficulty={self.difficulty.value})"
        )
