"""
pipeline -- GSM structural decomposition pipeline for MKAngel's GLM.

This module implements the GSM's 4-stage reasoning pipeline using grammar
operations instead of LLM API calls.  The grammar IS the reasoning engine:
rules fire, productions expand, loops are detected, and derivations are
traced -- all in pure Python, all auditable.

Stages:
    1. SKELETON  -- Strip to logical bones (S-R-O triples, noise removal)
    2. DAG       -- Map dependency graph (roots, cycles, critical path)
    3. DISCONFIRM -- Hunt structural weakness (falsification, fallacies)
    4. SYNTHESIS  -- Minimum viable logic (proven vs unproven, MNEMO compression)

Self-improvement:
    LEARN -- Per-session analysis, generates grammar rule patches
    SLEEP -- Cross-session consolidation, system-level grammar upgrades

Usage:
    >>> from glm.grammars import build_syntactic_grammar, build_morphological_grammar
    >>> from glm.pipeline import ReasoningPipeline
    >>>
    >>> pipeline = ReasoningPipeline(grammars={
    ...     "syntactic": build_syntactic_grammar(),
    ...     "morphological": build_morphological_grammar(),
    ... })
    >>>
    >>> result = pipeline.run("All birds can fly. Penguins are birds.")
    >>> print(result.summary)
"""

from .result import (
    ClaimStrength,
    DAGEdge,
    DAGNode,
    DAGResult,
    DisconfirmResult,
    EvidenceLevel,
    EvidenceNode,
    FallacyType,
    PipelineResult,
    SkeletonResult,
    SynthesisResult,
    Triple,
    WeaknessReport,
)
from .stages import (
    DAGStage,
    DisconfirmStage,
    PipelineStage,
    SkeletonStage,
    SynthesisStage,
)
from .pipeline import ReasoningPipeline
from .learner import GrammarPatch, LearnCycle, SleepCycle

__all__ = [
    # Pipeline
    "ReasoningPipeline",
    "PipelineStage",
    "PipelineResult",
    # Stages
    "SkeletonStage",
    "DAGStage",
    "DisconfirmStage",
    "SynthesisStage",
    # Results
    "SkeletonResult",
    "DAGResult",
    "DisconfirmResult",
    "SynthesisResult",
    "Triple",
    "DAGNode",
    "DAGEdge",
    "EvidenceNode",
    "WeaknessReport",
    # Enums
    "ClaimStrength",
    "EvidenceLevel",
    "FallacyType",
    # Learner
    "GrammarPatch",
    "LearnCycle",
    "SleepCycle",
]
