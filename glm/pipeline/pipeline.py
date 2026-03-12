"""
pipeline.py -- The ReasoningPipeline: GSM's 4-stage chain via grammar engine.

The ReasoningPipeline orchestrates the four stages:

    SKELETON -> DAG -> DISCONFIRM -> SYNTHESIS

Each stage receives the output of the previous stage and produces its
own typed result.  The pipeline collects all results into a single
PipelineResult for full audit-trail inspection.

The pipeline also supports:
    - Running individual stages in isolation (for testing or targeted use).
    - Injecting custom grammars per stage.
    - Timing each stage for performance profiling.
    - Configuring stage behavior via a config dict.

The pipeline does NOT call any LLM API.  All reasoning is performed by
grammar rules, productions, derivations, and loop detection -- the GLM
IS the reasoning engine.
"""

from __future__ import annotations

import time
from typing import Any, Dict, List, Optional

from glm.core.grammar import Grammar
from glm.core.engine import DerivationEngine

from .result import (
    DAGResult,
    DisconfirmResult,
    PipelineResult,
    SkeletonResult,
    SynthesisResult,
)
from .stages import (
    DAGStage,
    DisconfirmStage,
    PipelineStage,
    SkeletonStage,
    SynthesisStage,
)


class ReasoningPipeline:
    """The GSM reasoning pipeline: skeleton -> dag -> disconfirm -> synthesis.

    This is the top-level orchestrator.  It wires the four stages together,
    passes output from one to the next, and collects the full audit trail
    into a PipelineResult.

    The pipeline can be used in two modes:

    1. **Full pipeline** -- call ``run(input)`` to execute all four stages
       in sequence and get a complete PipelineResult.

    2. **Individual stages** -- call ``run_stage(stage_name, input)`` to
       execute a single stage.  Useful for testing, debugging, or when
       you only need one specific analysis (e.g., just skeleton extraction).

    Parameters:
        engine:     A shared DerivationEngine instance.  If not provided,
                    a new one is created.
        grammars:   Dict mapping grammar names to Grammar instances.
                    These are shared across all stages.  Stages use grammars
                    relevant to their operation (e.g., SkeletonStage uses
                    "syntactic" and "morphological").
        config:     Optional configuration dict.  Keys:
                    - ``max_derivation_steps``: int, caps derivation depth.
                    - ``skeleton``, ``dag``, ``disconfirm``, ``synthesis``:
                      per-stage config dicts.

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

    def __init__(
        self,
        engine: Optional[DerivationEngine] = None,
        grammars: Optional[Dict[str, Grammar]] = None,
        config: Optional[Dict[str, Any]] = None,
    ) -> None:
        self.engine = engine or DerivationEngine()
        self.grammars = grammars or {}
        self.config = config or {}

        # Instantiate stages with shared engine and grammars.
        self._stages: Dict[str, PipelineStage] = {
            "skeleton": SkeletonStage(
                engine=self.engine,
                grammars=self.grammars,
                config=self.config.get("skeleton", {}),
            ),
            "dag": DAGStage(
                engine=self.engine,
                grammars=self.grammars,
                config=self.config.get("dag", {}),
            ),
            "disconfirm": DisconfirmStage(
                engine=self.engine,
                grammars=self.grammars,
                config=self.config.get("disconfirm", {}),
            ),
            "synthesis": SynthesisStage(
                engine=self.engine,
                grammars=self.grammars,
                config=self.config.get("synthesis", {}),
            ),
        }

        # Stage execution order.
        self._stage_order = ["skeleton", "dag", "disconfirm", "synthesis"]

    # -- full pipeline execution --------------------------------------------

    def run(self, input_data: Any) -> PipelineResult:
        """Execute the full 4-stage reasoning pipeline.

        Parameters:
            input_data: Raw text or structured data to reason about.

        Returns:
            A PipelineResult containing all stage outputs and metadata.
        """
        result = PipelineResult(raw_input=input_data)
        current_input: Any = input_data

        for stage_name in self._stage_order:
            stage = self._stages[stage_name]
            t0 = time.time()

            try:
                stage_output = stage.run(current_input)
            except Exception as e:
                # Record the error but don't crash the pipeline.
                result.metadata[f"{stage_name}_error"] = str(e)
                break

            elapsed = time.time() - t0
            result.stage_timings[stage_name] = round(elapsed, 4)

            # Store the stage result.
            if stage_name == "skeleton":
                result.skeleton = stage_output
            elif stage_name == "dag":
                result.dag = stage_output
            elif stage_name == "disconfirm":
                result.disconfirm = stage_output
            elif stage_name == "synthesis":
                result.synthesis = stage_output

            # Feed output to the next stage.
            current_input = stage_output

        return result

    # -- individual stage execution -----------------------------------------

    def run_stage(
        self,
        stage_name: str,
        input_data: Any,
    ) -> Any:
        """Execute a single stage in isolation.

        Parameters:
            stage_name: One of "skeleton", "dag", "disconfirm", "synthesis".
            input_data: Appropriate input for that stage:
                        - skeleton: raw text
                        - dag: SkeletonResult
                        - disconfirm: DAGResult
                        - synthesis: DisconfirmResult

        Returns:
            The stage's typed result object.

        Raises:
            KeyError: If stage_name is not recognized.
        """
        if stage_name not in self._stages:
            raise KeyError(
                f"Unknown stage '{stage_name}'. "
                f"Valid stages: {list(self._stages.keys())}"
            )

        return self._stages[stage_name].run(input_data)

    # -- partial pipeline execution -----------------------------------------

    def run_through(
        self,
        input_data: Any,
        stop_after: str,
    ) -> PipelineResult:
        """Execute the pipeline but stop after a specific stage.

        Parameters:
            input_data: Raw text or structured data.
            stop_after: Stage name to stop after (inclusive).
                        E.g., "dag" runs skeleton + dag only.

        Returns:
            Partial PipelineResult with only the completed stages.
        """
        if stop_after not in self._stages:
            raise KeyError(
                f"Unknown stage '{stop_after}'. "
                f"Valid stages: {list(self._stages.keys())}"
            )

        result = PipelineResult(raw_input=input_data)
        current_input: Any = input_data

        for stage_name in self._stage_order:
            stage = self._stages[stage_name]
            t0 = time.time()

            try:
                stage_output = stage.run(current_input)
            except Exception as e:
                result.metadata[f"{stage_name}_error"] = str(e)
                break

            elapsed = time.time() - t0
            result.stage_timings[stage_name] = round(elapsed, 4)

            if stage_name == "skeleton":
                result.skeleton = stage_output
            elif stage_name == "dag":
                result.dag = stage_output
            elif stage_name == "disconfirm":
                result.disconfirm = stage_output
            elif stage_name == "synthesis":
                result.synthesis = stage_output

            current_input = stage_output

            if stage_name == stop_after:
                break

        return result

    # -- stage access -------------------------------------------------------

    @property
    def stages(self) -> Dict[str, PipelineStage]:
        """Access the stage instances (for introspection or testing)."""
        return dict(self._stages)

    @property
    def stage_names(self) -> List[str]:
        """Ordered list of stage names."""
        return list(self._stage_order)

    # -- grammar management -------------------------------------------------

    def add_grammar(self, name: str, grammar: Grammar) -> None:
        """Add or replace a grammar.  Propagates to all stages."""
        self.grammars[name] = grammar
        for stage in self._stages.values():
            stage.grammars[name] = grammar

    def remove_grammar(self, name: str) -> None:
        """Remove a grammar from the pipeline and all stages."""
        self.grammars.pop(name, None)
        for stage in self._stages.values():
            stage.grammars.pop(name, None)

    # -- repr ---------------------------------------------------------------

    def __repr__(self) -> str:
        return (
            f"ReasoningPipeline("
            f"stages={self._stage_order}, "
            f"grammars={list(self.grammars.keys())})"
        )
