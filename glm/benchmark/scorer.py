"""
scorer.py -- Scoring and metrics for GLM benchmark results.

Computes accuracy, efficiency, compression, and comparison metrics.
Each task type has its own scoring function because the notion of
"correct" varies: syllogism validation is binary (valid/invalid),
argument decomposition uses F1 over extracted triples, compression
tasks use fidelity ratios, and so on.

Metrics computed:
    - accuracy:              Fraction of correct answers (per task type).
    - f1_score:              Precision/recall/F1 for extraction tasks.
    - compression_ratio:     original_size / compressed_size.
    - round_trip_fidelity:   Semantic equivalence after compress/decompress.
    - time_ms:               Wall-clock time per instance.
    - compute_efficiency:    Accuracy per estimated MFLOP.
    - bits_per_char:         Information density of MNEMO representation.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from .tasks import BenchmarkTask, TaskType


# ---------------------------------------------------------------------------
# Scored result for a single instance
# ---------------------------------------------------------------------------

@dataclass
class ScoredResult:
    """The scored outcome of running a solver on a single BenchmarkTask.

    Attributes:
        task_id:        ID of the benchmark task.
        task_type:      Which task type this belongs to.
        solver:         Name of the solver ("glm" or "llm:model_name").
        answer:         The solver's raw answer.
        is_correct:     Whether the answer matches ground truth.
        score:          Numerical score in [0, 1] (may be partial credit).
        time_ms:        Wall-clock execution time in milliseconds.
        metrics:        Task-type-specific metrics (e.g., precision, recall).
        estimated_flops: Estimated FLOPs consumed by this answer.
    """

    task_id: str
    task_type: TaskType
    solver: str
    answer: Any
    is_correct: bool
    score: float
    time_ms: float
    metrics: Dict[str, Any] = field(default_factory=dict)
    estimated_flops: float = 0.0

    @property
    def compute_efficiency(self) -> float:
        """Accuracy per megaFLOP (higher is better)."""
        if self.estimated_flops <= 0:
            return float("inf") if self.score > 0 else 0.0
        mflops = self.estimated_flops / 1e6
        return self.score / mflops


# ---------------------------------------------------------------------------
# BenchmarkScorer
# ---------------------------------------------------------------------------

class BenchmarkScorer:
    """Scores solver outputs against ground truth.

    Dispatches to task-type-specific scoring functions and computes
    both per-instance and aggregate metrics.
    """

    # Estimated FLOPs per token for different solvers.
    # GLM: ~370K params, ~2 FLOPs per param per token = ~740K FLOPs/token.
    # LLM: ~200B params (Opus-class), ~2 FLOPs per param per token = ~400B FLOPs/token.
    FLOP_ESTIMATES = {
        "glm": 740_000,        # 370K params * 2
        "llm": 400_000_000_000,  # 200B params * 2
    }

    def score(
        self,
        task: BenchmarkTask,
        answer: Any,
        solver: str,
        time_ms: float,
    ) -> ScoredResult:
        """Score a single solver output against ground truth.

        Parameters:
            task:     The benchmark task with ground truth.
            answer:   The solver's answer.
            solver:   Solver identifier ("glm" or "llm:model_name").
            time_ms:  How long the solver took.

        Returns:
            A ScoredResult with detailed metrics.
        """
        dispatch = {
            TaskType.SYLLOGISM_VALIDATION: self._score_syllogism,
            TaskType.ARGUMENT_DECOMPOSITION: self._score_decomposition,
            TaskType.CIRCULAR_REASONING: self._score_circular,
            TaskType.CROSS_DOMAIN_ANALOGY: self._score_analogy,
            TaskType.DERIVATION_COMPLETION: self._score_derivation_completion,
            TaskType.ORIGIN_RECONSTRUCTION: self._score_reconstruction,
            TaskType.LOSSLESS_COMPRESSION: self._score_compression,
            TaskType.INFORMATION_DENSITY: self._score_density,
            TaskType.CHEMISTRY_BIOLOGY: self._score_multi_domain,
            TaskType.MATH_PHYSICS: self._score_multi_domain,
            TaskType.LINGUISTICS_ETYMOLOGY: self._score_multi_domain,
        }

        score_fn = dispatch.get(task.task_type, self._score_generic)
        is_correct, score, metrics = score_fn(task, answer)

        # Estimate compute.
        solver_base = solver.split(":")[0]
        flops_per_token = self.FLOP_ESTIMATES.get(solver_base, self.FLOP_ESTIMATES["llm"])
        # Rough token estimate: answer size / 4 chars per token.
        answer_tokens = max(1, len(str(answer)) // 4)
        estimated_flops = flops_per_token * answer_tokens

        return ScoredResult(
            task_id=task.id,
            task_type=task.task_type,
            solver=solver,
            answer=answer,
            is_correct=is_correct,
            score=score,
            time_ms=time_ms,
            metrics=metrics,
            estimated_flops=estimated_flops,
        )

    def aggregate(self, results: List[ScoredResult]) -> Dict[str, Any]:
        """Compute aggregate metrics across a list of scored results.

        Returns a dict with per-task-type and overall statistics.
        """
        if not results:
            return {"overall": {}, "by_task_type": {}, "by_solver": {}}

        overall = self._aggregate_group(results)

        by_task_type: Dict[str, Any] = {}
        by_solver: Dict[str, Any] = {}

        # Group by task type.
        task_groups: Dict[str, List[ScoredResult]] = {}
        for r in results:
            key = r.task_type.value
            task_groups.setdefault(key, []).append(r)

        for key, group in task_groups.items():
            by_task_type[key] = self._aggregate_group(group)

        # Group by solver.
        solver_groups: Dict[str, List[ScoredResult]] = {}
        for r in results:
            solver_groups.setdefault(r.solver, []).append(r)

        for solver, group in solver_groups.items():
            by_solver[solver] = self._aggregate_group(group)

        return {
            "overall": overall,
            "by_task_type": by_task_type,
            "by_solver": by_solver,
        }

    def _aggregate_group(self, results: List[ScoredResult]) -> Dict[str, Any]:
        """Compute aggregate stats for a group of results."""
        n = len(results)
        if n == 0:
            return {}

        correct = sum(1 for r in results if r.is_correct)
        scores = [r.score for r in results]
        times = [r.time_ms for r in results]
        flops = [r.estimated_flops for r in results]

        mean_score = sum(scores) / n
        mean_time = sum(times) / n
        total_flops = sum(flops)
        mean_flops = total_flops / n

        # Compute efficiency: mean accuracy per megaFLOP.
        mflops = total_flops / 1e6 if total_flops > 0 else 1.0
        efficiency = mean_score / mflops

        return {
            "count": n,
            "correct": correct,
            "accuracy": correct / n,
            "mean_score": round(mean_score, 4),
            "median_score": round(sorted(scores)[n // 2], 4),
            "mean_time_ms": round(mean_time, 2),
            "median_time_ms": round(sorted(times)[n // 2], 2),
            "total_flops": total_flops,
            "mean_flops": mean_flops,
            "compute_efficiency": efficiency,
        }

    # -----------------------------------------------------------------------
    # Task-type-specific scorers
    # -----------------------------------------------------------------------

    def _score_syllogism(
        self,
        task: BenchmarkTask,
        answer: Any,
    ) -> tuple[bool, float, Dict[str, Any]]:
        """Score syllogism validation: binary correct/incorrect."""
        expected = task.ground_truth.get("is_valid", None)

        if isinstance(answer, dict):
            predicted = answer.get("is_valid", answer.get("valid", None))
        elif isinstance(answer, bool):
            predicted = answer
        else:
            # Try to interpret string answers.
            ans_str = str(answer).lower().strip()
            predicted = ans_str in ("true", "yes", "valid", "1")

        is_correct = predicted == expected
        return is_correct, 1.0 if is_correct else 0.0, {"expected": expected, "predicted": predicted}

    def _score_decomposition(
        self,
        task: BenchmarkTask,
        answer: Any,
    ) -> tuple[bool, float, Dict[str, Any]]:
        """Score argument decomposition using F1 over extracted triples."""
        expected_triples = task.ground_truth.get("triples", [])
        if isinstance(answer, dict):
            predicted_triples = answer.get("triples", [])
        elif isinstance(answer, list):
            predicted_triples = answer
        else:
            predicted_triples = []

        # Convert to sets of (subject, relation, object) tuples for comparison.
        def triple_key(t: Dict[str, str]) -> tuple:
            return (
                t.get("subject", "").lower(),
                t.get("relation", "").lower(),
                t.get("object", "").lower(),
            )

        expected_set = {triple_key(t) for t in expected_triples}
        predicted_set = {triple_key(t) for t in predicted_triples}

        true_pos = len(expected_set & predicted_set)
        precision = true_pos / max(1, len(predicted_set))
        recall = true_pos / max(1, len(expected_set))
        f1 = (2 * precision * recall) / max(precision + recall, 1e-10)

        is_correct = expected_set == predicted_set
        return is_correct, f1, {
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "f1_score": round(f1, 4),
            "expected_count": len(expected_set),
            "predicted_count": len(predicted_set),
        }

    def _score_circular(
        self,
        task: BenchmarkTask,
        answer: Any,
    ) -> tuple[bool, float, Dict[str, Any]]:
        """Score circular reasoning detection."""
        expected_has_cycle = task.ground_truth.get("has_cycle", False)
        expected_cycle = task.ground_truth.get("cycle", [])

        if isinstance(answer, dict):
            predicted_has_cycle = answer.get("has_cycle", False)
            predicted_cycle = answer.get("cycle", [])
        elif isinstance(answer, bool):
            predicted_has_cycle = answer
            predicted_cycle = []
        else:
            predicted_has_cycle = bool(answer)
            predicted_cycle = []

        # Binary accuracy on cycle detection.
        detection_correct = predicted_has_cycle == expected_has_cycle

        # If there is a cycle, also check if the predicted cycle is valid.
        cycle_score = 0.0
        if expected_has_cycle and detection_correct:
            if predicted_cycle:
                # Check if predicted cycle nodes are a subset of actual cycle.
                expected_nodes = set(expected_cycle)
                predicted_nodes = set(predicted_cycle)
                overlap = len(expected_nodes & predicted_nodes)
                cycle_score = overlap / max(1, len(expected_nodes))
            else:
                cycle_score = 0.5  # Detected cycle but did not identify it.
        elif detection_correct:
            cycle_score = 1.0  # Correctly said no cycle.

        total_score = 0.5 * (1.0 if detection_correct else 0.0) + 0.5 * cycle_score

        return detection_correct, total_score, {
            "detection_correct": detection_correct,
            "cycle_precision": round(cycle_score, 4),
            "expected_has_cycle": expected_has_cycle,
            "predicted_has_cycle": predicted_has_cycle,
        }

    def _score_analogy(
        self,
        task: BenchmarkTask,
        answer: Any,
    ) -> tuple[bool, float, Dict[str, Any]]:
        """Score cross-domain analogy completion."""
        expected = task.ground_truth.get("answer", "")

        if isinstance(answer, dict):
            predicted = answer.get("answer", answer.get("concept", ""))
        elif isinstance(answer, str):
            predicted = answer
        else:
            predicted = str(answer)

        # Exact match.
        is_exact = predicted.lower().strip() == expected.lower().strip()

        # Partial match: check if the expected answer appears in the prediction.
        is_partial = expected.lower() in predicted.lower() if predicted else False

        score = 1.0 if is_exact else (0.5 if is_partial else 0.0)
        return is_exact, score, {
            "expected": expected,
            "predicted": predicted,
            "exact_match": is_exact,
            "partial_match": is_partial,
        }

    def _score_derivation_completion(
        self,
        task: BenchmarkTask,
        answer: Any,
    ) -> tuple[bool, float, Dict[str, Any]]:
        """Score derivation completion (forward prediction)."""
        expected_steps = task.ground_truth.get("remaining_steps", [])
        expected_final = task.ground_truth.get("final_form", "")

        if isinstance(answer, dict):
            predicted_steps = answer.get("remaining_steps", answer.get("steps", []))
            predicted_final = answer.get("final_form", "")
        elif isinstance(answer, list):
            predicted_steps = answer
            predicted_final = predicted_steps[-1].get("output", "") if predicted_steps else ""
        else:
            predicted_steps = []
            predicted_final = str(answer)

        # Score by matching step outputs.
        matched_steps = 0
        for i, exp_step in enumerate(expected_steps):
            if i < len(predicted_steps):
                exp_out = exp_step.get("output", "")
                pred_out = (predicted_steps[i].get("output", "")
                            if isinstance(predicted_steps[i], dict) else str(predicted_steps[i]))
                if str(exp_out).lower() == str(pred_out).lower():
                    matched_steps += 1

        step_accuracy = matched_steps / max(1, len(expected_steps))
        final_correct = str(predicted_final).lower().strip() == str(expected_final).lower().strip()

        score = 0.7 * step_accuracy + 0.3 * (1.0 if final_correct else 0.0)
        is_correct = step_accuracy == 1.0 and final_correct

        return is_correct, score, {
            "step_accuracy": round(step_accuracy, 4),
            "final_correct": final_correct,
            "matched_steps": matched_steps,
            "total_steps": len(expected_steps),
        }

    def _score_reconstruction(
        self,
        task: BenchmarkTask,
        answer: Any,
    ) -> tuple[bool, float, Dict[str, Any]]:
        """Score origin reconstruction (backward derivation)."""
        expected_origin = task.ground_truth.get("origin", "")
        expected_chain = task.ground_truth.get("full_chain", [])

        if isinstance(answer, dict):
            predicted_origin = answer.get("origin", "")
            predicted_chain = answer.get("chain", answer.get("full_chain", []))
        elif isinstance(answer, str):
            predicted_origin = answer
            predicted_chain = []
        else:
            predicted_origin = str(answer)
            predicted_chain = []

        origin_correct = str(predicted_origin).lower().strip() == str(expected_origin).lower().strip()

        # Partial credit for chain overlap.
        chain_score = 0.0
        if predicted_chain and expected_chain:
            matches = sum(
                1 for a, b in zip(predicted_chain, expected_chain)
                if str(a).lower() == str(b).lower()
            )
            chain_score = matches / max(1, len(expected_chain))

        score = 0.6 * (1.0 if origin_correct else 0.0) + 0.4 * chain_score
        return origin_correct, score, {
            "origin_correct": origin_correct,
            "chain_overlap": round(chain_score, 4),
            "expected_origin": expected_origin,
            "predicted_origin": predicted_origin,
        }

    def _score_compression(
        self,
        task: BenchmarkTask,
        answer: Any,
    ) -> tuple[bool, float, Dict[str, Any]]:
        """Score lossless compression round-trip fidelity."""
        original_text = task.ground_truth.get("text", "")

        if isinstance(answer, dict):
            compressed = answer.get("compressed", "")
            decompressed = answer.get("decompressed", "")
            ratio = answer.get("compression_ratio", 0.0)
        else:
            compressed = str(answer)
            decompressed = ""
            ratio = 0.0

        # Fidelity: how much of the original is recovered.
        if decompressed:
            # Simple character-level overlap for now.
            orig_lower = original_text.lower()
            decomp_lower = decompressed.lower()
            if orig_lower == decomp_lower:
                fidelity = 1.0
            else:
                # Jaccard on character trigrams.
                def trigrams(s: str) -> set:
                    return {s[i:i+3] for i in range(max(0, len(s) - 2))}

                orig_tri = trigrams(orig_lower)
                decomp_tri = trigrams(decomp_lower)
                if orig_tri or decomp_tri:
                    fidelity = len(orig_tri & decomp_tri) / max(1, len(orig_tri | decomp_tri))
                else:
                    fidelity = 0.0
        else:
            fidelity = 0.0

        # Compression ratio.
        if compressed and original_text:
            actual_ratio = len(original_text) / max(1, len(compressed))
        else:
            actual_ratio = 0.0

        is_correct = fidelity >= 0.95
        score = fidelity

        return is_correct, score, {
            "fidelity": round(fidelity, 4),
            "compression_ratio": round(actual_ratio, 2),
            "original_length": len(original_text),
            "compressed_length": len(compressed) if compressed else 0,
            "decompressed_length": len(decompressed) if decompressed else 0,
        }

    def _score_density(
        self,
        task: BenchmarkTask,
        answer: Any,
    ) -> tuple[bool, float, Dict[str, Any]]:
        """Score information density measurement."""
        expected_units = task.ground_truth.get("reasoning_units", 0)
        expected_density = task.ground_truth.get("original_density", 0.0)

        if isinstance(answer, dict):
            compressed_chars = answer.get("compressed_chars", 0)
            reasoning_units = answer.get("reasoning_units", 0)
        else:
            compressed_chars = len(str(answer))
            reasoning_units = 0

        if compressed_chars > 0 and reasoning_units > 0:
            compressed_density = reasoning_units / compressed_chars
            density_improvement = compressed_density / max(expected_density, 1e-10)
        else:
            compressed_density = 0.0
            density_improvement = 0.0

        # Score: density improvement > 1.0 means MNEMO is denser than raw text.
        score = min(1.0, density_improvement / 5.0)  # Normalize to [0, 1].
        is_correct = density_improvement > 1.0

        return is_correct, score, {
            "original_density": round(expected_density, 6),
            "compressed_density": round(compressed_density, 6),
            "density_improvement": round(density_improvement, 2),
            "expected_units": expected_units,
        }

    def _score_multi_domain(
        self,
        task: BenchmarkTask,
        answer: Any,
    ) -> tuple[bool, float, Dict[str, Any]]:
        """Score multi-domain (fugue) tasks."""
        expected = task.ground_truth
        expected_step_count = expected.get("step_count", 0)

        if isinstance(answer, dict):
            predicted_step_count = answer.get("step_count", 0)
            domains_covered = answer.get("domains_covered", [])
            derivation_valid = answer.get("derivation_valid", False)
        else:
            predicted_step_count = 0
            domains_covered = []
            derivation_valid = False

        # Check domain coverage.
        expected_domains = set(task.metadata.get("domains", []))
        covered_domains = set(domains_covered)
        domain_coverage = len(expected_domains & covered_domains) / max(1, len(expected_domains))

        # Check step count accuracy.
        step_accuracy = 1.0 - abs(predicted_step_count - expected_step_count) / max(1, expected_step_count)
        step_accuracy = max(0.0, step_accuracy)

        score = 0.4 * domain_coverage + 0.3 * step_accuracy + 0.3 * (1.0 if derivation_valid else 0.0)
        is_correct = domain_coverage == 1.0 and derivation_valid

        return is_correct, score, {
            "domain_coverage": round(domain_coverage, 4),
            "step_accuracy": round(step_accuracy, 4),
            "derivation_valid": derivation_valid,
            "expected_domains": list(expected_domains),
            "covered_domains": domains_covered,
        }

    def _score_generic(
        self,
        task: BenchmarkTask,
        answer: Any,
    ) -> tuple[bool, float, Dict[str, Any]]:
        """Fallback scorer: exact equality check."""
        is_correct = answer == task.ground_truth
        return is_correct, 1.0 if is_correct else 0.0, {}
