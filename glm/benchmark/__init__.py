"""
benchmark — Reasoning benchmark harness for the Grammar Language Model.

Measures how grammar-driven structural decomposition compares to brute-force
LLM reasoning across structural reasoning, compression, and multi-domain
tasks.

The thesis: a 370K parameter grammar-based model can match or exceed frontier
LLM reasoning (Opus 4.6, GPT 5.4) on structural reasoning tasks at a
fraction of the compute.

Usage::

    from glm.benchmark import BenchmarkSuite, run_benchmarks

    suite = BenchmarkSuite()
    results = run_benchmarks(suite)
    print(results.summary())

Architecture
------------
    tasks.py      Task type definitions and difficulty levels
    generator.py  Deterministic test case generation from grammar rules
    scorer.py     Accuracy, efficiency, and compression metrics
    runner.py     Execution harness for GLM and optional LLM baseline
    report.py     Text-based comparison reports
"""

from .tasks import (
    TaskType,
    DifficultyLevel,
    BenchmarkTask,
    STRUCTURAL_TASKS,
    COMPRESSION_TASKS,
    MULTIDOMAIN_TASKS,
    ALL_TASK_TYPES,
)
from .generator import BenchmarkGenerator
from .scorer import BenchmarkScorer, ScoredResult
from .runner import BenchmarkRunner, BenchmarkResult
from .report import BenchmarkReport


class BenchmarkSuite:
    """Top-level handle for the GLM reasoning benchmark suite.

    Combines generation, execution, scoring, and reporting into a single
    entry point.  All randomness is seeded for reproducibility.

    Attributes:
        seed:            RNG seed for deterministic generation.
        instances_per:   Number of instances per (task_type, difficulty) pair.
        difficulty_levels: Which difficulty levels to generate.
        generator:       The benchmark instance generator.
        runner:          The benchmark execution harness.
        scorer:          The scoring and metrics engine.
    """

    def __init__(
        self,
        seed: int = 42,
        instances_per: int = 100,
        difficulty_levels: list | None = None,
        llm_api_key: str | None = None,
        llm_model: str = "claude-opus-4-6",
    ) -> None:
        self.seed = seed
        self.instances_per = instances_per
        self.difficulty_levels = difficulty_levels or [
            DifficultyLevel.EASY,
            DifficultyLevel.MEDIUM,
            DifficultyLevel.HARD,
        ]
        self.generator = BenchmarkGenerator(seed=seed)
        self.scorer = BenchmarkScorer()
        self.runner = BenchmarkRunner(
            llm_api_key=llm_api_key,
            llm_model=llm_model,
        )

    def generate(self) -> list[BenchmarkTask]:
        """Generate all benchmark instances.

        Returns:
            A flat list of BenchmarkTask instances across all task types
            and difficulty levels.
        """
        tasks: list[BenchmarkTask] = []
        for task_type in ALL_TASK_TYPES:
            for difficulty in self.difficulty_levels:
                batch = self.generator.generate_batch(
                    task_type=task_type,
                    difficulty=difficulty,
                    count=self.instances_per,
                )
                tasks.extend(batch)
        return tasks

    def run(
        self,
        tasks: list[BenchmarkTask] | None = None,
        run_llm: bool = False,
    ) -> BenchmarkResult:
        """Generate (if needed) and execute benchmarks.

        Parameters:
            tasks:    Pre-generated tasks, or None to generate fresh.
            run_llm:  Whether to also run the LLM baseline.

        Returns:
            A BenchmarkResult with all scored outcomes.
        """
        if tasks is None:
            tasks = self.generate()
        return self.runner.run_all(
            tasks=tasks,
            scorer=self.scorer,
            run_llm=run_llm,
        )

    def report(self, result: BenchmarkResult) -> str:
        """Generate a text report from benchmark results.

        Returns:
            A formatted string with tables, per-task breakdowns, and
            GLM vs LLM comparison (if LLM baseline was run).
        """
        reporter = BenchmarkReport(result)
        return reporter.generate()


def run_benchmarks(
    suite: BenchmarkSuite | None = None,
    run_llm: bool = False,
) -> BenchmarkResult:
    """Convenience entry point: generate, run, and return results.

    Parameters:
        suite:    A configured BenchmarkSuite, or None for defaults.
        run_llm:  Whether to run the LLM baseline.

    Returns:
        A BenchmarkResult ready for reporting.
    """
    if suite is None:
        suite = BenchmarkSuite()
    tasks = suite.generate()
    return suite.run(tasks=tasks, run_llm=run_llm)


__all__ = [
    "BenchmarkSuite",
    "BenchmarkResult",
    "run_benchmarks",
    "TaskType",
    "DifficultyLevel",
    "BenchmarkTask",
    "BenchmarkGenerator",
    "BenchmarkScorer",
    "BenchmarkRunner",
    "BenchmarkReport",
]
