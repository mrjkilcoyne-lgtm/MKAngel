"""
Collaborative mode for MKAngel.

Multi-agent collaboration where multiple Angel instances focus on
different domains simultaneously -- like voices in a fugue, each
tackling the problem from their own grammatical perspective, then
harmonising their results.
"""

from __future__ import annotations

import time
import threading
from dataclasses import dataclass, field
from typing import Any


@dataclass
class AgentResult:
    """Result from a single agent's work."""

    domain: str
    output: str
    predictions: list[dict[str, Any]]
    strange_loops: int
    duration: float
    error: str | None = None


@dataclass
class CoworkResult:
    """Combined result from all agents in a session."""

    task: str
    agents: list[AgentResult]
    harmonics: list[dict[str, Any]]
    synthesis: str
    total_duration: float


class CoworkSession:
    """Multi-agent collaboration session.

    Spawns multiple Angel instances, each focusing on a different domain.
    Like a fugue: each voice carries the same theme through its own
    grammar, and the combined result reveals patterns no single
    voice could find alone.
    """

    def __init__(self, provider=None):
        """Initialise a cowork session.

        Args:
            provider: An LLM provider for enriched generation.
        """
        self._provider = provider
        self._agents: list[dict[str, Any]] = []
        self._results: list[AgentResult] = []
        self._task: str = ""
        self._active = False

    # ------------------------------------------------------------------
    # Session management
    # ------------------------------------------------------------------

    def start_session(self, task: str) -> str:
        """Start a new collaboration session around a task.

        Args:
            task: The problem or question for the agents to work on.

        Returns:
            Status message.
        """
        self._task = task
        self._results = []
        self._active = True

        # Add default agents covering the core domains
        if not self._agents:
            for domain in ("linguistic", "computational", "biological", "chemical"):
                self.add_agent(domain)

        domains = [a["domain"] for a in self._agents]
        return (
            f"Cowork session started.\n"
            f"Task: {task}\n"
            f"Agents: {', '.join(domains)}\n"
            f"Each agent will analyse the task through its domain grammar."
        )

    def add_agent(self, domain: str, config: dict[str, Any] | None = None) -> str:
        """Add an agent focused on a specific domain.

        Args:
            domain: Grammar domain (linguistic, computational, etc.).
            config: Optional agent configuration.

        Returns:
            Confirmation message.
        """
        # Check for duplicates
        for a in self._agents:
            if a["domain"] == domain:
                return f"Agent for domain '{domain}' already exists."

        self._agents.append({
            "domain": domain,
            "config": config or {},
            "added_at": time.time(),
        })
        return f"Agent added: {domain}"

    def remove_agent(self, domain: str) -> str:
        """Remove an agent by domain."""
        for i, a in enumerate(self._agents):
            if a["domain"] == domain:
                self._agents.pop(i)
                return f"Agent removed: {domain}"
        return f"No agent found for domain '{domain}'."

    def list_agents(self) -> list[dict[str, Any]]:
        """List all agents in the session."""
        return [
            {
                "domain": a["domain"],
                "config": a["config"],
                "added_at": a["added_at"],
            }
            for a in self._agents
        ]

    # ------------------------------------------------------------------
    # Execution
    # ------------------------------------------------------------------

    def run(self, task: str | None = None) -> CoworkResult:
        """Run all agents on the task and collect results.

        Each agent processes the task through its domain grammar
        concurrently (using threads for parallelism).

        Args:
            task: Override the session task, or use the current one.

        Returns:
            Combined CoworkResult with all agent outputs.
        """
        if task:
            self._task = task

        if not self._task:
            return CoworkResult(
                task="",
                agents=[],
                harmonics=[],
                synthesis="No task specified. Use start_session(task) first.",
                total_duration=0.0,
            )

        start = time.time()
        self._results = []

        # Create a single shared Angel -- avoids resource contention
        # from multiple threads each instantiating their own.
        from glm.angel import Angel
        angel = Angel()
        angel.awaken()

        # Run agents concurrently using threads
        threads: list[threading.Thread] = []
        results_lock = threading.Lock()

        for agent_info in self._agents:
            t = threading.Thread(
                target=self._run_agent,
                args=(agent_info, results_lock, angel),
                daemon=True,
            )
            threads.append(t)
            t.start()

        # Wait for all agents to complete (with timeout)
        for t in threads:
            t.join(timeout=30.0)

        total_duration = time.time() - start

        # Find harmonics -- where agents agree
        harmonics = self._find_harmonics()

        # Synthesise results
        synthesis = self._synthesise()

        return CoworkResult(
            task=self._task,
            agents=list(self._results),
            harmonics=harmonics,
            synthesis=synthesis,
            total_duration=total_duration,
        )

    def collect_results(self) -> CoworkResult:
        """Collect and synthesise results from all agents.

        Alias for run() that uses the existing task.
        """
        return self.run()

    def _run_agent(
        self,
        agent_info: dict[str, Any],
        lock: threading.Lock,
        angel: Any = None,
    ) -> None:
        """Run a single agent (executed in a thread)."""
        domain = agent_info["domain"]
        start = time.time()
        output = ""
        predictions: list[dict[str, Any]] = []
        loops = 0
        error = None

        try:
            if angel is None:
                from glm.angel import Angel
                angel = Angel()
                angel.awaken()

            tokens = self._task.lower().split()

            # Get predictions from this domain's grammar
            try:
                preds = angel.predict(tokens, domain=domain, horizon=8)
                predictions = preds
            except Exception:
                predictions = []

            # Get strange loop count
            loops = len(angel._strange_loops)

            # Build output from predictions
            if predictions:
                pred_lines = []
                for p in predictions[:5]:
                    pred_lines.append(
                        f"  {p.get('grammar','?')}: {p.get('predicted','?')} "
                        f"(conf={p.get('confidence',0):.2f})"
                    )
                output = (
                    f"Domain: {domain}\n"
                    f"Predictions:\n" + "\n".join(pred_lines)
                )
            else:
                output = (
                    f"Domain: {domain}\n"
                    f"No strong predictions from this domain's grammar."
                )

            # If we have an API provider, enrich the response
            if self._provider is not None:
                enriched = self._provider.generate(
                    f"As an expert in {domain}, analyse this task: "
                    f"{self._task}\n\nProvide insights from a {domain} "
                    f"perspective.",
                    system=(
                        f"You are an expert in {domain}. Provide analysis "
                        f"from the perspective of {domain} grammar and "
                        f"structure. Be concise and insightful."
                    ),
                    temperature=0.5,
                    max_tokens=512,
                )
                output = enriched

        except Exception as exc:
            error = str(exc)
            output = f"Agent error: {exc}"

        duration = time.time() - start
        result = AgentResult(
            domain=domain,
            output=output,
            predictions=predictions,
            strange_loops=loops,
            duration=duration,
            error=error,
        )

        with lock:
            self._results.append(result)

    # ------------------------------------------------------------------
    # Synthesis
    # ------------------------------------------------------------------

    def _find_harmonics(self) -> list[dict[str, Any]]:
        """Find where multiple agents' predictions agree."""
        # Collect all predicted outputs keyed by their string form
        prediction_map: dict[str, list[str]] = {}
        for result in self._results:
            for pred in result.predictions:
                key = str(pred.get("predicted", ""))
                if key:
                    if key not in prediction_map:
                        prediction_map[key] = []
                    if result.domain not in prediction_map[key]:
                        prediction_map[key].append(result.domain)

        # Harmonics are predictions shared across 2+ domains
        harmonics = []
        for prediction, domains in prediction_map.items():
            if len(domains) > 1:
                harmonics.append({
                    "prediction": prediction,
                    "domains": domains,
                    "strength": len(domains) / max(len(self._results), 1),
                })

        harmonics.sort(key=lambda h: h["strength"], reverse=True)
        return harmonics

    def _synthesise(self) -> str:
        """Synthesise all agent results into a unified response."""
        if not self._results:
            return "No agent results to synthesise."

        # If we have an API provider, do an intelligent synthesis
        if self._provider is not None:
            agent_summaries = []
            for r in self._results:
                agent_summaries.append(
                    f"[{r.domain}] ({r.duration:.1f}s): {r.output}"
                )
            combined = "\n\n".join(agent_summaries)

            harmonics_str = ""
            harmonics = self._find_harmonics()
            if harmonics:
                h_lines = [
                    f"  {h['prediction']} (domains: {', '.join(h['domains'])})"
                    for h in harmonics[:5]
                ]
                harmonics_str = (
                    "\n\nCross-domain harmonics found:\n" + "\n".join(h_lines)
                )

            return self._provider.generate(
                f"Synthesise these multi-domain analyses of the task "
                f"'{self._task}':\n\n{combined}{harmonics_str}\n\n"
                f"Provide a unified insight that combines all perspectives.",
                system=(
                    "You are synthesising insights from multiple domain "
                    "experts. Find common themes, highlight unique insights, "
                    "and provide a unified understanding. Be concise."
                ),
                temperature=0.4,
                max_tokens=1024,
            )

        # Local synthesis: combine structurally
        parts = []
        parts.append(f"Task: {self._task}")
        parts.append(f"Agents: {len(self._results)}")
        parts.append("")

        for r in self._results:
            status = "OK" if r.error is None else f"ERROR: {r.error}"
            parts.append(f"--- {r.domain.upper()} [{status}] ({r.duration:.2f}s) ---")
            parts.append(r.output)
            parts.append("")

        harmonics = self._find_harmonics()
        if harmonics:
            parts.append("--- HARMONICS (cross-domain agreement) ---")
            for h in harmonics[:5]:
                parts.append(
                    f"  '{h['prediction']}' found in: "
                    f"{', '.join(h['domains'])} "
                    f"(strength: {h['strength']:.2f})"
                )
            parts.append("")

        total_preds = sum(len(r.predictions) for r in self._results)
        total_loops = sum(r.strange_loops for r in self._results)
        parts.append(
            f"Total: {total_preds} predictions, {total_loops} strange loops, "
            f"{len(harmonics)} harmonics."
        )

        return "\n".join(parts)

    # ------------------------------------------------------------------
    # Display
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        status = "active" if self._active else "idle"
        agents = len(self._agents)
        return f"CoworkSession({status}, agents={agents})"
