"""
Swarm intelligence and harness system for MKAngel.

A swarm is a collection of specialised agents that coordinate to solve
problems no single agent could handle alone.  Like neurons in a brain,
bees in a hive, or instruments in an orchestra — the emergent behaviour
of the collective exceeds the sum of its parts.

Architecture:
- Swarm: a collection of agents with a coordination protocol
- Harness: a runtime that manages swarm lifecycle, routing, and synthesis
- Agent roles: explorer, analyst, synthesiser, critic, specialist
- Coordination: broadcast, relay, auction, consensus, hierarchical

The Borges Library principle: the swarm navigates the space of all
possible responses — the Library of Babel — finding the meaningful
needles in the infinite haystack.  Each agent explores a different
corridor of the Library.
"""

from __future__ import annotations

import time
import threading
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Callable


# ---------------------------------------------------------------------------
# Agent roles
# ---------------------------------------------------------------------------

class AgentRole(Enum):
    """Roles agents can play in a swarm."""
    EXPLORER = auto()      # Searches possibility space broadly
    ANALYST = auto()       # Deep analysis of a specific domain
    SYNTHESISER = auto()   # Combines findings from other agents
    CRITIC = auto()        # Evaluates and challenges conclusions
    SPECIALIST = auto()    # Domain expert (math, physics, code, etc.)
    NAVIGATOR = auto()     # Borges Library navigator — finds paths through possibility space
    ORACLE = auto()        # Uses strange loops for prediction
    SCRIBE = auto()        # Records and compresses to MNEMO


class CoordinationProtocol(Enum):
    """How agents coordinate within the swarm."""
    BROADCAST = auto()     # All agents receive all messages
    RELAY = auto()         # Messages pass through chain
    AUCTION = auto()       # Tasks assigned to best-suited agent
    CONSENSUS = auto()     # Agents vote on conclusions
    HIERARCHICAL = auto()  # Tree structure with delegation
    STIGMERGY = auto()     # Indirect coordination via shared state


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class SwarmMessage:
    """A message passed between agents in the swarm."""
    sender: str
    content: str
    message_type: str = "info"  # info, query, result, critique, consensus
    metadata: dict[str, Any] = field(default_factory=dict)
    timestamp: float = 0.0

    def __post_init__(self):
        if not self.timestamp:
            self.timestamp = time.time()


@dataclass
class SwarmAgent:
    """An agent within a swarm."""
    name: str
    role: AgentRole
    domain: str = "general"
    capabilities: list[str] = field(default_factory=list)
    config: dict[str, Any] = field(default_factory=dict)
    active: bool = True

    def describe(self) -> str:
        caps = ", ".join(self.capabilities) if self.capabilities else "general"
        return f"{self.name} [{self.role.name}] domain={self.domain} caps=[{caps}]"


@dataclass
class SwarmResult:
    """Result from a swarm operation."""
    task: str
    agents_used: list[str]
    individual_results: dict[str, Any]
    consensus: str
    confidence: float
    harmonics: list[dict[str, Any]]
    duration: float
    borges_paths_explored: int = 0


# ---------------------------------------------------------------------------
# The Library of Borges — possibility space navigator
# ---------------------------------------------------------------------------

class BorgesLibrary:
    """Navigator through the infinite Library of Babel.

    The Library contains every possible combination of characters —
    every book that could ever be written.  Most are gibberish.  The
    Angel's job is to find the meaningful volumes.

    This is the principle behind the swarm: each agent explores a
    different hexagonal room in the Library, searching for coherent
    text among the noise.  The grammar is the compass — it tells us
    which rooms are likely to contain meaning.

    Like the TARDIS: bigger on the inside than the outside.  A small
    set of grammar rules generates an infinite possibility space.
    The improbability drive: navigating to the improbable-but-correct
    answer by exploiting structural regularities.
    """

    def __init__(self) -> None:
        self._explored_paths: list[dict[str, Any]] = []
        self._promising_corridors: list[dict[str, Any]] = []

    def explore(
        self,
        query: str,
        grammars: list[Any] | None = None,
        depth: int = 5,
    ) -> list[dict[str, Any]]:
        """Explore the Library starting from a query.

        Uses grammar rules as a compass to navigate toward meaningful
        regions of the possibility space.  Without grammar, the Library
        is infinite noise.  With grammar, it becomes a structured search.

        Args:
            query: The question or seed to start exploration.
            grammars: Grammar objects to use as navigation compass.
            depth: How deep to explore each corridor.

        Returns:
            List of promising paths through the Library.
        """
        paths = []
        tokens = query.lower().split()

        if grammars:
            from glm.core.engine import DerivationEngine
            engine = DerivationEngine()
            for grammar in grammars:
                tree = engine.derive(tokens, grammar, direction="forward")
                for path in tree.paths()[:depth]:
                    if path:
                        paths.append({
                            "corridor": grammar.name,
                            "derivation": [d.output for d in path],
                            "depth": len(path),
                            "confidence": path[-1].metadata.get("weight", 0.5),
                        })

        # Record exploration
        exploration = {
            "query": query,
            "paths_found": len(paths),
            "timestamp": time.time(),
        }
        self._explored_paths.append(exploration)

        # Mark promising corridors (high confidence paths)
        for p in paths:
            if p.get("confidence", 0) > 0.6:
                self._promising_corridors.append(p)

        return paths

    def find_improbable(
        self,
        query: str,
        grammars: list[Any] | None = None,
    ) -> list[dict[str, Any]]:
        """The improbability drive — find unexpected but valid answers.

        Explores paths that grammar rules don't strongly prefer, looking
        for surprising connections.  The most interesting answers are
        often improbable — they connect domains that don't obviously relate.
        """
        paths = self.explore(query, grammars, depth=10)

        # Sort by inverse confidence — the most improbable valid paths
        improbable = sorted(paths, key=lambda p: p.get("confidence", 0.5))
        return improbable[:5]

    @property
    def rooms_explored(self) -> int:
        return len(self._explored_paths)

    @property
    def promising_corridors(self) -> list[dict[str, Any]]:
        return list(self._promising_corridors)


# ---------------------------------------------------------------------------
# Swarm — the collective
# ---------------------------------------------------------------------------

class Swarm:
    """A coordinated swarm of agents.

    Each agent is a voice in the fugue.  The swarm is the full orchestra.
    Coordination protocols determine how the voices harmonise.
    """

    def __init__(
        self,
        name: str = "angel_swarm",
        protocol: CoordinationProtocol = CoordinationProtocol.BROADCAST,
    ) -> None:
        self.name = name
        self.protocol = protocol
        self._agents: dict[str, SwarmAgent] = {}
        self._messages: list[SwarmMessage] = []
        self._library = BorgesLibrary()
        self._results: dict[str, Any] = {}
        self._angel: Any = None
        self._angel_lock = threading.Lock()

    # -- Agent management --------------------------------------------------

    def add_agent(
        self,
        name: str,
        role: AgentRole,
        domain: str = "general",
        capabilities: list[str] | None = None,
        config: dict[str, Any] | None = None,
    ) -> SwarmAgent:
        """Add an agent to the swarm."""
        agent = SwarmAgent(
            name=name,
            role=role,
            domain=domain,
            capabilities=capabilities or [],
            config=config or {},
        )
        self._agents[name] = agent
        return agent

    def remove_agent(self, name: str) -> bool:
        if name in self._agents:
            del self._agents[name]
            return True
        return False

    def list_agents(self) -> list[SwarmAgent]:
        return list(self._agents.values())

    # -- Swarm execution ---------------------------------------------------

    def run(
        self,
        task: str,
        provider: Any = None,
        grammars: list[Any] | None = None,
    ) -> SwarmResult:
        """Run the swarm on a task.

        Each agent processes the task according to its role and domain.
        Results are coordinated according to the swarm's protocol.
        The Borges Library navigator searches for unexpected connections.

        Args:
            task: The problem to solve.
            provider: Optional LLM provider for enriched generation.
            grammars: Grammar objects for navigation.

        Returns:
            SwarmResult with coordinated findings.
        """
        start = time.time()
        individual_results: dict[str, Any] = {}

        # If no agents, create a default swarm
        if not self._agents:
            self._create_default_swarm()

        # Run agents concurrently
        threads: list[threading.Thread] = []
        lock = threading.Lock()

        for agent in self._agents.values():
            if not agent.active:
                continue

            t = threading.Thread(
                target=self._run_agent,
                args=(agent, task, provider, grammars, individual_results, lock),
                daemon=True,
            )
            threads.append(t)
            t.start()

        for t in threads:
            t.join(timeout=30.0)

        # Navigate the Borges Library for unexpected connections
        library_paths = self._library.explore(task, grammars)

        # Synthesise results
        consensus = self._build_consensus(task, individual_results, provider)
        harmonics = self._find_swarm_harmonics(individual_results)

        # Calculate confidence
        confidences = []
        for r in individual_results.values():
            if isinstance(r, dict):
                confidences.append(r.get("confidence", 0.5))
        avg_confidence = sum(confidences) / max(len(confidences), 1)

        return SwarmResult(
            task=task,
            agents_used=list(individual_results.keys()),
            individual_results=individual_results,
            consensus=consensus,
            confidence=avg_confidence,
            harmonics=harmonics,
            duration=time.time() - start,
            borges_paths_explored=self._library.rooms_explored,
        )

    def _run_agent(
        self,
        agent: SwarmAgent,
        task: str,
        provider: Any,
        grammars: list[Any] | None,
        results: dict[str, Any],
        lock: threading.Lock,
    ) -> None:
        """Run a single agent on the task."""
        try:
            result = self._execute_agent_role(agent, task, provider, grammars)
            with lock:
                results[agent.name] = result

            # Broadcast result to other agents if using broadcast protocol
            if self.protocol == CoordinationProtocol.BROADCAST:
                self._messages.append(SwarmMessage(
                    sender=agent.name,
                    content=str(result.get("output", "")),
                    message_type="result",
                ))

        except Exception as exc:
            with lock:
                results[agent.name] = {"error": str(exc), "confidence": 0.0}

    def _execute_agent_role(
        self,
        agent: SwarmAgent,
        task: str,
        provider: Any,
        grammars: list[Any] | None,
    ) -> dict[str, Any]:
        """Execute based on agent's role."""
        if agent.role == AgentRole.EXPLORER:
            paths = self._library.explore(task, grammars, depth=8)
            return {
                "output": f"Explored {len(paths)} paths through possibility space",
                "paths": paths,
                "confidence": 0.6,
            }

        elif agent.role == AgentRole.NAVIGATOR:
            improbable = self._library.find_improbable(task, grammars)
            return {
                "output": f"Found {len(improbable)} improbable-but-valid paths",
                "paths": improbable,
                "confidence": 0.5,
            }

        elif agent.role == AgentRole.ANALYST:
            return self._analyse_task(agent, task, provider, grammars)

        elif agent.role == AgentRole.CRITIC:
            return {
                "output": f"Critic [{agent.domain}]: evaluating task assumptions",
                "critiques": [
                    "Are the premises well-defined?",
                    "Are there hidden assumptions?",
                    "What edge cases exist?",
                ],
                "confidence": 0.7,
            }

        elif agent.role == AgentRole.SYNTHESISER:
            return {
                "output": "Awaiting other agents' results for synthesis",
                "confidence": 0.5,
            }

        elif agent.role == AgentRole.SPECIALIST:
            return self._analyse_task(agent, task, provider, grammars)

        elif agent.role == AgentRole.ORACLE:
            return self._oracle_predict(agent, task, grammars)

        elif agent.role == AgentRole.SCRIBE:
            return self._scribe_compress(task)

        return {"output": "No specific handler", "confidence": 0.3}

    def _analyse_task(
        self,
        agent: SwarmAgent,
        task: str,
        provider: Any,
        grammars: list[Any] | None,
    ) -> dict[str, Any]:
        """Analyse task using grammar and optionally a provider."""
        tokens = task.lower().split()
        predictions = []

        if grammars:
            from glm.core.engine import DerivationEngine
            engine = DerivationEngine()
            for grammar in grammars:
                if grammar.domain == agent.domain or agent.domain == "general":
                    tree = engine.derive(tokens, grammar, direction="forward")
                    for path in tree.paths()[:5]:
                        if path:
                            predictions.append({
                                "grammar": grammar.name,
                                "output": path[-1].output,
                                "confidence": path[-1].metadata.get("weight", 0.5),
                            })

        output = f"Analyst [{agent.domain}]: {len(predictions)} grammar derivations"
        if provider:
            output = provider.generate(
                f"As a {agent.domain} specialist, analyse: {task}",
                system=f"You are a {agent.domain} domain expert. Be concise.",
                temperature=0.4,
                max_tokens=512,
            )

        return {
            "output": output,
            "predictions": predictions,
            "confidence": 0.7 if predictions else 0.4,
        }

    def _get_angel(self) -> Any:
        """Get or create the shared Angel instance (thread-safe)."""
        if self._angel is None:
            with self._angel_lock:
                if self._angel is None:
                    from glm.angel import Angel
                    self._angel = Angel()
                    self._angel.awaken()
        return self._angel

    def _oracle_predict(
        self,
        agent: SwarmAgent,
        task: str,
        grammars: list[Any] | None,
    ) -> dict[str, Any]:
        """Oracle uses strange loops for prediction."""
        try:
            angel = self._get_angel()
            forecast = angel.superforecast(
                task.lower().split(),
                domain=agent.domain if agent.domain != "general" else "linguistic",
            )
            return {
                "output": f"Oracle forecast: {len(forecast.get('predictions', []))} predictions",
                "forecast": forecast,
                "confidence": forecast.get("overall_confidence", 0.5),
            }
        except Exception as exc:
            return {"output": f"Oracle error: {exc}", "confidence": 0.2}

    def _scribe_compress(self, task: str) -> dict[str, Any]:
        """Scribe compresses task and results to MNEMO."""
        try:
            from glm.mnemo.language import encode
            mnemo = encode(task)
            return {
                "output": f"MNEMO encoding: {mnemo}",
                "mnemo": mnemo,
                "confidence": 0.6,
            }
        except Exception:
            return {"output": "Scribe: MNEMO encoding unavailable", "confidence": 0.3}

    # -- Coordination ------------------------------------------------------

    def _build_consensus(
        self,
        task: str,
        results: dict[str, Any],
        provider: Any,
    ) -> str:
        """Build consensus from individual agent results."""
        if not results:
            return "No agent results to synthesise."

        if provider:
            summaries = []
            for name, result in results.items():
                if isinstance(result, dict):
                    summaries.append(f"[{name}]: {result.get('output', 'no output')}")
            combined = "\n".join(summaries)
            return provider.generate(
                f"Synthesise these swarm agent results for task '{task}':\n{combined}",
                system="Combine insights from multiple specialist agents. Be concise.",
                temperature=0.3,
                max_tokens=1024,
            )

        parts = [f"Swarm consensus for: {task}", ""]
        for name, result in results.items():
            if isinstance(result, dict):
                parts.append(f"  [{name}]: {result.get('output', 'error')}")
        return "\n".join(parts)

    def _find_swarm_harmonics(
        self, results: dict[str, Any],
    ) -> list[dict[str, Any]]:
        """Find where swarm agents agree."""
        harmonics = []
        outputs = {}
        for name, result in results.items():
            if isinstance(result, dict):
                key = str(result.get("output", ""))[:100]
                if key not in outputs:
                    outputs[key] = []
                outputs[key].append(name)

        for output, agents in outputs.items():
            if len(agents) > 1:
                harmonics.append({
                    "finding": output,
                    "agents": agents,
                    "strength": len(agents) / max(len(results), 1),
                })
        return harmonics

    def _create_default_swarm(self) -> None:
        """Create a default swarm with standard agent composition."""
        self.add_agent("explorer", AgentRole.EXPLORER, "general", ["broad_search"])
        self.add_agent("navigator", AgentRole.NAVIGATOR, "general", ["borges_navigation"])
        self.add_agent("linguist", AgentRole.SPECIALIST, "linguistic", ["grammar_analysis"])
        self.add_agent("mathematician", AgentRole.SPECIALIST, "mathematical", ["proof", "algebra"])
        self.add_agent("physicist", AgentRole.SPECIALIST, "physics", ["mechanics", "quantum"])
        self.add_agent("coder", AgentRole.SPECIALIST, "computational", ["code_generation"])
        self.add_agent("oracle", AgentRole.ORACLE, "linguistic", ["strange_loops", "forecasting"])
        self.add_agent("scribe", AgentRole.SCRIBE, "meta", ["mnemo_encoding"])
        self.add_agent("critic", AgentRole.CRITIC, "general", ["evaluation"])
        self.add_agent("synthesiser", AgentRole.SYNTHESISER, "general", ["synthesis"])

    # -- Properties --------------------------------------------------------

    @property
    def library(self) -> BorgesLibrary:
        """Access the Borges Library navigator."""
        return self._library

    def __repr__(self) -> str:
        n = len(self._agents)
        active = sum(1 for a in self._agents.values() if a.active)
        return f"Swarm('{self.name}', agents={active}/{n}, protocol={self.protocol.name})"


# ---------------------------------------------------------------------------
# Harness — the swarm runtime manager
# ---------------------------------------------------------------------------

class SwarmHarness:
    """Runtime manager for swarm operations.

    The harness manages the lifecycle of swarms: creation, execution,
    monitoring, and result collection.  It can run multiple swarms
    concurrently and coordinate between them.
    """

    def __init__(self, provider: Any = None) -> None:
        self._provider = provider
        self._swarms: dict[str, Swarm] = {}
        self._history: list[SwarmResult] = []

    def create_swarm(
        self,
        name: str = "default",
        protocol: CoordinationProtocol = CoordinationProtocol.BROADCAST,
    ) -> Swarm:
        """Create a new swarm."""
        swarm = Swarm(name=name, protocol=protocol)
        self._swarms[name] = swarm
        return swarm

    def run_swarm(
        self,
        task: str,
        swarm_name: str = "default",
        grammars: list[Any] | None = None,
    ) -> SwarmResult:
        """Run a swarm on a task."""
        if swarm_name not in self._swarms:
            self.create_swarm(swarm_name)

        swarm = self._swarms[swarm_name]
        result = swarm.run(task, self._provider, grammars)
        self._history.append(result)
        return result

    def run_multi_swarm(
        self,
        task: str,
        swarm_names: list[str] | None = None,
        grammars: list[Any] | None = None,
    ) -> list[SwarmResult]:
        """Run multiple swarms concurrently on the same task."""
        names = swarm_names or list(self._swarms.keys())
        if not names:
            names = ["default"]

        results: list[SwarmResult] = []
        threads: list[threading.Thread] = []
        lock = threading.Lock()

        for name in names:
            if name not in self._swarms:
                self.create_swarm(name)

            def _run(n=name):
                r = self._swarms[n].run(task, self._provider, grammars)
                with lock:
                    results.append(r)

            t = threading.Thread(target=_run, daemon=True)
            threads.append(t)
            t.start()

        for t in threads:
            t.join(timeout=60.0)

        self._history.extend(results)
        return results

    @property
    def history(self) -> list[SwarmResult]:
        return list(self._history)

    def __repr__(self) -> str:
        return f"SwarmHarness(swarms={len(self._swarms)}, history={len(self._history)})"
