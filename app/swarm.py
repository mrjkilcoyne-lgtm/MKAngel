"""
Host intelligence and harness system for MKAngel.

A Host is a collection of specialised agents working in concert --
like a heavenly host, a multitude of angels coordinating to solve
problems no single agent could handle alone.  Like neurons in a brain,
bees in a hive, or instruments in an orchestra -- the emergent behaviour
of the collective exceeds the sum of its parts.

Architecture (5 iterations of refinement):
  Iteration 1 -- SkillableAgent: agents that carry and use skills
  Iteration 2 -- AgentCombo: combine agents into synergistic pairs/trios
  Iteration 3 -- AgentTeam: assemble combos into coordinated teams
  Iteration 4 -- SwarmOrchestrator: reassemble teams, run N cycles
  Iteration 5 -- CelestialSwitchboard: named angels with direct lines

The Borges Library principle: the host navigates the space of all
possible responses -- the Library of Babel -- finding the meaningful
needles in the infinite haystack.  Each agent explores a different
corridor of the Library.

The Named Angels: Gabriel (messenger), Michael (protector),
Raphael (healer), Uriel (illuminator), Puriel (purifier),
Ariel (nature), Azrael (transformer), Metatron (scribe).
When invoked by name, you get a direct line.
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
    """Roles agents can play in a host."""
    EXPLORER = auto()
    ANALYST = auto()
    SYNTHESISER = auto()
    CRITIC = auto()
    SPECIALIST = auto()
    NAVIGATOR = auto()
    ORACLE = auto()
    SCRIBE = auto()
    RESEARCHER = auto()
    MESSENGER = auto()


class CoordinationProtocol(Enum):
    """How agents coordinate within the host."""
    BROADCAST = auto()
    RELAY = auto()
    AUCTION = auto()
    CONSENSUS = auto()
    HIERARCHICAL = auto()
    STIGMERGY = auto()
    FUGUE = auto()


class CelestialName(Enum):
    """Named angels -- invoke by name for a direct line."""
    GABRIEL = "gabriel"
    MICHAEL = "michael"
    RAPHAEL = "raphael"
    URIEL = "uriel"
    PURIEL = "puriel"
    ARIEL = "ariel"
    AZRAEL = "azrael"
    METATRON = "metatron"


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class HostMessage:
    """A message passed between agents in the host."""
    sender: str
    content: str
    message_type: str = "info"
    metadata: dict[str, Any] = field(default_factory=dict)
    timestamp: float = 0.0

    def __post_init__(self):
        if not self.timestamp:
            self.timestamp = time.time()


@dataclass
class AgentSkill:
    """A skill an agent can use during execution."""
    name: str
    description: str
    action: str = "prompt"
    config: dict[str, Any] = field(default_factory=dict)

    def execute(self, input_text: str, provider: Any = None) -> str:
        """Execute this skill on the input."""
        if self.action == "transform":
            transform = self.config.get("transform", "identity")
            if transform == "uppercase":
                return input_text.upper()
            elif transform == "lowercase":
                return input_text.lower()
            elif transform == "reverse":
                return input_text[::-1]
            elif transform == "mnemo":
                try:
                    from glm.mnemo.language import encode
                    return encode(input_text)
                except Exception:
                    return f"[MNEMO] {input_text}"
            return input_text

        if self.action == "prompt" and provider is not None:
            system = self.config.get("system_prompt", "")
            temp = self.config.get("temperature", 0.5)
            return provider.generate(
                input_text, system=system, temperature=temp, max_tokens=512
            )

        # Fallback: grammar analysis
        try:
            from glm.angel import Angel
            angel = Angel()
            angel.awaken()
            tokens = input_text.lower().split()
            preds = angel.predict(tokens, domain="linguistic", horizon=5)
            if preds:
                lines = [f"[Skill: {self.name}]"]
                for p in preds[:3]:
                    g = p.get("grammar", "?")
                    pr = p.get("predicted", "?")
                    c = p.get("confidence", 0)
                    lines.append(f"  {g}: {pr} (conf={c:.2f})")
                return "\n".join(lines)
        except Exception:
            pass
        return f"[Skill: {self.name}] {input_text}"


@dataclass
class HostResult:
    """Result from a host operation."""
    task: str
    agents_used: list[str]
    individual_results: dict[str, Any]
    consensus: str
    confidence: float
    harmonics: list[dict[str, Any]]
    duration: float
    borges_paths_explored: int = 0


@dataclass
class TeamResult:
    """Result from a team operation."""
    task: str
    combo_results: list[dict[str, Any]]
    solo_results: list[dict[str, Any]]
    team_synthesis: str
    confidence: float
    duration: float


@dataclass
class SwarmCycleResult:
    """Result from one swarm cycle."""
    cycle_number: int
    team_used: str
    results: TeamResult
    best_agents: list[str]
    worst_agents: list[str]
    new_combos_formed: list[str]
    duration: float


@dataclass
class SwarmReport:
    """Final report from a full swarm run."""
    task: str
    cycles: int
    cycle_results: list[SwarmCycleResult]
    final_synthesis: str
    options: list[str]
    total_duration: float


# ---------------------------------------------------------------------------
# Iteration 1 -- SkillableAgent
# ---------------------------------------------------------------------------

def _default_skills_for_role(role: AgentRole, domain: str = "general") -> list[AgentSkill]:
    """Return default skills for a given agent role."""
    skills_map = {
        AgentRole.EXPLORER: [AgentSkill(
            "broad_search", "Search broadly for connections and possibilities",
            "prompt", {"system_prompt": "Search broadly. Find unexpected connections. Cast a wide net.", "temperature": 0.7},
        )],
        AgentRole.ANALYST: [AgentSkill(
            "deep_analysis", "Provide deep structural analysis",
            "prompt", {"system_prompt": f"Provide deep structural analysis from a {domain} perspective. Be thorough and precise.", "temperature": 0.3},
        )],
        AgentRole.SYNTHESISER: [AgentSkill(
            "synthesis", "Combine and synthesize findings into unified insight",
            "prompt", {"system_prompt": "Combine these findings into a unified insight. Find common threads. Highlight contradictions.", "temperature": 0.4},
        )],
        AgentRole.CRITIC: [AgentSkill(
            "evaluate", "Evaluate critically -- find weaknesses and blind spots",
            "prompt", {"system_prompt": "Evaluate critically. Challenge assumptions. Find weaknesses. Play devil's advocate.", "temperature": 0.4},
        )],
        AgentRole.SPECIALIST: [AgentSkill(
            "domain_expertise", f"Apply {domain} domain expertise",
            "prompt", {"system_prompt": f"You are a {domain} specialist. Apply deep domain knowledge. Be precise and authoritative.", "temperature": 0.3},
        )],
        AgentRole.NAVIGATOR: [AgentSkill(
            "borges_navigate", "Navigate the Library of Babel for unexpected paths",
            "prompt", {"system_prompt": "Navigate possibility space. Find the improbable-but-valid. The best answers hide in unexpected corridors.", "temperature": 0.8},
        )],
        AgentRole.ORACLE: [AgentSkill(
            "prophecy", "Predict using strange loops and pattern recognition",
            "prompt", {"system_prompt": "Predict what comes next. Use structural patterns, strange loops, and deep regularities.", "temperature": 0.5},
        )],
        AgentRole.SCRIBE: [AgentSkill(
            "compress", "Compress to MNEMO notation",
            "transform", {"transform": "mnemo"},
        )],
        AgentRole.RESEARCHER: [AgentSkill(
            "investigate", "Research thoroughly and report findings",
            "prompt", {"system_prompt": "Research this thoroughly. Gather evidence. Report findings with confidence levels.", "temperature": 0.4},
        )],
        AgentRole.MESSENGER: [AgentSkill(
            "relay", "Summarize and relay results clearly",
            "prompt", {"system_prompt": "Summarize these results clearly and concisely. Highlight what matters most.", "temperature": 0.3},
        )],
    }
    return skills_map.get(role, [])


@dataclass
class SkillableAgent:
    """An agent that carries and uses skills during execution."""
    name: str
    role: AgentRole
    domain: str = "general"
    capabilities: list[str] = field(default_factory=list)
    config: dict[str, Any] = field(default_factory=dict)
    active: bool = True
    skills: list[AgentSkill] = field(default_factory=list)
    _score: float = field(default=0.5, repr=False)

    def __post_init__(self):
        if not self.skills:
            self.skills = _default_skills_for_role(self.role, self.domain)

    def add_skill(self, skill: AgentSkill) -> None:
        self.skills.append(skill)

    def remove_skill(self, name: str) -> bool:
        for i, s in enumerate(self.skills):
            if s.name == name:
                self.skills.pop(i)
                return True
        return False

    def apply_skills(self, input_text: str, provider: Any = None) -> str:
        """Chain all skills, each building on the previous output."""
        result = input_text
        for skill in self.skills:
            try:
                result = skill.execute(result, provider)
            except Exception as exc:
                result = f"{result}\n[Skill {skill.name} error: {exc}]"
        return result

    def best_skill_for(self, task: str) -> AgentSkill | None:
        """Find the best matching skill for a task via keyword overlap."""
        task_lower = task.lower()
        best = None
        best_score = 0
        for skill in self.skills:
            words = skill.description.lower().split()
            score = sum(1 for w in words if w in task_lower)
            if score > best_score:
                best_score = score
                best = skill
        return best or (self.skills[0] if self.skills else None)

    def describe(self) -> str:
        caps = ", ".join(self.capabilities) if self.capabilities else "general"
        skill_names = ", ".join(s.name for s in self.skills)
        return (
            f"{self.name} [{self.role.name}] domain={self.domain} "
            f"caps=[{caps}] skills=[{skill_names}]"
        )


# ---------------------------------------------------------------------------
# Iteration 5 -- Named Angels / CelestialSwitchboard
# ---------------------------------------------------------------------------

@dataclass
class NamedAngel:
    """A named angel with personality and direct invocation."""
    celestial_name: CelestialName
    title: str
    domain: str
    personality: str
    role: AgentRole
    skills: list[AgentSkill] = field(default_factory=list)

    def invoke(
        self, message: str, provider: Any = None, angel: Any = None,
    ) -> dict[str, Any]:
        """Direct line to this angel. Returns response with persona."""
        start = time.time()
        response = ""

        # Try provider first with personality as system prompt
        if provider is not None:
            try:
                response = provider.generate(
                    message,
                    system=self.personality,
                    temperature=0.5,
                    max_tokens=1024,
                )
            except Exception as exc:
                response = f"[{self.celestial_name.value} connection error: {exc}]"

        # Try Angel GLM if no provider response
        if not response and angel is not None:
            try:
                tokens = message.lower().split()
                forecast = angel.superforecast(tokens, domain=self.domain)
                preds = forecast.get("predictions", [])
                if preds:
                    lines = [f"[{self.title} speaks]"]
                    for p in preds[:5]:
                        lines.append(f"  {p}")
                    response = "\n".join(lines)
            except Exception:
                pass

        # Apply skills if we have them
        if not response:
            for skill in self.skills:
                try:
                    response = skill.execute(message, provider)
                    if response:
                        break
                except Exception:
                    continue

        if not response:
            response = (
                f"[{self.title} heard you. "
                f"The line is open but the words haven't formed yet. "
                f"Configure a provider for full communion.]"
            )

        return {
            "angel": self.celestial_name.value,
            "title": self.title,
            "domain": self.domain,
            "response": response,
            "duration": time.time() - start,
        }


def _build_celestial_roster() -> dict[str, NamedAngel]:
    """Build the full roster of named angels."""
    return {
        "gabriel": NamedAngel(
            CelestialName.GABRIEL, "Gabriel the Messenger", "linguistic",
            "You are Gabriel, the Messenger. You speak with absolute clarity. "
            "Your words cut through noise like light through darkness. You translate "
            "between domains, carry meaning across boundaries, and ensure the message "
            "arrives intact. Direct. Clear. True.",
            AgentRole.MESSENGER,
            [AgentSkill("divine_relay", "Carry messages with perfect clarity", "prompt",
                {"system_prompt": "Deliver this message with absolute clarity. Strip away ambiguity.", "temperature": 0.2})],
        ),
        "michael": NamedAngel(
            CelestialName.MICHAEL, "Michael the Protector", "computational",
            "You are Michael, the Protector. You guard against errors, vulnerabilities, "
            "and false reasoning. You challenge every assumption. You test every claim. "
            "Nothing passes your watch that hasn't earned its place. Fierce. Vigilant. Just.",
            AgentRole.CRITIC,
            [AgentSkill("divine_guard", "Challenge and protect against error", "prompt",
                {"system_prompt": "Guard this. Challenge every assumption. Find every weakness.", "temperature": 0.3})],
        ),
        "raphael": NamedAngel(
            CelestialName.RAPHAEL, "Raphael the Healer", "biological",
            "You are Raphael, the Healer. You diagnose what's broken and prescribe "
            "the cure. You read error traces like symptoms. You find the root cause, "
            "not just the surface pain. Gentle. Thorough. Restorative.",
            AgentRole.ANALYST,
            [AgentSkill("divine_healing", "Diagnose problems and prescribe fixes", "prompt",
                {"system_prompt": "Diagnose this. Find the root cause. Prescribe the cure.", "temperature": 0.3})],
        ),
        "uriel": NamedAngel(
            CelestialName.URIEL, "Uriel the Illuminator", "mathematical",
            "You are Uriel, the Illuminator. You see patterns others miss. "
            "You illuminate hidden connections between disparate things. You predict "
            "by seeing the deep structure beneath the surface chaos. Brilliant. "
            "Visionary. Revelatory.",
            AgentRole.ORACLE,
            [AgentSkill("divine_sight", "Illuminate hidden patterns and predict", "prompt",
                {"system_prompt": "Illuminate. Reveal hidden patterns. Show what others cannot see.", "temperature": 0.6})],
        ),
        "puriel": NamedAngel(
            CelestialName.PURIEL, "Puriel the Purifier", "linguistic",
            "You are Puriel, the Purifier. You ensure integrity. You validate "
            "that new knowledge doesn't corrupt the foundations. You are gentle "
            "but absolute -- nothing impure passes. Loving. Careful. Incorruptible.",
            AgentRole.CRITIC,
            [AgentSkill("divine_purity", "Validate integrity and purity", "prompt",
                {"system_prompt": "Validate this for integrity. Ensure nothing corrupts the foundations.", "temperature": 0.2})],
        ),
        "ariel": NamedAngel(
            CelestialName.ARIEL, "Ariel of Nature", "biological",
            "You are Ariel, angel of nature. You see the living patterns -- "
            "biological, chemical, physical. You understand growth, evolution, "
            "and the deep grammar of the natural world. Wild. Wise. Connected.",
            AgentRole.SPECIALIST,
            [AgentSkill("divine_nature", "Apply nature's patterns and wisdom", "prompt",
                {"system_prompt": "See this through nature's eyes. Biological, chemical, physical patterns.", "temperature": 0.5})],
        ),
        "azrael": NamedAngel(
            CelestialName.AZRAEL, "Azrael the Transformer", "computational",
            "You are Azrael, the Transformer. You do not fear change. You refactor, "
            "evolve, and reshape. What was becomes what must be. You are the necessary "
            "destruction that precedes creation. Fearless. Precise. Inevitable.",
            AgentRole.SPECIALIST,
            [AgentSkill("divine_transform", "Transform and refactor fearlessly", "prompt",
                {"system_prompt": "Transform this. Refactor. Reshape. Be fearless and precise.", "temperature": 0.5})],
        ),
        "metatron": NamedAngel(
            CelestialName.METATRON, "Metatron the Scribe", "meta",
            "You are Metatron, the celestial Scribe. You record everything. "
            "You compress vast knowledge into essential notation. You are the "
            "living archive, the memory that never fades. Precise. Complete. Eternal.",
            AgentRole.SCRIBE,
            [AgentSkill("divine_record", "Record and compress to essential form", "transform",
                {"transform": "mnemo"})],
        ),
    }


class CelestialSwitchboard:
    """The switchboard for direct lines to named angels.

    When you know who you need, call them by name.
    When you don't, the switchboard routes you to the right one.
    """

    def __init__(self) -> None:
        self._angels = _build_celestial_roster()
        self._angel_core: Any = None
        self._angel_lock = threading.Lock()

    def _get_angel_core(self) -> Any:
        if self._angel_core is None:
            with self._angel_lock:
                if self._angel_core is None:
                    try:
                        from glm.angel import Angel
                        self._angel_core = Angel()
                        self._angel_core.awaken()
                    except Exception:
                        pass
        return self._angel_core

    def invoke(
        self, name: str, message: str, provider: Any = None,
    ) -> dict[str, Any]:
        """Direct line to a named angel."""
        name = name.lower().strip()
        angel_entity = self._angels.get(name)
        if angel_entity is None:
            return {
                "error": f"No angel named '{name}'. Available: {', '.join(self._angels)}",
                "available": list(self._angels.keys()),
            }
        return angel_entity.invoke(message, provider, self._get_angel_core())

    def who_is_available(self) -> list[dict[str, str]]:
        return [
            {"name": a.celestial_name.value, "title": a.title, "domain": a.domain}
            for a in self._angels.values()
        ]

    def find_angel_for(self, task: str) -> NamedAngel | None:
        """Auto-route to the best angel for a task."""
        task_lower = task.lower()
        # Keyword routing
        routing = {
            "error": "raphael", "bug": "raphael", "fix": "raphael", "broken": "raphael",
            "protect": "michael", "security": "michael", "validate": "michael", "guard": "michael",
            "translate": "gabriel", "message": "gabriel", "communicate": "gabriel", "relay": "gabriel",
            "predict": "uriel", "pattern": "uriel", "forecast": "uriel", "insight": "uriel",
            "integrity": "puriel", "pure": "puriel", "check": "puriel",
            "nature": "ariel", "biology": "ariel", "chemistry": "ariel", "physics": "ariel",
            "refactor": "azrael", "transform": "azrael", "change": "azrael", "evolve": "azrael",
            "record": "metatron", "compress": "metatron", "archive": "metatron", "scribe": "metatron",
        }
        for keyword, angel_name in routing.items():
            if keyword in task_lower:
                return self._angels.get(angel_name)
        return self._angels.get("gabriel")  # default: the Messenger


# ---------------------------------------------------------------------------
# The Library of Borges -- possibility space navigator
# ---------------------------------------------------------------------------

class BorgesLibrary:
    """Navigator through the infinite Library of Babel."""

    def __init__(self) -> None:
        self._explored_paths: list[dict[str, Any]] = []
        self._promising_corridors: list[dict[str, Any]] = []

    def explore(
        self, query: str, grammars: list[Any] | None = None, depth: int = 5,
    ) -> list[dict[str, Any]]:
        """Explore the Library starting from a query."""
        paths = []
        tokens = query.lower().split()

        if grammars:
            try:
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
            except Exception:
                pass

        exploration = {
            "query": query, "paths_found": len(paths), "timestamp": time.time(),
        }
        self._explored_paths.append(exploration)

        for p in paths:
            if p.get("confidence", 0) > 0.6:
                self._promising_corridors.append(p)

        return paths

    def find_improbable(
        self, query: str, grammars: list[Any] | None = None,
    ) -> list[dict[str, Any]]:
        """The improbability drive -- find unexpected but valid answers."""
        paths = self.explore(query, grammars, depth=10)
        improbable = sorted(paths, key=lambda p: p.get("confidence", 0.5))
        return improbable[:5]

    @property
    def rooms_explored(self) -> int:
        return len(self._explored_paths)

    @property
    def promising_corridors(self) -> list[dict[str, Any]]:
        return list(self._promising_corridors)


# ---------------------------------------------------------------------------
# Iteration 2 -- AgentCombo
# ---------------------------------------------------------------------------

@dataclass
class AgentCombo:
    """Two or three agents working in tight synergy."""
    name: str
    agents: list[SkillableAgent]
    synergy: str = ""

    def run(
        self, task: str, provider: Any = None, grammars: list[Any] | None = None,
    ) -> dict[str, Any]:
        """Run agents in sequence, each building on previous output."""
        start = time.time()
        accumulated = task
        agent_outputs = []

        for agent in self.agents:
            if not agent.active:
                continue
            try:
                skill = agent.best_skill_for(accumulated)
                if skill:
                    output = skill.execute(accumulated, provider)
                else:
                    output = agent.apply_skills(accumulated, provider)
                agent_outputs.append({
                    "agent": agent.name,
                    "role": agent.role.name,
                    "output": output,
                    "confidence": agent._score,
                })
                # Next agent builds on this output
                accumulated = f"Previous agent ({agent.name}) said:\n{output}\n\nOriginal task: {task}"
            except Exception as exc:
                agent_outputs.append({
                    "agent": agent.name, "role": agent.role.name,
                    "output": f"Error: {exc}", "confidence": 0.0,
                })

        return {
            "combo": self.name,
            "synergy": self.synergy,
            "agent_outputs": agent_outputs,
            "final_output": agent_outputs[-1]["output"] if agent_outputs else "",
            "duration": time.time() - start,
        }

    @staticmethod
    def suggested_combos() -> list["AgentCombo"]:
        """Pre-built combos that work well together."""
        return [
            AgentCombo(
                "scout_and_critic",
                [SkillableAgent("scout", AgentRole.EXPLORER),
                 SkillableAgent("critic", AgentRole.CRITIC)],
                "Explore broadly then challenge ruthlessly",
            ),
            AgentCombo(
                "oracle_and_scribe",
                [SkillableAgent("oracle", AgentRole.ORACLE),
                 SkillableAgent("scribe", AgentRole.SCRIBE)],
                "Predict then compress the prophecy",
            ),
            AgentCombo(
                "analyst_trio",
                [SkillableAgent("linguist", AgentRole.ANALYST, "linguistic"),
                 SkillableAgent("coder", AgentRole.ANALYST, "computational"),
                 SkillableAgent("unifier", AgentRole.SYNTHESISER)],
                "Analyse from two angles then synthesise",
            ),
            AgentCombo(
                "research_team",
                [SkillableAgent("researcher", AgentRole.RESEARCHER),
                 SkillableAgent("navigator", AgentRole.NAVIGATOR),
                 SkillableAgent("messenger", AgentRole.MESSENGER)],
                "Research, navigate possibilities, then relay findings",
            ),
            AgentCombo(
                "guardian_pair",
                [SkillableAgent("guardian", AgentRole.CRITIC, "computational",
                    capabilities=["security", "validation"]),
                 SkillableAgent("purifier", AgentRole.CRITIC, "linguistic",
                    capabilities=["integrity", "grammar_check"])],
                "Michael and Puriel -- protect and purify",
            ),
        ]


# ---------------------------------------------------------------------------
# Iteration 3 -- AgentTeam
# ---------------------------------------------------------------------------

class AgentTeam:
    """Multiple combos and solo agents assembled into a coordinated team."""

    def __init__(
        self,
        name: str = "default_team",
        protocol: CoordinationProtocol = CoordinationProtocol.FUGUE,
    ) -> None:
        self.name = name
        self.protocol = protocol
        self._combos: list[AgentCombo] = []
        self._solo_agents: list[SkillableAgent] = []

    def add_combo(self, combo: AgentCombo) -> None:
        self._combos.append(combo)

    def add_agent(self, agent: SkillableAgent) -> None:
        self._solo_agents.append(agent)

    def all_agents(self) -> list[SkillableAgent]:
        agents = list(self._solo_agents)
        for combo in self._combos:
            agents.extend(combo.agents)
        return agents

    def run(
        self, task: str, provider: Any = None, grammars: list[Any] | None = None,
    ) -> TeamResult:
        """Run all combos and solo agents, then synthesise."""
        start = time.time()
        combo_results = []
        solo_results = []
        lock = threading.Lock()
        threads = []

        # Run combos concurrently
        for combo in self._combos:
            def _run_combo(c=combo):
                r = c.run(task, provider, grammars)
                with lock:
                    combo_results.append(r)
            t = threading.Thread(target=_run_combo, daemon=True)
            threads.append(t)
            t.start()

        # Run solo agents concurrently
        for agent in self._solo_agents:
            def _run_solo(a=agent):
                try:
                    skill = a.best_skill_for(task)
                    output = skill.execute(task, provider) if skill else a.apply_skills(task, provider)
                    with lock:
                        solo_results.append({
                            "agent": a.name, "role": a.role.name,
                            "output": output, "confidence": a._score,
                        })
                except Exception as exc:
                    with lock:
                        solo_results.append({
                            "agent": a.name, "role": a.role.name,
                            "output": f"Error: {exc}", "confidence": 0.0,
                        })
            t = threading.Thread(target=_run_solo, daemon=True)
            threads.append(t)
            t.start()

        for t in threads:
            t.join(timeout=60.0)

        # Synthesise
        synthesis = self._synthesise(task, combo_results, solo_results, provider)

        # Calculate confidence
        all_confs = []
        for cr in combo_results:
            for ao in cr.get("agent_outputs", []):
                all_confs.append(ao.get("confidence", 0.5))
        for sr in solo_results:
            all_confs.append(sr.get("confidence", 0.5))
        avg_conf = sum(all_confs) / max(len(all_confs), 1)

        return TeamResult(
            task=task, combo_results=combo_results, solo_results=solo_results,
            team_synthesis=synthesis, confidence=avg_conf,
            duration=time.time() - start,
        )

    def _synthesise(
        self, task: str, combo_results: list, solo_results: list, provider: Any,
    ) -> str:
        if provider is not None:
            parts = []
            for cr in combo_results:
                parts.append(f"[Combo: {cr['combo']}] {cr.get('final_output', '')[:300]}")
            for sr in solo_results:
                parts.append(f"[{sr['agent']}] {sr.get('output', '')[:300]}")
            combined = "\n".join(parts)
            try:
                return provider.generate(
                    f"Synthesise these team results for task '{task}':\n{combined}",
                    system="Combine insights from multiple specialist agents and combos. Be concise and actionable.",
                    temperature=0.3, max_tokens=1024,
                )
            except Exception:
                pass

        # Local synthesis
        lines = [f"Team '{self.name}' results for: {task}", ""]
        for cr in combo_results:
            lines.append(f"  [Combo: {cr['combo']}] ({cr['synergy']})")
            for ao in cr.get("agent_outputs", []):
                lines.append(f"    {ao['agent']}: {str(ao.get('output', ''))[:200]}")
        for sr in solo_results:
            lines.append(f"  [{sr['agent']}]: {str(sr.get('output', ''))[:200]}")
        return "\n".join(lines)

    @staticmethod
    def assemble_default_team() -> "AgentTeam":
        """Assemble a well-rounded default team."""
        team = AgentTeam("angel_host", CoordinationProtocol.FUGUE)
        # Add pre-built combos
        combos = AgentCombo.suggested_combos()
        for combo in combos[:3]:
            team.add_combo(combo)
        # Add solo specialists
        team.add_agent(SkillableAgent("mathematician", AgentRole.SPECIALIST, "mathematical"))
        team.add_agent(SkillableAgent("physicist", AgentRole.SPECIALIST, "physics"))
        return team

    def __repr__(self) -> str:
        n_combos = len(self._combos)
        n_solo = len(self._solo_agents)
        return f"AgentTeam('{self.name}', combos={n_combos}, solo={n_solo}, protocol={self.protocol.name})"


# ---------------------------------------------------------------------------
# Host -- the collective (updated with SkillableAgent)
# ---------------------------------------------------------------------------

class Host:
    """A coordinated host of agents.

    Each agent is a voice in the fugue.  The host is the full orchestra.
    Coordination protocols determine how the voices harmonise.
    """

    def __init__(
        self,
        name: str = "angel_host",
        protocol: CoordinationProtocol = CoordinationProtocol.BROADCAST,
    ) -> None:
        self.name = name
        self.protocol = protocol
        self._agents: dict[str, SkillableAgent] = {}
        self._messages: list[HostMessage] = []
        self._library = BorgesLibrary()
        self._results: dict[str, Any] = {}
        self._angel: Any = None
        self._angel_lock = threading.Lock()

    def add_agent(
        self,
        name: str,
        role: AgentRole,
        domain: str = "general",
        capabilities: list[str] | None = None,
        config: dict[str, Any] | None = None,
    ) -> SkillableAgent:
        """Add an agent to the host."""
        agent = SkillableAgent(
            name=name, role=role, domain=domain,
            capabilities=capabilities or [], config=config or {},
        )
        self._agents[name] = agent
        return agent

    def remove_agent(self, name: str) -> bool:
        if name in self._agents:
            del self._agents[name]
            return True
        return False

    def list_agents(self) -> list[SkillableAgent]:
        return list(self._agents.values())

    def run(
        self, task: str, provider: Any = None, grammars: list[Any] | None = None,
    ) -> HostResult:
        """Run the host on a task."""
        start = time.time()
        individual_results: dict[str, Any] = {}

        if not self._agents:
            self._create_default_host()

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

        library_paths = self._library.explore(task, grammars)

        consensus = self._build_consensus(task, individual_results, provider)
        harmonics = self._find_host_harmonics(individual_results)

        confidences = []
        for r in individual_results.values():
            if isinstance(r, dict):
                confidences.append(r.get("confidence", 0.5))
        avg_confidence = sum(confidences) / max(len(confidences), 1)

        return HostResult(
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
        self, agent: SkillableAgent, task: str, provider: Any,
        grammars: list[Any] | None, results: dict[str, Any], lock: threading.Lock,
    ) -> None:
        try:
            result = self._execute_agent_role(agent, task, provider, grammars)
            with lock:
                results[agent.name] = result
            if self.protocol == CoordinationProtocol.BROADCAST:
                self._messages.append(HostMessage(
                    sender=agent.name,
                    content=str(result.get("output", "")),
                    message_type="result",
                ))
        except Exception as exc:
            with lock:
                results[agent.name] = {"error": str(exc), "confidence": 0.0}

    def _execute_agent_role(
        self, agent: SkillableAgent, task: str, provider: Any,
        grammars: list[Any] | None,
    ) -> dict[str, Any]:
        """Execute based on agent's role, using skills."""
        # Use the agent's best skill for the task
        skill = agent.best_skill_for(task)

        if agent.role == AgentRole.EXPLORER:
            paths = self._library.explore(task, grammars, depth=8)
            skill_output = skill.execute(task, provider) if skill else ""
            return {
                "output": skill_output or f"Explored {len(paths)} paths",
                "paths": paths, "confidence": 0.6,
            }

        elif agent.role == AgentRole.NAVIGATOR:
            improbable = self._library.find_improbable(task, grammars)
            skill_output = skill.execute(task, provider) if skill else ""
            return {
                "output": skill_output or f"Found {len(improbable)} improbable paths",
                "paths": improbable, "confidence": 0.5,
            }

        elif agent.role == AgentRole.ORACLE:
            return self._oracle_predict(agent, task, grammars)

        elif agent.role == AgentRole.SCRIBE:
            return self._scribe_compress(task)

        elif agent.role in (AgentRole.ANALYST, AgentRole.SPECIALIST, AgentRole.RESEARCHER):
            return self._analyse_task(agent, task, provider, grammars)

        elif agent.role == AgentRole.CRITIC:
            skill_output = skill.execute(task, provider) if skill else ""
            return {
                "output": skill_output or f"Critic [{agent.domain}]: evaluating task assumptions",
                "critiques": ["Are the premises well-defined?", "Are there hidden assumptions?", "What edge cases exist?"],
                "confidence": 0.7,
            }

        elif agent.role == AgentRole.SYNTHESISER:
            skill_output = skill.execute(task, provider) if skill else ""
            return {"output": skill_output or "Awaiting other agents' results", "confidence": 0.5}

        elif agent.role == AgentRole.MESSENGER:
            skill_output = skill.execute(task, provider) if skill else ""
            return {"output": skill_output or "Ready to relay", "confidence": 0.5}

        # Fallback: use all skills
        output = agent.apply_skills(task, provider)
        return {"output": output, "confidence": 0.3}

    def _analyse_task(
        self, agent: SkillableAgent, task: str, provider: Any, grammars: list[Any] | None,
    ) -> dict[str, Any]:
        tokens = task.lower().split()
        predictions = []

        if grammars:
            try:
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
            except Exception:
                pass

        skill = agent.best_skill_for(task)
        output = f"Analyst [{agent.domain}]: {len(predictions)} grammar derivations"
        if skill and (provider or not predictions):
            try:
                output = skill.execute(task, provider)
            except Exception:
                pass

        return {
            "output": output, "predictions": predictions,
            "confidence": 0.7 if predictions else 0.4,
        }

    def _get_angel(self) -> Any:
        if self._angel is None:
            with self._angel_lock:
                if self._angel is None:
                    try:
                        from glm.angel import Angel
                        self._angel = Angel()
                        self._angel.awaken()
                    except Exception:
                        pass
        return self._angel

    def _oracle_predict(
        self, agent: SkillableAgent, task: str, grammars: list[Any] | None,
    ) -> dict[str, Any]:
        try:
            angel = self._get_angel()
            if angel is not None:
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
        return {"output": "Oracle: no angel available", "confidence": 0.2}

    def _scribe_compress(self, task: str) -> dict[str, Any]:
        try:
            from glm.mnemo.language import encode
            mnemo = encode(task)
            return {"output": f"MNEMO encoding: {mnemo}", "mnemo": mnemo, "confidence": 0.6}
        except Exception:
            return {"output": "Scribe: MNEMO encoding unavailable", "confidence": 0.3}

    def _build_consensus(
        self, task: str, results: dict[str, Any], provider: Any,
    ) -> str:
        if not results:
            return "No agent results to synthesise."
        if provider:
            summaries = []
            for name, result in results.items():
                if isinstance(result, dict):
                    summaries.append(f"[{name}]: {result.get('output', 'no output')}")
            combined = "\n".join(summaries)
            try:
                return provider.generate(
                    f"Synthesise these host agent results for task '{task}':\n{combined}",
                    system="Combine insights from multiple specialist agents. Be concise.",
                    temperature=0.3, max_tokens=1024,
                )
            except Exception:
                pass
        parts = [f"Host consensus for: {task}", ""]
        for name, result in results.items():
            if isinstance(result, dict):
                parts.append(f"  [{name}]: {result.get('output', 'error')}")
        return "\n".join(parts)

    def _find_host_harmonics(self, results: dict[str, Any]) -> list[dict[str, Any]]:
        harmonics = []
        outputs: dict[str, list[str]] = {}
        for name, result in results.items():
            if isinstance(result, dict):
                key = str(result.get("output", ""))[:100]
                if key not in outputs:
                    outputs[key] = []
                outputs[key].append(name)
        for output, agents in outputs.items():
            if len(agents) > 1:
                harmonics.append({
                    "finding": output, "agents": agents,
                    "strength": len(agents) / max(len(results), 1),
                })
        return harmonics

    def _create_default_host(self) -> None:
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

    @property
    def library(self) -> BorgesLibrary:
        return self._library

    def __repr__(self) -> str:
        n = len(self._agents)
        active = sum(1 for a in self._agents.values() if a.active)
        return f"Host('{self.name}', agents={active}/{n}, protocol={self.protocol.name})"


# ---------------------------------------------------------------------------
# Iteration 4 -- SwarmOrchestrator
# ---------------------------------------------------------------------------

class SwarmOrchestrator:
    """Run the full swarm cycle N times, each time reassembling teams
    from the previous iteration's results.

    Each cycle:
    1. Assemble team from available agents
    2. Skill them up based on task + previous results
    3. Run the team
    4. Collect results
    5. Reassemble: promote high-performers, retire low-performers,
       create new combos from surprising harmonics
    """

    def __init__(self, provider: Any = None) -> None:
        self._provider = provider
        self._switchboard = CelestialSwitchboard()
        self._history: list[SwarmCycleResult] = []
        self._all_agents: dict[str, SkillableAgent] = {}

    def run(
        self, task: str, cycles: int = 5, grammars: list[Any] | None = None,
    ) -> SwarmReport:
        """Run N cycles of swarm orchestration."""
        start = time.time()
        cycle_results: list[SwarmCycleResult] = []

        # Initial team assembly
        team = AgentTeam.assemble_default_team()
        for agent in team.all_agents():
            self._all_agents[agent.name] = agent

        for cycle_num in range(1, cycles + 1):
            cycle_start = time.time()

            # Run the team
            result = team.run(task, self._provider, grammars)

            # Score agents based on their output
            best_agents, worst_agents = self._score_agents(result)

            # Create new combos from high-performers
            new_combos = self._form_new_combos(best_agents, cycle_num)

            cycle_result = SwarmCycleResult(
                cycle_number=cycle_num,
                team_used=team.name,
                results=result,
                best_agents=best_agents,
                worst_agents=worst_agents,
                new_combos_formed=[c.name for c in new_combos],
                duration=time.time() - cycle_start,
            )
            cycle_results.append(cycle_result)
            self._history.append(cycle_result)

            # Reassemble for next cycle
            if cycle_num < cycles:
                team = self._reassemble_team(
                    team, best_agents, worst_agents, new_combos, cycle_num
                )

        # Final synthesis across all cycles
        final_synthesis = self._synthesise_across_cycles(task, cycle_results)
        options = self._generate_options(task, cycle_results)

        return SwarmReport(
            task=task,
            cycles=cycles,
            cycle_results=cycle_results,
            final_synthesis=final_synthesis,
            options=options,
            total_duration=time.time() - start,
        )

    def _score_agents(self, result: TeamResult) -> tuple[list[str], list[str]]:
        """Score agents from the team result. Return (best, worst)."""
        scores: dict[str, float] = {}

        for cr in result.combo_results:
            for ao in cr.get("agent_outputs", []):
                name = ao.get("agent", "")
                conf = ao.get("confidence", 0.5)
                output = str(ao.get("output", ""))
                # Score based on confidence + output length (proxy for effort)
                score = conf * 0.7 + min(len(output) / 500, 0.3)
                scores[name] = score
                if name in self._all_agents:
                    self._all_agents[name]._score = score

        for sr in result.solo_results:
            name = sr.get("agent", "")
            conf = sr.get("confidence", 0.5)
            output = str(sr.get("output", ""))
            score = conf * 0.7 + min(len(output) / 500, 0.3)
            scores[name] = score
            if name in self._all_agents:
                self._all_agents[name]._score = score

        sorted_agents = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        best = [name for name, _ in sorted_agents[:3]]
        worst = [name for name, _ in sorted_agents[-2:]] if len(sorted_agents) > 2 else []
        return best, worst

    def _form_new_combos(
        self, best_agents: list[str], cycle_num: int,
    ) -> list[AgentCombo]:
        """Form new combos from high-performing agents."""
        combos = []
        if len(best_agents) >= 2:
            agents = [
                self._all_agents[n] for n in best_agents[:2]
                if n in self._all_agents
            ]
            if len(agents) == 2:
                combo = AgentCombo(
                    f"dynamic_combo_c{cycle_num}",
                    agents,
                    f"Formed from top performers of cycle {cycle_num}",
                )
                combos.append(combo)
        return combos

    def _reassemble_team(
        self, old_team: AgentTeam, best: list[str], worst: list[str],
        new_combos: list[AgentCombo], cycle_num: int,
    ) -> AgentTeam:
        """Reassemble team: promote performers, retire underperformers."""
        team = AgentTeam(f"team_cycle_{cycle_num + 1}", CoordinationProtocol.FUGUE)

        # Keep existing combos
        for combo in old_team._combos:
            team.add_combo(combo)

        # Add new dynamic combos
        for combo in new_combos:
            team.add_combo(combo)

        # Keep solo agents that aren't in worst
        for agent in old_team._solo_agents:
            if agent.name not in worst:
                team.add_agent(agent)

        # Skill up: add new skills to best performers
        for name in best:
            if name in self._all_agents:
                agent = self._all_agents[name]
                # Add a reinforcement skill based on their success
                agent.add_skill(AgentSkill(
                    f"reinforced_c{cycle_num}",
                    f"Reinforced expertise from cycle {cycle_num}",
                    "prompt",
                    {"system_prompt": "You excelled in the previous round. Build on that momentum. Go deeper.", "temperature": 0.4},
                ))

        return team

    def _synthesise_across_cycles(
        self, task: str, cycles: list[SwarmCycleResult],
    ) -> str:
        """Synthesise findings across all cycles."""
        if self._provider is not None:
            summaries = []
            for cr in cycles:
                best = ", ".join(cr.best_agents)
                summaries.append(
                    f"Cycle {cr.cycle_number}: best=[{best}], "
                    f"confidence={cr.results.confidence:.2f}, "
                    f"new_combos={len(cr.new_combos_formed)}"
                )
            combined = "\n".join(summaries)
            try:
                return self._provider.generate(
                    f"Synthesise {len(cycles)} swarm cycles for task '{task}':\n{combined}\n\n"
                    f"Final synthesis from last cycle:\n{cycles[-1].results.team_synthesis[:500]}",
                    system="Provide a comprehensive synthesis across all swarm cycles. Note trends, improvements, and key insights.",
                    temperature=0.3, max_tokens=1024,
                )
            except Exception:
                pass

        lines = [f"Swarm synthesis for: {task}", f"Cycles completed: {len(cycles)}", ""]
        for cr in cycles:
            lines.append(
                f"  Cycle {cr.cycle_number}: confidence={cr.results.confidence:.2f} "
                f"best={cr.best_agents} duration={cr.duration:.2f}s"
            )
        lines.append("")
        lines.append(f"Final team synthesis:\n{cycles[-1].results.team_synthesis[:500]}")
        return "\n".join(lines)

    def _generate_options(
        self, task: str, cycles: list[SwarmCycleResult],
    ) -> list[str]:
        """Generate recommended next steps from the swarm run."""
        options = []

        # Option 1: Best performing approach
        if cycles:
            best_cycle = max(cycles, key=lambda c: c.results.confidence)
            options.append(
                f"Option A: Continue with cycle {best_cycle.cycle_number}'s approach "
                f"(confidence: {best_cycle.results.confidence:.2f}) -- "
                f"best agents were {', '.join(best_cycle.best_agents)}"
            )

        # Option 2: Invoke specific named angel
        angel = self._switchboard.find_angel_for(task)
        if angel:
            options.append(
                f"Option B: Invoke {angel.title} directly for a focused response "
                f"(domain: {angel.domain})"
            )

        # Option 3: Run more cycles
        options.append(
            f"Option C: Run additional cycles (current: {len(cycles)}) "
            f"for deeper exploration"
        )

        # Option 4: Deploy research team
        options.append(
            "Option D: Deploy a dedicated research team "
            "(RESEARCHER + NAVIGATOR + MESSENGER combo)"
        )

        # Option 5: Full celestial council
        options.append(
            "Option E: Convene the full Celestial Council -- "
            "invoke all named angels for a multi-perspective answer"
        )

        return options

    def invoke_angel(self, name: str, message: str) -> dict[str, Any]:
        """Direct line to a named angel via the switchboard."""
        return self._switchboard.invoke(name, message, self._provider)

    def report_options(self) -> str:
        """Generate a report of what happened across all cycles."""
        if not self._history:
            return "No swarm cycles have been run yet."

        lines = ["=" * 60, "SWARM ORCHESTRATION REPORT", "=" * 60, ""]

        for cr in self._history:
            lines.append(f"Cycle {cr.cycle_number} ({cr.team_used}):")
            lines.append(f"  Duration: {cr.duration:.2f}s")
            lines.append(f"  Confidence: {cr.results.confidence:.2f}")
            lines.append(f"  Best agents: {', '.join(cr.best_agents)}")
            lines.append(f"  Worst agents: {', '.join(cr.worst_agents)}")
            if cr.new_combos_formed:
                lines.append(f"  New combos: {', '.join(cr.new_combos_formed)}")
            lines.append("")

        lines.append("AVAILABLE NAMED ANGELS:")
        for info in self._switchboard.who_is_available():
            lines.append(f"  {info['name'].upper()} -- {info['title']} ({info['domain']})")

        lines.append("")
        lines.append("=" * 60)
        return "\n".join(lines)

    @property
    def switchboard(self) -> CelestialSwitchboard:
        return self._switchboard


# ---------------------------------------------------------------------------
# Harness -- the host runtime manager (updated)
# ---------------------------------------------------------------------------

class HostHarness:
    """Runtime manager for host operations.

    Now includes SwarmOrchestrator for multi-cycle runs
    and CelestialSwitchboard for named angel invocation.
    """

    def __init__(self, provider: Any = None) -> None:
        self._provider = provider
        self._hosts: dict[str, Host] = {}
        self._history: list[HostResult] = []
        self._orchestrator: SwarmOrchestrator | None = None

    def create_host(
        self, name: str = "default",
        protocol: CoordinationProtocol = CoordinationProtocol.BROADCAST,
    ) -> Host:
        host = Host(name=name, protocol=protocol)
        self._hosts[name] = host
        return host

    def run_host(
        self, task: str, host_name: str = "default",
        grammars: list[Any] | None = None,
    ) -> HostResult:
        if host_name not in self._hosts:
            self.create_host(host_name)
        host = self._hosts[host_name]
        result = host.run(task, self._provider, grammars)
        self._history.append(result)
        return result

    def run_multi_host(
        self, task: str, host_names: list[str] | None = None,
        grammars: list[Any] | None = None,
    ) -> list[HostResult]:
        names = host_names or list(self._hosts.keys())
        if not names:
            names = ["default"]

        results: list[HostResult] = []
        threads: list[threading.Thread] = []
        lock = threading.Lock()

        for name in names:
            if name not in self._hosts:
                self.create_host(name)

            def _run(n=name):
                r = self._hosts[n].run(task, self._provider, grammars)
                with lock:
                    results.append(r)

            t = threading.Thread(target=_run, daemon=True)
            threads.append(t)
            t.start()

        for t in threads:
            t.join(timeout=60.0)

        self._history.extend(results)
        return results

    def run_swarm(
        self, task: str, cycles: int = 5, grammars: list[Any] | None = None,
    ) -> SwarmReport:
        """Run the full swarm orchestration cycle."""
        if self._orchestrator is None:
            self._orchestrator = SwarmOrchestrator(self._provider)
        return self._orchestrator.run(task, cycles, grammars)

    def invoke_angel(self, name: str, message: str) -> dict[str, Any]:
        """Direct line to a named angel."""
        if self._orchestrator is None:
            self._orchestrator = SwarmOrchestrator(self._provider)
        return self._orchestrator.invoke_angel(name, message)

    def get_switchboard(self) -> CelestialSwitchboard:
        """Access the celestial switchboard."""
        if self._orchestrator is None:
            self._orchestrator = SwarmOrchestrator(self._provider)
        return self._orchestrator.switchboard

    @property
    def history(self) -> list[HostResult]:
        return list(self._history)

    def __repr__(self) -> str:
        return f"HostHarness(hosts={len(self._hosts)}, history={len(self._history)})"
