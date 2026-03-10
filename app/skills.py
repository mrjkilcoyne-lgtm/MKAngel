"""
Skill creation and management for MKAngel.

Skills are named capabilities with trigger patterns and actions.
They extend the Angel's abilities through a simple plugin system.
Stored as JSON files in ~/.mkangel/skills/.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any


_SKILLS_DIR = Path.home() / ".mkangel" / "skills"


@dataclass
class Skill:
    """A named capability with a trigger pattern and action.

    Skills are lightweight plugins: a name, a trigger (regex or keyword),
    an action type, and configuration for how to execute.
    """

    name: str
    description: str
    trigger: str               # keyword or pattern that activates this skill
    action: str                # action type: "prompt", "transform", "chain"
    config: dict[str, Any] = field(default_factory=dict)
    created_at: float = 0.0
    enabled: bool = True

    def matches(self, text: str) -> bool:
        """Check if the input text matches this skill's trigger."""
        trigger_lower = self.trigger.lower()
        text_lower = text.lower()

        # Simple keyword matching
        if trigger_lower in text_lower:
            return True

        # Check if trigger words are all present
        trigger_words = trigger_lower.split()
        if len(trigger_words) > 1:
            return all(w in text_lower for w in trigger_words)

        return False

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dictionary."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Skill":
        """Deserialize from dictionary."""
        return cls(
            name=data.get("name", "unnamed"),
            description=data.get("description", ""),
            trigger=data.get("trigger", ""),
            action=data.get("action", "prompt"),
            config=data.get("config", {}),
            created_at=data.get("created_at", 0.0),
            enabled=data.get("enabled", True),
        )


# ---------------------------------------------------------------------------
# Default skills
# ---------------------------------------------------------------------------

_DEFAULT_SKILLS = [
    Skill(
        name="summarize",
        description="Summarize text or conversation into key points",
        trigger="summarize",
        action="prompt",
        config={
            "system_prompt": (
                "Summarize the following into clear, concise bullet points. "
                "Focus on key information and actionable items."
            ),
            "temperature": 0.3,
        },
    ),
    Skill(
        name="translate",
        description="Translate text between languages or domains",
        trigger="translate",
        action="prompt",
        config={
            "system_prompt": (
                "Translate the following text. If no target language is "
                "specified, translate to English. Preserve meaning and tone."
            ),
            "temperature": 0.2,
        },
    ),
    Skill(
        name="analyze",
        description="Deep analysis of text, code, or data",
        trigger="analyze",
        action="prompt",
        config={
            "system_prompt": (
                "Provide a thorough analysis of the following. Cover structure, "
                "patterns, strengths, weaknesses, and suggestions."
            ),
            "temperature": 0.4,
        },
    ),
    Skill(
        name="predict",
        description="Predict next elements or outcomes using grammar patterns",
        trigger="predict next",
        action="prompt",
        config={
            "system_prompt": (
                "Based on the patterns in this input, predict what comes next. "
                "Explain your reasoning using structural and grammatical patterns."
            ),
            "temperature": 0.5,
        },
    ),
]


class SkillManager:
    """Manages custom skills / plugins for MKAngel.

    Skills are stored as individual JSON files in ~/.mkangel/skills/.
    Default skills are always available and regenerated if missing.
    """

    def __init__(self, skills_dir: Path | str | None = None):
        self._dir = Path(skills_dir) if skills_dir else _SKILLS_DIR
        self._dir.mkdir(parents=True, exist_ok=True)
        self._skills: dict[str, Skill] = {}
        self._load_all()

    # ------------------------------------------------------------------
    # Skill management
    # ------------------------------------------------------------------

    def create_skill(
        self,
        name: str,
        trigger: str,
        action: str = "prompt",
        description: str = "",
        config: dict[str, Any] | None = None,
    ) -> Skill:
        """Create and persist a new skill.

        Args:
            name: Unique skill name.
            trigger: Keyword or pattern that activates this skill.
            action: Action type ("prompt", "transform", "chain").
            description: Human-readable description.
            config: Additional configuration.

        Returns:
            The created Skill.

        Raises:
            ValueError: If a skill with this name already exists.
        """
        name = name.strip().lower().replace(" ", "_")
        if name in self._skills:
            raise ValueError(
                f"Skill '{name}' already exists. Delete it first or choose "
                f"a different name."
            )

        skill = Skill(
            name=name,
            description=description or f"Custom skill: {name}",
            trigger=trigger,
            action=action,
            config=config or {},
            created_at=time.time(),
        )
        self._skills[name] = skill
        self._save_skill(skill)
        return skill

    def delete_skill(self, name: str) -> bool:
        """Delete a skill by name. Returns True if it existed."""
        name = name.strip().lower()
        if name not in self._skills:
            return False

        del self._skills[name]
        skill_file = self._dir / f"{name}.json"
        if skill_file.exists():
            skill_file.unlink()
        return True

    def list_skills(self) -> list[Skill]:
        """Return all registered skills."""
        return list(self._skills.values())

    def get_skill(self, name: str) -> Skill | None:
        """Get a skill by name."""
        return self._skills.get(name.strip().lower())

    def find_matching_skills(self, text: str) -> list[Skill]:
        """Find all skills whose triggers match the input text."""
        return [
            s for s in self._skills.values()
            if s.enabled and s.matches(text)
        ]

    def execute_skill(
        self,
        name: str,
        user_input: str,
        provider=None,
    ) -> str:
        """Execute a skill with the given input.

        Args:
            name: Skill name.
            user_input: The user's input text.
            provider: An LLM provider to use for generation.

        Returns:
            The skill's output as a string.
        """
        skill = self.get_skill(name)
        if skill is None:
            return f"Skill '{name}' not found. Use /skills to list available skills."

        if not skill.enabled:
            return f"Skill '{skill.name}' is currently disabled."

        system_prompt = skill.config.get("system_prompt", "")
        temperature = skill.config.get("temperature", 0.5)

        if skill.action == "prompt":
            return self._execute_prompt(
                skill, user_input, system_prompt, temperature, provider
            )
        elif skill.action == "transform":
            return self._execute_transform(skill, user_input)
        elif skill.action == "chain":
            return self._execute_chain(skill, user_input, provider)
        else:
            return f"Unknown action type: {skill.action}"

    def toggle_skill(self, name: str) -> bool | None:
        """Toggle a skill's enabled state. Returns new state or None."""
        skill = self.get_skill(name)
        if skill is None:
            return None
        skill.enabled = not skill.enabled
        self._save_skill(skill)
        return skill.enabled

    # ------------------------------------------------------------------
    # Execution methods
    # ------------------------------------------------------------------

    def _execute_prompt(
        self,
        skill: Skill,
        user_input: str,
        system_prompt: str,
        temperature: float,
        provider,
    ) -> str:
        """Execute a prompt-based skill."""
        if provider is not None:
            return provider.generate(
                user_input,
                system=system_prompt,
                temperature=temperature,
            )

        # Fallback: local processing
        lines = [f"[Skill: {skill.name}]"]
        lines.append(f"Action: {skill.action}")
        lines.append(f"Input: {user_input}")
        lines.append("")
        if system_prompt:
            lines.append(f"System context: {system_prompt}")
            lines.append("")
        lines.append(
            "Note: For full skill execution, configure an API provider."
        )
        lines.append("The local GLM provides structural analysis only.")

        # Try local grammar analysis
        try:
            from glm.angel import Angel
            angel = Angel()
            angel.awaken()
            tokens = user_input.lower().split()
            predictions = angel.predict(tokens, domain="linguistic", horizon=5)
            if predictions:
                lines.append("")
                lines.append("Grammar analysis:")
                for pred in predictions[:3]:
                    lines.append(
                        f"  {pred.get('grammar','?')}: "
                        f"{pred.get('predicted','?')} "
                        f"(conf={pred.get('confidence',0):.2f})"
                    )
        except Exception:
            pass

        return "\n".join(lines)

    def _execute_transform(self, skill: Skill, user_input: str) -> str:
        """Execute a transform-based skill (text manipulation)."""
        transform = skill.config.get("transform", "identity")
        if transform == "uppercase":
            return user_input.upper()
        elif transform == "lowercase":
            return user_input.lower()
        elif transform == "reverse":
            return user_input[::-1]
        elif transform == "word_count":
            words = user_input.split()
            return f"Word count: {len(words)}"
        else:
            return user_input  # identity

    def _execute_chain(
        self, skill: Skill, user_input: str, provider,
    ) -> str:
        """Execute a chain-based skill (sequence of operations)."""
        steps = skill.config.get("steps", [])
        if not steps:
            return f"Skill '{skill.name}' has no chain steps configured."

        result = user_input
        outputs = []
        for i, step in enumerate(steps, 1):
            step_prompt = step.get("prompt", "Process this: {input}")
            step_prompt = step_prompt.replace("{input}", result)

            if provider is not None:
                result = provider.generate(
                    step_prompt,
                    system=step.get("system", ""),
                    temperature=step.get("temperature", 0.5),
                )
            else:
                result = f"[Step {i}: {step.get('name', 'unnamed')}] {step_prompt}"

            outputs.append(f"Step {i}: {step.get('name', f'step_{i}')}")

        return f"Chain completed ({len(steps)} steps):\n\n{result}"

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def _save_skill(self, skill: Skill) -> None:
        """Save a skill to disk as JSON."""
        path = self._dir / f"{skill.name}.json"
        with open(path, "w") as f:
            json.dump(skill.to_dict(), f, indent=2)

    def _load_all(self) -> None:
        """Load all skills from disk, creating defaults if needed."""
        # Ensure defaults exist on disk
        for default in _DEFAULT_SKILLS:
            default_path = self._dir / f"{default.name}.json"
            if not default_path.exists():
                default.created_at = default.created_at or time.time()
                self._save_skill(default)

        # Load all skill files
        for path in sorted(self._dir.glob("*.json")):
            try:
                with open(path) as f:
                    data = json.load(f)
                skill = Skill.from_dict(data)
                self._skills[skill.name] = skill
            except (json.JSONDecodeError, OSError, KeyError):
                continue  # Skip broken files silently
