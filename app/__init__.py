"""
MKAngel Application Layer

The user-facing shell around the Grammar Language Model.
Works on Termux (Android), desktop Linux/macOS/Windows,
and is installable via pip.
"""

__version__ = "0.1.0"

from app.settings import Settings, PROVIDERS
from app.memory import Memory, MemoryEntry
from app.providers import (
    Provider,
    LocalProvider,
    APIProvider,
    HybridProvider,
    get_provider,
)
from app.coder import Coder
from app.skills import SkillManager, Skill
from app.cowork import CoworkSession
from app.chat import ChatSession
from app.voice import VoiceEngine, AudioSegment, TranscriptionResult, VoiceProfile
from app.swarm import Swarm, SwarmHarness, BorgesLibrary, AgentRole
from app.cloud import CloudManager, CloudStorage, LocalStorage, CloudConfig
from app.self_improve import SelfImprover, LearnedPattern, SkillRequest

__all__ = [
    "Settings",
    "PROVIDERS",
    "Memory",
    "MemoryEntry",
    "Provider",
    "LocalProvider",
    "APIProvider",
    "HybridProvider",
    "get_provider",
    "Coder",
    "SkillManager",
    "Skill",
    "CoworkSession",
    "ChatSession",
    # Voice
    "VoiceEngine",
    "AudioSegment",
    "TranscriptionResult",
    "VoiceProfile",
    # Swarm
    "Swarm",
    "SwarmHarness",
    "BorgesLibrary",
    "AgentRole",
    # Cloud
    "CloudManager",
    "CloudStorage",
    "LocalStorage",
    "CloudConfig",
    # Self-improvement
    "SelfImprover",
    "LearnedPattern",
    "SkillRequest",
]
