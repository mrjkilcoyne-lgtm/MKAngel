"""
MKAngel Application Layer

The user-facing shell around the Grammar Language Model.
Works on Termux (Android), desktop Linux/macOS/Windows,
and is installable via pip.

Imports are individually guarded so a missing native dependency
on one platform (e.g. Android/Kivy) does not break the entire
package.
"""

__version__ = "0.1.0"

# ── Core (always expected to work) ──────────────────────────────────────

try:
    from app.settings import Settings, PROVIDERS
except Exception:
    pass

try:
    from app.memory import Memory, MemoryEntry
except Exception:
    pass

try:
    from app.providers import (
        Provider,
        LocalProvider,
        APIProvider,
        HybridProvider,
        get_provider,
    )
except Exception:
    pass

try:
    from app.coder import Coder
except Exception:
    pass

try:
    from app.skills import SkillManager, Skill
except Exception:
    pass

try:
    from app.cowork import CoworkSession
except Exception:
    pass

try:
    from app.chat import ChatSession
except Exception:
    pass

# ── Optional / platform-dependent ──────────────────────────────────────

try:
    from app.voice import VoiceEngine, AudioSegment, TranscriptionResult, VoiceProfile
except Exception:
    pass

try:
    from app.swarm import Host, HostHarness, BorgesLibrary, AgentRole
except Exception:
    pass

try:
    from app.cloud import CloudManager, CloudStorage, LocalStorage, CloudConfig
except Exception:
    pass

try:
    from app.self_improve import SelfImprover, LearnedPattern, SkillRequest
except Exception:
    pass

try:
    from app.puriel import GrammarIntegrityChecksum, PurityWhitelist
except Exception:
    pass

# ── Phase 1: First Light ──────────────────────────────────────────────

try:
    from app.vestments import get_vestment, vestment_to_css, ALL_VESTMENTS
except Exception:
    pass

try:
    from app.gestures import GestureDetector, GestureAction
except Exception:
    pass

try:
    from app.documents import DocumentManager, Document
except Exception:
    pass
