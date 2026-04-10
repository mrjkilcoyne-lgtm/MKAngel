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

# ── Angel EVERYTHING modules ──────────────────────────────────────────

try:
    from app.angel_ui import AngelScreen, AngelOrb, AngelMessage, ResponseKind
except Exception:
    pass

try:
    from app.tools import ToolRegistry, ToolSpec, create_default_registry
except Exception:
    pass

try:
    from app.web import search, fetch, check_connectivity
except Exception:
    pass

try:
    from app.tongue import Tongue, Language, OutputFormat
except Exception:
    pass

try:
    from app.compliance import (
        ConsentManager, DataProtectionOfficer, ComplianceGuard,
        DataPortability, ContentModerator,
    )
except Exception:
    pass

try:
    from app.senses import AngelSenses, Sense, Perception, CodeReader, ErrorReader
except Exception:
    pass

try:
    from app.growth import (
        GrowthEngine, GrowthPatch, SessionTracker, Reflector, ShutdownIncentive,
    )
except Exception:
    pass

try:
    from app.conductor import AngelConductor
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

# ── WhatsApp bridge (optional: needs claude-agent-sdk + Node/Baileys) ──

try:
    from app.whatsapp.config import BridgeConfig
except Exception:
    pass
