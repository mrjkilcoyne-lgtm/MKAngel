"""
Legal compliance layer -- UK GDPR, EU GDPR, CCPA/CPRA, COPPA.

The Angel respects your data as she respects grammar:
with structure, purpose, and care.
"""

from __future__ import annotations

import functools
import json
import re
import time
from pathlib import Path
from typing import Any, Callable

from app.paths import mkangel_dir

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

RETENTION_DAYS_DEFAULT = 90
COPPA_AGE = 13
GDPR_MINOR_AGE = 16
UK_GDPR_MINOR_AGE = 13

CONSENT_PURPOSES = (
    "data_processing",
    "external_api",
    "analytics",
    "language_profiling",
    "web_search",
)

# Known third-party providers and their DPA status.
_DPA_REGISTRY: dict[str, dict[str, Any]] = {
    "openai": {
        "provider": "OpenAI",
        "dpa_available": True,
        "data_location": "US",
        "adequacy_decision": False,
        "notes": "Standard Contractual Clauses available.",
    },
    "anthropic": {
        "provider": "Anthropic",
        "dpa_available": True,
        "data_location": "US",
        "adequacy_decision": False,
        "notes": "Standard Contractual Clauses available.",
    },
    "google": {
        "provider": "Google",
        "dpa_available": True,
        "data_location": "US/EU",
        "adequacy_decision": False,
        "notes": "EU data residency option exists.",
    },
    "mistral": {
        "provider": "Mistral AI",
        "dpa_available": True,
        "data_location": "EU",
        "adequacy_decision": True,
        "notes": "EU-based provider.",
    },
    "groq": {
        "provider": "Groq",
        "dpa_available": False,
        "data_location": "US",
        "adequacy_decision": False,
        "notes": "No public DPA at time of writing.",
    },
    "local": {
        "provider": "Local (on-device)",
        "dpa_available": True,
        "data_location": "device",
        "adequacy_decision": True,
        "notes": "No data leaves the device.",
    },
}

PRIVACY_NOTICE = """\
MKAngel Privacy Notice
======================

What data we collect
--------------------
- Conversation history (stored locally on your device).
- Learned grammar patterns and user preferences.
- Consent choices.

How we use it
-------------
- To provide grammar analysis, language assistance, and chat.
- To improve responses through learned patterns.
- When you grant consent, prompts may be sent to third-party
  LLM providers (OpenAI, Anthropic, Google, Mistral, Groq).

Third-party sharing
-------------------
- Data is ONLY sent to external providers when you have given
  explicit consent for "external_api" processing.
- We do not sell your data.
- External providers may process data under their own terms;
  see their respective privacy policies.

Data retention
--------------
- Conversation and memory data is retained for {retention} days
  by default.  You may change this in settings.
- You may request deletion at any time (Right to Erasure).

Your rights
-----------
Under UK GDPR, EU GDPR, and CCPA/CPRA you have the right to:
  - Access your data (Right of Access / Right to Know).
  - Export your data in a portable format (Data Portability).
  - Delete your data (Right to Erasure / Right to Delete).
  - Withdraw consent at any time.
  - Object to processing.

Children
--------
- Users under 16 (EU GDPR) or under 13 (UK GDPR / US COPPA)
  receive additional protections including content filtering
  and restricted external API access.

Contact
-------
For data protection queries, contact the app maintainer via
the project's GitHub repository.
"""

# ---------------------------------------------------------------------------
# PII regex patterns
# ---------------------------------------------------------------------------

_EMAIL_RE = re.compile(
    r"[a-zA-Z0-9._%+\-]+@[a-zA-Z0-9.\-]+\.[a-zA-Z]{2,}"
)

# International phone: optional +, country code, then digits/spaces/dashes
_PHONE_RE = re.compile(
    r"(?<!\d)(?:\+?\d{1,3}[\s\-]?)?"
    r"(?:\(?\d{2,5}\)?[\s\-]?)"
    r"\d{3,4}[\s\-]?\d{3,4}(?!\d)"
)

# UK postcode: e.g. SW1A 1AA, EC2R 8AH, M1 1AE
_UK_POSTCODE_RE = re.compile(
    r"\b[A-Z]{1,2}\d[A-Z\d]?\s?\d[A-Z]{2}\b",
    re.IGNORECASE,
)

# UK National Insurance Number: AB 12 34 56 C
_UK_NINO_RE = re.compile(
    r"\b[A-CEGHJ-PR-TW-Z]{2}\s?\d{2}\s?\d{2}\s?\d{2}\s?[A-D]\b",
    re.IGNORECASE,
)

# US Social Security Number: 123-45-6789
_US_SSN_RE = re.compile(
    r"\b\d{3}[\-\s]?\d{2}[\-\s]?\d{4}\b"
)

_PII_PATTERNS: list[tuple[re.Pattern[str], str]] = [
    (_EMAIL_RE, "[EMAIL]"),
    (_PHONE_RE, "[PHONE]"),
    (_UK_POSTCODE_RE, "[POSTCODE]"),
    (_UK_NINO_RE, "[NINO]"),
    (_US_SSN_RE, "[SSN]"),
]

# Words that suggest unsafe content for minors.
_UNSAFE_KEYWORDS = {
    "violence", "violent", "gore", "murder", "kill",
    "drug", "drugs", "narcotic",
    "gambling", "casino", "betting",
    "porn", "pornography", "xxx", "nsfw",
    "suicide", "self-harm", "self harm",
    "weapon", "weapons", "firearm", "firearms",
    "extremism", "terrorism",
}

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

_CONFIG_DIR = mkangel_dir()
_CONSENT_FILE = _CONFIG_DIR / "consent.json"


# ===========================================================================
# ConsentManager
# ===========================================================================

class ConsentManager:
    """Tracks and persists user consent per processing purpose."""

    def __init__(self) -> None:
        self._consent: dict[str, bool] = {p: False for p in CONSENT_PURPOSES}
        self._age: int | None = None
        self._jurisdiction: str = "ALL"
        self._first_run: bool = True
        self._load()

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def _load(self) -> None:
        if not _CONSENT_FILE.exists():
            self._first_run = True
            return
        try:
            with open(_CONSENT_FILE, "r") as fh:
                data = json.load(fh)
            for purpose in CONSENT_PURPOSES:
                self._consent[purpose] = bool(data.get("consent", {}).get(purpose, False))
            self._age = data.get("age")
            self._jurisdiction = data.get("jurisdiction", "ALL")
            self._first_run = False
        except (json.JSONDecodeError, OSError):
            self._first_run = True

    def _save(self) -> None:
        _CONFIG_DIR.mkdir(parents=True, exist_ok=True)
        data = {
            "consent": self._consent,
            "age": self._age,
            "jurisdiction": self._jurisdiction,
            "updated_at": time.time(),
        }
        with open(_CONSENT_FILE, "w") as fh:
            json.dump(data, fh, indent=2)
        self._first_run = False

    # ------------------------------------------------------------------
    # Consent operations
    # ------------------------------------------------------------------

    def request_consent(self, purpose: str) -> bool:
        """Return True if the user has granted consent for *purpose*."""
        return self._consent.get(purpose, False)

    def grant_consent(self, purpose: str) -> None:
        """Record that the user has granted consent for *purpose*."""
        if purpose not in CONSENT_PURPOSES:
            raise ValueError(
                f"Unknown consent purpose '{purpose}'. "
                f"Valid: {', '.join(CONSENT_PURPOSES)}"
            )
        self._consent[purpose] = True
        self._save()

    def revoke_consent(self, purpose: str) -> None:
        """Withdraw consent for *purpose*."""
        if purpose in self._consent:
            self._consent[purpose] = False
            self._save()

    def get_consent_status(self) -> dict[str, bool]:
        """Return a copy of all consent states."""
        return dict(self._consent)

    def is_first_run(self) -> bool:
        """True if consent has never been collected."""
        return self._first_run

    # ------------------------------------------------------------------
    # Age verification
    # ------------------------------------------------------------------

    def set_user_age(self, age: int) -> None:
        """Record the user's age for minor-status checks."""
        if age < 0:
            raise ValueError("Age must be non-negative.")
        self._age = age
        self._save()

    def is_minor(self) -> bool:
        """True if the user qualifies as a minor under applicable law.

        EU GDPR: under 16.
        UK GDPR / US COPPA: under 13.
        If jurisdiction is ALL, use the strictest threshold (16).
        """
        if self._age is None:
            return False  # age unknown -- cannot assert minor status
        j = self._jurisdiction.upper()
        if j == "EU":
            return self._age < GDPR_MINOR_AGE
        if j == "UK":
            return self._age < UK_GDPR_MINOR_AGE
        if j == "US":
            return self._age < COPPA_AGE
        # ALL or unknown -- strictest
        return self._age < GDPR_MINOR_AGE

    def set_jurisdiction(self, jurisdiction: str) -> None:
        """Set the legal jurisdiction: UK, EU, US, or ALL."""
        jurisdiction = jurisdiction.upper()
        if jurisdiction not in ("UK", "EU", "US", "ALL"):
            raise ValueError(
                f"Unsupported jurisdiction '{jurisdiction}'. Use UK, EU, US, or ALL."
            )
        self._jurisdiction = jurisdiction
        self._save()

    @property
    def jurisdiction(self) -> str:
        return self._jurisdiction

    @property
    def age(self) -> int | None:
        return self._age


# ===========================================================================
# DataProtectionOfficer
# ===========================================================================

class DataProtectionOfficer:
    """The compliance enforcer -- checks, sanitises, and advises."""

    def __init__(self, consent_manager: ConsentManager | None = None) -> None:
        self._cm = consent_manager or ConsentManager()
        self._retention_days = RETENTION_DAYS_DEFAULT

    # ------------------------------------------------------------------
    # Pre-flight checks
    # ------------------------------------------------------------------

    def check_before_api_call(
        self, provider: str, prompt: str
    ) -> tuple[bool, str]:
        """Check whether an API call is permitted.

        Returns (allowed, reason).
        """
        if not self._cm.request_consent("external_api"):
            return False, "User has not consented to external API calls."

        if not self._cm.request_consent("data_processing"):
            return False, "User has not consented to data processing."

        if self._cm.is_minor():
            provider_lower = provider.lower()
            if provider_lower != "local":
                return False, (
                    "External API calls are blocked for users identified "
                    "as minors under applicable data protection law."
                )

        dpa = self.get_dpa_status(provider)
        if not dpa.get("dpa_available", False):
            return False, (
                f"No Data Processing Agreement on file for provider "
                f"'{provider}'.  Proceed with caution or use a different provider."
            )

        # Check for PII in the prompt
        sanitised = self.sanitize_for_external(prompt)
        if sanitised != prompt:
            return False, (
                "The prompt appears to contain personal data (PII).  "
                "Sanitise before sending to an external provider."
            )

        return True, "OK"

    def check_before_web_request(self, url: str) -> tuple[bool, str]:
        """Check whether a web request is permitted."""
        if not self._cm.request_consent("web_search"):
            return False, "User has not consented to web search / requests."

        if self._cm.is_minor():
            # Very basic domain blocking for minors -- real-world would
            # use a proper safe-search API.
            lower_url = url.lower()
            for kw in _UNSAFE_KEYWORDS:
                if kw in lower_url:
                    return False, (
                        f"URL blocked for minor user: contains '{kw}'."
                    )

        return True, "OK"

    # ------------------------------------------------------------------
    # PII sanitisation
    # ------------------------------------------------------------------

    def sanitize_for_external(self, text: str) -> str:
        """Strip PII patterns from *text* before external transmission."""
        result = text
        for pattern, replacement in _PII_PATTERNS:
            result = pattern.sub(replacement, result)
        return result

    # ------------------------------------------------------------------
    # Data retention
    # ------------------------------------------------------------------

    def check_data_retention(self, memory_entry_age_days: float) -> bool:
        """Return True if the entry should be RETAINED (not yet expired)."""
        return memory_entry_age_days <= self._retention_days

    def get_retention_days(self) -> int:
        """Current retention period in days."""
        return self._retention_days

    def set_retention_days(self, days: int) -> None:
        """Override the default retention period."""
        if days < 1:
            raise ValueError("Retention period must be at least 1 day.")
        self._retention_days = days

    # ------------------------------------------------------------------
    # Privacy notice & DPA status
    # ------------------------------------------------------------------

    def get_privacy_notice(self) -> str:
        """Return a plain-text privacy notice."""
        return PRIVACY_NOTICE.replace(
            "{retention}", str(self._retention_days)
        )

    def get_dpa_status(self, provider: str) -> dict[str, Any]:
        """Return DPA information for *provider*."""
        key = provider.lower().strip()
        if key in _DPA_REGISTRY:
            return dict(_DPA_REGISTRY[key])
        return {
            "provider": provider,
            "dpa_available": False,
            "data_location": "unknown",
            "adequacy_decision": False,
            "notes": "Provider not in registry.  Manual review required.",
        }

    # ------------------------------------------------------------------
    # Jurisdiction (delegates to ConsentManager)
    # ------------------------------------------------------------------

    def set_jurisdiction(self, jurisdiction: str) -> None:
        self._cm.set_jurisdiction(jurisdiction)

    @property
    def jurisdiction(self) -> str:
        return self._cm.jurisdiction


# ===========================================================================
# ComplianceGuard -- decorator & wrappers
# ===========================================================================

class ComplianceGuard:
    """High-level guards that combine consent checks and sanitisation."""

    def __init__(
        self,
        consent_manager: ConsentManager | None = None,
        dpo: DataProtectionOfficer | None = None,
    ) -> None:
        self.cm = consent_manager or ConsentManager()
        self.dpo = dpo or DataProtectionOfficer(self.cm)

    def guard_api_call(
        self, provider: str, prompt: str
    ) -> tuple[bool, str, str]:
        """Wrap a provider call with consent + sanitisation.

        Returns (allowed, reason, sanitised_prompt).
        """
        allowed, reason = self.dpo.check_before_api_call(provider, prompt)
        if not allowed:
            return False, reason, prompt
        sanitised = self.dpo.sanitize_for_external(prompt)
        return True, "OK", sanitised

    def guard_web_request(self, url: str) -> tuple[bool, str]:
        """Wrap a web request with consent check.

        Returns (allowed, reason).
        """
        return self.dpo.check_before_web_request(url)


def requires_consent(purpose: str) -> Callable:
    """Decorator that blocks execution unless consent is granted.

    Usage::

        @requires_consent("external_api")
        def call_provider(prompt):
            ...

    Raises RuntimeError if consent is not granted.
    """

    def decorator(fn: Callable) -> Callable:
        @functools.wraps(fn)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            cm = ConsentManager()
            if not cm.request_consent(purpose):
                raise RuntimeError(
                    f"Consent required for '{purpose}' but not granted.  "
                    f"Call ConsentManager.grant_consent('{purpose}') first."
                )
            return fn(*args, **kwargs)

        return wrapper

    return decorator


# ===========================================================================
# DataPortability -- GDPR Art.17 & Art.20
# ===========================================================================

class DataPortability:
    """Implements the right to data portability and right to erasure."""

    def export_user_data(self, memory: Any) -> str:
        """Export all user data as a JSON string (Art.20 portability).

        Expects *memory* to be an ``app.memory.Memory`` instance (or
        anything with ``stats``, ``search_memory``, ``list_sessions``
        methods).
        """
        export: dict[str, Any] = {
            "export_format": "MKAngel Data Export v1",
            "exported_at": time.time(),
            "consent": ConsentManager().get_consent_status(),
        }

        try:
            export["stats"] = memory.stats()
        except Exception:
            export["stats"] = {}

        try:
            export["sessions"] = memory.list_sessions(limit=10000)
        except Exception:
            export["sessions"] = []

        try:
            export["memories"] = [
                {
                    "id": e.id,
                    "category": e.category,
                    "key": e.key,
                    "value": e.value,
                    "metadata": e.metadata,
                    "created_at": e.created_at,
                }
                for e in memory.search_memory("", limit=100000)
            ]
        except Exception:
            export["memories"] = []

        return json.dumps(export, indent=2, default=str)

    def delete_user_data(self, memory: Any) -> dict[str, Any]:
        """Delete all user data (Art.17 right to erasure).

        Returns a summary of what was deleted.
        """
        deleted: dict[str, Any] = {"deleted_at": time.time(), "items": {}}

        # Delete sessions
        try:
            sessions = memory.list_sessions(limit=100000)
            for s in sessions:
                memory.delete_session(s["session_id"])
            deleted["items"]["sessions"] = len(sessions)
        except Exception:
            deleted["items"]["sessions"] = "error"

        # Delete consent file
        try:
            if _CONSENT_FILE.exists():
                _CONSENT_FILE.unlink()
            deleted["items"]["consent_file"] = True
        except Exception:
            deleted["items"]["consent_file"] = "error"

        return deleted

    def get_data_inventory(self) -> dict[str, Any]:
        """Describe what data is stored and the legal basis."""
        return {
            "conversation_history": {
                "location": "~/.mkangel/memory.db",
                "purpose": "Provide contextual assistance across sessions.",
                "legal_basis": "Legitimate interest / user consent.",
                "retention": f"{RETENTION_DAYS_DEFAULT} days (configurable).",
            },
            "learned_patterns": {
                "location": "~/.mkangel/memory.db",
                "purpose": "Improve grammar analysis quality over time.",
                "legal_basis": "Legitimate interest / user consent.",
                "retention": f"{RETENTION_DAYS_DEFAULT} days (configurable).",
            },
            "user_preferences": {
                "location": "~/.mkangel/memory.db",
                "purpose": "Remember user settings between sessions.",
                "legal_basis": "Necessary for service provision.",
                "retention": "Until user deletes or resets.",
            },
            "consent_records": {
                "location": "~/.mkangel/consent.json",
                "purpose": "Record user consent decisions.",
                "legal_basis": "Legal obligation (GDPR Art.7).",
                "retention": "Duration of account plus 3 years.",
            },
            "api_keys": {
                "location": "~/.mkangel/settings.json",
                "purpose": "Authenticate with chosen LLM providers.",
                "legal_basis": "Necessary for service provision.",
                "retention": "Until user removes key.",
            },
        }


# ===========================================================================
# ContentModerator
# ===========================================================================

class ContentModerator:
    """Basic content safety checks, especially for minor users."""

    def check_safe_search(self, query: str) -> bool:
        """Return True if *query* appears safe for minor users."""
        lower = query.lower()
        for kw in _UNSAFE_KEYWORDS:
            if kw in lower:
                return False
        return True

    def filter_response(self, text: str, is_minor: bool) -> str:
        """Apply content filtering to a response.

        For minor users, redacts lines containing unsafe keywords.
        For adult users, returns text unchanged.
        """
        if not is_minor:
            return text

        lines = text.split("\n")
        filtered: list[str] = []
        for line in lines:
            lower_line = line.lower()
            if any(kw in lower_line for kw in _UNSAFE_KEYWORDS):
                filtered.append("[Content filtered for age-appropriate viewing.]")
            else:
                filtered.append(line)
        return "\n".join(filtered)
