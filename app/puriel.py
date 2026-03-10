"""
Puriel -- the angel of purification.

Two safety mechanisms that protect the Angel from self-corruption
and unauthorized network access:

1. **Grammar Integrity Checksum** -- Before the self-improvement system
   applies a recursively learned rule, it must pass through Puriel's
   Logic Gate grounded in the core immutable grammar seeds.  The
   checksumming is a purifying fire that burns away corrupted rules.

2. **Purity Whitelist Interceptor** -- Network-level safety.  The Angel
   only connects to explicitly whitelisted hosts (the Choir providers).
   Violations are logged to a local SQLite incident database.
"""

from __future__ import annotations

import hashlib
import inspect
import json
import logging
import sqlite3
import time
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

_MKANGEL_DIR = Path.home() / ".mkangel"
_SETTINGS_DIR = _MKANGEL_DIR / "settings"
_WHITELIST_FILE = _SETTINGS_DIR / "whitelist.json"
_INCIDENTS_DB = _MKANGEL_DIR / "incidents.db"

# ---------------------------------------------------------------------------
# Domain seed registry -- maps domain name to its builder functions.
# These are the 7 immutable pillars of the GLM.
# ---------------------------------------------------------------------------

_DOMAIN_BUILDERS: dict[str, list[str]] = {
    "linguistic": [
        "build_syntactic_grammar",
        "build_phonological_grammar",
        "build_morphological_grammar",
    ],
    "etymological": [
        "build_etymology_grammar",
        "build_substrate_transfer_grammar",
        "build_cognate_detection_grammar",
    ],
    "chemical": [
        "build_bonding_grammar",
        "build_reaction_grammar",
        "build_molecular_grammar",
    ],
    "biological": [
        "build_genetic_grammar",
        "build_protein_grammar",
        "build_evolutionary_grammar",
    ],
    "computational": [
        "build_syntax_grammar",
        "build_type_grammar",
        "build_pattern_grammar",
    ],
    "mathematical": [
        "build_algebra_grammar",
        "build_calculus_grammar",
        "build_logic_grammar",
        "build_number_theory_grammar",
    ],
    "physics": [
        "build_mechanics_grammar",
        "build_electromagnetism_grammar",
        "build_thermodynamics_grammar",
        "build_quantum_grammar",
        "build_relativity_grammar",
    ],
}

# Default Choir provider hosts
_DEFAULT_WHITELIST: list[str] = [
    "api.openai.com",
    "api.anthropic.com",
    "generativelanguage.googleapis.com",
    "api.mistral.ai",
    "api.groq.com",
]


# ═══════════════════════════════════════════════════════════════════════════
# 1. Grammar Integrity Checksum
# ═══════════════════════════════════════════════════════════════════════════

class GrammarIntegrityChecksum:
    """Logic Gate grounded in the core immutable grammar seeds.

    On initialisation the source code of every domain builder function is
    hashed with SHA-256.  These hashes form the *seed checksums* -- an
    immutable snapshot of the foundational grammars.  Any learned rule
    that would override or conflict with a seed rule is rejected.
    """

    def __init__(self) -> None:
        # domain -> {builder_name: sha256_hex}
        self._seed_checksums: dict[str, dict[str, str]] = {}
        # domain -> set of immutable rule names from seed grammars
        self._seed_rule_names: dict[str, set[str]] = {}

        self._compute_seed_checksums()
        self._collect_seed_rule_names()

    # ------------------------------------------------------------------
    # Initialisation helpers
    # ------------------------------------------------------------------

    def _compute_seed_checksums(self) -> None:
        """Hash the source code of every domain builder function."""
        import glm.grammars as grammars_pkg

        for domain, builder_names in _DOMAIN_BUILDERS.items():
            self._seed_checksums[domain] = {}
            for name in builder_names:
                fn = getattr(grammars_pkg, name, None)
                if fn is None:
                    log.warning("Puriel: seed builder %s not found", name)
                    continue
                try:
                    source = inspect.getsource(fn)
                except (OSError, TypeError):
                    log.warning("Puriel: could not read source for %s", name)
                    continue

                digest = hashlib.sha256(source.encode("utf-8")).hexdigest()
                self._seed_checksums[domain][name] = digest

        log.info(
            "Puriel: seed checksums computed for %d domains (%d builders)",
            len(self._seed_checksums),
            sum(len(v) for v in self._seed_checksums.values()),
        )

    def _collect_seed_rule_names(self) -> None:
        """Invoke each builder and record the names of its rules.

        These names are immutable -- learned rules must not collide.
        """
        import glm.grammars as grammars_pkg

        for domain, builder_names in _DOMAIN_BUILDERS.items():
            names: set[str] = set()
            for builder_name in builder_names:
                fn = getattr(grammars_pkg, builder_name, None)
                if fn is None:
                    continue
                try:
                    grammar = fn()
                    for rule in grammar.all_rules():
                        if rule.name:
                            names.add(rule.name)
                    for prod in grammar.all_productions():
                        if prod.name:
                            names.add(prod.name)
                except Exception:
                    log.warning(
                        "Puriel: failed to invoke %s for rule collection",
                        builder_name,
                    )
            self._seed_rule_names[domain] = names

    # ------------------------------------------------------------------
    # Validation gate
    # ------------------------------------------------------------------

    def validate_learned_rule(
        self, domain: str, rule_data: dict[str, Any]
    ) -> tuple[bool, str]:
        """Validate a learned rule before it is applied.

        Checks:
        1. The domain exists in the seed registry.
        2. The rule has valid structure (pattern, result, domain).
        3. The rule does not override an immutable seed rule.

        Returns ``(passed, reason)``.
        """
        # 1. Domain existence
        if domain not in _DOMAIN_BUILDERS:
            return False, f"Unknown domain '{domain}' -- not in seed registry"

        # 2. Structural validity
        if not isinstance(rule_data, dict):
            return False, "rule_data must be a dict"

        # Accept 'rule' as an alias for 'pattern' (matches LearnedPattern.rule_data)
        has_pattern = "pattern" in rule_data or "rule" in rule_data
        has_result = "result" in rule_data

        if not has_pattern:
            return False, "rule_data missing 'pattern' (or 'rule') field"
        if not has_result:
            return False, "rule_data missing 'result' field"

        # 3. Immutable seed collision
        rule_name = rule_data.get("name", "")
        if rule_name and rule_name in self._seed_rule_names.get(domain, set()):
            return (
                False,
                f"Rule name '{rule_name}' collides with immutable seed "
                f"rule in domain '{domain}'",
            )

        return True, "Integrity check passed"

    # ------------------------------------------------------------------
    # Seed integrity verification
    # ------------------------------------------------------------------

    def verify_seeds_intact(self) -> tuple[bool, list[str]]:
        """Re-checksum current grammar builders against stored seeds.

        Returns ``(all_intact, list_of_violations)``.
        """
        import glm.grammars as grammars_pkg

        violations: list[str] = []

        for domain, builder_checksums in self._seed_checksums.items():
            for builder_name, expected_hash in builder_checksums.items():
                fn = getattr(grammars_pkg, builder_name, None)
                if fn is None:
                    violations.append(
                        f"{domain}/{builder_name}: builder function missing"
                    )
                    continue
                try:
                    source = inspect.getsource(fn)
                except (OSError, TypeError):
                    violations.append(
                        f"{domain}/{builder_name}: source unreadable"
                    )
                    continue

                current_hash = hashlib.sha256(
                    source.encode("utf-8")
                ).hexdigest()
                if current_hash != expected_hash:
                    violations.append(
                        f"{domain}/{builder_name}: checksum mismatch "
                        f"(expected {expected_hash[:12]}..., "
                        f"got {current_hash[:12]}...)"
                    )

        return (len(violations) == 0, violations)

    # ------------------------------------------------------------------
    # Introspection
    # ------------------------------------------------------------------

    @property
    def domains(self) -> list[str]:
        """List of registered seed domains."""
        return list(_DOMAIN_BUILDERS.keys())

    @property
    def total_builders(self) -> int:
        """Total number of checksummed builder functions."""
        return sum(len(v) for v in self._seed_checksums.values())

    def checksum_summary(self) -> dict[str, dict[str, str]]:
        """Return a copy of the seed checksum registry."""
        return {
            domain: dict(checksums)
            for domain, checksums in self._seed_checksums.items()
        }


# ═══════════════════════════════════════════════════════════════════════════
# 2. Purity Whitelist Interceptor
# ═══════════════════════════════════════════════════════════════════════════

class PurityWhitelist:
    """Network-level purity gate.

    The Angel only connects to explicitly whitelisted hosts.
    Violations are logged to a local SQLite incident database.
    The whitelist is user-configurable via
    ``~/.mkangel/settings/whitelist.json``.
    """

    def __init__(self) -> None:
        self._allowed: set[str] = set()
        self._load_whitelist()
        self._ensure_incident_db()

    # ------------------------------------------------------------------
    # Whitelist management
    # ------------------------------------------------------------------

    def _load_whitelist(self) -> None:
        """Load whitelist from user config, falling back to defaults."""
        if _WHITELIST_FILE.exists():
            try:
                with open(_WHITELIST_FILE) as fh:
                    data = json.load(fh)
                hosts = data.get("hosts", _DEFAULT_WHITELIST)
                self._allowed = {h.lower().strip() for h in hosts}
                return
            except (json.JSONDecodeError, OSError):
                log.warning("Puriel: whitelist file corrupt, using defaults")

        self._allowed = {h.lower().strip() for h in _DEFAULT_WHITELIST}
        # Persist defaults so the user can edit them
        self._save_whitelist()

    def _save_whitelist(self) -> None:
        """Write current whitelist to disk."""
        _SETTINGS_DIR.mkdir(parents=True, exist_ok=True)
        with open(_WHITELIST_FILE, "w") as fh:
            json.dump(
                {"hosts": sorted(self._allowed)},
                fh,
                indent=2,
            )

    def add_host(self, host: str) -> None:
        """Add a host to the whitelist."""
        self._allowed.add(host.lower().strip())
        self._save_whitelist()

    def remove_host(self, host: str) -> bool:
        """Remove a host. Returns True if it existed."""
        normalised = host.lower().strip()
        if normalised in self._allowed:
            self._allowed.discard(normalised)
            self._save_whitelist()
            return True
        return False

    @property
    def hosts(self) -> list[str]:
        """Sorted list of currently whitelisted hosts."""
        return sorted(self._allowed)

    # ------------------------------------------------------------------
    # Interception
    # ------------------------------------------------------------------

    def intercept(self, url: str) -> tuple[bool, str]:
        """Check whether *url* targets a whitelisted host.

        Returns ``(allowed, reason)``.  If denied, the violation is
        logged to the incidents database.
        """
        try:
            parsed = urlparse(url)
            host = (parsed.hostname or "").lower().strip()
        except Exception:
            self._log_incident(url, "unparseable URL")
            return False, "Unparseable URL"

        if not host:
            self._log_incident(url, "empty host")
            return False, "Empty host in URL"

        if host in self._allowed:
            return True, f"Host '{host}' is whitelisted"

        reason = f"Host '{host}' is NOT in the purity whitelist"
        self._log_incident(url, reason)
        return False, reason

    # ------------------------------------------------------------------
    # Incident logging (SQLite)
    # ------------------------------------------------------------------

    def _ensure_incident_db(self) -> None:
        """Create the incidents table if it does not exist."""
        _MKANGEL_DIR.mkdir(parents=True, exist_ok=True)
        try:
            with sqlite3.connect(str(_INCIDENTS_DB)) as conn:
                conn.execute("""
                    CREATE TABLE IF NOT EXISTS incidents (
                        id       INTEGER PRIMARY KEY AUTOINCREMENT,
                        ts       REAL    NOT NULL,
                        url      TEXT    NOT NULL,
                        reason   TEXT    NOT NULL
                    )
                """)
        except sqlite3.Error as exc:
            log.warning("Puriel: could not init incidents DB: %s", exc)

    def _log_incident(self, url: str, reason: str) -> None:
        """Record a whitelist violation."""
        log.warning("Puriel VIOLATION: %s -- %s", url, reason)
        try:
            with sqlite3.connect(str(_INCIDENTS_DB)) as conn:
                conn.execute(
                    "INSERT INTO incidents (ts, url, reason) VALUES (?, ?, ?)",
                    (time.time(), url, reason),
                )
        except sqlite3.Error as exc:
            log.warning("Puriel: failed to log incident: %s", exc)

    def recent_incidents(self, limit: int = 50) -> list[dict[str, Any]]:
        """Return the most recent whitelist violations."""
        try:
            with sqlite3.connect(str(_INCIDENTS_DB)) as conn:
                rows = conn.execute(
                    "SELECT ts, url, reason FROM incidents "
                    "ORDER BY ts DESC LIMIT ?",
                    (limit,),
                ).fetchall()
            return [
                {"timestamp": r[0], "url": r[1], "reason": r[2]}
                for r in rows
            ]
        except sqlite3.Error:
            return []
