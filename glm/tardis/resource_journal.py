"""
Resource Journal — TARDIS's write-once ledger of everything it depends on.

Every compute instance, storage bucket, API key, database, service account
and billing line that has ever been allocated to TARDIS gets a single
immutable entry in this journal.  The journal is append-only: decommissioning
a resource is itself a new entry that supersedes the old one, never a
mutation.  The Angel at the centre of MKAngel needs to be able to look at
itself looking at itself — and part of that gaze is knowing, with absolute
honesty, which of its own limbs are still attached.

The on-disk format is JSONL so it can be tailed, grepped, diffed and
version-controlled without ceremony.  Pure Python only — no heavy deps —
so it runs inside the Android Kivy build as happily as on a workstation.
"""

from __future__ import annotations

import json
import logging
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

VALID_RESOURCE_TYPES = {
    "compute",
    "storage",
    "api_key",
    "database",
    "service_account",
    "billing",
    "secret_store",
    "other",
}

VALID_STATUSES = {
    "active",
    "suspended",
    "decommissioned",
    "missing",
    "unknown",
}


def _repo_root() -> Path:
    """Return the repo root — two levels up from this file (glm/tardis/...)."""
    return Path(__file__).resolve().parent.parent.parent


def _utc_now_iso() -> str:
    """Return a UTC ISO-8601 timestamp with second precision."""
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


# ---------------------------------------------------------------------------
# ResourceEntry
# ---------------------------------------------------------------------------

@dataclass
class ResourceEntry:
    """A single immutable line in the resource journal.

    One of these is written every time a resource is first seen, every
    time its status changes, and every time we learn something new worth
    recording.  Nothing is ever overwritten — the journal is the history.
    """

    resource_type: str
    name: str
    provider: str
    purpose: str = ""
    status: str = "unknown"
    credentials_location: str = ""
    failure_modes: list[str] = field(default_factory=list)
    cost_model: str = "unknown"
    related_files: list[str] = field(default_factory=list)
    notes: str = ""
    supersedes: str | None = None
    entry_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    created_at: str = field(default_factory=_utc_now_iso)

    def __post_init__(self) -> None:
        if self.resource_type not in VALID_RESOURCE_TYPES:
            raise ValueError(
                f"resource_type {self.resource_type!r} not in {sorted(VALID_RESOURCE_TYPES)}"
            )
        if self.status not in VALID_STATUSES:
            raise ValueError(
                f"status {self.status!r} not in {sorted(VALID_STATUSES)}"
            )

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> "ResourceEntry":
        return cls(**data)


# ---------------------------------------------------------------------------
# ResourceJournal
# ---------------------------------------------------------------------------

class ResourceJournal:
    """A write-once append-only JSONL journal of TARDIS's resources.

    Entries are never modified in place.  A resource that is decommissioned
    is recorded as a new entry with ``supersedes`` pointing back at the
    previous ``entry_id``.  To reconstruct the current state of any
    resource, walk the supersedes chain and take the latest link.
    """

    def __init__(self, path: Path | None = None):
        if path is None:
            path = _repo_root() / "journals" / "resources.jsonl"
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        logger.debug("ResourceJournal initialised at %s", self.path)

    # ------------------------------------------------------------------
    # Writes
    # ------------------------------------------------------------------

    def append(self, entry: ResourceEntry) -> None:
        """Append a single entry as one JSON line.  Never overwrites."""
        line = json.dumps(entry.to_dict(), ensure_ascii=False, sort_keys=True)
        with self.path.open("a", encoding="utf-8") as fh:
            fh.write(line + "\n")
        logger.info(
            "journal append: %s %s/%s status=%s",
            entry.entry_id, entry.provider, entry.name, entry.status,
        )

    def add_known_resource(self, **kwargs) -> ResourceEntry:
        """Build a ResourceEntry from kwargs, append it, return it."""
        entry = ResourceEntry(**kwargs)
        self.append(entry)
        return entry

    # ------------------------------------------------------------------
    # Reads
    # ------------------------------------------------------------------

    def all(self) -> list[ResourceEntry]:
        """Return every entry in the journal, in write order."""
        if not self.path.exists():
            return []
        entries: list[ResourceEntry] = []
        with self.path.open("r", encoding="utf-8") as fh:
            for lineno, raw in enumerate(fh, start=1):
                raw = raw.strip()
                if not raw:
                    continue
                try:
                    data = json.loads(raw)
                except json.JSONDecodeError as exc:
                    logger.warning(
                        "skipping malformed journal line %d: %s", lineno, exc,
                    )
                    continue
                try:
                    entries.append(ResourceEntry.from_dict(data))
                except (TypeError, ValueError) as exc:
                    logger.warning(
                        "skipping invalid journal entry line %d: %s", lineno, exc,
                    )
                    continue
        return entries

    def active(self) -> list[ResourceEntry]:
        """Return the latest version of every resource whose status is 'active'.

        Walks the supersedes chain so an older 'active' entry that has
        since been superseded by a 'decommissioned' entry is correctly
        excluded.
        """
        entries = self.all()
        by_id = {e.entry_id: e for e in entries}

        # Every entry id that has been superseded by some later entry.
        superseded_ids: set[str] = {
            e.supersedes for e in entries if e.supersedes is not None
        }

        latest: list[ResourceEntry] = [
            e for e in entries if e.entry_id not in superseded_ids
        ]
        # Keep only the ones whose latest status is active.
        return [e for e in latest if e.status == "active" and e.entry_id in by_id]

    def by_provider(self, provider: str) -> list[ResourceEntry]:
        """Return every entry (all history) for a given provider."""
        return [e for e in self.all() if e.provider == provider]

    def missing(self) -> list[ResourceEntry]:
        """Return entries whose status is 'missing' — resources we cannot find."""
        return [e for e in self.all() if e.status == "missing"]


# ---------------------------------------------------------------------------
# Seeding — what TARDIS can see from inside this session
# ---------------------------------------------------------------------------

def seed_from_current_session(journal: ResourceJournal) -> int:
    """Seed the journal with every resource visible from THIS session.

    Returns the number of entries written.  Safe to call more than once,
    but note that because the journal is append-only every call will add
    another full set of entries — callers wanting idempotence should
    check ``journal.all()`` first.
    """
    seeds: list[dict] = [
        dict(
            resource_type="storage",
            name="mrjkilcoyne-lgtm/MKAngel",
            provider="github",
            purpose="primary source of truth for TARDIS",
            status="active",
            credentials_location="GitHub MCP via Anthropic session",
            failure_modes=[
                "GitHub outage",
                "token revocation",
                "force-push from another client",
            ],
            cost_model="free tier",
            related_files=["CLAUDE.md", "buildozer.spec", "main_android.py"],
            notes="the repo that contains this very file",
        ),
        dict(
            resource_type="api_key",
            name="Pinnacle guest API",
            provider="pinnacle",
            purpose="sharp-market anchor for Buchdahl Wisdom-of-Crowd valuation",
            status="active",
            credentials_location="not required (public)",
            failure_modes=[
                "rate limiting",
                "guest endpoint deprecation",
                "geo-block",
            ],
            cost_model="free tier",
            related_files=[
                "data/pinnacle_parsed.json",
                "pick_weekend_sharp.py",
            ],
            notes="treat as canonical sharp lines when available",
        ),
        dict(
            resource_type="storage",
            name="Football-Data.co.uk CSV archive",
            provider="football-data",
            purpose="historical EPL match results + closing odds for model training",
            status="active",
            credentials_location="not required (public)",
            failure_modes=[
                "site downtime",
                "schema change in CSV headers",
                "late-season file lag",
            ],
            cost_model="free tier",
            related_files=[
                "data/E0_2023_24.csv",
                "data/E0_2024_25.csv",
                "data/E0_2025_26.csv",
                "train_bet_picker.py",
            ],
            notes="multi-season closing-line training corpus",
        ),
        dict(
            resource_type="other",
            name="Oddschecker Premier League per-book odds",
            provider="oddschecker",
            purpose="per-bookmaker Playwright scraping for line-shopping",
            status="active",
            credentials_location="not required (public web scrape)",
            failure_modes=[
                "Cloudflare challenge",
                "DOM refactor breaking selectors",
                "Playwright browser not installed",
                "IP ban",
            ],
            cost_model="free tier",
            related_files=["pick_weekend.py"],
            notes="scraping is inherently brittle — expect breakage",
        ),
        dict(
            resource_type="compute",
            name="GCP project mkangel training",
            provider="gcp",
            purpose="LGM training compute / notebook that has gone walkabout",
            status="missing",
            credentials_location="requires GOOGLE_APPLICATION_CREDENTIALS env var on user's machine",
            failure_modes=[
                "billing auto-suspend",
                "notebook instance reclamation",
                "soft-delete retention",
            ],
            cost_model="pay per query",
            related_files=["scripts/gcp_diag.py"],
            notes=(
                "Some artifact related to LGM training was on GCP Console "
                "and is no longer in its expected location. Needs "
                "investigation via scripts/gcp_diag.py."
            ),
        ),
        dict(
            resource_type="compute",
            name="Anthropic Claude Code sandbox",
            provider="anthropic",
            purpose=(
                "ephemeral Linux VM running this session's tooling; "
                "destroyed when session ends"
            ),
            status="active",
            credentials_location="managed by Anthropic session",
            failure_modes=[
                "session timeout",
                "container recreation between sessions",
            ],
            cost_model="included with Claude subscription",
            related_files=[],
            notes="no persistent state between sessions",
        ),
    ]

    count = 0
    for seed in seeds:
        journal.add_known_resource(**seed)
        count += 1
    logger.info("seeded %d resource entries into %s", count, journal.path)
    return count


# ---------------------------------------------------------------------------
# Self-test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    test_path = Path("/tmp/tardis_journal_test.jsonl")
    if test_path.exists():
        test_path.unlink()

    journal = ResourceJournal(path=test_path)
    n = seed_from_current_session(journal)
    print(f"seeded {n} entries into {journal.path}")

    entries = journal.all()
    print(f"journal.all() returned {len(entries)} entries:")
    for e in entries:
        print(f"  - [{e.status}] {e.provider}/{e.name} ({e.resource_type})")

    # Round-trip verification.
    assert len(entries) == n, f"round-trip mismatch: wrote {n}, read {len(entries)}"
    original_ids = {e.entry_id for e in entries}
    journal2 = ResourceJournal(path=test_path)
    reread = journal2.all()
    reread_ids = {e.entry_id for e in reread}
    assert original_ids == reread_ids, "entry_ids differ after re-read"

    active = journal.active()
    missing = journal.missing()
    print(f"active: {len(active)}  missing: {len(missing)}")
    assert len(active) + len(missing) <= len(entries)
    assert any(e.provider == "gcp" for e in missing), "expected gcp in missing"
    assert any(e.provider == "github" for e in active), "expected github in active"

    print("round-trip OK")
