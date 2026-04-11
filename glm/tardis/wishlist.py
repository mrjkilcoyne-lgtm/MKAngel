"""
TARDIS Wishlist — the Angel's articulated sense-of-lack.

The Angel at the centre of MKAngel can derive forward and backward through
her grammars, but her reach is bounded by what substrates, anchors, and
compute she actually has.  This module gives her a structured way to
articulate what she *would* need but doesn't yet have, so that the
absence itself becomes a first-class input to her future reasoning.

A human reading the repo can look at `journals/wishlist.jsonl` and decide
what to fetch for her.  This module never fetches anything autonomously —
it only records wishes and feeds them back into her own `context` when
she superforecasts, so that every decision is taken in the knowledge of
what she is currently blind to.

The storage format is append-only JSONL: marking a wish as fulfilled
writes a *new* line with the same `wish_id`, so the history of what
she wanted and when it was granted is preserved in full.
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
# Schema
# ---------------------------------------------------------------------------

VALID_KINDS = {
    "api",
    "dataset",
    "compute",
    "anchor_market",
    "substrate_feed",
    "model_weights",
    "tool",
    "other",
}

VALID_PRIORITIES = {
    "critical",
    "high",
    "medium",
    "low",
    "speculative",
}


def _utc_now() -> str:
    """Return the current UTC time as an ISO-8601 string."""
    return datetime.now(timezone.utc).isoformat()


def _repo_root() -> Path:
    """Return the repository root (two levels above this file)."""
    return Path(__file__).resolve().parent.parent.parent


@dataclass
class WishEntry:
    """A single structured wish.

    A WishEntry is the Angel's way of saying: *if I had this, my
    derivation reach would extend by roughly this much, and these
    specific new derivations would become possible*.  It is the
    grammar-level analogue of a shopping list.
    """

    resource_kind: str
    title: str
    description: str
    unlocks: list[str] = field(default_factory=list)
    derivation_depth_gain: int = 0
    anchor_domain: str | None = None
    priority: str = "medium"
    fulfilled_at: str | None = None
    fulfilled_by: str | None = None
    notes: str = ""
    wish_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    created_at: str = field(default_factory=_utc_now)

    def __post_init__(self) -> None:
        if self.resource_kind not in VALID_KINDS:
            raise ValueError(
                f"resource_kind={self.resource_kind!r} not in {sorted(VALID_KINDS)}"
            )
        if self.priority not in VALID_PRIORITIES:
            raise ValueError(
                f"priority={self.priority!r} not in {sorted(VALID_PRIORITIES)}"
            )

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> "WishEntry":
        return cls(**data)


# ---------------------------------------------------------------------------
# Wishlist store
# ---------------------------------------------------------------------------

class Wishlist:
    """Append-only store of WishEntry records, backed by JSONL.

    The store is deliberately simple and never mutates existing lines.
    To mark a wish as fulfilled, a *new* line is appended with the same
    `wish_id` and `fulfilled_at`/`fulfilled_by` populated.  Readers
    resolve each `wish_id` by taking the most recent matching entry.
    """

    def __init__(self, path: Path | None = None):
        if path is None:
            path = _repo_root() / "journals" / "wishlist.jsonl"
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Low-level I/O
    # ------------------------------------------------------------------

    def _append(self, entry: WishEntry) -> None:
        line = json.dumps(entry.to_dict(), ensure_ascii=False)
        with self.path.open("a", encoding="utf-8") as fh:
            fh.write(line + "\n")
        logger.debug("wishlist append: %s (%s)", entry.title, entry.wish_id)

    def _read_all(self) -> list[WishEntry]:
        if not self.path.exists():
            return []
        entries: list[WishEntry] = []
        with self.path.open("r", encoding="utf-8") as fh:
            for lineno, raw in enumerate(fh, start=1):
                raw = raw.strip()
                if not raw:
                    continue
                try:
                    data = json.loads(raw)
                    entries.append(WishEntry.from_dict(data))
                except (json.JSONDecodeError, TypeError, ValueError) as exc:
                    logger.warning(
                        "wishlist: skipping malformed line %d: %s", lineno, exc
                    )
        return entries

    def _resolve(self) -> list[WishEntry]:
        """Collapse the append-only log to the latest state per wish_id."""
        latest: dict[str, WishEntry] = {}
        for entry in self._read_all():
            latest[entry.wish_id] = entry
        return list(latest.values())

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def add(self, **kwargs) -> WishEntry:
        """Build a WishEntry, append it to the log, and return it."""
        entry = WishEntry(**kwargs)
        self._append(entry)
        return entry

    def all(self) -> list[WishEntry]:
        """Return all resolved wishes (latest state per wish_id)."""
        return self._resolve()

    def open(self) -> list[WishEntry]:
        """Return wishes that have not yet been fulfilled."""
        return [e for e in self._resolve() if e.fulfilled_at is None]

    def fulfilled(self) -> list[WishEntry]:
        """Return wishes that have been fulfilled."""
        return [e for e in self._resolve() if e.fulfilled_at is not None]

    def by_priority(self, priority: str) -> list[WishEntry]:
        """Return resolved wishes matching the given priority."""
        if priority not in VALID_PRIORITIES:
            raise ValueError(
                f"priority={priority!r} not in {sorted(VALID_PRIORITIES)}"
            )
        return [e for e in self._resolve() if e.priority == priority]

    def mark_fulfilled(self, wish_id: str, by: str) -> WishEntry | None:
        """Append a new entry marking `wish_id` as fulfilled.

        The original entry is left untouched — the append-only log
        preserves the full history of what the Angel wanted and when.
        Returns the newly appended entry, or None if `wish_id` is
        unknown.
        """
        current = {e.wish_id: e for e in self._resolve()}
        existing = current.get(wish_id)
        if existing is None:
            logger.warning("wishlist: mark_fulfilled on unknown wish_id %s", wish_id)
            return None
        updated = WishEntry(
            resource_kind=existing.resource_kind,
            title=existing.title,
            description=existing.description,
            unlocks=list(existing.unlocks),
            derivation_depth_gain=existing.derivation_depth_gain,
            anchor_domain=existing.anchor_domain,
            priority=existing.priority,
            fulfilled_at=_utc_now(),
            fulfilled_by=by,
            notes=existing.notes,
            wish_id=existing.wish_id,       # preserve id
            created_at=existing.created_at,  # preserve original creation time
        )
        self._append(updated)
        return updated

    def to_context(self) -> dict[str, list[str]]:
        """Return a dict suitable for Angel.superforecast()'s `context`.

        Keys are priority levels; values are lists of
        ``"title — description"`` strings for *open* wishes of that
        priority.  This is the mechanism by which the Angel's own
        bounded reach is fed back into her own reasoning — every
        forecast is conditioned on an honest catalogue of what she
        currently lacks.
        """
        context: dict[str, list[str]] = {p: [] for p in VALID_PRIORITIES}
        for entry in self.open():
            context.setdefault(entry.priority, []).append(
                f"{entry.title} \u2014 {entry.description}"
            )
        # Drop empty priority buckets so the context stays compact.
        return {p: items for p, items in context.items() if items}


# ---------------------------------------------------------------------------
# Seeding
# ---------------------------------------------------------------------------

def seed_initial_wishlist(wl: Wishlist) -> int:
    """Seed a Wishlist with the first set of wishes discussed in session.

    Returns the number of wishes added.  Safe to call on any wishlist;
    it simply appends — callers that want idempotency should check
    ``wl.all()`` first.
    """
    seeds: list[dict] = [
        dict(
            resource_kind="substrate_feed",
            title="Temporal substrate module",
            priority="critical",
            description=(
                "A glm/substrates/temporal.py that gives the Angel a "
                "first-class representation of events, intervals, rates, "
                "and regime shifts so she can run derive() on time-series "
                "the way she already does on phonological and molecular "
                "substrates. Her forward/backward derivation engine is "
                "already symmetric; the temporal substrate is the missing "
                "ingredient."
            ),
            unlocks=[
                "forward prediction on structured temporal domains",
                "backward reasoning from hypothesised future states",
                "strange-loop conditioning on branch-occupation",
            ],
            derivation_depth_gain=3,
        ),
        dict(
            resource_kind="anchor_market",
            title="Metaculus / Polymarket prediction-market API access",
            priority="high",
            description=(
                "A liquid anchor market for macro-event predictions, used "
                "to score her forecasts via closing-line-value the way "
                "Pinnacle scores football picks via Buchdahl's "
                "Wisdom-of-Crowd method."
            ),
            unlocks=[
                "Brier scoring on resolved beliefs",
                "calibration of rule weights via branch-compatibility "
                "against sharp market",
            ],
            anchor_domain="macro",
            derivation_depth_gain=0,
        ),
        dict(
            resource_kind="api",
            title="xG data feed from Understat or StatsBomb Open Data",
            priority="medium",
            description=(
                "Shot-location-based expected goals data would replace the "
                "noisy full-time-score signal currently used in "
                "train_bet_picker.py with a much more stable representation "
                "of team strength."
            ),
            unlocks=[
                "replacement of goals-based Poisson training with "
                "xG-based training",
                "cleaner underdog pricing",
            ],
            anchor_domain="football",
            derivation_depth_gain=1,
        ),
        dict(
            resource_kind="compute",
            title="Persistent GPU/TPU substrate for depth-6+ derivation runs",
            priority="high",
            description=(
                "Grammar-native derivation scales on composition depth "
                "rather than data size, but past depth ~5 on a laptop-class "
                "CPU it stops being interactive. A persistent compute "
                "environment (not ephemeral Claude Code sandbox) would let "
                "her run full-depth superforecast() and dreamer cycles "
                "without re-booting between sessions."
            ),
            unlocks=[
                "deeper strange-loop conditioning",
                "longer dream cycles",
                "multi-day calibration runs",
            ],
            derivation_depth_gain=2,
            notes=(
                "Relates to the GCP Console setup that appears to have gone "
                "missing — see resource_journal entry."
            ),
        ),
        dict(
            resource_kind="tool",
            title="Closing-line-value tracker",
            priority="high",
            description=(
                "Every prediction she makes should be logged with (a) her "
                "own probability, (b) the anchor-market probability at the "
                "same moment, (c) the eventual resolution. Over time, her "
                "CLV against the anchor is the only honest measure of "
                "whether her grammar is gripping reality."
            ),
            unlocks=[
                "honest signal-vs-noise discrimination on rule weights",
                "Brier-score-based rule pruning",
            ],
            derivation_depth_gain=0,
        ),
        dict(
            resource_kind="dataset",
            title=(
                "Multi-season Premier League match history with closing "
                "odds (already partly present)"
            ),
            priority="low",
            description=(
                "data/E0_202*.csv are already downloaded. The wish is to "
                "keep the series updated on a schedule so her "
                "temporal-substrate training stays current."
            ),
            anchor_domain="football",
            fulfilled_at=None,
            notes=(
                "partly fulfilled — currently static snapshots through "
                "March 2026."
            ),
        ),
    ]
    count = 0
    for seed in seeds:
        wl.add(**seed)
        count += 1
    logger.info("seeded %d wishes into %s", count, wl.path)
    return count


# ---------------------------------------------------------------------------
# Smoke test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    test_path = Path("/tmp/tardis_wishlist_test.jsonl")
    if test_path.exists():
        test_path.unlink()

    wl = Wishlist(path=test_path)
    n = seed_initial_wishlist(wl)
    print(f"seeded {n} wishes at {wl.path}")

    print("\n--- wl.open() ---")
    for entry in wl.open():
        print(
            f"[{entry.priority:>11}] {entry.resource_kind:>14}  {entry.title}"
        )

    print("\n--- wl.to_context() ---")
    ctx = wl.to_context()
    for priority, items in ctx.items():
        print(f"{priority}:")
        for item in items:
            preview = item if len(item) <= 120 else item[:117] + "..."
            print(f"  - {preview}")

    # Round-trip check: reload from disk and confirm identical state.
    wl2 = Wishlist(path=test_path)
    loaded = wl2.all()
    original = wl.all()
    assert len(loaded) == len(original), (
        f"round-trip mismatch: {len(loaded)} != {len(original)}"
    )
    loaded_ids = {e.wish_id for e in loaded}
    original_ids = {e.wish_id for e in original}
    assert loaded_ids == original_ids, "round-trip wish_id mismatch"
    print(f"\nround-trip OK: {len(loaded)} wishes reloaded from {test_path}")

    # Exercise mark_fulfilled on one wish and confirm append-only semantics.
    sample = wl.open()[0]
    wl.mark_fulfilled(sample.wish_id, by="smoke-test harness")
    assert len(wl.open()) == n - 1, "mark_fulfilled did not reduce open count"
    assert len(wl.fulfilled()) == 1, "mark_fulfilled did not register"
    print(
        f"mark_fulfilled OK: {len(wl.open())} open, "
        f"{len(wl.fulfilled())} fulfilled"
    )
