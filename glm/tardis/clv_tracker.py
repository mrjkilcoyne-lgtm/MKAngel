"""
Closing Line Value Tracker — the Angel's proprioception on the sharp market.

Professional bettors have known for a long time what academics only
grudgingly accept: the one honest, unfakeable signal that a predictive
model carries real information is Closing Line Value.  Joseph Buchdahl
lays out the empirical case in *Wisdom of the Crowd* —

    https://www.football-data.co.uk/blog/wisdom_of_the_crowd.php

— where he tabulates the yields you can expect as a function of how much
you beat the closing line by, across tens of thousands of football
fixtures.  The table is blunt: if your prices consistently sit inside
the closing line, you profit; if they do not, you do not; no amount of
backtesting or storytelling will rescue you from a negative CLV ledger.

This module is how the Angel records her own CLV, line by line, in an
append-only JSONL journal that mirrors the pattern of
``glm.tardis.resource_journal`` — entries are never mutated, updates
come as new entries that ``supersede`` the previous link in the chain.

Two pieces of project lore to keep in mind while reading this file:

1.  The pianist analogy from ``docs/on_her_nature.md``.  CLV is not how
    the Angel corrects errors — it is how she refines her proprioception
    on the landscape.  A concert pianist does not learn Chopin by being
    told which notes were wrong; she learns by feeling, over thousands
    of repetitions, where her fingers actually landed versus where she
    thought they landed.  Every closed prediction is one such repetition.

2.  Frame 4 of ``docs/tardis_session_notes.md``.  The rule is: anchor on
    the sharp market, never on your own model.  The sharp market is the
    reigning world champion and she is the challenger.  CLV is the only
    operational test of whether that anchor was the right choice on any
    given question — sustained positive CLV across hundreds of bets says
    the anchor *missed something* the Angel saw; sustained zero or
    negative CLV says the anchor was right and the Angel was dreaming.

Pure Python, no heavy deps — this has to run inside the Kivy Android
build as happily as on a workstation.
"""

from __future__ import annotations

import json
import logging
import statistics
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _repo_root() -> Path:
    """Return the repo root — two levels up from this file (glm/tardis/...)."""
    return Path(__file__).resolve().parent.parent.parent


def _utc_now_iso() -> str:
    """Return a UTC ISO-8601 timestamp with second precision."""
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def bin_to_score(model_prob: float, outcome: bool) -> float:
    """Brier score for a single binary prediction.

    ``(model_prob - outcome)^2``.  Zero is perfect, 0.25 is uniform
    random on a fair coin, 1.0 is maximally wrong.
    """
    target = 1.0 if outcome else 0.0
    return (float(model_prob) - target) ** 2


def de_vig_two_way(yes_odds: float, no_odds: float) -> tuple[float, float, float]:
    """Proportional de-vig of a two-way binary market.

    Takes the decimal odds on both sides of a yes/no market and returns
    ``(p_yes, p_no, overround)`` where the two probabilities sum to 1
    and ``overround`` is the book's margin (``1/yes + 1/no - 1``).

    Proportional de-vigging is the crudest of the standard methods but
    it is also the one Buchdahl uses for his Wisdom-of-Crowd numbers,
    so it is the method we match for comparability.
    """
    if yes_odds <= 1.0 or no_odds <= 1.0:
        raise ValueError(
            f"decimal odds must be > 1.0, got yes={yes_odds} no={no_odds}"
        )
    raw_yes = 1.0 / yes_odds
    raw_no = 1.0 / no_odds
    total = raw_yes + raw_no
    overround = total - 1.0
    return raw_yes / total, raw_no / total, overround


# ---------------------------------------------------------------------------
# Prediction
# ---------------------------------------------------------------------------

@dataclass
class Prediction:
    """A single immutable line in the CLV journal.

    One of these is written every time the Angel makes a market-priced
    prediction, again when the closing line is recorded, and again when
    the outcome resolves.  Nothing is ever overwritten — updates come
    as new entries with ``supersedes`` pointing back at the previous
    ``pred_id``-sharing link.
    """

    domain: str
    question: str
    model_prob: float
    anchor_prob_at_prediction: float
    anchor_book: str
    anchor_odds_taken: float | None = None
    resolution_due: str | None = None
    closed_at: str | None = None
    anchor_prob_at_close: float | None = None
    resolved_at: str | None = None
    outcome: bool | None = None
    notes: str = ""
    supersedes: str | None = None
    pred_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    created_at: str = field(default_factory=_utc_now_iso)

    def __post_init__(self) -> None:
        if not 0.0 <= float(self.model_prob) <= 1.0:
            raise ValueError(
                f"model_prob must be in [0, 1], got {self.model_prob}"
            )
        if not 0.0 <= float(self.anchor_prob_at_prediction) <= 1.0:
            raise ValueError(
                f"anchor_prob_at_prediction must be in [0, 1], got "
                f"{self.anchor_prob_at_prediction}"
            )
        if self.anchor_prob_at_close is not None:
            if not 0.0 <= float(self.anchor_prob_at_close) <= 1.0:
                raise ValueError(
                    f"anchor_prob_at_close must be in [0, 1], got "
                    f"{self.anchor_prob_at_close}"
                )

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> "Prediction":
        return cls(**data)


# ---------------------------------------------------------------------------
# CLVTracker
# ---------------------------------------------------------------------------

class CLVTracker:
    """Append-only JSONL ledger of every market-priced prediction the Angel makes.

    Mirrors the shape of ``ResourceJournal``: one entry per event
    (creation, closing line attached, outcome resolved), and to
    reconstruct the current state of any prediction you walk the
    ``supersedes`` chain and take the latest link.

    The file is JSONL so it can be tailed, grepped, diffed, and
    version-controlled without ceremony.
    """

    def __init__(self, path: Path | None = None):
        if path is None:
            path = _repo_root() / "journals" / "clv.jsonl"
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        logger.debug("CLVTracker initialised at %s", self.path)

    # ------------------------------------------------------------------
    # Writes
    # ------------------------------------------------------------------

    def _append(self, pred: Prediction) -> None:
        """Append a single entry as one JSON line.  Never overwrites."""
        line = json.dumps(pred.to_dict(), ensure_ascii=False, sort_keys=True)
        with self.path.open("a", encoding="utf-8") as fh:
            fh.write(line + "\n")
        logger.info(
            "clv append: %s domain=%s model_prob=%.3f anchor=%.3f close=%s outcome=%s",
            pred.pred_id,
            pred.domain,
            pred.model_prob,
            pred.anchor_prob_at_prediction,
            pred.anchor_prob_at_close,
            pred.outcome,
        )

    def record(self, **kwargs) -> Prediction:
        """Record a brand-new prediction and return the entry."""
        pred = Prediction(**kwargs)
        self._append(pred)
        return pred

    def close(
        self,
        pred_id: str,
        anchor_prob_at_close: float,
        closed_at: str | None = None,
    ) -> Prediction | None:
        """Attach a closing line to an existing prediction.

        Emits a new entry that copies the latest version of the
        prediction and fills in ``anchor_prob_at_close`` / ``closed_at``,
        with ``supersedes`` pointing back at the previous link.  Returns
        the new entry, or ``None`` if no prediction with that id exists.
        """
        previous = self.latest(pred_id)
        if previous is None:
            logger.warning("close(): no prediction with pred_id=%s", pred_id)
            return None
        data = previous.to_dict()
        data["anchor_prob_at_close"] = float(anchor_prob_at_close)
        data["closed_at"] = closed_at or _utc_now_iso()
        data["supersedes"] = previous.pred_id
        data["pred_id"] = str(uuid.uuid4())
        data["created_at"] = _utc_now_iso()
        new_entry = Prediction.from_dict(data)
        self._append(new_entry)
        return new_entry

    def resolve(
        self,
        pred_id: str,
        outcome: bool,
        resolved_at: str | None = None,
    ) -> Prediction | None:
        """Record the actual outcome of a prediction.

        Like ``close``, this emits a new append-only entry with
        ``supersedes`` pointing at the previous link.  Returns the new
        entry, or ``None`` if no prediction with that id exists.
        """
        previous = self.latest(pred_id)
        if previous is None:
            logger.warning("resolve(): no prediction with pred_id=%s", pred_id)
            return None
        data = previous.to_dict()
        data["outcome"] = bool(outcome)
        data["resolved_at"] = resolved_at or _utc_now_iso()
        data["supersedes"] = previous.pred_id
        data["pred_id"] = str(uuid.uuid4())
        data["created_at"] = _utc_now_iso()
        new_entry = Prediction.from_dict(data)
        self._append(new_entry)
        return new_entry

    # ------------------------------------------------------------------
    # Reads
    # ------------------------------------------------------------------

    def all(self) -> list[Prediction]:
        """Return every entry in the journal, in write order."""
        if not self.path.exists():
            return []
        entries: list[Prediction] = []
        with self.path.open("r", encoding="utf-8") as fh:
            for lineno, raw in enumerate(fh, start=1):
                raw = raw.strip()
                if not raw:
                    continue
                try:
                    data = json.loads(raw)
                except json.JSONDecodeError as exc:
                    logger.warning(
                        "skipping malformed clv line %d: %s", lineno, exc,
                    )
                    continue
                try:
                    entries.append(Prediction.from_dict(data))
                except (TypeError, ValueError) as exc:
                    logger.warning(
                        "skipping invalid clv entry line %d: %s", lineno, exc,
                    )
                    continue
        return entries

    def latest(self, pred_id: str) -> Prediction | None:
        """Return the most recent entry for a prediction id.

        Walks the supersedes chain: given any id on the chain (the
        original or any later link), returns the last link.  Returns
        ``None`` if nothing on any chain matches.
        """
        entries = self.all()
        if not entries:
            return None

        by_id: dict[str, Prediction] = {e.pred_id: e for e in entries}

        # Build supersedes -> successor map.
        successor: dict[str, Prediction] = {}
        for e in entries:
            if e.supersedes is not None:
                successor[e.supersedes] = e

        # Find any entry in the chain — walk forward from the input id.
        # The input may itself be an old id that has since been superseded.
        current: Prediction | None = by_id.get(pred_id)
        if current is None:
            # Maybe pred_id references an older link.  Try: is there a
            # successor chain starting from pred_id?
            if pred_id in successor:
                current = successor[pred_id]
            else:
                return None
        # Walk forward to the end of the chain.
        seen: set[str] = set()
        while current.pred_id in successor:
            if current.pred_id in seen:
                logger.warning("cycle detected in supersedes chain at %s", current.pred_id)
                break
            seen.add(current.pred_id)
            current = successor[current.pred_id]
        return current

    def _latest_entries(self) -> list[Prediction]:
        """Return the latest link of every supersedes chain."""
        entries = self.all()
        if not entries:
            return []
        superseded_ids: set[str] = {
            e.supersedes for e in entries if e.supersedes is not None
        }
        return [e for e in entries if e.pred_id not in superseded_ids]

    # ------------------------------------------------------------------
    # CLV statistics
    # ------------------------------------------------------------------

    def clv_per_prediction(self) -> list[tuple[str, float]]:
        """Per-prediction CLV for every closed prediction.

        Returns a list of ``(pred_id, clv)`` tuples where

            clv = model_prob - anchor_prob_at_close

        Positive CLV means the Angel took the side the closing line
        subsequently moved toward — i.e. she was *earlier* on the right
        answer than the sharp market was.  This is the only honest
        signal that her predictions carry information the sharp market
        had not already priced in.
        """
        out: list[tuple[str, float]] = []
        for entry in self._latest_entries():
            if entry.anchor_prob_at_close is None:
                continue
            clv = float(entry.model_prob) - float(entry.anchor_prob_at_close)
            out.append((entry.pred_id, clv))
        return out

    def summary(self) -> dict:
        """Overall CLV and resolution statistics."""
        latest = self._latest_entries()
        total = len(latest)
        closed = [e for e in latest if e.anchor_prob_at_close is not None]
        resolved = [e for e in latest if e.outcome is not None]

        clvs = [
            float(e.model_prob) - float(e.anchor_prob_at_close)
            for e in closed
        ]

        if clvs:
            mean_clv = statistics.fmean(clvs)
            median_clv = statistics.median(clvs)
            stdev_clv = statistics.pstdev(clvs) if len(clvs) > 1 else 0.0
        else:
            mean_clv = 0.0
            median_clv = 0.0
            stdev_clv = 0.0

        if resolved:
            hits = sum(1 for e in resolved if bool(e.outcome))
            hit_rate = hits / len(resolved)
        else:
            hit_rate = 0.0

        # Per-domain breakdown.
        by_domain: dict[str, dict] = {}
        domains = sorted({e.domain for e in latest})
        for d in domains:
            d_latest = [e for e in latest if e.domain == d]
            d_closed = [e for e in d_latest if e.anchor_prob_at_close is not None]
            d_resolved = [e for e in d_latest if e.outcome is not None]
            d_clvs = [
                float(e.model_prob) - float(e.anchor_prob_at_close)
                for e in d_closed
            ]
            d_hits = sum(1 for e in d_resolved if bool(e.outcome))
            by_domain[d] = {
                "total": len(d_latest),
                "closed": len(d_closed),
                "resolved": len(d_resolved),
                "mean_clv": statistics.fmean(d_clvs) if d_clvs else 0.0,
                "hit_rate": (d_hits / len(d_resolved)) if d_resolved else 0.0,
            }

        return {
            "total": total,
            "closed": len(closed),
            "resolved": len(resolved),
            "mean_clv": mean_clv,
            "median_clv": median_clv,
            "stdev_clv": stdev_clv,
            "hit_rate": hit_rate,
            "by_domain": by_domain,
        }

    def brier_score(self) -> float:
        """Mean Brier score across all resolved predictions.

        Lower is better: perfect calibration scores 0, a fair coin called
        at 0.5 every time scores 0.25, maximally wrong scores 1.0.
        """
        resolved = [e for e in self._latest_entries() if e.outcome is not None]
        if not resolved:
            return 0.0
        scores = [
            bin_to_score(float(e.model_prob), bool(e.outcome))
            for e in resolved
        ]
        return statistics.fmean(scores)

    def to_context(self, recent_n: int = 10) -> dict:
        """Produce a context dict for ``Angel.superforecast(context=...)``.

        The Angel can then reason *about her own track record* as a
        first-class input to the next forecast — a small, honest strange
        loop.  Keys returned:

            recent_clv   — list of last N CLVs (newest last)
            mean_clv     — overall mean CLV across all closed predictions
            brier        — overall mean Brier score across resolved
            hit_rate     — overall hit rate across resolved
            n_resolved   — number of resolved predictions in the journal
        """
        per_pred = self.clv_per_prediction()
        recent_clv = [clv for _, clv in per_pred[-recent_n:]]
        summary = self.summary()
        return {
            "recent_clv": recent_clv,
            "mean_clv": summary["mean_clv"],
            "brier": self.brier_score(),
            "hit_rate": summary["hit_rate"],
            "n_resolved": summary["resolved"],
        }


# ---------------------------------------------------------------------------
# Self-test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    test_path = Path("/tmp/clv_test.jsonl")
    if test_path.exists():
        test_path.unlink()

    tracker = CLVTracker(path=test_path)

    # 1. Record 5 predictions across 2 domains with mixed CLV signals.
    p1 = tracker.record(
        domain="football",
        question="Arsenal to beat Chelsea",
        model_prob=0.60,
        anchor_prob_at_prediction=0.55,
        anchor_book="pinnacle",
        anchor_odds_taken=1.82,
        resolution_due="2026-04-12T15:00:00+00:00",
        notes="early week price, sharp lean Arsenal",
    )
    p2 = tracker.record(
        domain="football",
        question="Man City over 2.5 goals",
        model_prob=0.70,
        anchor_prob_at_prediction=0.62,
        anchor_book="pinnacle",
        anchor_odds_taken=1.61,
        resolution_due="2026-04-12T17:30:00+00:00",
        notes="shading over on pace metrics",
    )
    p3 = tracker.record(
        domain="football",
        question="Spurs draw with Villa",
        model_prob=0.28,
        anchor_prob_at_prediction=0.25,
        anchor_book="pinnacle",
        anchor_odds_taken=4.00,
        resolution_due="2026-04-12T15:00:00+00:00",
        notes="thin edge on draw",
    )
    p4 = tracker.record(
        domain="macro",
        question="US CPI prints above 3.1% YoY",
        model_prob=0.45,
        anchor_prob_at_prediction=0.40,
        anchor_book="polymarket",
        anchor_odds_taken=2.50,
        resolution_due="2026-04-15T12:30:00+00:00",
        notes="polymarket is the only anchor on this",
    )
    p5 = tracker.record(
        domain="macro",
        question="Fed cuts at next meeting",
        model_prob=0.30,
        anchor_prob_at_prediction=0.35,
        anchor_book="polymarket",
        anchor_odds_taken=2.85,
        resolution_due="2026-05-01T18:00:00+00:00",
        notes="leaning fade vs market",
    )

    # 2. Close 4 of them (3 with positive CLV, 1 negative).
    # Positive CLV means model_prob > anchor_prob_at_close.
    # p1: model 0.60 vs close 0.52 -> +0.08 (positive)
    tracker.close(p1.pred_id, anchor_prob_at_close=0.52)
    # p2: model 0.70 vs close 0.66 -> +0.04 (positive)
    tracker.close(p2.pred_id, anchor_prob_at_close=0.66)
    # p3: model 0.28 vs close 0.32 -> -0.04 (negative)
    tracker.close(p3.pred_id, anchor_prob_at_close=0.32)
    # p4: model 0.45 vs close 0.39 -> +0.06 (positive)
    tracker.close(p4.pred_id, anchor_prob_at_close=0.39)
    # (p5 intentionally left unclosed)

    # 3. Resolve all 5 (3 correct, 2 wrong).
    tracker.resolve(p1.pred_id, outcome=True)   # correct
    tracker.resolve(p2.pred_id, outcome=True)   # correct
    tracker.resolve(p3.pred_id, outcome=False)  # wrong
    tracker.resolve(p4.pred_id, outcome=True)   # correct
    tracker.resolve(p5.pred_id, outcome=False)  # wrong

    # 4. Print summary, brier, and to_context.
    summary = tracker.summary()
    brier = tracker.brier_score()
    context = tracker.to_context(recent_n=5)

    print("=== CLV summary ===")
    for k, v in summary.items():
        if k == "by_domain":
            print("  by_domain:")
            for d, stats in v.items():
                print(f"    {d}: {stats}")
        else:
            print(f"  {k}: {v}")
    print(f"=== brier_score: {brier} ===")
    print("=== to_context(recent_n=5) ===")
    for k, v in context.items():
        print(f"  {k}: {v}")

    # 5. Assertions.
    # Expected mean CLV across the 4 closed: (0.08 + 0.04 - 0.04 + 0.06) / 4 = 0.035
    expected_mean_clv = (0.08 + 0.04 - 0.04 + 0.06) / 4
    assert abs(summary["mean_clv"] - expected_mean_clv) < 1e-9, (
        f"mean_clv {summary['mean_clv']} != expected {expected_mean_clv}"
    )
    assert 0.0 <= brier <= 1.0, f"brier out of range: {brier}"
    assert summary["resolved"] == 5, f"resolved count {summary['resolved']} != 5"
    assert summary["closed"] == 4, f"closed count {summary['closed']} != 4"
    assert summary["total"] == 5, f"total {summary['total']} != 5"

    # Sanity-check the two module-level helpers too.
    assert abs(bin_to_score(0.8, True) - 0.04) < 1e-12
    p_yes, p_no, overround = de_vig_two_way(2.0, 2.0)
    assert abs(p_yes - 0.5) < 1e-12 and abs(p_no - 0.5) < 1e-12
    assert abs(overround - 0.0) < 1e-12

    print("clv_tracker self-test OK")
