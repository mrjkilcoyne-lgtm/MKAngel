#!/usr/bin/env python3
"""Train the MKAngel football bet picker for UK leagues.

Usage
-----

Default -- train on the built-in demo season (simulated 2023-24-style
20-team Premier League, 380 matches) and print sample predictions::

    python train_bet_picker.py

Train on your own results CSV (columns: ``home_team,away_team,
home_goals,away_goals``, optional ``season,weight``)::

    python train_bet_picker.py --csv data/epl_2023_24.csv \\
        --out checkpoints/bet_picker.json --epochs 600

Load a saved model and price a set of upcoming fixtures::

    python train_bet_picker.py --load checkpoints/bet_picker.json --predict

The demo season is synthetic: each match is sampled from a bivariate
Poisson using plausible Premier League 2023-24 team priors. This lets
you verify the training loop end-to-end without any network access or
real match data. For real betting, swap in a CSV of completed matches
-- Football-Data.co.uk publishes free historical UK results suitable
for this purpose.

IMPORTANT: Gambling is risky. The UK national problem gambling helpline
is 0808 8020 133 (GamCare); see also gambleaware.org. Never stake more
than you can afford to lose.
"""

from __future__ import annotations

import argparse
import csv
import math
import os
import random
import sys
from typing import Dict, List, Optional, Tuple

# Allow running this script directly from the repo root.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from app.bet_picker import (  # noqa: E402
    BetRecommendation,
    Fixture,
    Match,
    PoissonBetPicker,
    format_bet,
    implied_probabilities,
)


# =======================================================================
# Built-in demo priors: Premier League 2023-24 team strengths
# =======================================================================
#
# These are PRIORS used only to synthesise a demonstration season.
# Numbers are rough, perception-based ratings on a log-goals scale:
#   attack > 0 means "scores more than league average",
#   defence < 0 means "concedes less than league average".
# They are NOT fitted to real match data. For real betting, train on
# actual results via --csv.

PL_2023_24_PRIORS: List[Tuple[str, float, float]] = [
    # team                     attack   defence  (lower defence = tighter)
    ("Manchester City",          0.55,  -0.35),
    ("Arsenal",                  0.45,  -0.40),
    ("Liverpool",                0.50,  -0.20),
    ("Aston Villa",              0.25,  -0.10),
    ("Tottenham",                0.30,   0.00),
    ("Chelsea",                  0.15,  -0.05),
    ("Newcastle",                0.20,  -0.05),
    ("Manchester United",        0.05,   0.00),
    ("West Ham",                 0.00,   0.05),
    ("Brighton",                 0.05,   0.05),
    ("Bournemouth",             -0.05,   0.10),
    ("Fulham",                  -0.05,   0.05),
    ("Wolves",                  -0.10,   0.10),
    ("Crystal Palace",          -0.10,   0.05),
    ("Everton",                 -0.20,  -0.05),
    ("Brentford",               -0.05,   0.15),
    ("Nottingham Forest",       -0.15,   0.15),
    ("Luton Town",              -0.25,   0.30),
    ("Burnley",                 -0.30,   0.25),
    ("Sheffield United",        -0.40,   0.40),
]

DEMO_BASE_RATE = math.log(1.35)   # league avg goals/team/match ≈ 1.35
DEMO_HOME_ADVANTAGE = 0.27        # ~30% boost to home expected goals


# =======================================================================
# Demo season generation
# =======================================================================

def generate_demo_season(
    priors: List[Tuple[str, float, float]] = PL_2023_24_PRIORS,
    base_rate: float = DEMO_BASE_RATE,
    home_advantage: float = DEMO_HOME_ADVANTAGE,
    seed: int = 2024,
) -> List[Match]:
    """Simulate a full double round-robin season.

    Each team plays every other team home and away (380 matches for
    20 teams). Goals are sampled from independent Poisson distributions
    using the prior strengths.
    """
    rng = random.Random(seed)
    teams = [p[0] for p in priors]
    attack = {p[0]: p[1] for p in priors}
    defence = {p[0]: p[2] for p in priors}

    matches: List[Match] = []
    for home in teams:
        for away in teams:
            if home == away:
                continue
            lh = math.exp(
                base_rate + attack[home] + defence[away] + home_advantage
            )
            la = math.exp(base_rate + attack[away] + defence[home])
            hg = _sample_poisson(lh, rng)
            ag = _sample_poisson(la, rng)
            matches.append(Match(
                home_team=home,
                away_team=away,
                home_goals=hg,
                away_goals=ag,
                season="demo-2023-24",
                weight=1.0,
            ))
    rng.shuffle(matches)
    return matches


def _sample_poisson(lam: float, rng: random.Random) -> int:
    """Knuth's rejection sampler for small Poisson means."""
    if lam <= 0:
        return 0
    L = math.exp(-lam)
    k = 0
    p = 1.0
    while True:
        k += 1
        p *= rng.random()
        if p <= L:
            return k - 1
        if k > 30:
            return k - 1  # very unlikely safeguard


# =======================================================================
# CSV loading
# =======================================================================

def load_matches_csv(path: str) -> List[Match]:
    """Load matches from a CSV file.

    Expected columns (case-insensitive, in any order):
        home_team, away_team, home_goals, away_goals
    Optional columns: season, weight
    """
    matches: List[Match] = []
    with open(path, "r", newline="") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None:
            raise ValueError(f"CSV {path} has no header row")
        cols = {c.lower(): c for c in reader.fieldnames}
        for req in ("home_team", "away_team", "home_goals", "away_goals"):
            if req not in cols:
                raise ValueError(f"CSV {path} missing required column: {req}")
        for row in reader:
            try:
                matches.append(Match(
                    home_team=row[cols["home_team"]].strip(),
                    away_team=row[cols["away_team"]].strip(),
                    home_goals=int(row[cols["home_goals"]]),
                    away_goals=int(row[cols["away_goals"]]),
                    season=row.get(cols.get("season", ""), "").strip() if "season" in cols else "",
                    weight=float(row[cols["weight"]]) if "weight" in cols else 1.0,
                ))
            except (KeyError, ValueError) as e:
                print(f"  skipping malformed row: {e}", file=sys.stderr)
    return matches


# =======================================================================
# Evaluation helpers
# =======================================================================

def evaluate_accuracy(picker: PoissonBetPicker, matches: List[Match]) -> Dict[str, float]:
    """Score the model on held-out matches.

    Returns:
        result_accuracy: fraction of matches where the argmax 1X2
            prediction matched the observed result.
        log_loss: mean negative log-probability of the observed result
            under the model's 1X2 distribution (lower is better).
        rps: mean Rank Probability Score over 1X2 (lower is better;
            standard football forecasting metric).
    """
    if not matches:
        return {"result_accuracy": 0.0, "log_loss": 0.0, "rps": 0.0}

    n = 0
    hits = 0
    log_loss = 0.0
    rps = 0.0

    for m in matches:
        try:
            probs = picker.probabilities(m.home_team, m.away_team)
        except KeyError:
            continue
        n += 1

        if m.home_goals > m.away_goals:
            true = "H"
            p_true = probs.home
        elif m.home_goals == m.away_goals:
            true = "D"
            p_true = probs.draw
        else:
            true = "A"
            p_true = probs.away

        # Top pick
        ranked = max(
            [("H", probs.home), ("D", probs.draw), ("A", probs.away)],
            key=lambda kv: kv[1],
        )
        if ranked[0] == true:
            hits += 1

        log_loss += -math.log(max(p_true, 1e-9))

        # Rank Probability Score (Epstein 1969) for ordered 1X2:
        # order is H, D, A
        p_cum = [probs.home, probs.home + probs.draw]
        if true == "H":
            o_cum = [1.0, 1.0]
        elif true == "D":
            o_cum = [0.0, 1.0]
        else:
            o_cum = [0.0, 0.0]
        rps += 0.5 * sum((p_cum[i] - o_cum[i]) ** 2 for i in range(2))

    if n == 0:
        return {"result_accuracy": 0.0, "log_loss": 0.0, "rps": 0.0}

    return {
        "result_accuracy": hits / n,
        "log_loss": log_loss / n,
        "rps": rps / n,
    }


def demo_fixtures() -> List[Fixture]:
    """A small set of hypothetical fixtures with sample bookmaker odds."""
    return [
        Fixture(
            home_team="Manchester City", away_team="Sheffield United",
            kickoff="Sat 15:00",
            odds_home=1.20, odds_draw=8.00, odds_away=15.00,
            odds_btts_yes=2.30, odds_btts_no=1.55,
            odds_over_2_5=1.40, odds_under_2_5=2.85,
        ),
        Fixture(
            home_team="Arsenal", away_team="Liverpool",
            kickoff="Sun 16:30",
            odds_home=2.55, odds_draw=3.60, odds_away=2.70,
            odds_btts_yes=1.55, odds_btts_no=2.40,
            odds_over_2_5=1.70, odds_under_2_5=2.15,
        ),
        Fixture(
            home_team="Burnley", away_team="Luton Town",
            kickoff="Sat 15:00",
            odds_home=2.10, odds_draw=3.40, odds_away=3.40,
            odds_btts_yes=1.90, odds_btts_no=1.90,
            odds_over_2_5=2.05, odds_under_2_5=1.80,
        ),
        Fixture(
            home_team="Aston Villa", away_team="Chelsea",
            kickoff="Sun 14:00",
            odds_home=2.25, odds_draw=3.50, odds_away=3.05,
            odds_btts_yes=1.50, odds_btts_no=2.55,
            odds_over_2_5=1.65, odds_under_2_5=2.25,
        ),
        Fixture(
            home_team="Tottenham", away_team="Brighton",
            kickoff="Sat 12:30",
            odds_home=1.95, odds_draw=3.80, odds_away=3.60,
            odds_btts_yes=1.50, odds_btts_no=2.55,
            odds_over_2_5=1.55, odds_under_2_5=2.40,
        ),
    ]


# =======================================================================
# Reporting
# =======================================================================

def print_team_table(picker: PoissonBetPicker, top: int = 20) -> None:
    print("\nLearned team ratings (sorted by attack − defence):")
    print(f"  {'Team':25s}  {'Attack':>8s}  {'Defence':>8s}  {'Net':>8s}")
    print(f"  {'-' * 25}  {'-' * 8}  {'-' * 8}  {'-' * 8}")
    for i, (team, atk, dfc) in enumerate(picker.team_table()[:top]):
        net = atk - dfc
        print(f"  {team:25s}  {atk:+8.3f}  {dfc:+8.3f}  {net:+8.3f}")


def print_fixture_predictions(
    picker: PoissonBetPicker,
    fixtures: List[Fixture],
) -> None:
    print("\nFixture predictions:")
    for fx in fixtures:
        try:
            p = picker.probabilities(fx.home_team, fx.away_team)
        except KeyError as e:
            print(f"  skipping {fx.home_team} v {fx.away_team}: {e}")
            continue
        print(f"\n  {fx.home_team} v {fx.away_team}   ({fx.kickoff})")
        print(f"    xG: {p.expected_home_goals:.2f} - {p.expected_away_goals:.2f}")
        print(
            f"    1X2:  H {p.home * 100:5.1f}%   "
            f"D {p.draw * 100:5.1f}%   A {p.away * 100:5.1f}%"
        )
        print(
            f"    BTTS: Yes {p.btts_yes * 100:5.1f}%   "
            f"No {p.btts_no * 100:5.1f}%"
        )
        print(
            f"    O/U 2.5: Over {p.over_2_5 * 100:5.1f}%   "
            f"Under {p.under_2_5 * 100:5.1f}%"
        )

        if fx.odds_home and fx.odds_draw and fx.odds_away:
            ih, id_, ia, margin = implied_probabilities(
                fx.odds_home, fx.odds_draw, fx.odds_away
            )
            print(
                f"    book: H {ih * 100:5.1f}%   D {id_ * 100:5.1f}%   "
                f"A {ia * 100:5.1f}%   (margin {margin * 100:.1f}%)"
            )


def print_recommendations(recs: List[BetRecommendation], bank: float) -> None:
    if not recs:
        print("\nNo value bets found — the bookmaker's prices contain no "
              "edge above the minimum threshold.")
        return
    total_stake = sum(r.stake for r in recs)
    total_ev = sum(r.expected_profit for r in recs)
    print(f"\nRecommended bets (bank = £{bank:.2f}):")
    for r in recs:
        print("  " + format_bet(r))
    print(f"\n  Total stake:            £{total_stake:.2f}  "
          f"({total_stake / bank * 100:.1f}% of bank)")
    print(f"  Sum of expected profit: £{total_ev:.2f}")
    print("\n  NOTE: expected profit assumes the model is correctly "
          "calibrated. It usually is not.")
    print("  Gamble responsibly -- gambleaware.org, 0808 8020 133.")


# =======================================================================
# Main
# =======================================================================

def main(argv: Optional[List[str]] = None) -> int:
    p = argparse.ArgumentParser(description="Train the MKAngel football bet picker.")
    p.add_argument("--csv", type=str, default=None,
                   help="Path to a match-results CSV. If omitted, uses the "
                        "built-in synthetic Premier League demo season.")
    p.add_argument("--out", type=str, default="checkpoints/bet_picker.json",
                   help="Where to save the trained model.")
    p.add_argument("--load", type=str, default=None,
                   help="Skip training and load a saved model from this path.")
    p.add_argument("--epochs", type=int, default=500)
    p.add_argument("--lr", type=float, default=0.05)
    p.add_argument("--l2", type=float, default=1e-3)
    p.add_argument("--rho", type=float, default=-0.08,
                   help="Dixon-Coles low-score correction. 0 disables it.")
    p.add_argument("--bank", type=float, default=100.0,
                   help="Bankroll for stake sizing.")
    p.add_argument("--min-edge", type=float, default=0.05,
                   help="Minimum edge to trigger a bet recommendation.")
    p.add_argument("--kelly", type=float, default=0.25,
                   help="Kelly fraction. 0.25 = quarter Kelly (recommended).")
    p.add_argument("--seed", type=int, default=2024)
    p.add_argument("--predict-only", action="store_true",
                   help="Print predictions but no bet recommendations.")
    args = p.parse_args(argv)

    # -- Load or train --
    if args.load:
        print(f"Loading model from {args.load}")
        picker = PoissonBetPicker.load(args.load)
    else:
        if args.csv:
            print(f"Loading matches from {args.csv}")
            matches = load_matches_csv(args.csv)
        else:
            print("Generating synthetic Premier League demo season "
                  "(20 teams, 380 matches) ...")
            matches = generate_demo_season(seed=args.seed)
        print(f"  loaded {len(matches)} matches "
              f"across {len({m.home_team for m in matches})} teams")

        # Small held-out split for evaluation
        random.Random(args.seed).shuffle(matches)
        split = int(len(matches) * 0.9)
        train_matches = matches[:split]
        val_matches = matches[split:]

        picker = PoissonBetPicker(l2=args.l2, rho=args.rho)
        print("\nTraining bivariate-Poisson model (SGD + momentum)...")
        picker.fit(
            train_matches,
            epochs=args.epochs,
            lr=args.lr,
            verbose=True,
            log_every=max(1, args.epochs // 10),
            seed=args.seed,
        )

        train_metrics = evaluate_accuracy(picker, train_matches)
        val_metrics = evaluate_accuracy(picker, val_matches)
        print("\nTraining-set metrics:")
        for k, v in train_metrics.items():
            print(f"  {k:16s} {v:.4f}")
        print("Validation-set metrics:")
        for k, v in val_metrics.items():
            print(f"  {k:16s} {v:.4f}")

        picker.save(args.out)
        print(f"\nModel saved to {args.out}")

    # -- Inspect --
    print_team_table(picker)

    # -- Predict on demo fixtures --
    fixtures = demo_fixtures()
    print_fixture_predictions(picker, fixtures)

    if args.predict_only:
        return 0

    # -- Bet picks --
    recs = picker.pick_bets(
        fixtures,
        bank=args.bank,
        min_edge=args.min_edge,
        kelly_fraction=args.kelly,
    )
    print_recommendations(recs, bank=args.bank)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
