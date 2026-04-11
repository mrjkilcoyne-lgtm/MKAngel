#!/usr/bin/env python3
"""Sharp-anchor weekend evaluation — Joseph Buchdahl's "Wisdom of the Crowd"
method applied to live weekend Premier League prices.

What this does
--------------
Rather than trusting our own Poisson model as ground truth, this script
uses Pinnacle's de-vigged prices as the sharp-market consensus (the
same approach Buchdahl documented in his 67,000-bet European football
backtest, and the approach Starlizard / Smartodds / professional
football syndicates build their systems around).

For each of the weekend picks we already identified, the script:

    1. Strips Pinnacle's ~3-3.5% margin off the 1X2 market to get
       the sharp-consensus fair price for each outcome.
    2. Takes the best available price across 25 UK books (from
       the live Oddschecker scrape baked into this script) for
       the outcome we want to back.
    3. Computes the Buchdahl gap  (offered - fair) / fair  in %.
    4. FIRE  if gap >= +2%  (Buchdahl's Football-Data cutoff)
       SKIP  otherwise (even a +1.9% gap is not a value bet).
    5. Quarter-Kelly stakes against the sharp-anchor fair probability
       on the fired bets, capped at 3% of bank per bet.

Reference
---------
Buchdahl, J. (2015). *The Wisdom of the Crowd in a Football Betting
Market.* football-data.co.uk. Reported empirical yields at 67,000
bets:

       edge threshold   bets       yield
       >0%              22,281     +3.40%
       >1%              14,837     +4.60%
       >2%               9,196     +6.63%   <- default threshold
       >3%               5,474     +8.83%
       >4%               3,243    +12.70%
       >5%               1,927    +13.22%
"""

from __future__ import annotations

import os
import sys
from typing import Dict, List, Tuple

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from app.bet_picker import buchdahl_fair_odds, pick_sharp_value, sharp_edge  # noqa: E402


# -----------------------------------------------------------------------
# Live weekend data — captured from Pinnacle's guest API and the
# Oddschecker per-book Playwright scrape earlier in this session.
# All prices are the live Saturday-morning best-of-book across 25 UK
# bookmakers plus Pinnacle's own quote for the same match.
# -----------------------------------------------------------------------

WEEKEND = [
    {
        "label": "Nottingham Forest v Aston Villa",
        "kickoff": "Sun 14:00",
        "pinnacle": [2.67, 3.33, 2.79],    # H, D, A
        "pick": "Aston Villa",
        "outcome_index": 2,
        "panel": {
            "Betfair Sportsbook": 2.80,
            "Matchbook":          2.80,
            "Sky Bet":            2.75,
            "Spreadex":           2.75,
            "AK Bets":            2.75,
            "888sport":           2.74,
            "Unibet":             2.65,
            "Bet365":             2.63,
            "William Hill":       2.63,
            "BetVictor":          2.63,
            "Ladbrokes":          2.60,
        },
    },
    {
        "label": "Burnley v Brighton",
        "kickoff": "Sat 15:00",
        "pinnacle": [4.50, 4.15, 1.75],
        "pick": "Burnley",
        "outcome_index": 0,
        "panel": {
            "AK Bets":    4.80,
            "BetVictor":  4.75,
            "Kwiff":      4.75,
            "888sport":   4.65,
            "Bet365":     4.60,
            "Betfred":    4.60,
            "Sky Bet":    4.50,
            "Spreadex":   4.50,
            "Paddy Power":4.50,
            "Ladbrokes":  4.20,
        },
    },
    {
        "label": "Liverpool v Fulham",
        "kickoff": "Sat 17:30",
        "pinnacle": [1.65, 4.31, 5.07],
        "pick": "Fulham",
        "outcome_index": 2,
        "panel": {
            "Betfair Sportsbook": 5.01,
            "Matchbook":          5.01,
            "Betfred":            5.00,
            "Sky Bet":            5.00,
            "Kwiff":              5.00,
            "888sport":           5.00,
            "Spreadex":           5.00,
            "AK Bets":            5.00,
            "Coral":              4.80,
            "Bet365":             4.75,
            "Ladbrokes":          4.60,
        },
    },
]


def main(bank: float = 200.0, min_edge_pct: float = 2.0) -> int:
    print("=" * 78)
    print("SHARP EVALUATION — Buchdahl Wisdom of the Crowd")
    print("Pinnacle de-vigged fair price = sharp-market consensus")
    print(f"Value threshold: gap >= +{min_edge_pct:.1f}%  (Buchdahl default)")
    print(f"Bank: £{bank:.0f}   |   Staking: quarter-Kelly, 3% single-bet cap")
    print("=" * 78)

    fires: List[Dict] = []
    total_stake = 0.0
    total_ev = 0.0

    for fx in WEEKEND:
        fair_vec, margin = buchdahl_fair_odds(fx["pinnacle"])
        print(f"\n{fx['label']}   ({fx['kickoff']})")
        print(f"  Pinnacle  H {fx['pinnacle'][0]:.2f}   D {fx['pinnacle'][1]:.2f}"
              f"   A {fx['pinnacle'][2]:.2f}   (margin {margin * 100:.2f}%)")
        print(f"  Fair      H {fair_vec[0]:.3f}   D {fair_vec[1]:.3f}"
              f"   A {fair_vec[2]:.3f}")

        outcome_names = ["Home", "Draw", "Away"]
        fair_pick = fair_vec[fx["outcome_index"]]
        print(f"  Pick      {fx['pick']} ({outcome_names[fx['outcome_index']]})"
              f"   sharp fair = {fair_pick:.3f}   "
              f"sharp p = {1.0 / fair_pick * 100:.1f}%")

        print("\n  Panel vs sharp fair:")
        panel_sorted = sorted(fx["panel"].items(), key=lambda kv: -kv[1])
        for book, odds in panel_sorted:
            info = sharp_edge(odds, fx["pinnacle"], fx["outcome_index"])
            mark = "FIRE" if info["gap_pct"] >= min_edge_pct else (
                "MARG" if info["gap_pct"] >= 0 else "SKIP"
            )
            print(f"    {book:22s} {odds:5.2f}   "
                  f"gap {info['gap_pct']:+6.2f}%   {mark}")

        best = pick_sharp_value(
            fx["panel"], fx["pinnacle"], fx["outcome_index"],
            min_edge_pct=min_edge_pct,
        )
        if best is None:
            print(f"\n  VERDICT: SKIP — no book clears +{min_edge_pct:.1f}%")
            continue

        # Quarter-Kelly stake on the Buchdahl fair probability
        D = best["offered_odds"]
        p = best["fair_prob"]
        full_k = max(0.0, (p * D - 1.0) / (D - 1.0))
        stake = min(0.03 * bank, 0.25 * full_k * bank)
        stake = round(stake, 2)
        ev = round(p * (D - 1.0) * stake - (1.0 - p) * stake, 2)
        fires.append({**best, "fixture": fx["label"], "pick": fx["pick"],
                      "stake": stake, "ev": ev})
        total_stake += stake
        total_ev += ev
        print(f"\n  VERDICT: FIRE   {fx['pick']} @ {D:.2f} on {best['book']}"
              f"   gap {best['gap_pct']:+.2f}%")
        print(f"           stake £{stake:.2f}  EV £{ev:+.2f}")

    print("\n" + "=" * 78)
    print(f"SHARP CARD — {len(fires)} fire / {len(WEEKEND) - len(fires)} skip")
    print("=" * 78)
    if fires:
        for f in fires:
            print(f"  {f['fixture']:38s}  {f['pick']:14s}  "
                  f"@{f['offered_odds']:.2f} ({f['book']})  "
                  f"edge {f['gap_pct']:+.2f}%  stake £{f['stake']:.2f}")
        print(f"\n  Total stake:   £{total_stake:.2f} ({total_stake / bank * 100:.1f}% of bank)")
        print(f"  Expected P/L:  £{total_ev:+.2f}")
    else:
        print("  No fires — sharp market consensus is already longer than every "
              "offered price.")
        print("  The disciplined answer this weekend is to NOT bet.")

    print()
    print("Buchdahl empirical yields at different edge thresholds")
    print("(67,034 European football bets, 2012/13 to 2014/15):")
    print("    >0%  3.40%   >1%  4.60%   >2%  6.63%   >3%  8.83%")
    print("    >4% 12.70%   >5% 13.22%")
    print()
    print("Gamble responsibly — gambleaware.org, 0808 8020 133.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
