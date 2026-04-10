#!/usr/bin/env python3
"""Run the football bet picker on real weekend odds.

Reads:
  - checkpoints/bet_picker_real.json (trained on 2023-26 EPL real data)
  - data/pinnacle_parsed.json        (Pinnacle H/D/A, O/U, AH per fixture)
  - data/panel_odds.json             (other bookmakers, keyed by fixture)

Writes nothing; prints the top 3 picks from ``pick_bets_aggregated``.

No stake is ever recommended above the configured Kelly cap, and the
aggregator requires a minimum edge so thin or noisy opportunities are
ignored. Gamble responsibly.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Dict, List

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from app.bet_picker import (  # noqa: E402
    MARKET_AH_AWAY_M1,
    MARKET_AH_AWAY_P1,
    MARKET_AH_HOME_M1,
    MARKET_AH_HOME_P1,
    MARKET_AWAY,
    MARKET_BTTS_NO,
    MARKET_BTTS_YES,
    MARKET_DC_12,
    MARKET_DC_1X,
    MARKET_DC_X2,
    MARKET_DNB_AWAY,
    MARKET_DNB_HOME,
    MARKET_DRAW,
    MARKET_HOME,
    MARKET_OVER_2_5,
    MARKET_OVER_3_5,
    MARKET_UNDER_2_5,
    MARKET_UNDER_3_5,
    BetRecommendation,
    FixtureQuote,
    PoissonBetPicker,
    implied_probabilities,
    shin_devig,
)

# Map the compact keys we use in data files into the canonical market
# constants understood by pick_bets_aggregated.
KEY_MAP = {
    "H": MARKET_HOME,
    "D": MARKET_DRAW,
    "A": MARKET_AWAY,
    "OVER_2_5": MARKET_OVER_2_5,
    "UNDER_2_5": MARKET_UNDER_2_5,
    "OVER_3_5": MARKET_OVER_3_5,
    "UNDER_3_5": MARKET_UNDER_3_5,
    "BTTS_YES": MARKET_BTTS_YES,
    "BTTS_NO": MARKET_BTTS_NO,
    "AH_H_-1.0": MARKET_AH_HOME_M1,
    "AH_A_1.0": MARKET_AH_AWAY_P1,
    "AH_H_1.0": MARKET_AH_HOME_P1,
    "AH_A_-1.0": MARKET_AH_AWAY_M1,
    # Allow float / dotted variants
    "OVER_2.5": MARKET_OVER_2_5,
    "UNDER_2.5": MARKET_UNDER_2_5,
    "OVER_3.5": MARKET_OVER_3_5,
    "UNDER_3.5": MARKET_UNDER_3_5,
}

# Pinnacle / comparison-site team names mapped to the canonical form
# used in the training CSV.
TEAM_ALIASES = {
    "Manchester City": "Manchester City",
    "Manchester United": "Manchester United",
    "Newcastle United": "Newcastle",
    "Newcastle": "Newcastle",
    "Nottingham Forest": "Nottingham Forest",
    "Nott'm Forest": "Nottingham Forest",
    "Tottenham Hotspur": "Tottenham",
    "Tottenham": "Tottenham",
    "Spurs": "Tottenham",
    "AFC Bournemouth": "Bournemouth",
    "Bournemouth": "Bournemouth",
    "Wolverhampton Wanderers": "Wolves",
    "Wolves": "Wolves",
}


def canon(team: str) -> str:
    return TEAM_ALIASES.get(team.strip(), team.strip())


def parse_fixture_label(label: str):
    parts = label.split(" v ")
    if len(parts) != 2:
        parts = label.split(" vs ")
    return canon(parts[0].strip()), canon(parts[1].strip())


def build_quotes(
    pinnacle_file: str,
    panel_file: str | None,
) -> List[FixtureQuote]:
    quotes: List[FixtureQuote] = []

    pin = json.load(open(pinnacle_file))
    for label, market_odds in pin.items():
        home, away = parse_fixture_label(label)
        mapped = {}
        for k, v in market_odds.items():
            if k in KEY_MAP and v is not None and v > 1.0:
                mapped[KEY_MAP[k]] = float(v)
        if mapped:
            quotes.append(FixtureQuote(
                home_team=home, away_team=away,
                book="Pinnacle",
                odds=mapped,
            ))

    if panel_file and os.path.exists(panel_file):
        panel = json.load(open(panel_file))
        for label, books in panel.items():
            home, away = parse_fixture_label(label)
            for book_name, market_odds in books.items():
                mapped = {}
                for k, v in market_odds.items():
                    if k in KEY_MAP and v is not None and v > 1.0:
                        mapped[KEY_MAP[k]] = float(v)
                if mapped:
                    quotes.append(FixtureQuote(
                        home_team=home, away_team=away,
                        book=book_name,
                        odds=mapped,
                    ))
    return quotes


def pinnacle_fair_prob_map(pinnacle_file: str) -> Dict[tuple, Dict[str, float]]:
    """Build a {(home, away): {market: fair_prob}} map by Shin-de-vigging
    every market Pinnacle quotes. This is our "sharp anchor" -- the
    closest thing we have to true probabilities -- and any pick where
    our model disagrees with this by more than a threshold is rejected
    as likely model error.
    """
    out: Dict[tuple, Dict[str, float]] = {}
    if not os.path.exists(pinnacle_file):
        return out
    pin = json.load(open(pinnacle_file))
    for label, odds in pin.items():
        home, away = parse_fixture_label(label)
        probs: Dict[str, float] = {}
        # 1X2 via Shin
        if all(k in odds for k in ("H", "D", "A")):
            ph, pd, pa = shin_devig([odds["H"], odds["D"], odds["A"]])
            probs[MARKET_HOME] = ph
            probs[MARKET_DRAW] = pd
            probs[MARKET_AWAY] = pa
            probs[MARKET_DC_1X] = ph + pd
            probs[MARKET_DC_X2] = pd + pa
            probs[MARKET_DC_12] = ph + pa
            if pd < 0.999:
                probs[MARKET_DNB_HOME] = ph / (1.0 - pd)
                probs[MARKET_DNB_AWAY] = pa / (1.0 - pd)
        # Over/Under via pair devig
        for over_k, under_k, mo, mu in [
            ("OVER_2_5", "UNDER_2_5", MARKET_OVER_2_5, MARKET_UNDER_2_5),
            ("OVER_2.5", "UNDER_2.5", MARKET_OVER_2_5, MARKET_UNDER_2_5),
            ("OVER_3_5", "UNDER_3_5", MARKET_OVER_3_5, MARKET_UNDER_3_5),
            ("OVER_3.5", "UNDER_3.5", MARKET_OVER_3_5, MARKET_UNDER_3_5),
        ]:
            if over_k in odds and under_k in odds:
                po, pu = shin_devig([odds[over_k], odds[under_k]])
                probs[mo] = po
                probs[mu] = pu
        # Asian handicap ±1 via pair devig. The resulting p_h and p_a
        # are effective probabilities conditional on not pushing --
        # which is exactly what the picker's AH model_prob reports.
        if "AH_H_-1.0" in odds and "AH_A_1.0" in odds:
            ph, pa = shin_devig([odds["AH_H_-1.0"], odds["AH_A_1.0"]])
            probs[MARKET_AH_HOME_M1] = ph
            probs[MARKET_AH_AWAY_P1] = pa
        if "AH_H_1.0" in odds and "AH_A_-1.0" in odds:
            ph, pa = shin_devig([odds["AH_H_1.0"], odds["AH_A_-1.0"]])
            probs[MARKET_AH_HOME_P1] = ph
            probs[MARKET_AH_AWAY_M1] = pa
        out[(home, away)] = probs
    return out


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="checkpoints/bet_picker_real.json")
    p.add_argument("--pinnacle", default="data/pinnacle_parsed.json")
    p.add_argument("--panel", default="data/panel_odds.json")
    p.add_argument("--bank", type=float, default=200.0)
    p.add_argument("--min-edge", type=float, default=0.03)
    p.add_argument("--kelly", type=float, default=0.25)
    p.add_argument("--max-stake", type=float, default=0.03)
    p.add_argument("--max-total", type=float, default=0.15)
    p.add_argument("--top", type=int, default=3)
    p.add_argument("--sanity-pp", type=float, default=0.07,
                   help="Reject picks where model probability differs "
                        "from Pinnacle Shin-de-vigged fair probability "
                        "by more than this many percentage points.")
    args = p.parse_args()

    picker = PoissonBetPicker.load(args.model)
    quotes = build_quotes(args.pinnacle, args.panel)
    fair_probs = pinnacle_fair_prob_map(args.pinnacle)
    print(f"Loaded {len(quotes)} bookmaker quotes across "
          f"{len({(q.home_team, q.away_team) for q in quotes})} fixtures")
    print(f"Pinnacle Shin-de-vigged fair probabilities for "
          f"{len(fair_probs)} fixtures")

    # Per-fixture preview: model's view vs sharpest available book (Pinnacle)
    print("\nModel vs Pinnacle (sharpest book, de-vigged):")
    print("-" * 78)
    seen = set()
    for q in quotes:
        if q.book != "Pinnacle":
            continue
        key = (q.home_team, q.away_team)
        if key in seen:
            continue
        seen.add(key)
        try:
            mp = picker.probabilities(q.home_team, q.away_team)
        except KeyError as e:
            print(f"  skip {q.home_team} v {q.away_team}: {e}")
            continue
        h = q.odds.get(MARKET_HOME)
        d = q.odds.get(MARKET_DRAW)
        a = q.odds.get(MARKET_AWAY)
        if h and d and a:
            ih, id_, ia, margin = implied_probabilities(h, d, a)
            print(f"  {q.home_team:22s} v {q.away_team:22s}")
            print(f"    model : H {mp.home*100:5.1f}%  D {mp.draw*100:5.1f}%  "
                  f"A {mp.away*100:5.1f}%  (xG {mp.expected_home_goals:.2f}-"
                  f"{mp.expected_away_goals:.2f})")
            print(f"    Pinn  : H {ih*100:5.1f}%  D {id_*100:5.1f}%  "
                  f"A {ia*100:5.1f}%  (margin {margin*100:.1f}%)")

    raw_recs = picker.pick_bets_aggregated(
        quotes,
        bank=args.bank,
        min_edge=args.min_edge,
        kelly_fraction=args.kelly,
        max_stake_fraction=args.max_stake,
        max_total_stake_fraction=args.max_total,
    )

    # --- Pinnacle-sanity filter ---
    # A value bet is only trusted when our model's probability is close
    # to what Pinnacle (the sharpest book) implies once its margin is
    # stripped out. Any pick where we disagree with Pinnacle by more
    # than sanity_pp percentage points is almost certainly model error,
    # not a real edge, and gets rejected.
    accepted: List[BetRecommendation] = []
    rejected: List[tuple] = []
    for r in raw_recs:
        home, away = parse_fixture_label(r.fixture)
        anchor = fair_probs.get((home, away), {})
        sharp_p = anchor.get(r.market)
        if sharp_p is None:
            # No Pinnacle reference for this market -- accept with
            # caution (common for BTTS, which Pinnacle doesn't price)
            accepted.append(r)
            continue
        diff = abs(r.model_prob - sharp_p)
        if diff > args.sanity_pp:
            rejected.append((r, sharp_p, diff))
            continue
        accepted.append(r)
    recs = accepted

    if rejected:
        print(f"\nRejected {len(rejected)} picks for disagreeing with "
              f"Pinnacle by > {args.sanity_pp*100:.0f}pp "
              f"(likely model error, not true edge):")
        for r, sp, diff in rejected[:8]:
            print(f"  - {r.fixture:35s}  {r.market:13s}  "
                  f"model {r.model_prob*100:5.1f}%  "
                  f"Pinn {sp*100:5.1f}%  diff {diff*100:+.1f}pp  "
                  f"(would have been edge {r.edge*100:+.1f}%)")

    print(f"\nFound {len(recs)} value bets passing sanity filter "
          f"(min edge {args.min_edge*100:.1f}%, bank £{args.bank:.0f}).")
    print()
    print(f"TOP {args.top} PICKS — weekend of 11-12 April 2026")
    print("=" * 78)
    for i, r in enumerate(recs[:args.top], 1):
        fair = 1.0 / r.model_prob if r.model_prob > 0 else float("inf")
        print(f"\n  Pick #{i}  ({r.edge*100:+.1f}% edge)")
        print(f"    Fixture : {r.fixture}")
        print(f"    Market  : {r.market}")
        print(f"    Price   : {r.decimal_odds:.2f}  (model fair {fair:.2f})")
        print(f"    Book    : {r.book}")
        print(f"    Stake   : £{r.stake:.2f}  (quarter-Kelly, £{args.bank:.0f} bank)")
        print(f"    EV      : £{r.expected_profit:+.2f}")
    print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
