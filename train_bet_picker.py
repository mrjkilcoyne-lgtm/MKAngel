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
    MARKET_OVER_1_5,
    MARKET_OVER_2_5,
    MARKET_OVER_3_5,
    MARKET_UNDER_1_5,
    MARKET_UNDER_2_5,
    MARKET_UNDER_3_5,
    BetRecommendation,
    Fixture,
    FixtureQuote,
    Match,
    PoissonBetPicker,
    format_bet,
    implied_probabilities,
    shin_devig,
    walk_forward_backtest,
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
    season_tag: str = "demo-2023-24",
    season_start: str = "2023-08-12",
    match_noise_sd: float = 0.25,
) -> List[Match]:
    """Simulate a full double round-robin season with dated fixtures.

    Each team plays every other team home and away (380 matches for
    20 teams). Fixtures are laid out one match week per calendar week
    across 38 weeks so the walk-forward backtest has realistic dates.
    """
    from datetime import datetime, timedelta

    rng = random.Random(seed)
    teams = [p[0] for p in priors]
    attack = {p[0]: p[1] for p in priors}
    defence = {p[0]: p[2] for p in priors}

    pairings: List[Tuple[str, str]] = []
    for home in teams:
        for away in teams:
            if home == away:
                continue
            pairings.append((home, away))
    rng.shuffle(pairings)

    # Distribute pairings roughly evenly across 38 match weeks.
    per_week = max(1, len(pairings) // 38)
    start = datetime.fromisoformat(season_start)
    matches: List[Match] = []
    for idx, (home, away) in enumerate(pairings):
        week = idx // per_week
        day_offset = week * 7 + rng.randint(0, 2)
        date = (start + timedelta(days=day_offset)).date().isoformat()

        # Per-match log-normal shock on both teams' expected goals.
        # This models the irreducible noise a goal-only Poisson cannot
        # capture: injuries, suspensions, weather, tactical mismatches,
        # red cards, referee, travel fatigue, motivation. Without it
        # a Poisson picker trained on Poisson data has zero residual
        # uncertainty and the backtest becomes pathological.
        shock_h = rng.gauss(0.0, match_noise_sd)
        shock_a = rng.gauss(0.0, match_noise_sd)
        lh = math.exp(
            base_rate + attack[home] + defence[away] + home_advantage + shock_h
        )
        la = math.exp(base_rate + attack[away] + defence[home] + shock_a)
        hg = _sample_poisson(lh, rng)
        ag = _sample_poisson(la, rng)
        matches.append(Match(
            home_team=home,
            away_team=away,
            home_goals=hg,
            away_goals=ag,
            date=date,
            season=season_tag,
            weight=1.0,
        ))
    matches.sort(key=lambda m: m.date)
    return matches


def generate_multi_season(
    n_seasons: int = 3,
    priors: List[Tuple[str, float, float]] = PL_2023_24_PRIORS,
    base_rate: float = DEMO_BASE_RATE,
    home_advantage: float = DEMO_HOME_ADVANTAGE,
    seed: int = 2024,
    drift: float = 0.05,
    match_noise_sd: float = 0.25,
) -> List[Match]:
    """Generate several consecutive synthetic seasons with mild drift.

    Each new season perturbs the team strengths by a small random walk
    (scale ``drift``). This mimics the way real Premier League form
    evolves year over year and forces the backtest to adapt via
    retraining with time decay.
    """
    rng = random.Random(seed)
    priors_now = [(t, a, d) for t, a, d in priors]
    all_matches: List[Match] = []
    for s in range(n_seasons):
        year = 2021 + s
        start = f"{year}-08-12"
        tag = f"demo-{year}-{(year + 1) % 100:02d}"
        season_matches = generate_demo_season(
            priors=priors_now,
            base_rate=base_rate,
            home_advantage=home_advantage,
            seed=seed + s,
            season_tag=tag,
            season_start=start,
            match_noise_sd=match_noise_sd,
        )
        all_matches.extend(season_matches)
        # Drift for next season
        priors_now = [
            (
                t,
                a + rng.gauss(0, drift),
                d + rng.gauss(0, drift),
            )
            for t, a, d in priors_now
        ]
    all_matches.sort(key=lambda m: m.date)
    return all_matches


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
# Bookmaker panel simulation
# =======================================================================

# Mean bookmaker margins roughly match the UK market: Pinnacle ~2%,
# the big chain books 5-7%. Noise adds idiosyncratic per-book price
# dispersion; that dispersion is where the aggregator's edge lives.

DEFAULT_BOOK_PANEL: List[Tuple[str, float, float]] = [
    # name       margin   noise_sd (log odds)
    ("Pinnacle",   0.025,  0.020),
    ("Bet365",     0.055,  0.035),
    ("William Hill", 0.060, 0.040),
    ("Ladbrokes",  0.065,  0.040),
    ("Paddy Power", 0.065, 0.045),
    ("Coral",      0.060,  0.045),
    ("Sky Bet",    0.055,  0.040),
    ("Unibet",     0.055,  0.040),
    ("Betfair Sportsbook", 0.050, 0.035),
    ("BoyleSports", 0.065, 0.050),
]


def _true_match_probabilities(
    home: str,
    away: str,
    attack: Dict[str, float],
    defence: Dict[str, float],
    base_rate: float,
    home_advantage: float,
    goal_cap: int = 8,
) -> Dict[str, float]:
    """Ground-truth fair probabilities for a fixture.

    Used by the bookmaker simulator so the panel's prices are
    centred on the true market with each book adding its own margin
    and per-book noise. Returns a dict keyed by market constant.
    """
    lh = math.exp(base_rate + attack[home] + defence[away] + home_advantage)
    la = math.exp(base_rate + attack[away] + defence[home])

    # Manually compute Poisson pmfs
    def pmf(lam: float) -> List[float]:
        out = [math.exp(-lam)]
        for k in range(1, goal_cap + 1):
            out.append(out[-1] * lam / k)
        return out

    ph = pmf(lh)
    pa = pmf(la)

    p_home = p_draw = p_away = 0.0
    p_btts_yes = 0.0
    totals = [0.0] * (2 * goal_cap + 1)
    diffs: Dict[int, float] = {}
    for i in range(goal_cap + 1):
        for j in range(goal_cap + 1):
            p = ph[i] * pa[j]
            if i > j:
                p_home += p
            elif i == j:
                p_draw += p
            else:
                p_away += p
            if i > 0 and j > 0:
                p_btts_yes += p
            totals[i + j] += p
            diffs[i - j] = diffs.get(i - j, 0.0) + p

    s = p_home + p_draw + p_away
    p_home /= s
    p_draw /= s
    p_away /= s
    return {
        MARKET_HOME: p_home,
        MARKET_DRAW: p_draw,
        MARKET_AWAY: p_away,
        MARKET_BTTS_YES: p_btts_yes,
        MARKET_BTTS_NO: max(0.0, 1.0 - p_btts_yes),
        MARKET_OVER_1_5: sum(totals[2:]),
        MARKET_UNDER_1_5: max(0.0, 1.0 - sum(totals[2:])),
        MARKET_OVER_2_5: sum(totals[3:]),
        MARKET_UNDER_2_5: max(0.0, 1.0 - sum(totals[3:])),
        MARKET_OVER_3_5: sum(totals[4:]),
        MARKET_UNDER_3_5: max(0.0, 1.0 - sum(totals[4:])),
        MARKET_DC_1X: p_home + p_draw,
        MARKET_DC_X2: p_draw + p_away,
        MARKET_DC_12: p_home + p_away,
        MARKET_DNB_HOME: p_home / max(1e-9, 1.0 - p_draw),
        MARKET_DNB_AWAY: p_away / max(1e-9, 1.0 - p_draw),
        MARKET_AH_HOME_M1: sum(p for d, p in diffs.items() if d >= 2),
        MARKET_AH_AWAY_P1: sum(p for d, p in diffs.items() if d <= 0),
        MARKET_AH_HOME_P1: sum(p for d, p in diffs.items() if d >= 0),
        MARKET_AH_AWAY_M1: sum(p for d, p in diffs.items() if d <= -2),
    }


def simulate_book_quotes(
    match: Match,
    true_probs: Dict[str, float],
    panel: List[Tuple[str, float, float]] = DEFAULT_BOOK_PANEL,
    rng: Optional[random.Random] = None,
    markets: Optional[List[str]] = None,
) -> List[FixtureQuote]:
    """Simulate one bookmaker panel's decimal-odds quote for a match.

    For each book, each market price is:

        fair_prob  ~  true_prob * exp(N(0, noise_sd))
        decimal    =  1 / (fair_prob * (1 + margin))

    i.e. log-normal noise around the true probability, then an
    inverse-price hit by the book's margin. The realised best-of-book
    line across the panel can therefore be occasionally longer than
    the true fair price -- the value the aggregator exists to capture.
    """
    if rng is None:
        rng = random.Random()
    if markets is None:
        markets = list(true_probs.keys())

    quotes: List[FixtureQuote] = []
    for name, margin, noise in panel:
        odds: Dict[str, float] = {}
        for m in markets:
            p = true_probs.get(m, 0.0)
            if p <= 1e-6:
                continue
            # Log-normal perturbation on the fair probability.
            noisy = p * math.exp(rng.gauss(0.0, noise))
            noisy = max(1e-6, min(0.999, noisy))
            # Apply the book's margin to the *priced* probability.
            priced = noisy * (1.0 + margin)
            odds[m] = round(1.0 / priced, 2)
        quotes.append(FixtureQuote(
            home_team=match.home_team,
            away_team=match.away_team,
            book=name,
            kickoff=match.date,
            odds=odds,
        ))
    return quotes


def build_quote_panel(
    matches: List[Match],
    priors: List[Tuple[str, float, float]],
    base_rate: float = DEMO_BASE_RATE,
    home_advantage: float = DEMO_HOME_ADVANTAGE,
    panel: List[Tuple[str, float, float]] = DEFAULT_BOOK_PANEL,
    seed: int = 7,
) -> Dict[int, List[FixtureQuote]]:
    """Build the full backtest quote panel keyed by match index."""
    rng = random.Random(seed)
    attack = {p[0]: p[1] for p in priors}
    defence = {p[0]: p[2] for p in priors}

    out: Dict[int, List[FixtureQuote]] = {}
    for idx, m in enumerate(matches):
        # Guard: only build quotes for teams the priors know.
        if m.home_team not in attack or m.away_team not in attack:
            continue
        true_probs = _true_match_probabilities(
            m.home_team, m.away_team,
            attack, defence, base_rate, home_advantage,
        )
        out[idx] = simulate_book_quotes(m, true_probs, panel=panel, rng=rng)
    return out


# =======================================================================
# CSV loading
# =======================================================================

def load_matches_csv(path: str) -> List[Match]:
    """Load matches from a CSV file.

    Expected columns (case-insensitive, in any order):
        home_team, away_team, home_goals, away_goals
    Optional columns: date, season, weight
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
                    date=row[cols["date"]].strip() if "date" in cols else "",
                    season=row[cols["season"]].strip() if "season" in cols else "",
                    weight=float(row[cols["weight"]]) if "weight" in cols else 1.0,
                ))
            except (KeyError, ValueError) as e:
                print(f"  skipping malformed row: {e}", file=sys.stderr)
    return matches


# -----------------------------------------------------------------------
# Football-Data.co.uk format
# -----------------------------------------------------------------------
#
# The files at https://www.football-data.co.uk/englandm.php (season CSVs
# for the Premier League, EFL Championship, League One and League Two)
# are the standard free source for UK football results + closing odds.
#
# Key columns: Date, HomeTeam, AwayTeam, FTHG, FTAG, plus many
# bookmaker columns like B365H / B365D / B365A, BWH / BWD / BWA, ...
# We also recognise the over/under columns (B365>2.5 / B365<2.5 etc.).

_FD_BOOK_PREFIXES = [
    "B365",  # Bet365
    "BW",    # Betway
    "IW",    # Interwetten
    "LB",    # Ladbrokes
    "PS",    # Pinnacle
    "WH",    # William Hill
    "VC",    # VC Bet
    "SJ",    # Stan James
]

_FD_BOOK_NAME = {
    "B365": "Bet365",
    "BW": "Betway",
    "IW": "Interwetten",
    "LB": "Ladbrokes",
    "PS": "Pinnacle",
    "WH": "William Hill",
    "VC": "VC Bet",
    "SJ": "Stan James",
}


def load_football_data_csv(
    path: str,
) -> Tuple[List[Match], Dict[int, List[FixtureQuote]]]:
    """Load a Football-Data.co.uk results CSV with closing odds.

    Returns ``(matches, quotes_by_index)`` ready for walk_forward_backtest.
    Unknown/missing columns are skipped silently.
    """
    matches: List[Match] = []
    quotes: Dict[int, List[FixtureQuote]] = {}
    with open(path, "r", newline="") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None:
            raise ValueError(f"CSV {path} has no header row")
        fields = {c.strip(): c for c in reader.fieldnames}

        required = ["Date", "HomeTeam", "AwayTeam", "FTHG", "FTAG"]
        for r in required:
            if r not in fields:
                raise ValueError(f"{path}: missing required FD column {r}")

        for row in reader:
            try:
                date = row[fields["Date"]].strip()
                home = row[fields["HomeTeam"]].strip()
                away = row[fields["AwayTeam"]].strip()
                fthg = int(row[fields["FTHG"]])
                ftag = int(row[fields["FTAG"]])
            except (KeyError, ValueError):
                continue

            idx = len(matches)
            matches.append(Match(
                home_team=home,
                away_team=away,
                home_goals=fthg,
                away_goals=ftag,
                date=date,
            ))

            book_quotes: List[FixtureQuote] = []
            for prefix in _FD_BOOK_PREFIXES:
                odds: Dict[str, float] = {}

                def _get(col: str) -> Optional[float]:
                    if col not in fields:
                        return None
                    raw = row[fields[col]].strip()
                    if not raw:
                        return None
                    try:
                        v = float(raw)
                    except ValueError:
                        return None
                    return v if v > 1.0 else None

                h = _get(f"{prefix}H")
                d = _get(f"{prefix}D")
                a = _get(f"{prefix}A")
                if h and d and a:
                    odds[MARKET_HOME] = h
                    odds[MARKET_DRAW] = d
                    odds[MARKET_AWAY] = a
                over = _get(f"{prefix}>2.5")
                under = _get(f"{prefix}<2.5")
                if over and under:
                    odds[MARKET_OVER_2_5] = over
                    odds[MARKET_UNDER_2_5] = under
                if odds:
                    book_quotes.append(FixtureQuote(
                        home_team=home,
                        away_team=away,
                        book=_FD_BOOK_NAME[prefix],
                        kickoff=date,
                        odds=odds,
                    ))
            if book_quotes:
                quotes[idx] = book_quotes

    # Sort chronologically and re-key quotes by new index.
    indexed = sorted(range(len(matches)), key=lambda i: matches[i].date or "")
    new_matches = [matches[i] for i in indexed]
    new_quotes = {
        new_idx: quotes[old_idx]
        for new_idx, old_idx in enumerate(indexed)
        if old_idx in quotes
    }
    return new_matches, new_quotes


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

def run_backtest(args: argparse.Namespace) -> int:
    """Walk-forward backtest of the aggregator strategy."""
    if args.fd_csv:
        print(f"Loading Football-Data.co.uk CSV {args.fd_csv} ...")
        matches, quotes = load_football_data_csv(args.fd_csv)
    else:
        print(f"Generating {args.seasons}-season synthetic league "
              f"with drift + bookmaker panel ...")
        matches = generate_multi_season(
            n_seasons=args.seasons,
            seed=args.seed,
            match_noise_sd=args.match_noise,
        )
        quotes = build_quote_panel(
            matches,
            priors=PL_2023_24_PRIORS,
            seed=args.seed + 1,
        )
    print(f"  {len(matches)} matches across "
          f"{len({m.home_team for m in matches})} teams "
          f"with {len(quotes)} quoted fixtures "
          f"(~{len(DEFAULT_BOOK_PANEL)} books each)")

    print("\nWalk-forward backtest (retraining periodically with time decay)...")
    result = walk_forward_backtest(
        matches,
        quotes,
        initial_train=args.initial_train,
        retrain_every=args.retrain_every,
        starting_bank=args.bank,
        min_edge=args.min_edge,
        kelly_fraction=args.kelly,
        max_stake_fraction=args.max_stake,
        max_total_stake_fraction=args.max_total_stake,
        half_life_days=args.half_life,
        flat_stake=args.flat_stake,
        fit_kwargs={"epochs": args.epochs, "lr": args.lr},
        picker_kwargs={"l2": args.l2, "rho": args.rho},
        verbose=True,
    )

    print()
    print(result.summary())

    # Extra: closing-line comparison and calibration summary.
    print("\nStaked-weighted realised vs expected profit:")
    if result.total_staked > 0:
        ev_total = sum(
            r["profit"] for r in result.per_market.values()
        )
        print(f"  realised P/L £{ev_total:+,.2f}  "
              f"on £{result.total_staked:,.2f} staked  "
              f"({ev_total / result.total_staked * 100:+.2f}% ROI)")
    print("\nGamble responsibly -- gambleaware.org, 0808 8020 133.")
    return 0


def main(argv: Optional[List[str]] = None) -> int:
    p = argparse.ArgumentParser(description="Train the MKAngel football bet picker.")

    # Mode flags
    p.add_argument("--backtest", action="store_true",
                   help="Walk-forward backtest against past performance. "
                        "Uses multi-season synthetic data unless --fd-csv "
                        "is provided.")
    p.add_argument("--fd-csv", type=str, default=None,
                   help="Football-Data.co.uk CSV file with historical "
                        "results and bookmaker odds (for --backtest).")

    p.add_argument("--csv", type=str, default=None,
                   help="Path to a match-results CSV (training mode).")
    p.add_argument("--out", type=str, default="checkpoints/bet_picker.json",
                   help="Where to save the trained model.")
    p.add_argument("--load", type=str, default=None,
                   help="Skip training and load a saved model from this path.")
    p.add_argument("--epochs", type=int, default=500)
    p.add_argument("--lr", type=float, default=0.05)
    p.add_argument("--l2", type=float, default=1e-3)
    p.add_argument("--rho", type=float, default=-0.08,
                   help="Dixon-Coles low-score correction. 0 disables it.")
    p.add_argument("--bank", type=float, default=1000.0,
                   help="Bankroll for stake sizing.")
    p.add_argument("--min-edge", type=float, default=0.03,
                   help="Minimum edge to trigger a bet recommendation.")
    p.add_argument("--kelly", type=float, default=0.25,
                   help="Kelly fraction. 0.25 = quarter Kelly (recommended).")
    p.add_argument("--max-stake", type=float, default=0.03,
                   help="Hard cap on single stake as fraction of bank.")
    p.add_argument("--max-total-stake", type=float, default=0.30,
                   help="Hard cap on total stake per match card.")
    p.add_argument("--half-life", type=float, default=365.0,
                   help="Time-decay half life in days (backtest & training).")
    p.add_argument("--seasons", type=int, default=3,
                   help="Number of synthetic seasons for --backtest when "
                        "no --fd-csv supplied.")
    p.add_argument("--initial-train", type=int, default=380,
                   help="Initial training window size for backtest.")
    p.add_argument("--retrain-every", type=int, default=10,
                   help="Retrain after every N walked matches.")
    p.add_argument("--flat-stake", type=float, default=None,
                   help="Use a fixed unit stake instead of compounding "
                        "Kelly (honest ROI reporting).")
    p.add_argument("--match-noise", type=float, default=0.25,
                   help="Per-match log-lambda shock in synthetic data "
                        "(models injuries/tactics/form; 0 = pure Poisson).")
    p.add_argument("--seed", type=int, default=2024)
    p.add_argument("--predict-only", action="store_true",
                   help="Print predictions but no bet recommendations.")
    args = p.parse_args(argv)

    if args.backtest:
        return run_backtest(args)

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
            half_life_days=args.half_life,
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
