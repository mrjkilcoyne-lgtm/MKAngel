"""Football bet picker -- UK leagues.

A transparent, pure-Python value-betting model for UK association football.
Uses a Dixon-Coles-style bivariate Poisson goal model:

    log(lambda_home) = base + attack[H] + defence[A] + home_advantage
    log(lambda_away) = base + attack[A] + defence[H]

where ``attack`` is scoring strength (higher = scores more) and
``defence`` is leakiness (higher = concedes more). Parameters are fit by
minimising the negative log-likelihood of observed full-time scores via
SGD with momentum. No numpy, no torch -- fits the MKAngel pure-Python
stack so it runs on Android.

From the fitted goal model we derive market prices for the common UK
football markets:

- **1X2** (home / draw / away full-time result)
- **BTTS** (both teams to score, yes/no)
- **Over/Under 2.5 goals** (can be parameterised for 1.5, 3.5, ...)
- **Correct score** (up to a configurable goal cap)

Bets are picked by comparing the model's probability to the bookmaker
price. When ``model_prob * decimal_odds > 1`` there is a positive edge;
stake size is set via fractional Kelly.

RESPONSIBLE GAMBLING
--------------------
This module is a research/educational tool. No model beats the market
reliably -- past performance does not predict future returns, and
bookmaker margins mean most bets lose over time. UK users:
gambleaware.org, 0808 8020 133. The picker enforces a hard Kelly cap
and refuses stakes on negative-edge bets.
"""

from __future__ import annotations

import json
import math
import os
import random
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any, Callable, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple


# =======================================================================
# Data structures
# =======================================================================

@dataclass
class Match:
    """A completed football match used for training.

    Attributes
    ----------
    home_team, away_team : str
        Team names (case-sensitive, must be consistent across matches).
    home_goals, away_goals : int
        Full-time goals scored.
    date : str, optional
        ISO-format date "YYYY-MM-DD". Used for time-decay weighting
        and walk-forward backtesting. May also be any parseable
        ``datetime.fromisoformat`` string.
    season : str, optional
        Season tag (e.g. "2023-24"). Used only for bookkeeping.
    weight : float
        Sample weight -- use <1.0 to down-weight older seasons.
    """

    home_team: str
    away_team: str
    home_goals: int
    away_goals: int
    date: str = ""
    season: str = ""
    weight: float = 1.0


# Canonical market keys used by the aggregator and pick_bets.
MARKET_HOME = "1X2:home"
MARKET_DRAW = "1X2:draw"
MARKET_AWAY = "1X2:away"
MARKET_BTTS_YES = "BTTS:yes"
MARKET_BTTS_NO = "BTTS:no"
MARKET_OVER_1_5 = "O/U1.5:over"
MARKET_UNDER_1_5 = "O/U1.5:under"
MARKET_OVER_2_5 = "O/U2.5:over"
MARKET_UNDER_2_5 = "O/U2.5:under"
MARKET_OVER_3_5 = "O/U3.5:over"
MARKET_UNDER_3_5 = "O/U3.5:under"
MARKET_DC_1X = "DC:1X"
MARKET_DC_X2 = "DC:X2"
MARKET_DC_12 = "DC:12"
MARKET_DNB_HOME = "DNB:home"
MARKET_DNB_AWAY = "DNB:away"
MARKET_AH_HOME_M1 = "AH:home-1"
MARKET_AH_AWAY_P1 = "AH:away+1"
MARKET_AH_HOME_P1 = "AH:home+1"
MARKET_AH_AWAY_M1 = "AH:away-1"

# Markets where a push (stake refund) is possible.
_PUSH_MARKETS = {
    MARKET_AH_HOME_M1,
    MARKET_AH_AWAY_P1,
    MARKET_AH_HOME_P1,
    MARKET_AH_AWAY_M1,
    MARKET_DNB_HOME,
    MARKET_DNB_AWAY,
}


@dataclass
class Fixture:
    """An upcoming match to price and potentially bet on.

    Odds are in decimal (UK standard): ``2.50`` means a £1 stake returns
    £2.50 total (£1.50 profit) if the bet wins. Missing markets can be
    left as ``None``.

    The primary API is the ``odds`` dict which maps a canonical market
    key (see ``MARKET_*`` constants) to a decimal price. The legacy
    scalar ``odds_*`` keyword arguments are still accepted and merged
    into the dict at construction time so older calling code keeps
    working.
    """

    home_team: str
    away_team: str
    kickoff: str = ""              # free-form, e.g. "2024-08-17 15:00"
    odds: Dict[str, float] = field(default_factory=dict)

    # Legacy scalar fields (auto-merged into ``odds`` in __post_init__)
    odds_home: Optional[float] = None
    odds_draw: Optional[float] = None
    odds_away: Optional[float] = None
    odds_btts_yes: Optional[float] = None
    odds_btts_no: Optional[float] = None
    odds_over_2_5: Optional[float] = None
    odds_under_2_5: Optional[float] = None

    def __post_init__(self) -> None:
        legacy = {
            MARKET_HOME: self.odds_home,
            MARKET_DRAW: self.odds_draw,
            MARKET_AWAY: self.odds_away,
            MARKET_BTTS_YES: self.odds_btts_yes,
            MARKET_BTTS_NO: self.odds_btts_no,
            MARKET_OVER_2_5: self.odds_over_2_5,
            MARKET_UNDER_2_5: self.odds_under_2_5,
        }
        for k, v in legacy.items():
            if v is not None and k not in self.odds:
                self.odds[k] = float(v)


@dataclass
class FixtureQuote:
    """A single bookmaker's quote for a fixture.

    The aggregator collects many of these across books and, per market,
    selects the *best* (highest decimal) price. Every stake is placed
    at the best-of-book rather than with any single bookmaker -- that
    is where most of the bet picker's realised edge comes from.
    """

    home_team: str
    away_team: str
    book: str
    kickoff: str = ""
    odds: Dict[str, float] = field(default_factory=dict)


@dataclass
class BetRecommendation:
    """A single recommended bet."""

    fixture: str                   # "Home v Away"
    market: str                    # "1X2:home", "BTTS:yes", "O/U2.5:over", ...
    model_prob: float              # fair probability (effective after push handling)
    decimal_odds: float            # best-of-book bookmaker price
    edge: float                    # effective_prob * odds - 1
    kelly_fraction: float          # fractional-Kelly stake as fraction of bank
    stake: float                   # recommended stake in currency
    expected_profit: float         # stake * edge
    book: str = ""                 # bookmaker offering the best price
    push_prob: float = 0.0         # probability of stake refund (AH / DNB)


@dataclass
class MatchProbabilities:
    """Full set of market probabilities for a single fixture."""

    home: float
    draw: float
    away: float
    btts_yes: float
    btts_no: float
    over_2_5: float
    under_2_5: float
    expected_home_goals: float
    expected_away_goals: float
    correct_score: Dict[Tuple[int, int], float] = field(default_factory=dict)

    # Extended markets (populated by probabilities() when available)
    over_1_5: float = 0.0
    under_1_5: float = 0.0
    over_3_5: float = 0.0
    under_3_5: float = 0.0
    dc_1x: float = 0.0
    dc_x2: float = 0.0
    dc_12: float = 0.0
    # Draw-no-bet: effective probabilities (conditional on non-draw)
    dnb_home: float = 0.0
    dnb_away: float = 0.0
    # Asian handicap: (win, push) per side at ±1 lines
    ah_home_minus_1_win: float = 0.0
    ah_home_minus_1_push: float = 0.0
    ah_away_plus_1_win: float = 0.0
    ah_away_plus_1_push: float = 0.0
    ah_home_plus_1_win: float = 0.0
    ah_home_plus_1_push: float = 0.0
    ah_away_minus_1_win: float = 0.0
    ah_away_minus_1_push: float = 0.0

    def as_dict(self) -> Dict[str, float]:
        return {
            "home": self.home,
            "draw": self.draw,
            "away": self.away,
            "btts_yes": self.btts_yes,
            "btts_no": self.btts_no,
            "over_1_5": self.over_1_5,
            "under_1_5": self.under_1_5,
            "over_2_5": self.over_2_5,
            "under_2_5": self.under_2_5,
            "over_3_5": self.over_3_5,
            "under_3_5": self.under_3_5,
            "dc_1x": self.dc_1x,
            "dc_x2": self.dc_x2,
            "dc_12": self.dc_12,
            "dnb_home": self.dnb_home,
            "dnb_away": self.dnb_away,
            "xg_home": self.expected_home_goals,
            "xg_away": self.expected_away_goals,
        }


@dataclass
class BacktestResult:
    """Outcome of a walk-forward backtest against past match results."""

    starting_bank: float
    final_bank: float
    total_staked: float
    total_profit: float
    num_bets: int
    num_wins: int
    num_pushes: int
    num_losses: int
    hit_rate: float
    roi: float                # total_profit / total_staked
    yield_pct: float          # total_profit / num_bets (average per bet)
    max_drawdown: float       # largest peak-to-trough drop in bank
    bank_history: List[float] = field(default_factory=list)
    per_market: Dict[str, Dict[str, float]] = field(default_factory=dict)

    # Out-of-sample 1X2 accuracy across all walked matches (not just
    # the ones we bet on). Lets us see how many "past results" the
    # model would have predicted correctly in its argmax.
    matches_scored: int = 0
    result_hits: int = 0         # top-1 1X2 prediction matched actual
    result_rps: float = 0.0      # mean Rank Probability Score
    result_log_loss: float = 0.0 # mean neg-log-prob of observed outcome

    def summary(self) -> str:
        pnl = self.final_bank - self.starting_bank
        lines = [
            f"Backtest summary",
            f"  Starting bank : £{self.starting_bank:,.2f}",
            f"  Final bank    : £{self.final_bank:,.2f}",
            f"  P/L           : £{pnl:+,.2f}",
            f"  Total staked  : £{self.total_staked:,.2f}",
            f"  Bets placed   : {self.num_bets}",
            f"  W / Push / L  : {self.num_wins} / {self.num_pushes} / {self.num_losses}",
            f"  Hit rate      : {self.hit_rate * 100:.1f}%",
            f"  ROI on stake  : {self.roi * 100:+.2f}%",
            f"  Yield / bet   : £{self.yield_pct:+.3f}",
            f"  Max drawdown  : £{self.max_drawdown:,.2f}",
        ]
        if self.matches_scored:
            result_acc = self.result_hits / self.matches_scored
            lines.append(
                f"  Past-results match rate:  "
                f"{result_acc * 100:.1f}% "
                f"({self.result_hits}/{self.matches_scored})  "
                f"[argmax 1X2 vs actual]"
            )
            lines.append(
                f"  Mean RPS     : {self.result_rps:.4f}  "
                f"(lower = better; 0.20 is strong for football)"
            )
            lines.append(
                f"  Mean log-loss: {self.result_log_loss:.4f}  "
                f"(lower = better)"
            )
        if self.per_market:
            lines.append("  Per-market ROI:")
            rows = sorted(
                self.per_market.items(),
                key=lambda kv: kv[1]["profit"],
                reverse=True,
            )
            for name, m in rows:
                lines.append(
                    f"    {name:16s}  n={int(m['bets']):4d}  "
                    f"stake=£{m['stake']:7.2f}  "
                    f"pnl=£{m['profit']:+7.2f}  "
                    f"roi={m['roi'] * 100:+6.2f}%"
                )
        return "\n".join(lines)


# =======================================================================
# Poisson bet picker
# =======================================================================

class PoissonBetPicker:
    """Bivariate-Poisson football model with SGD training.

    Parameters
    ----------
    goal_cap : int
        Maximum goals per side used when marginalising the score matrix
        to compute market probabilities. 10 is plenty for football.
    l2 : float
        L2 regularisation coefficient on team parameters. Shrinks weak
        signals toward the league average.
    rho : float
        Dixon-Coles low-score adjustment. Slightly inflates 0-0 / 1-1
        and deflates 1-0 / 0-1 to match observed football data.
        ``rho = 0`` reduces to pure independent Poisson.
    """

    def __init__(
        self,
        goal_cap: int = 10,
        l2: float = 1e-3,
        rho: float = -0.08,
    ) -> None:
        self.goal_cap = goal_cap
        self.l2 = l2
        self.rho = rho

        self.teams: List[str] = []
        self.team_index: Dict[str, int] = {}

        # Parameters (populated by fit)
        self.attack: List[float] = []          # alpha_i (shared attack)
        self.defence: List[float] = []         # beta_i (shared defence)
        self.home_advantage: float = 0.25
        self.base_rate: float = 0.0            # log global mean goals

        # Training diagnostics
        self.loss_history: List[float] = []

    # -------------------------------------------------------------------
    # Vocab / parameter initialisation
    # -------------------------------------------------------------------

    def _register_teams(self, matches: Sequence[Match]) -> None:
        seen: List[str] = []
        for m in matches:
            for t in (m.home_team, m.away_team):
                if t not in self.team_index:
                    self.team_index[t] = len(seen)
                    seen.append(t)
        self.teams = seen
        n = len(self.teams)
        if not self.attack or len(self.attack) != n:
            self.attack = [0.0] * n
            self.defence = [0.0] * n

        # Initialise base_rate from league mean goals per team per match
        if matches:
            total = sum(m.home_goals + m.away_goals for m in matches)
            per_team = total / (2.0 * len(matches))
            self.base_rate = math.log(max(per_team, 0.1))
        else:
            self.base_rate = math.log(1.35)

    # -------------------------------------------------------------------
    # Core goal-expectation function
    # -------------------------------------------------------------------

    def _lambdas(self, home_idx: int, away_idx: int) -> Tuple[float, float]:
        """Expected goals for (home, away) given current parameters.

        ``defence[i]`` is leakiness: higher means team ``i`` concedes more,
        so it boosts the opponent's expected goals.
        """
        lh = math.exp(
            self.base_rate
            + self.attack[home_idx]
            + self.defence[away_idx]
            + self.home_advantage
        )
        la = math.exp(
            self.base_rate
            + self.attack[away_idx]
            + self.defence[home_idx]
        )
        # Guard against blow-up in early training
        lh = min(max(lh, 1e-4), 15.0)
        la = min(max(la, 1e-4), 15.0)
        return lh, la

    # -------------------------------------------------------------------
    # Training
    # -------------------------------------------------------------------

    def fit(
        self,
        matches: Sequence[Match],
        epochs: int = 400,
        lr: float = 0.05,
        momentum: float = 0.9,
        verbose: bool = True,
        log_every: int = 50,
        seed: Optional[int] = 42,
        half_life_days: Optional[float] = None,
        reference_date: Optional[str] = None,
    ) -> Dict[str, float]:
        """Fit model parameters by SGD on negative log-likelihood.

        Parameters
        ----------
        matches : sequence of Match
            Training data.
        epochs, lr, momentum, verbose, log_every, seed
            Standard SGD controls.
        half_life_days : float, optional
            If set, each match's ``weight`` is multiplied by
            ``0.5 ** (age_days / half_life_days)`` where ``age_days`` is
            the gap between ``match.date`` and ``reference_date``.
            Matches with no date get full weight.
        reference_date : str, optional
            Anchor date for time decay. If ``None``, the latest date
            observed in ``matches`` is used.

        Returns a dict with ``final_loss`` and ``epochs`` run.
        """
        if not matches:
            raise ValueError("No matches provided to fit().")

        self._register_teams(matches)
        rng = random.Random(seed)

        n_teams = len(self.teams)

        # Momentum buffers
        v_attack = [0.0] * n_teams
        v_defence = [0.0] * n_teams
        v_ha = 0.0
        v_base = 0.0

        match_list = list(matches)

        # -- Time-decay weights --
        effective_weights: List[float] = [m.weight for m in match_list]
        if half_life_days and half_life_days > 0:
            ref = _parse_date(reference_date) if reference_date else None
            if ref is None:
                dates = [_parse_date(m.date) for m in match_list]
                dates = [d for d in dates if d is not None]
                ref = max(dates) if dates else None
            if ref is not None:
                lam = math.log(2.0) / half_life_days
                for i, m in enumerate(match_list):
                    d = _parse_date(m.date)
                    if d is None:
                        continue
                    age = max(0.0, (ref - d).days)
                    effective_weights[i] = m.weight * math.exp(-lam * age)

        for epoch in range(epochs):
            order = list(range(len(match_list)))
            rng.shuffle(order)
            total_loss = 0.0
            total_weight = 0.0

            # Per-epoch gradient accumulators
            g_attack = [0.0] * n_teams
            g_defence = [0.0] * n_teams
            g_ha = 0.0
            g_base = 0.0

            for idx in order:
                m = match_list[idx]
                h = self.team_index[m.home_team]
                a = self.team_index[m.away_team]
                lh, la = self._lambdas(h, a)

                gh = m.home_goals
                ga = m.away_goals
                w = effective_weights[idx]

                # NLL of Poisson(lh) at gh and Poisson(la) at ga:
                #   -[gh*log(lh) - lh - log(gh!)] - [ga*log(la) - la - log(ga!)]
                # The factorial terms do not depend on parameters, but we
                # include them in the reported loss for interpretability.
                nll = (
                    lh - gh * math.log(lh) + _log_factorial(gh)
                    + la - ga * math.log(la) + _log_factorial(ga)
                )
                total_loss += w * nll
                total_weight += w

                # dNLL/d(log lh) = lh - gh, and log lh depends linearly on
                # attack[h], defence[a], home_advantage and base_rate. Same
                # for log la w.r.t. attack[a] and defence[h].
                dh = (lh - gh) * w
                da = (la - ga) * w

                g_attack[h] += dh
                g_attack[a] += da
                g_defence[a] += dh
                g_defence[h] += da
                g_ha += dh
                g_base += dh + da

            # Normalise
            denom = max(total_weight, 1.0)
            for i in range(n_teams):
                g_attack[i] = g_attack[i] / denom + self.l2 * self.attack[i]
                g_defence[i] = g_defence[i] / denom + self.l2 * self.defence[i]
            g_ha /= denom
            g_base /= denom

            # SGD + momentum updates
            for i in range(n_teams):
                v_attack[i] = momentum * v_attack[i] + g_attack[i]
                v_defence[i] = momentum * v_defence[i] + g_defence[i]
                self.attack[i] -= lr * v_attack[i]
                self.defence[i] -= lr * v_defence[i]
            v_ha = momentum * v_ha + g_ha
            v_base = momentum * v_base + g_base
            self.home_advantage -= lr * v_ha
            self.base_rate -= lr * v_base

            # Identifiability: force sum(attack) = sum(defence) = 0
            mean_a = sum(self.attack) / n_teams
            mean_d = sum(self.defence) / n_teams
            for i in range(n_teams):
                self.attack[i] -= mean_a
                self.defence[i] -= mean_d
            self.base_rate += mean_a  # attack shift absorbs into base rate

            avg_loss = total_loss / denom
            self.loss_history.append(avg_loss)

            if verbose and (epoch % log_every == 0 or epoch == epochs - 1):
                print(
                    f"  epoch {epoch + 1:4d}/{epochs}  "
                    f"nll={avg_loss:.4f}  "
                    f"ha={self.home_advantage:+.3f}  "
                    f"base={self.base_rate:+.3f}"
                )

        return {
            "final_loss": self.loss_history[-1] if self.loss_history else 0.0,
            "epochs": float(epochs),
            "teams": float(n_teams),
        }

    # -------------------------------------------------------------------
    # Prediction
    # -------------------------------------------------------------------

    def _require(self, home: str, away: str) -> Tuple[int, int]:
        if home not in self.team_index:
            raise KeyError(f"Unknown home team: {home!r}")
        if away not in self.team_index:
            raise KeyError(f"Unknown away team: {away!r}")
        return self.team_index[home], self.team_index[away]

    def expected_goals(self, home: str, away: str) -> Tuple[float, float]:
        h, a = self._require(home, away)
        return self._lambdas(h, a)

    def score_matrix(self, home: str, away: str) -> List[List[float]]:
        """Return P(home_goals=i, away_goals=j) for i, j in [0, goal_cap].

        Applies the Dixon-Coles low-score correction controlled by
        ``self.rho``.
        """
        lh, la = self.expected_goals(home, away)
        cap = self.goal_cap

        # Precompute Poisson pmfs
        ph = _poisson_pmf_vector(lh, cap)
        pa = _poisson_pmf_vector(la, cap)

        # Joint matrix = outer product, then Dixon-Coles tweak on low
        # scores:  tau(i, j, lh, la, rho)
        matrix = [[ph[i] * pa[j] for j in range(cap + 1)] for i in range(cap + 1)]

        rho = self.rho
        if rho != 0.0:
            # Only the four cells (0,0), (0,1), (1,0), (1,1) are affected.
            matrix[0][0] *= 1.0 - lh * la * rho
            matrix[0][1] *= 1.0 + lh * rho
            matrix[1][0] *= 1.0 + la * rho
            matrix[1][1] *= 1.0 - rho

            # Clip negatives that may appear for extreme rho values.
            for i in range(2):
                for j in range(2):
                    if matrix[i][j] < 0:
                        matrix[i][j] = 0.0

            # Renormalise to account for both the tweak and truncation
            # at ``goal_cap``.
            total = sum(sum(row) for row in matrix)
            if total > 0:
                for i in range(cap + 1):
                    for j in range(cap + 1):
                        matrix[i][j] /= total

        return matrix

    def probabilities(self, home: str, away: str) -> MatchProbabilities:
        """Compute all market probabilities for a fixture.

        Covers 1X2, BTTS, Over/Under {1.5, 2.5, 3.5}, double chance,
        draw-no-bet and Asian handicap at ±1. All numbers are derived
        from the Dixon-Coles-adjusted score matrix.
        """
        matrix = self.score_matrix(home, away)
        cap = self.goal_cap

        p_home = 0.0
        p_draw = 0.0
        p_away = 0.0
        p_btts_yes = 0.0

        # Goal-sum buckets for over/under lines.
        total_buckets = [0.0] * (2 * cap + 1)
        # Goal-difference buckets (home - away) for Asian handicap.
        diff_buckets: Dict[int, float] = {}

        for i in range(cap + 1):
            for j in range(cap + 1):
                p = matrix[i][j]
                if p <= 0:
                    continue
                if i > j:
                    p_home += p
                elif i == j:
                    p_draw += p
                else:
                    p_away += p
                if i > 0 and j > 0:
                    p_btts_yes += p
                total_buckets[i + j] += p
                d = i - j
                diff_buckets[d] = diff_buckets.get(d, 0.0) + p

        # Tidy up rounding drift.
        p_total = p_home + p_draw + p_away
        if p_total > 0:
            p_home /= p_total
            p_draw /= p_total
            p_away /= p_total

        over_1_5 = sum(total_buckets[2:])
        over_2_5 = sum(total_buckets[3:])
        over_3_5 = sum(total_buckets[4:])

        # Double chance
        dc_1x = p_home + p_draw
        dc_x2 = p_draw + p_away
        dc_12 = p_home + p_away

        # Draw-no-bet effective probabilities (conditional on no draw)
        if p_draw < 1.0:
            dnb_home = p_home / (1.0 - p_draw)
            dnb_away = p_away / (1.0 - p_draw)
        else:
            dnb_home = dnb_away = 0.0

        # Asian handicap ±1 (whole-number lines, push possible)
        #  Home -1 : home must win by 2+    (push on exactly +1)
        #  Away +1 : away gets +1 goal      (push on exactly +1)
        #  Home +1 : home gets +1 goal      (push on exactly -1)
        #  Away -1 : away must win by 2+    (push on exactly -1)
        ah_h_m1_win = sum(p for d, p in diff_buckets.items() if d >= 2)
        ah_h_m1_push = diff_buckets.get(1, 0.0)
        ah_a_p1_win = sum(p for d, p in diff_buckets.items() if d <= 0)
        ah_a_p1_push = diff_buckets.get(1, 0.0)
        ah_h_p1_win = sum(p for d, p in diff_buckets.items() if d >= 0)
        ah_h_p1_push = diff_buckets.get(-1, 0.0)
        ah_a_m1_win = sum(p for d, p in diff_buckets.items() if d <= -2)
        ah_a_m1_push = diff_buckets.get(-1, 0.0)

        lh, la = self.expected_goals(home, away)

        # Keep only the top correct-score cells for report sanity.
        flat = []
        for i in range(min(cap + 1, 6)):
            for j in range(min(cap + 1, 6)):
                flat.append(((i, j), matrix[i][j]))
        flat.sort(key=lambda kv: kv[1], reverse=True)
        top_scores = dict(flat[:10])

        return MatchProbabilities(
            home=p_home,
            draw=p_draw,
            away=p_away,
            btts_yes=p_btts_yes,
            btts_no=max(0.0, 1.0 - p_btts_yes),
            over_1_5=over_1_5,
            under_1_5=max(0.0, 1.0 - over_1_5),
            over_2_5=over_2_5,
            under_2_5=max(0.0, 1.0 - over_2_5),
            over_3_5=over_3_5,
            under_3_5=max(0.0, 1.0 - over_3_5),
            dc_1x=dc_1x,
            dc_x2=dc_x2,
            dc_12=dc_12,
            dnb_home=dnb_home,
            dnb_away=dnb_away,
            ah_home_minus_1_win=ah_h_m1_win,
            ah_home_minus_1_push=ah_h_m1_push,
            ah_away_plus_1_win=ah_a_p1_win,
            ah_away_plus_1_push=ah_a_p1_push,
            ah_home_plus_1_win=ah_h_p1_win,
            ah_home_plus_1_push=ah_h_p1_push,
            ah_away_minus_1_win=ah_a_m1_win,
            ah_away_minus_1_push=ah_a_m1_push,
            expected_home_goals=lh,
            expected_away_goals=la,
            correct_score=top_scores,
        )

    def _market_probabilities(
        self, probs: MatchProbabilities
    ) -> Dict[str, Tuple[float, float]]:
        """Return ``{market_key: (p_win, p_push)}`` for pick_bets().

        Markets without push always get ``p_push = 0``.
        """
        return {
            MARKET_HOME: (probs.home, 0.0),
            MARKET_DRAW: (probs.draw, 0.0),
            MARKET_AWAY: (probs.away, 0.0),
            MARKET_BTTS_YES: (probs.btts_yes, 0.0),
            MARKET_BTTS_NO: (probs.btts_no, 0.0),
            MARKET_OVER_1_5: (probs.over_1_5, 0.0),
            MARKET_UNDER_1_5: (probs.under_1_5, 0.0),
            MARKET_OVER_2_5: (probs.over_2_5, 0.0),
            MARKET_UNDER_2_5: (probs.under_2_5, 0.0),
            MARKET_OVER_3_5: (probs.over_3_5, 0.0),
            MARKET_UNDER_3_5: (probs.under_3_5, 0.0),
            MARKET_DC_1X: (probs.dc_1x, 0.0),
            MARKET_DC_X2: (probs.dc_x2, 0.0),
            MARKET_DC_12: (probs.dc_12, 0.0),
            MARKET_DNB_HOME: (probs.dnb_home, probs.draw),
            MARKET_DNB_AWAY: (probs.dnb_away, probs.draw),
            MARKET_AH_HOME_M1: (
                probs.ah_home_minus_1_win, probs.ah_home_minus_1_push
            ),
            MARKET_AH_AWAY_P1: (
                probs.ah_away_plus_1_win, probs.ah_away_plus_1_push
            ),
            MARKET_AH_HOME_P1: (
                probs.ah_home_plus_1_win, probs.ah_home_plus_1_push
            ),
            MARKET_AH_AWAY_M1: (
                probs.ah_away_minus_1_win, probs.ah_away_minus_1_push
            ),
        }

    # -------------------------------------------------------------------
    # Bet selection
    # -------------------------------------------------------------------

    def _score_market(
        self,
        p_win: float,
        p_push: float,
        odds: float,
        bank: float,
        min_edge: float,
        kelly_fraction: float,
        max_stake_fraction: float,
    ) -> Optional[Tuple[float, float, float, float]]:
        """Edge + Kelly stake for a single market quote.

        Handles push markets (AH ±1, DNB) by computing an effective
        probability conditional on the bet not being refunded.

        Returns ``(effective_prob, edge, kelly_fraction, stake)`` or
        ``None`` if the quote should be skipped.
        """
        if odds is None or odds <= 1.0 or p_win <= 0.0:
            return None

        active = 1.0 - p_push
        if active <= 1e-9:
            return None
        p_eff = p_win / active
        if p_eff <= 0.0 or p_eff >= 1.0:
            return None

        edge = p_eff * odds - 1.0
        if edge < min_edge:
            return None

        full_kelly = (p_eff * odds - 1.0) / (odds - 1.0)
        # For push markets, scale Kelly by the active fraction so the
        # capital at risk per unit stake matches the standard formula.
        full_kelly *= active
        if full_kelly <= 0:
            return None

        f = full_kelly * kelly_fraction
        f = min(f, max_stake_fraction)
        stake = round(bank * f, 2)
        if stake <= 0:
            return None
        return p_eff, edge, f, stake

    def pick_bets(
        self,
        fixtures: Iterable[Fixture],
        bank: float = 100.0,
        min_edge: float = 0.03,
        kelly_fraction: float = 0.25,
        max_stake_fraction: float = 0.05,
    ) -> List[BetRecommendation]:
        """Score fixtures against bookmaker odds and recommend bets.

        Each ``Fixture`` is treated as one bookmaker's full quote --
        iterate over its ``odds`` dict (which is auto-populated from
        the legacy ``odds_home``/``odds_draw``/... scalar arguments).
        For the aggregator workflow where many bookmakers offer quotes
        on the same match, use ``pick_bets_aggregated`` instead.

        Parameters
        ----------
        fixtures : iterable of Fixture
            Upcoming matches with at least one market price set.
        bank : float
            Bankroll in pounds (or any currency).
        min_edge : float
            Minimum required edge before a bet is recommended. Bets
            below this threshold are ignored.
        kelly_fraction : float
            Fraction of full Kelly to stake. 0.25 (quarter-Kelly) is a
            standard safe default; full Kelly is too aggressive.
        max_stake_fraction : float
            Hard cap on any single stake as a fraction of bank.
        """
        if bank <= 0:
            return []

        recs: List[BetRecommendation] = []

        for fx in fixtures:
            try:
                probs = self.probabilities(fx.home_team, fx.away_team)
            except KeyError:
                continue
            label = f"{fx.home_team} v {fx.away_team}"
            market_probs = self._market_probabilities(probs)

            for market_name, odds in fx.odds.items():
                if market_name not in market_probs:
                    continue
                p_win, p_push = market_probs[market_name]
                scored = self._score_market(
                    p_win, p_push, odds, bank,
                    min_edge, kelly_fraction, max_stake_fraction,
                )
                if scored is None:
                    continue
                p_eff, edge, f, stake = scored
                recs.append(BetRecommendation(
                    fixture=label,
                    market=market_name,
                    model_prob=p_eff,
                    decimal_odds=odds,
                    edge=edge,
                    kelly_fraction=f,
                    stake=stake,
                    expected_profit=round(stake * edge, 2),
                    push_prob=p_push,
                ))

        recs.sort(key=lambda r: r.edge, reverse=True)
        return recs

    def pick_bets_aggregated(
        self,
        quotes: Iterable[FixtureQuote],
        bank: float = 100.0,
        min_edge: float = 0.03,
        kelly_fraction: float = 0.25,
        max_stake_fraction: float = 0.05,
        max_total_stake_fraction: float = 0.5,
    ) -> List[BetRecommendation]:
        """Aggregator-first pick: line-shop across multiple bookmakers.

        For every (home, away, market) triple, pick the *best* decimal
        odds on offer across all supplied ``FixtureQuote`` objects, then
        score that best price against the model. This is where most of
        the realised edge lives -- a single-book picker usually loses to
        the margin, but an aggregator that routes each stake to whoever
        is offering the longest price can turn a break-even model into
        a positive-EV one.

        Parameters
        ----------
        quotes : iterable of FixtureQuote
            Every bookmaker snapshot for every fixture. The same match
            may appear many times (once per book).
        bank, min_edge, kelly_fraction, max_stake_fraction
            See ``pick_bets``.
        max_total_stake_fraction : float
            Safety rail: if recommendations would stake more than this
            fraction of bank in aggregate, every stake is scaled down
            proportionally so the cap is respected.
        """
        if bank <= 0:
            return []

        best: Dict[Tuple[str, str, str], Tuple[float, str, str]] = {}
        for q in quotes:
            key_prefix = (q.home_team, q.away_team)
            for market_name, odds in q.odds.items():
                if odds is None or odds <= 1.0:
                    continue
                key = (key_prefix[0], key_prefix[1], market_name)
                current = best.get(key)
                if current is None or odds > current[0]:
                    best[key] = (odds, q.book, q.kickoff)

        # Group keys by fixture so we only call probabilities() once.
        by_fixture: Dict[Tuple[str, str], List[Tuple[str, float, str, str]]] = {}
        for (h, a, m), (odds, book, kickoff) in best.items():
            by_fixture.setdefault((h, a), []).append((m, odds, book, kickoff))

        recs: List[BetRecommendation] = []
        for (h, a), markets in by_fixture.items():
            try:
                probs = self.probabilities(h, a)
            except KeyError:
                continue
            label = f"{h} v {a}"
            market_probs = self._market_probabilities(probs)

            for market_name, odds, book, _kickoff in markets:
                if market_name not in market_probs:
                    continue
                p_win, p_push = market_probs[market_name]
                scored = self._score_market(
                    p_win, p_push, odds, bank,
                    min_edge, kelly_fraction, max_stake_fraction,
                )
                if scored is None:
                    continue
                p_eff, edge, f, stake = scored
                recs.append(BetRecommendation(
                    fixture=label,
                    market=market_name,
                    model_prob=p_eff,
                    decimal_odds=odds,
                    edge=edge,
                    kelly_fraction=f,
                    stake=stake,
                    expected_profit=round(stake * edge, 2),
                    book=book,
                    push_prob=p_push,
                ))

        # Enforce an overall cap on exposure per card.
        total = sum(r.stake for r in recs)
        cap = bank * max_total_stake_fraction
        if total > cap and total > 0:
            scale = cap / total
            for r in recs:
                r.stake = round(r.stake * scale, 2)
                r.kelly_fraction *= scale
                r.expected_profit = round(r.stake * r.edge, 2)

        recs.sort(key=lambda r: r.edge, reverse=True)
        return recs

    # -------------------------------------------------------------------
    # Persistence
    # -------------------------------------------------------------------

    def save(self, path: str) -> None:
        payload = {
            "version": 1,
            "teams": self.teams,
            "attack": self.attack,
            "defence": self.defence,
            "home_advantage": self.home_advantage,
            "base_rate": self.base_rate,
            "goal_cap": self.goal_cap,
            "l2": self.l2,
            "rho": self.rho,
            "loss_history": self.loss_history[-200:],
        }
        parent = os.path.dirname(path)
        if parent:
            os.makedirs(parent, exist_ok=True)
        with open(path, "w") as f:
            json.dump(payload, f, indent=2)

    @classmethod
    def load(cls, path: str) -> "PoissonBetPicker":
        with open(path, "r") as f:
            data = json.load(f)
        picker = cls(
            goal_cap=int(data.get("goal_cap", 10)),
            l2=float(data.get("l2", 1e-3)),
            rho=float(data.get("rho", -0.08)),
        )
        picker.teams = list(data["teams"])
        picker.team_index = {t: i for i, t in enumerate(picker.teams)}
        picker.attack = list(data["attack"])
        picker.defence = list(data["defence"])
        picker.home_advantage = float(data["home_advantage"])
        picker.base_rate = float(data["base_rate"])
        picker.loss_history = list(data.get("loss_history", []))
        return picker

    # -------------------------------------------------------------------
    # Introspection
    # -------------------------------------------------------------------

    def team_table(self) -> List[Tuple[str, float, float]]:
        """Return ``[(team, attack, defence), ...]`` sorted by net rating.

        Net rating = attack - defence; higher is stronger overall.
        Useful for a quick sanity check that the model has learnt
        something sensible.
        """
        rows = [
            (t, self.attack[i], self.defence[i])
            for i, t in enumerate(self.teams)
        ]
        rows.sort(key=lambda r: r[1] - r[2], reverse=True)
        return rows


# =======================================================================
# Numerical helpers (pure Python)
# =======================================================================

def _log_factorial(n: int) -> float:
    if n < 2:
        return 0.0
    return sum(math.log(k) for k in range(2, n + 1))


def _poisson_pmf_vector(lam: float, cap: int) -> List[float]:
    """P(X=k) for k in [0, cap] under Poisson(lam). Stable for small cap."""
    out = [0.0] * (cap + 1)
    out[0] = math.exp(-lam)
    for k in range(1, cap + 1):
        out[k] = out[k - 1] * lam / k
    return out


def _parse_date(s: Optional[str]) -> Optional[datetime]:
    """Forgiving date parser used by time decay and the backtest.

    Accepts ISO ("YYYY-MM-DD"), Football-Data.co.uk style ("DD/MM/YY"
    or "DD/MM/YYYY"), and full timestamps. Returns ``None`` if the
    input is empty or unparseable.
    """
    if not s:
        return None
    s = s.strip()
    if not s:
        return None
    # Fast path for ISO
    for fmt in ("%Y-%m-%d", "%Y-%m-%dT%H:%M:%S", "%Y-%m-%d %H:%M:%S",
                "%d/%m/%Y", "%d/%m/%y"):
        try:
            return datetime.strptime(s, fmt)
        except ValueError:
            continue
    try:
        return datetime.fromisoformat(s)
    except ValueError:
        return None


# =======================================================================
# Market utilities
# =======================================================================

def implied_probabilities(
    odds_home: float,
    odds_draw: float,
    odds_away: float,
    method: str = "proportional",
) -> Tuple[float, float, float, float]:
    """Strip bookmaker margin from a 1X2 price.

    Parameters
    ----------
    odds_home, odds_draw, odds_away : float
        Decimal odds.
    method : {"proportional", "shin"}
        ``"proportional"`` divides each raw inverse price by the sum,
        which is simple but biases toward longshots. ``"shin"`` uses
        Shin (1993)'s model of informed bettors; it generally produces
        better-calibrated fair probabilities and is the default in
        serious odds aggregator stacks.

    Returns (p_home, p_draw, p_away, overround) where ``overround``
    is the bookmaker edge (e.g. 0.05 means a 5% margin).
    """
    raw = [1.0 / odds_home, 1.0 / odds_draw, 1.0 / odds_away]
    total = sum(raw)
    overround = total - 1.0
    if method == "shin":
        ps = shin_devig([odds_home, odds_draw, odds_away])
        return ps[0], ps[1], ps[2], overround
    return (raw[0] / total, raw[1] / total, raw[2] / total, overround)


def shin_devig(
    decimal_odds: Sequence[float],
    tol: float = 1e-9,
    max_iter: int = 200,
) -> List[float]:
    """Shin (1993) de-vigging: fair probabilities from decimal odds.

    Models the bookmaker's margin as an allowance for a ``z`` fraction
    of informed bettors. Given market prices ``q_i = 1/odds_i`` summing
    to ``1 + overround``, solve for ``z`` and return the fair ``p_i``
    using the closed form::

        p_i = (sqrt(z^2 + 4*(1-z)*q_i^2/Q) - z) / (2*(1-z))

    with ``Q = sum(q_i)``. ``z`` is found by bisection on the
    constraint ``sum(p_i) = 1``. Falls back to proportional devig if
    the book has zero or negative margin.
    """
    n = len(decimal_odds)
    if n == 0:
        return []
    q = [1.0 / o for o in decimal_odds]
    Q = sum(q)
    if Q <= 1.0 + 1e-9:
        return [qi / Q for qi in q]  # no margin to strip

    def p_given_z(z: float) -> List[float]:
        out = []
        for qi in q:
            inner = z * z + 4.0 * (1.0 - z) * qi * qi / Q
            root = math.sqrt(max(inner, 0.0))
            out.append((root - z) / (2.0 * (1.0 - z)))
        return out

    lo, hi = 0.0, 0.5
    for _ in range(max_iter):
        mid = 0.5 * (lo + hi)
        s = sum(p_given_z(mid))
        if abs(s - 1.0) < tol:
            return p_given_z(mid)
        if s > 1.0:
            lo = mid
        else:
            hi = mid
    return p_given_z(0.5 * (lo + hi))


def aggregate_best_odds(
    quotes: Iterable[FixtureQuote],
) -> List[FixtureQuote]:
    """Collapse many per-book quotes into one best-of-book quote per fixture.

    For every ``(home, away, market)`` triple the highest decimal price
    wins. Book name for the winning market is attached via the
    ``book`` field of the returned aggregated ``FixtureQuote``, but
    since a single aggregated quote cannot belong to one book, the
    book field is set to ``"best"`` and per-market provenance is
    encoded in a sibling list returned as the second tuple element
    if needed. For most callers the single-result form is sufficient.
    """
    best: Dict[Tuple[str, str, str], Tuple[float, str, str]] = {}
    for q in quotes:
        for market_name, odds in q.odds.items():
            if odds is None or odds <= 1.0:
                continue
            key = (q.home_team, q.away_team, market_name)
            current = best.get(key)
            if current is None or odds > current[0]:
                best[key] = (odds, q.book, q.kickoff)

    by_fixture: Dict[Tuple[str, str, str], Dict[str, float]] = {}
    for (h, a, m), (odds, _book, kickoff) in best.items():
        key = (h, a, kickoff)
        by_fixture.setdefault(key, {})[m] = odds

    return [
        FixtureQuote(
            home_team=h,
            away_team=a,
            book="best",
            kickoff=kickoff,
            odds=dict(markets),
        )
        for (h, a, kickoff), markets in by_fixture.items()
    ]


def format_bet(rec: BetRecommendation) -> str:
    """One-line human-readable summary of a bet recommendation."""
    book = f"@{rec.book}" if rec.book else ""
    return (
        f"{rec.fixture:32s}  {rec.market:13s}  "
        f"p={rec.model_prob:.3f}  odds={rec.decimal_odds:.2f}{book:>10s}  "
        f"edge={rec.edge * 100:+.1f}%  stake=£{rec.stake:.2f}"
    )


# =======================================================================
# Walk-forward backtest
# =======================================================================

def settle_bet(
    rec: BetRecommendation,
    home_goals: int,
    away_goals: int,
) -> Tuple[str, float]:
    """Settle a single recommendation against the real full-time result.

    Returns ``(outcome, pnl)`` where ``outcome`` is one of
    ``"win"``, ``"lose"``, ``"push"`` and ``pnl`` is the profit or
    loss on the stake (positive = win, negative = loss, zero = push).
    """
    diff = home_goals - away_goals
    total = home_goals + away_goals
    m = rec.market

    def won(win: bool) -> Tuple[str, float]:
        if win:
            return "win", round(rec.stake * (rec.decimal_odds - 1.0), 2)
        return "lose", -rec.stake

    def push_or(resolver: Callable[[], bool], is_push: Callable[[], bool]) -> Tuple[str, float]:
        if is_push():
            return "push", 0.0
        return won(resolver())

    if m == MARKET_HOME:
        return won(diff > 0)
    if m == MARKET_DRAW:
        return won(diff == 0)
    if m == MARKET_AWAY:
        return won(diff < 0)
    if m == MARKET_BTTS_YES:
        return won(home_goals > 0 and away_goals > 0)
    if m == MARKET_BTTS_NO:
        return won(not (home_goals > 0 and away_goals > 0))
    if m == MARKET_OVER_1_5:
        return won(total >= 2)
    if m == MARKET_UNDER_1_5:
        return won(total <= 1)
    if m == MARKET_OVER_2_5:
        return won(total >= 3)
    if m == MARKET_UNDER_2_5:
        return won(total <= 2)
    if m == MARKET_OVER_3_5:
        return won(total >= 4)
    if m == MARKET_UNDER_3_5:
        return won(total <= 3)
    if m == MARKET_DC_1X:
        return won(diff >= 0)
    if m == MARKET_DC_X2:
        return won(diff <= 0)
    if m == MARKET_DC_12:
        return won(diff != 0)
    if m == MARKET_DNB_HOME:
        return push_or(lambda: diff > 0, lambda: diff == 0)
    if m == MARKET_DNB_AWAY:
        return push_or(lambda: diff < 0, lambda: diff == 0)
    if m == MARKET_AH_HOME_M1:
        return push_or(lambda: diff >= 2, lambda: diff == 1)
    if m == MARKET_AH_AWAY_P1:
        return push_or(lambda: diff <= 0, lambda: diff == 1)
    if m == MARKET_AH_HOME_P1:
        return push_or(lambda: diff >= 0, lambda: diff == -1)
    if m == MARKET_AH_AWAY_M1:
        return push_or(lambda: diff <= -2, lambda: diff == -1)

    # Unknown market -- refund.
    return "push", 0.0


def walk_forward_backtest(
    matches: Sequence[Match],
    quotes_by_match: Mapping[int, List[FixtureQuote]],
    initial_train: int,
    retrain_every: int = 10,
    starting_bank: float = 1000.0,
    min_edge: float = 0.03,
    kelly_fraction: float = 0.25,
    max_stake_fraction: float = 0.03,
    max_total_stake_fraction: float = 0.30,
    half_life_days: Optional[float] = 365.0,
    flat_stake: Optional[float] = None,
    fit_kwargs: Optional[Dict[str, Any]] = None,
    picker_kwargs: Optional[Dict[str, Any]] = None,
    verbose: bool = True,
) -> BacktestResult:
    """Walk forward through ``matches``, retraining periodically.

    Parameters
    ----------
    matches : sequence of Match
        Must already be sorted by ``match.date`` ascending.
    quotes_by_match : mapping of index -> list of FixtureQuote
        The bookmaker panel available *at the time* the match was
        predicted. The key is the index into ``matches``.
    initial_train : int
        Number of matches used for the initial fit before any bet is
        considered. Typical values: 1-2 full seasons.
    retrain_every : int
        Refit the model after this many settled matches. 10 is cheap
        and gives a fresh model roughly every match week.
    starting_bank, min_edge, kelly_fraction,
    max_stake_fraction, max_total_stake_fraction :
        Forwarded to ``pick_bets_aggregated``.
    half_life_days : float, optional
        Time decay applied during every refit (recent matches weigh
        more).
    flat_stake : float, optional
        If set, all bets use this fixed unit size (in bank currency)
        instead of Kelly sizing. Kelly is still used as the *filter*
        -- a bet is only placed if Kelly would have recommended some
        stake -- but the actual wager is ``flat_stake``. This yields
        a cleaner ROI signal since it removes the compounding effect
        and is what most serious tipsters report.
    fit_kwargs, picker_kwargs : dict, optional
        Extra keyword arguments for ``PoissonBetPicker`` construction
        and ``.fit()``.

    Returns
    -------
    BacktestResult
    """
    if initial_train >= len(matches):
        raise ValueError("initial_train must be smaller than len(matches)")

    fit_kwargs = dict(fit_kwargs or {})
    picker_kwargs = dict(picker_kwargs or {})

    bank = starting_bank
    peak = starting_bank
    max_dd = 0.0
    history = [bank]
    total_staked = 0.0
    total_profit = 0.0
    num_bets = 0
    num_wins = 0
    num_pushes = 0
    num_losses = 0
    per_market: Dict[str, Dict[str, float]] = {}

    picker: Optional[PoissonBetPicker] = None
    matches_since_fit = retrain_every  # force an initial fit

    # Past-results tracking (every walked match, regardless of whether
    # we placed a bet on it).
    matches_scored = 0
    result_hits = 0
    result_rps_sum = 0.0
    result_log_loss_sum = 0.0

    for idx in range(initial_train, len(matches)):
        if matches_since_fit >= retrain_every or picker is None:
            picker = PoissonBetPicker(**picker_kwargs)
            ref_date = matches[idx - 1].date if idx > 0 else None
            picker.fit(
                matches[:idx],
                half_life_days=half_life_days,
                reference_date=ref_date,
                verbose=False,
                **fit_kwargs,
            )
            matches_since_fit = 0

        match = matches[idx]

        # Score this match out-of-sample regardless of whether we bet.
        try:
            probs = picker.probabilities(match.home_team, match.away_team)
        except KeyError:
            probs = None
        if probs is not None:
            matches_scored += 1
            if match.home_goals > match.away_goals:
                true = "H"
                p_true = probs.home
            elif match.home_goals == match.away_goals:
                true = "D"
                p_true = probs.draw
            else:
                true = "A"
                p_true = probs.away
            top = max(
                [("H", probs.home), ("D", probs.draw), ("A", probs.away)],
                key=lambda kv: kv[1],
            )
            if top[0] == true:
                result_hits += 1
            result_log_loss_sum += -math.log(max(p_true, 1e-9))
            p_cum = [probs.home, probs.home + probs.draw]
            if true == "H":
                o_cum = [1.0, 1.0]
            elif true == "D":
                o_cum = [0.0, 1.0]
            else:
                o_cum = [0.0, 0.0]
            result_rps_sum += 0.5 * sum(
                (p_cum[i] - o_cum[i]) ** 2 for i in range(2)
            )

        quotes = quotes_by_match.get(idx, [])
        if not quotes:
            matches_since_fit += 1
            continue

        # Sizing note: when flat_stake is used we still run Kelly as a
        # filter but replace the stake with a constant unit. We size
        # Kelly against the *starting* bank, not the running bank, so
        # both modes look at the same threshold and the only difference
        # is compounding vs flat.
        size_bank = starting_bank if flat_stake is not None else bank
        recs = picker.pick_bets_aggregated(
            quotes,
            bank=size_bank,
            min_edge=min_edge,
            kelly_fraction=kelly_fraction,
            max_stake_fraction=max_stake_fraction,
            max_total_stake_fraction=max_total_stake_fraction,
        )

        if flat_stake is not None:
            for rec in recs:
                rec.stake = flat_stake
                rec.expected_profit = round(flat_stake * rec.edge, 2)

        for rec in recs:
            if rec.stake <= 0:
                continue
            outcome, pnl = settle_bet(rec, match.home_goals, match.away_goals)
            bank += pnl
            history.append(bank)
            total_staked += rec.stake
            total_profit += pnl
            num_bets += 1
            if outcome == "win":
                num_wins += 1
            elif outcome == "push":
                num_pushes += 1
            else:
                num_losses += 1

            stats = per_market.setdefault(
                rec.market, {"bets": 0.0, "stake": 0.0, "profit": 0.0, "roi": 0.0}
            )
            stats["bets"] += 1
            stats["stake"] += rec.stake
            stats["profit"] += pnl

            if bank > peak:
                peak = bank
            dd = peak - bank
            if dd > max_dd:
                max_dd = dd

        matches_since_fit += 1
        if verbose and idx % max(1, (len(matches) - initial_train) // 10) == 0:
            print(
                f"  [{idx - initial_train + 1}/{len(matches) - initial_train}]  "
                f"bank=£{bank:,.2f}  bets={num_bets}  "
                f"roi={(total_profit / total_staked * 100) if total_staked else 0:+.2f}%"
            )

    for name, stats in per_market.items():
        stats["roi"] = stats["profit"] / stats["stake"] if stats["stake"] else 0.0

    return BacktestResult(
        starting_bank=starting_bank,
        final_bank=bank,
        total_staked=total_staked,
        total_profit=total_profit,
        num_bets=num_bets,
        num_wins=num_wins,
        num_pushes=num_pushes,
        num_losses=num_losses,
        hit_rate=num_wins / max(1, num_bets - num_pushes),
        roi=total_profit / total_staked if total_staked else 0.0,
        yield_pct=total_profit / num_bets if num_bets else 0.0,
        max_drawdown=max_dd,
        bank_history=history,
        per_market=per_market,
        matches_scored=matches_scored,
        result_hits=result_hits,
        result_rps=result_rps_sum / max(1, matches_scored),
        result_log_loss=result_log_loss_sum / max(1, matches_scored),
    )
