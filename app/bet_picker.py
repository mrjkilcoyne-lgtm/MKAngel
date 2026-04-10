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
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple


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
    season : str, optional
        Season tag (e.g. "2023-24"). Used only for bookkeeping.
    weight : float
        Sample weight -- use <1.0 to down-weight older seasons.
    """

    home_team: str
    away_team: str
    home_goals: int
    away_goals: int
    season: str = ""
    weight: float = 1.0


@dataclass
class Fixture:
    """An upcoming match to price and potentially bet on.

    Odds are in decimal (UK standard): ``2.50`` means a £1 stake
    returns £2.50 total (£1.50 profit) if the bet wins. Missing
    markets can be left as ``None``.
    """

    home_team: str
    away_team: str
    kickoff: str = ""              # free-form, e.g. "2024-08-17 15:00"
    odds_home: Optional[float] = None
    odds_draw: Optional[float] = None
    odds_away: Optional[float] = None
    odds_btts_yes: Optional[float] = None
    odds_btts_no: Optional[float] = None
    odds_over_2_5: Optional[float] = None
    odds_under_2_5: Optional[float] = None


@dataclass
class BetRecommendation:
    """A single recommended bet."""

    fixture: str                   # "Home v Away"
    market: str                    # "1X2:home", "BTTS:yes", "O/U2.5:over", ...
    model_prob: float              # fair probability
    decimal_odds: float            # bookmaker price
    edge: float                    # model_prob * odds - 1
    kelly_fraction: float          # full Kelly stake as fraction of bank
    stake: float                   # recommended stake in currency
    expected_profit: float         # stake * edge


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

    def as_dict(self) -> Dict[str, float]:
        return {
            "home": self.home,
            "draw": self.draw,
            "away": self.away,
            "btts_yes": self.btts_yes,
            "btts_no": self.btts_no,
            "over_2_5": self.over_2_5,
            "under_2_5": self.under_2_5,
            "xg_home": self.expected_home_goals,
            "xg_away": self.expected_away_goals,
        }


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
    ) -> Dict[str, float]:
        """Fit model parameters by SGD on negative log-likelihood.

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

        for epoch in range(epochs):
            rng.shuffle(match_list)
            total_loss = 0.0
            total_weight = 0.0

            # Per-epoch gradient accumulators
            g_attack = [0.0] * n_teams
            g_defence = [0.0] * n_teams
            g_ha = 0.0
            g_base = 0.0

            for m in match_list:
                h = self.team_index[m.home_team]
                a = self.team_index[m.away_team]
                lh, la = self._lambdas(h, a)

                gh = m.home_goals
                ga = m.away_goals
                w = m.weight

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
        """Compute all market probabilities for a fixture."""
        matrix = self.score_matrix(home, away)
        cap = self.goal_cap

        p_home = 0.0
        p_draw = 0.0
        p_away = 0.0
        p_btts_yes = 0.0
        p_over_2_5 = 0.0

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
                if i + j >= 3:
                    p_over_2_5 += p

        # Tidy up rounding drift.
        p_total = p_home + p_draw + p_away
        if p_total > 0:
            p_home /= p_total
            p_draw /= p_total
            p_away /= p_total

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
            over_2_5=p_over_2_5,
            under_2_5=max(0.0, 1.0 - p_over_2_5),
            expected_home_goals=lh,
            expected_away_goals=la,
            correct_score=top_scores,
        )

    # -------------------------------------------------------------------
    # Bet selection
    # -------------------------------------------------------------------

    def pick_bets(
        self,
        fixtures: Iterable[Fixture],
        bank: float = 100.0,
        min_edge: float = 0.03,
        kelly_fraction: float = 0.25,
        max_stake_fraction: float = 0.05,
    ) -> List[BetRecommendation]:
        """Score fixtures against bookmaker odds and recommend bets.

        Parameters
        ----------
        fixtures : iterable of Fixture
            Upcoming matches with at least one market price set.
        bank : float
            Bankroll in pounds (or any currency).
        min_edge : float
            Minimum required edge (model_prob * odds - 1). Bets below
            this are ignored -- accounts for model uncertainty.
        kelly_fraction : float
            Fraction of full Kelly to stake. 0.25 (quarter-Kelly) is a
            standard safe default; full Kelly is too aggressive.
        max_stake_fraction : float
            Hard cap on any single stake as a fraction of bank. Prevents
            the model from ever recommending more than, e.g., 5% on one
            bet, regardless of edge size.
        """
        if bank <= 0:
            return []

        recs: List[BetRecommendation] = []

        for fx in fixtures:
            try:
                probs = self.probabilities(fx.home_team, fx.away_team)
            except KeyError:
                continue
            fixture_label = f"{fx.home_team} v {fx.away_team}"

            markets = [
                ("1X2:home", probs.home, fx.odds_home),
                ("1X2:draw", probs.draw, fx.odds_draw),
                ("1X2:away", probs.away, fx.odds_away),
                ("BTTS:yes", probs.btts_yes, fx.odds_btts_yes),
                ("BTTS:no", probs.btts_no, fx.odds_btts_no),
                ("O/U2.5:over", probs.over_2_5, fx.odds_over_2_5),
                ("O/U2.5:under", probs.under_2_5, fx.odds_under_2_5),
            ]

            for name, p, odds in markets:
                if odds is None or odds <= 1.0 or p <= 0.0:
                    continue
                edge = p * odds - 1.0
                if edge < min_edge:
                    continue

                # Kelly fraction for a binary bet at decimal odds D:
                #   f* = (p * (D - 1) - (1 - p)) / (D - 1)
                #      = (p * D - 1) / (D - 1)
                full_kelly = (p * odds - 1.0) / (odds - 1.0)
                if full_kelly <= 0:
                    continue

                f = full_kelly * kelly_fraction
                f = min(f, max_stake_fraction)
                stake = round(bank * f, 2)
                if stake <= 0:
                    continue

                recs.append(BetRecommendation(
                    fixture=fixture_label,
                    market=name,
                    model_prob=p,
                    decimal_odds=odds,
                    edge=edge,
                    kelly_fraction=f,
                    stake=stake,
                    expected_profit=round(stake * edge, 2),
                ))

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


# =======================================================================
# Market utilities
# =======================================================================

def implied_probabilities(
    odds_home: float,
    odds_draw: float,
    odds_away: float,
) -> Tuple[float, float, float, float]:
    """Strip bookmaker margin from a 1X2 price.

    Returns (p_home, p_draw, p_away, overround) where ``overround``
    is the bookmaker edge (e.g. 0.05 means a 5% margin).
    """
    raw_h = 1.0 / odds_home
    raw_d = 1.0 / odds_draw
    raw_a = 1.0 / odds_away
    total = raw_h + raw_d + raw_a
    return (raw_h / total, raw_d / total, raw_a / total, total - 1.0)


def format_bet(rec: BetRecommendation) -> str:
    """One-line human-readable summary of a bet recommendation."""
    return (
        f"{rec.fixture:35s}  {rec.market:14s}  "
        f"p={rec.model_prob:.3f}  odds={rec.decimal_odds:.2f}  "
        f"edge={rec.edge * 100:+.1f}%  stake=£{rec.stake:.2f}"
    )
