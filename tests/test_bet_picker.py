"""Smoke tests for the football bet picker."""

from __future__ import annotations

import math
import os
import tempfile
import unittest

from app.bet_picker import (
    MARKET_AH_HOME_M1,
    MARKET_AWAY,
    MARKET_DC_1X,
    MARKET_DNB_HOME,
    MARKET_HOME,
    MARKET_OVER_2_5,
    Fixture,
    FixtureQuote,
    Match,
    PoissonBetPicker,
    aggregate_best_odds,
    implied_probabilities,
    settle_bet,
    shin_devig,
    walk_forward_backtest,
    _poisson_pmf_vector,
)


def _toy_matches() -> list:
    """A tiny fabricated league with a clear strength gradient.

    Strong beats Medium beats Weak, at home and away, with a consistent
    goal gradient. Enough to verify the model picks up who is better.
    """
    matches = []
    pairings = [
        ("Strong", "Medium", 3, 1),
        ("Strong", "Weak",   4, 0),
        ("Medium", "Weak",   2, 0),
        ("Medium", "Strong", 0, 2),
        ("Weak",   "Strong", 0, 3),
        ("Weak",   "Medium", 1, 2),
    ]
    for home, away, hg, ag in pairings * 6:  # replicate to stabilise SGD
        matches.append(Match(home, away, hg, ag))
    return matches


class PoissonHelperTests(unittest.TestCase):

    def test_poisson_pmf_sums_to_one(self):
        pmf = _poisson_pmf_vector(1.5, cap=20)
        self.assertAlmostEqual(sum(pmf), 1.0, places=6)

    def test_poisson_pmf_matches_formula(self):
        lam = 2.3
        pmf = _poisson_pmf_vector(lam, cap=6)
        for k, got in enumerate(pmf):
            expected = math.exp(-lam) * lam ** k / math.factorial(k)
            self.assertAlmostEqual(got, expected, places=8)


class ImpliedProbabilityTests(unittest.TestCase):

    def test_strip_margin(self):
        ph, pd, pa, margin = implied_probabilities(2.0, 3.5, 4.0)
        self.assertAlmostEqual(ph + pd + pa, 1.0, places=6)
        self.assertGreater(margin, 0.0)
        # Home should still be the shortest implied price.
        self.assertGreater(ph, pd)
        self.assertGreater(ph, pa)


class BetPickerTrainingTests(unittest.TestCase):

    def setUp(self):
        self.picker = PoissonBetPicker(l2=1e-3, rho=0.0)
        self.picker.fit(
            _toy_matches(),
            epochs=300,
            lr=0.1,
            verbose=False,
            seed=7,
        )

    def test_teams_registered(self):
        self.assertEqual(
            set(self.picker.teams), {"Strong", "Medium", "Weak"}
        )

    def test_strength_ordering(self):
        """The model should rank Strong > Medium > Weak by net rating."""
        table = self.picker.team_table()
        order = [row[0] for row in table]
        self.assertEqual(order, ["Strong", "Medium", "Weak"])

    def test_probabilities_sum_to_one(self):
        p = self.picker.probabilities("Strong", "Weak")
        self.assertAlmostEqual(p.home + p.draw + p.away, 1.0, places=5)
        self.assertAlmostEqual(p.btts_yes + p.btts_no, 1.0, places=5)
        self.assertAlmostEqual(p.over_2_5 + p.under_2_5, 1.0, places=5)

    def test_strong_dominates_weak(self):
        p = self.picker.probabilities("Strong", "Weak")
        self.assertGreater(p.home, 0.6)
        self.assertGreater(p.home, p.away)

    def test_unknown_team_raises(self):
        with self.assertRaises(KeyError):
            self.picker.probabilities("Atlantis", "Weak")


class BetPickingTests(unittest.TestCase):

    def setUp(self):
        self.picker = PoissonBetPicker(l2=1e-3, rho=0.0)
        self.picker.fit(
            _toy_matches(), epochs=300, lr=0.1, verbose=False, seed=11,
        )

    def test_zero_edge_returns_nothing(self):
        # Price the fixture at the exact model probability.
        probs = self.picker.probabilities("Strong", "Weak")
        fx = Fixture(
            home_team="Strong", away_team="Weak",
            odds_home=1.0 / probs.home,   # zero edge
        )
        recs = self.picker.pick_bets([fx], bank=100.0, min_edge=0.01)
        self.assertEqual(recs, [])

    def test_positive_edge_produces_bet(self):
        probs = self.picker.probabilities("Strong", "Weak")
        overpriced_home = 1.0 / probs.home * 1.5  # 50% overround in our favour
        fx = Fixture(
            home_team="Strong", away_team="Weak",
            odds_home=overpriced_home,
        )
        recs = self.picker.pick_bets([fx], bank=100.0, min_edge=0.05)
        self.assertEqual(len(recs), 1)
        r = recs[0]
        self.assertEqual(r.market, "1X2:home")
        self.assertGreater(r.edge, 0.4)
        self.assertGreater(r.stake, 0.0)
        # Hard cap on single stake is 5% by default.
        self.assertLessEqual(r.stake, 5.0 + 1e-9)

    def test_negative_edge_ignored(self):
        probs = self.picker.probabilities("Strong", "Weak")
        fx = Fixture(
            home_team="Strong", away_team="Weak",
            odds_home=1.0 / probs.home * 0.5,  # bookie overpriced, negative edge
        )
        recs = self.picker.pick_bets([fx], bank=100.0)
        self.assertEqual(recs, [])


class PersistenceTests(unittest.TestCase):

    def test_save_load_round_trip(self):
        picker = PoissonBetPicker(l2=1e-3, rho=-0.05)
        picker.fit(
            _toy_matches(), epochs=50, lr=0.1, verbose=False, seed=3,
        )
        p_before = picker.probabilities("Strong", "Weak")

        with tempfile.TemporaryDirectory() as td:
            path = os.path.join(td, "picker.json")
            picker.save(path)
            loaded = PoissonBetPicker.load(path)

        p_after = loaded.probabilities("Strong", "Weak")
        self.assertAlmostEqual(p_before.home, p_after.home, places=6)
        self.assertAlmostEqual(p_before.draw, p_after.draw, places=6)
        self.assertAlmostEqual(p_before.away, p_after.away, places=6)
        self.assertEqual(loaded.teams, picker.teams)


class ExtendedMarketsTests(unittest.TestCase):
    """Double chance, DNB, Asian handicap, extra over/under lines."""

    def setUp(self):
        self.picker = PoissonBetPicker(l2=1e-3, rho=-0.05)
        self.picker.fit(
            _toy_matches(), epochs=200, lr=0.1, verbose=False, seed=19,
        )

    def test_double_chance_consistency(self):
        p = self.picker.probabilities("Strong", "Weak")
        self.assertAlmostEqual(p.dc_1x, p.home + p.draw, places=6)
        self.assertAlmostEqual(p.dc_x2, p.draw + p.away, places=6)
        self.assertAlmostEqual(p.dc_12, p.home + p.away, places=6)

    def test_over_under_lines_are_monotone(self):
        p = self.picker.probabilities("Strong", "Weak")
        self.assertGreaterEqual(p.over_1_5, p.over_2_5)
        self.assertGreaterEqual(p.over_2_5, p.over_3_5)
        self.assertAlmostEqual(p.over_1_5 + p.under_1_5, 1.0, places=5)
        self.assertAlmostEqual(p.over_3_5 + p.under_3_5, 1.0, places=5)

    def test_draw_no_bet_is_conditional(self):
        p = self.picker.probabilities("Strong", "Weak")
        expected = p.home / (1.0 - p.draw)
        self.assertAlmostEqual(p.dnb_home, expected, places=6)

    def test_asian_handicap_pushes_reported(self):
        p = self.picker.probabilities("Strong", "Weak")
        # Pushes are strictly positive when any draw-margin scores exist.
        self.assertGreaterEqual(p.ah_home_minus_1_push, 0.0)
        self.assertGreaterEqual(p.ah_away_plus_1_push, 0.0)
        # For AH home -1, push is exactly P(home wins by 1).
        self.assertAlmostEqual(
            p.ah_home_minus_1_push, p.ah_away_plus_1_push, places=6,
        )


class ShinDeVigTests(unittest.TestCase):

    def test_sums_to_one(self):
        ps = shin_devig([2.1, 3.4, 3.6])
        self.assertAlmostEqual(sum(ps), 1.0, places=6)

    def test_shin_sharpens_favourite(self):
        """Shin should shift probability away from longshots compared
        to the crude proportional de-vig."""
        odds = [1.50, 4.50, 8.00]
        prop_total = sum(1 / o for o in odds)
        prop = [1 / o / prop_total for o in odds]
        shin = shin_devig(odds)
        self.assertGreater(shin[0], prop[0])
        self.assertLess(shin[2], prop[2])


class AggregatorTests(unittest.TestCase):

    def setUp(self):
        self.picker = PoissonBetPicker(l2=1e-3, rho=0.0)
        self.picker.fit(
            _toy_matches(), epochs=300, lr=0.1, verbose=False, seed=5,
        )
        self.true = self.picker.probabilities("Strong", "Weak")

    def _book(self, name, home_odds):
        return FixtureQuote(
            home_team="Strong", away_team="Weak",
            book=name,
            odds={MARKET_HOME: home_odds},
        )

    def test_aggregate_best_odds_picks_max(self):
        quotes = [
            self._book("A", 1.80),
            self._book("B", 1.95),
            self._book("C", 1.70),
        ]
        agg = aggregate_best_odds(quotes)
        self.assertEqual(len(agg), 1)
        self.assertAlmostEqual(agg[0].odds[MARKET_HOME], 1.95, places=6)

    def test_aggregator_tags_best_book(self):
        fair = 1.0 / self.true.home
        quotes = [
            self._book("Short", fair * 0.95),   # negative-edge book
            self._book("Long",  fair * 1.20),   # positive-edge book
            self._book("Even",  fair),
        ]
        recs = self.picker.pick_bets_aggregated(
            quotes, bank=1000.0, min_edge=0.05,
        )
        self.assertEqual(len(recs), 1)
        self.assertEqual(recs[0].book, "Long")
        self.assertGreater(recs[0].edge, 0.1)

    def test_total_stake_cap_respected(self):
        fair = 1.0 / self.true.home
        quotes = [
            FixtureQuote(
                home_team="Strong", away_team="Weak", book="X",
                odds={
                    MARKET_HOME: fair * 2.0,
                    MARKET_OVER_2_5: fair * 2.0,
                    MARKET_DC_1X: fair * 2.0,
                },
            ),
        ]
        recs = self.picker.pick_bets_aggregated(
            quotes, bank=100.0, min_edge=0.01,
            max_total_stake_fraction=0.1,
        )
        total = sum(r.stake for r in recs)
        self.assertLessEqual(total, 10.0 + 1e-6)


class SettleBetTests(unittest.TestCase):

    def _rec(self, market, odds=2.0, stake=10.0):
        from app.bet_picker import BetRecommendation
        return BetRecommendation(
            fixture="Home v Away",
            market=market,
            model_prob=0.5,
            decimal_odds=odds,
            edge=0.0,
            kelly_fraction=0.0,
            stake=stake,
            expected_profit=0.0,
        )

    def test_home_win(self):
        outcome, pnl = settle_bet(self._rec(MARKET_HOME), 2, 1)
        self.assertEqual(outcome, "win")
        self.assertAlmostEqual(pnl, 10.0)

    def test_away_loses_on_home_win(self):
        outcome, pnl = settle_bet(self._rec(MARKET_AWAY), 2, 1)
        self.assertEqual(outcome, "lose")
        self.assertAlmostEqual(pnl, -10.0)

    def test_dnb_push_on_draw(self):
        outcome, pnl = settle_bet(self._rec(MARKET_DNB_HOME), 1, 1)
        self.assertEqual(outcome, "push")
        self.assertAlmostEqual(pnl, 0.0)

    def test_ah_home_minus_1_push_on_exact_1_goal_win(self):
        outcome, pnl = settle_bet(self._rec(MARKET_AH_HOME_M1), 1, 0)
        self.assertEqual(outcome, "push")
        self.assertAlmostEqual(pnl, 0.0)

    def test_ah_home_minus_1_win_on_2_plus(self):
        outcome, pnl = settle_bet(self._rec(MARKET_AH_HOME_M1), 3, 1)
        self.assertEqual(outcome, "win")
        self.assertAlmostEqual(pnl, 10.0)


class TimeDecayFitTests(unittest.TestCase):

    def test_recent_matches_outweigh_ancient(self):
        """A recent 180 matches where Weak always thumps Strong should
        flip the ranking compared to older equal-weighted data."""
        old = []
        for home, away, hg, ag in [
            ("Strong", "Weak",   3, 0),
            ("Weak",   "Strong", 0, 3),
        ] * 30:
            old.append(Match(home, away, hg, ag, date="2020-01-01"))

        # Much more recent, opposite-direction data.
        recent = []
        for home, away, hg, ag in [
            ("Strong", "Weak",   0, 3),
            ("Weak",   "Strong", 4, 0),
        ] * 60:
            recent.append(Match(home, away, hg, ag, date="2024-05-01"))

        picker = PoissonBetPicker(l2=1e-4, rho=0.0)
        picker.fit(
            old + recent,
            epochs=400, lr=0.1, verbose=False,
            half_life_days=90.0,
            reference_date="2024-06-01",
            seed=13,
        )
        p = picker.probabilities("Strong", "Weak")
        # With a 90-day half life and the recent block favouring Weak,
        # Weak should be priced ahead of Strong.
        self.assertGreater(p.away, p.home)


class BacktestTests(unittest.TestCase):

    def test_walk_forward_runs_and_scores(self):
        """Mini smoke test of the full walk-forward loop."""
        from datetime import datetime, timedelta
        import random as _r

        teams = ["A", "B", "C", "D"]
        # Strengths: A > B > C > D
        atk = {"A": 0.6, "B": 0.2, "C": -0.2, "D": -0.6}
        dfc = {"A": -0.3, "B": -0.1, "C": 0.1, "D": 0.3}
        rng = _r.Random(42)
        start = datetime(2023, 1, 1)
        matches = []
        for week in range(30):
            date = (start + timedelta(days=week * 7)).date().isoformat()
            for h in teams:
                for a in teams:
                    if h == a:
                        continue
                    lh = math.exp(0.3 + atk[h] + dfc[a] + 0.2)
                    la = math.exp(0.3 + atk[a] + dfc[h])
                    hg = min(rng.expovariate(1.0 / lh).__floor__(), 6)
                    ag = min(rng.expovariate(1.0 / la).__floor__(), 6)
                    matches.append(Match(h, a, hg, ag, date=date))
        matches.sort(key=lambda m: m.date)

        # Build a trivial 2-book panel with different margins.
        quotes = {}
        for idx, m in enumerate(matches):
            lh = math.exp(0.3 + atk[m.home_team] + dfc[m.away_team] + 0.2)
            la = math.exp(0.3 + atk[m.away_team] + dfc[m.home_team])
            # Coarse fair probs
            ph = lh / (lh + la + 0.8)
            pa = la / (lh + la + 0.8)
            pd = max(0.01, 1 - ph - pa)
            quotes[idx] = [
                FixtureQuote(
                    m.home_team, m.away_team, book="A",
                    odds={
                        MARKET_HOME: round(1 / (ph * 1.04), 2),
                        "1X2:draw": round(1 / (pd * 1.04), 2),
                        MARKET_AWAY: round(1 / (pa * 1.04), 2),
                    },
                ),
                FixtureQuote(
                    m.home_team, m.away_team, book="B",
                    odds={
                        MARKET_HOME: round(1 / (ph * 1.07), 2),
                        "1X2:draw": round(1 / (pd * 1.07), 2),
                        MARKET_AWAY: round(1 / (pa * 1.07), 2),
                    },
                ),
            ]

        result = walk_forward_backtest(
            matches,
            quotes,
            initial_train=len(matches) // 2,
            retrain_every=20,
            starting_bank=500.0,
            min_edge=0.01,
            flat_stake=1.0,
            half_life_days=180.0,
            fit_kwargs={"epochs": 100, "lr": 0.1},
            verbose=False,
        )
        # Smoke properties: runs, records some matches, numbers are
        # finite.
        self.assertGreater(result.matches_scored, 0)
        self.assertGreaterEqual(result.hit_rate, 0.0)
        self.assertLessEqual(result.hit_rate, 1.0)
        self.assertFalse(math.isnan(result.roi))


if __name__ == "__main__":
    unittest.main()
