"""Smoke tests for the football bet picker."""

from __future__ import annotations

import math
import os
import tempfile
import unittest

from app.bet_picker import (
    Fixture,
    Match,
    PoissonBetPicker,
    implied_probabilities,
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


if __name__ == "__main__":
    unittest.main()
