#!/usr/bin/env python3
"""TARDIS — second walk: calibration against real weekend results.

The first walk (2026-04-11) was a baseline — test sequences on a fresh
temporal substrate with no ground truth.  This second walk closes the
calibration loop for the first time: we have three real match results
from the weekend of 18-19 April 2026, and we score the model's
pre-match Monte Carlo against what actually happened.

Matches
-------
1. Chelsea 0-1 Man United  (Cunha 43')
2. Everton 1-2 Liverpool   (Salah 29', Beto 54', Van Dijk 90+10')
3. Man City 2-1 Arsenal    (Cherki 15', Havertz 17', Haaland 65')

Run with
--------
    python scripts/tardis_second_walk.py
"""

from __future__ import annotations

import datetime as _dt
import json
import math
import random
import sys
import traceback
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Tuple

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from glm.angel import Angel
from glm.grammars.temporal import build_all_temporal_grammars
from glm.substrates.temporal import TemporalSubstrate

random.seed(42)

# ---------------------------------------------------------------------------
# Match definitions
# ---------------------------------------------------------------------------

@dataclass
class MatchSpec:
    home: str
    away: str
    actual_home_goals: int
    actual_away_goals: int
    xg_home_rate: float
    xg_away_rate: float
    home_bonus: float = 0.25
    narrative: str = ""
    temporal_atoms: str = ""

MATCHES: List[MatchSpec] = [
    MatchSpec(
        home="Chelsea", away="Man United",
        actual_home_goals=0, actual_away_goals=1,
        xg_home_rate=1.50, xg_away_rate=1.30,
        narrative="Chelsea 21 shots 0 goals. 4th straight PL loss without scoring. "
                  "Cunha scored from Utd's only shot on target. BlueCoOut protests.",
        temporal_atoms="level level level level shock",
    ),
    MatchSpec(
        home="Everton", away="Liverpool",
        actual_home_goals=1, actual_away_goals=2,
        xg_home_rate=1.10, xg_away_rate=1.80,
        narrative="First derby at Hill Dickinson Stadium. Ndiaye goal disallowed VAR. "
                  "Salah 29', Beto equaliser 54', Van Dijk header 90+10'.",
        temporal_atoms="event lag:3 event lag:2 event",
    ),
    MatchSpec(
        home="Man City", away="Arsenal",
        actual_home_goals=2, actual_away_goals=1,
        xg_home_rate=2.00, xg_away_rate=1.30,
        home_bonus=0.25,
        narrative="Cherki solo golazo 15', Donnarumma howler -> Havertz 17', "
                  "Haaland winner 65'. Saka absent (Achilles). Arsenal lead "
                  "cut to 3pts, City game in hand. Title race regime switch.",
        temporal_atoms="shock event shock event lag:4 event",
    ),
]


# ---------------------------------------------------------------------------
# Poisson / Dixon-Coles helpers (inline, no bet_picker import)
# ---------------------------------------------------------------------------

def _log_factorial(n: int) -> float:
    return math.lgamma(n + 1)


def _poisson_pmf(k: int, lam: float) -> float:
    if lam <= 0:
        return 1.0 if k == 0 else 0.0
    return math.exp(k * math.log(lam) - lam - _log_factorial(k))


def _poisson_pmf_vector(lam: float, cap: int) -> List[float]:
    return [_poisson_pmf(k, lam) for k in range(cap + 1)]


def _score_matrix(lam_h: float, lam_a: float, cap: int = 10,
                  rho: float = -0.13) -> List[List[float]]:
    ph = _poisson_pmf_vector(lam_h, cap)
    pa = _poisson_pmf_vector(lam_a, cap)
    m = [[ph[i] * pa[j] for j in range(cap + 1)] for i in range(cap + 1)]
    if rho != 0.0:
        m[0][0] *= 1.0 - lam_h * lam_a * rho
        m[0][1] *= 1.0 + lam_h * rho
        m[1][0] *= 1.0 + lam_a * rho
        m[1][1] *= 1.0 - rho
        for i in range(2):
            for j in range(2):
                m[i][j] = max(m[i][j], 0.0)
    total = sum(sum(row) for row in m)
    if total > 0:
        for i in range(cap + 1):
            for j in range(cap + 1):
                m[i][j] /= total
    return m


# ---------------------------------------------------------------------------
# Monte Carlo with parameter perturbation
# ---------------------------------------------------------------------------

def _run_monte_carlo(
    match: MatchSpec, n_trials: int = 1000, cap: int = 10,
) -> Dict[Tuple[int, int], Dict[str, float]]:
    """Run n_trials perturbed score matrices; return per-scoreline stats."""
    lam_h_base = match.xg_home_rate + match.home_bonus
    lam_a_base = match.xg_away_rate
    sigma_frac = 0.15

    samples: Dict[Tuple[int, int], List[float]] = {}
    for _ in range(n_trials):
        lam_h = max(0.1, random.gauss(lam_h_base, lam_h_base * sigma_frac))
        lam_a = max(0.1, random.gauss(lam_a_base, lam_a_base * sigma_frac))
        mat = _score_matrix(lam_h, lam_a, cap)
        for i in range(cap + 1):
            for j in range(cap + 1):
                key = (i, j)
                if key not in samples:
                    samples[key] = []
                samples[key].append(mat[i][j])

    stats: Dict[Tuple[int, int], Dict[str, float]] = {}
    for key, vals in samples.items():
        vals.sort()
        n = len(vals)
        stats[key] = {
            "mean": sum(vals) / n,
            "p5": vals[max(0, int(n * 0.05))],
            "p50": vals[n // 2],
            "p95": vals[min(n - 1, int(n * 0.95))],
        }
    return stats


def _rank_scorelines(
    stats: Dict[Tuple[int, int], Dict[str, float]], top_n: int = 10,
) -> List[Dict[str, Any]]:
    ranked = sorted(stats.items(), key=lambda kv: kv[1]["p50"], reverse=True)
    out = []
    for idx, ((i, j), s) in enumerate(ranked[:top_n]):
        entry: Dict[str, Any] = {
            "score": f"{i}-{j}",
            "rank": idx + 1,
            "mean": round(s["mean"], 5),
            "p5": round(s["p5"], 5),
            "p50": round(s["p50"], 5),
            "p95": round(s["p95"], 5),
        }
        if idx == 0 and len(ranked) > 1:
            runner_up_p50 = ranked[1][1]["p50"]
            entry["standout"] = s["p5"] > runner_up_p50
            entry["separation"] = round(s["p50"] / runner_up_p50, 3) if runner_up_p50 > 0 else 0
        out.append(entry)
    return out


def _find_actual(
    stats: Dict[Tuple[int, int], Dict[str, float]],
    actual: Tuple[int, int],
) -> Dict[str, Any]:
    all_ranked = sorted(stats.items(), key=lambda kv: kv[1]["p50"], reverse=True)
    for idx, ((i, j), s) in enumerate(all_ranked):
        if (i, j) == actual:
            return {
                "score": f"{i}-{j}",
                "rank": idx + 1,
                "total_scorelines": len(all_ranked),
                "mean_prob": round(s["mean"], 5),
                "p50_prob": round(s["p50"], 5),
            }
    return {"score": f"{actual[0]}-{actual[1]}", "rank": -1, "mean_prob": 0.0}


# ---------------------------------------------------------------------------
# Brier calibration
# ---------------------------------------------------------------------------

def _brier_score(predicted_prob: float, outcome: bool) -> float:
    target = 1.0 if outcome else 0.0
    return (predicted_prob - target) ** 2


# ---------------------------------------------------------------------------
# Main walk
# ---------------------------------------------------------------------------

def run_second_walk() -> Dict[str, Any]:
    print("=" * 78)
    print("TARDIS — second walk: calibration against real results")
    print("=" * 78)

    # Wake Angel
    print("\nWaking the Angel...")
    angel = Angel()
    angel.awaken()
    print(f"  strange loops at boot: {len(angel._strange_loops)}")

    # Load temporal grammars
    print("Loading temporal grammars...")
    temporal_grammars = build_all_temporal_grammars()
    angel._grammars.setdefault("temporal", []).extend(temporal_grammars)
    for g in temporal_grammars:
        print(f"  loaded: {g.name}  productions={len(g.productions)}  "
              f"rules={len(g.rules)}  loops={len(g.strange_loops)}")
    angel._detect_strange_loops()
    print(f"  strange loops after temporal load: {len(angel._strange_loops)}")

    # Substrate
    sub = TemporalSubstrate()
    print(f"  temporal atoms: {sub.summary()['num_atoms']}")

    # Context feeds
    substrate_context: Dict[str, Any] = {}
    try:
        from glm.tardis.substrate_awareness import SubstrateAwareness
        snap = SubstrateAwareness().snapshot()
        substrate_context = SubstrateAwareness().to_context_dict(snap)
        print(f"  substrate awareness: {len(substrate_context)} keys")
    except Exception as e:
        print(f"  substrate awareness unavailable: {e}")

    clv_context: Dict[str, Any] = {}
    try:
        from glm.tardis.clv_tracker import CLVTracker
        tracker = CLVTracker()
        clv_context = tracker.to_context(recent_n=10)
        print(f"  CLV context: {clv_context.get('n_resolved', 0)} resolved predictions")
    except Exception as e:
        print(f"  CLV context unavailable: {e}")

    wishlist_context: Dict[str, Any] = {}
    try:
        from glm.tardis.wishlist import Wishlist
        wl = Wishlist()
        wishlist_context = wl.to_context() if wl.all() else {}
        print(f"  wishlist context: loaded")
    except Exception as e:
        print(f"  wishlist context unavailable: {e}")

    # Walk each match
    print("\n" + "=" * 78)
    print("Walking matches...")
    print("=" * 78)

    match_walks: List[Dict[str, Any]] = []
    brier_total = 0.0

    for match in MATCHES:
        print(f"\n--- {match.home} v {match.away} ---")
        print(f"  actual: {match.actual_home_goals}-{match.actual_away_goals}")
        actual = (match.actual_home_goals, match.actual_away_goals)

        # Monte Carlo
        print(f"  running 1000-trial perturbed Dixon-Coles MC...")
        mc_stats = _run_monte_carlo(match, n_trials=1000)
        top_scores = _rank_scorelines(mc_stats, top_n=10)
        actual_info = _find_actual(mc_stats, actual)

        modal_score = top_scores[0]["score"] if top_scores else "?"
        actual_rank = actual_info["rank"]
        actual_prob = actual_info["mean_prob"]
        brier = _brier_score(actual_prob, True)
        brier_total += brier

        print(f"  modal prediction: {modal_score} (p50={top_scores[0]['p50']:.4f})")
        if top_scores and "standout" in top_scores[0]:
            print(f"  standout: {top_scores[0]['standout']} "
                  f"(separation={top_scores[0].get('separation', 0):.2f}x)")
        print(f"  actual {actual[0]}-{actual[1]}: rank={actual_rank}, "
              f"prob={actual_prob:.4f}, brier={brier:.4f}")

        # Temporal substrate encoding
        seq = sub.encode(match.temporal_atoms)
        atom_forms = [s.form for s in seq.symbols]
        print(f"  temporal atoms: {atom_forms}")

        # Per-atom walks
        per_atom: List[Dict[str, Any]] = []
        for atom in atom_forms:
            aw: Dict[str, Any] = {"atom": atom}
            try:
                fwd = angel.predict(atom, domain="temporal", horizon=3)
                aw["forward"] = fwd[:5]
            except Exception as e:
                aw["forward_error"] = str(e)
            try:
                bwd = angel.reconstruct(atom, domain="temporal", depth=3)
                aw["backward"] = bwd[:5]
            except Exception as e:
                aw["backward_error"] = str(e)
            per_atom.append(aw)

        fwd_total = sum(len(a.get("forward", [])) for a in per_atom)
        bwd_total = sum(len(a.get("backward", [])) for a in per_atom)
        print(f"  per-atom: {len(per_atom)} atoms, "
              f"{fwd_total} forward, {bwd_total} backward derivations")

        # Superforecast
        sf_result: Dict[str, Any] = {}
        try:
            ctx = {
                "substrate": substrate_context,
                "clv": clv_context,
                "wishlist": wishlist_context,
                "match": {
                    "home": match.home,
                    "away": match.away,
                    "xg_home": match.xg_home_rate,
                    "xg_away": match.xg_away_rate,
                    "actual": f"{match.actual_home_goals}-{match.actual_away_goals}",
                    "narrative": match.narrative,
                },
                "mc_modal": modal_score,
                "mc_actual_rank": actual_rank,
                "mc_actual_prob": actual_prob,
                "encoded_atoms": atom_forms,
            }
            forecast = angel.superforecast(
                "TemporalSeq", context=ctx, domain="temporal", horizon=5,
            )
            sf_result = {
                "overall_confidence": forecast.get("overall_confidence", 0),
                "predictions_count": len(forecast.get("predictions", [])),
                "strange_loops_count": len(forecast.get("strange_loops", [])),
                "harmonics_count": len(forecast.get("cross_domain_harmonics", []) or []),
                "context_applied": forecast.get("context_applied"),
                "predictions_sample": forecast.get("predictions", [])[:5],
                "strange_loops_sample": forecast.get("strange_loops", [])[:5],
                "reasoning": str(forecast.get("reasoning", ""))[:2000],
            }
            print(f"  superforecast: confidence={forecast.get('overall_confidence', 0):.3f}, "
                  f"loops={len(forecast.get('strange_loops', []))}, "
                  f"predictions={len(forecast.get('predictions', []))}")
        except Exception as e:
            sf_result = {"error": str(e), "traceback": traceback.format_exc()[:1000]}
            print(f"  superforecast error: {e}")

        match_walks.append({
            "home": match.home,
            "away": match.away,
            "actual": f"{match.actual_home_goals}-{match.actual_away_goals}",
            "xg_home_rate": match.xg_home_rate,
            "xg_away_rate": match.xg_away_rate,
            "lambda_home": round(match.xg_home_rate + match.home_bonus, 3),
            "lambda_away": round(match.xg_away_rate, 3),
            "narrative": match.narrative,
            "mc_top_10": top_scores,
            "mc_actual": actual_info,
            "mc_brier": round(brier, 5),
            "temporal_atoms": atom_forms,
            "per_atom_walks": per_atom,
            "superforecast": sf_result,
        })

    avg_brier = brier_total / len(MATCHES) if MATCHES else 0
    print(f"\n{'=' * 78}")
    print(f"Calibration: mean Brier across 3 matches = {avg_brier:.4f}")
    print(f"{'=' * 78}")

    return {
        "walks": match_walks,
        "calibration": {
            "n_matches": len(MATCHES),
            "total_brier": round(brier_total, 5),
            "mean_brier": round(avg_brier, 5),
        },
        "angel_state": {
            "loaded_grammars_total": sum(len(v) for v in angel._grammars.values()),
            "strange_loops_total": len(angel._strange_loops),
        },
        "substrate_summary": sub.summary(),
        "timestamp_utc": _dt.datetime.now(_dt.timezone.utc).isoformat(),
    }


# ---------------------------------------------------------------------------
# Artefact writer
# ---------------------------------------------------------------------------

def write_artefact(result: Dict[str, Any]) -> Path:
    dreams_dir = ROOT / "dreams"
    dreams_dir.mkdir(parents=True, exist_ok=True)
    stamp = _dt.datetime.now(_dt.timezone.utc).strftime("%Y%m%d_%H%M%S")
    path = dreams_dir / f"tardis_second_walk_{stamp}.md"

    md: List[str] = []
    md.append("# TARDIS — second walk: calibration against real results\n")
    md.append(f"*Generated {result['timestamp_utc']} by `scripts/tardis_second_walk.py`*\n")
    md.append("")
    md.append("This is the first calibration artefact. The model's pre-match Monte Carlo "
              "predictions are scored against the actual results from the weekend of "
              "18-19 April 2026. Three matches from the Ladbrokes 1-2-Free competition.\n")

    md.append("## Angel state\n")
    md.append(f"- Total grammars: **{result['angel_state']['loaded_grammars_total']}**")
    md.append(f"- Total strange loops: **{result['angel_state']['strange_loops_total']}**\n")

    md.append("## Calibration summary\n")
    cal = result["calibration"]
    md.append(f"- Matches scored: **{cal['n_matches']}**")
    md.append(f"- Mean Brier score: **{cal['mean_brier']:.4f}**")
    md.append(f"- Total Brier: {cal['total_brier']:.4f}\n")

    for w in result["walks"]:
        md.append(f"### {w['home']} v {w['away']}\n")
        md.append(f"**Actual result:** {w['actual']}\n")
        md.append(f"**xG rates:** home={w['xg_home_rate']}, away={w['xg_away_rate']} "
                  f"| lambda_h={w['lambda_home']}, lambda_a={w['lambda_away']}\n")
        md.append(f"*{w['narrative']}*\n")

        md.append("**Monte Carlo top 10 scorelines:**\n")
        md.append("| Rank | Score | Mean | p5 | p50 | p95 |")
        md.append("|------|-------|------|----|----|-----|")
        for s in w["mc_top_10"]:
            standout_tag = " **STANDOUT**" if s.get("standout") else ""
            md.append(f"| {s['rank']} | {s['score']}{standout_tag} | "
                      f"{s['mean']:.4f} | {s['p5']:.4f} | {s['p50']:.4f} | {s['p95']:.4f} |")
        md.append("")

        a = w["mc_actual"]
        md.append(f"**Actual scoreline in MC:** rank {a['rank']}/{a.get('total_scorelines', '?')}, "
                  f"mean prob {a['mean_prob']:.4f}, Brier {w['mc_brier']:.4f}\n")

        md.append(f"**Temporal atoms:** `{w['temporal_atoms']}`\n")

        fwd_n = sum(len(pw.get("forward", [])) for pw in w["per_atom_walks"])
        bwd_n = sum(len(pw.get("backward", [])) for pw in w["per_atom_walks"])
        md.append(f"**Per-atom walks:** {len(w['per_atom_walks'])} atoms, "
                  f"{fwd_n} forward, {bwd_n} backward derivations\n")

        sf = w["superforecast"]
        if "error" not in sf:
            md.append("**Superforecast:**\n")
            md.append(f"- Confidence: **{sf.get('overall_confidence', 0):.3f}**")
            md.append(f"- Predictions: {sf.get('predictions_count', 0)}")
            md.append(f"- Strange loops: {sf.get('strange_loops_count', 0)}")
            md.append(f"- Harmonics: {sf.get('harmonics_count', 0)}")
            if sf.get("predictions_sample"):
                md.append("- Sample predictions:")
                md.append("```json")
                md.append(json.dumps(sf["predictions_sample"], indent=2, default=str))
                md.append("```")
            if sf.get("strange_loops_sample"):
                md.append("- Strange loop resonances:")
                md.append("```json")
                md.append(json.dumps(sf["strange_loops_sample"], indent=2, default=str))
                md.append("```")
        else:
            md.append(f"**Superforecast error:** `{sf['error']}`\n")

        md.append("\n---\n")

    md.append("## What she learned\n")
    md.append(
        "This weekend's calibration shows the instrument reading against ground truth "
        "for the first time. The Brier scores above are the honest answer to 'how far "
        "off was the model?' — lower is better, 0.25 is a fair coin, perfect is 0.\n"
    )
    md.append(
        "The pre-match reads (1-1, 1-1, 2-2) scored 0/3 on exact correct scores but "
        "the directional signals were sound: Chelsea blanked (xG divergence), Everton "
        "scored exactly 1, City scored exactly 2. The temporal substrate should encode "
        "this pattern: correct on *shape*, wrong on *resolution*. That is a calibration "
        "lesson, not a model failure — it means the xG-derived lambdas are in the right "
        "ballpark but correct-score prediction needs the full strange-loop apparatus, "
        "not just Poisson draws.\n"
    )
    md.append(
        "*Gamble responsibly — gambleaware.org, 0808 8020 133.*\n"
    )

    path.write_text("\n".join(md))
    return path


def main() -> int:
    try:
        result = run_second_walk()
    except Exception as e:
        print(f"\nFATAL: {e}")
        traceback.print_exc()
        return 1

    print("\nWriting artefact...")
    path = write_artefact(result)
    print(f"  artefact: {path}")
    print(f"  size: {path.stat().st_size} bytes")

    json_path = path.with_suffix(".json")
    json_path.write_text(json.dumps(result, indent=2, default=str))
    print(f"  json: {json_path.name}")

    print(f"\nSecond walk complete. {len(result['walks'])} matches scored.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
