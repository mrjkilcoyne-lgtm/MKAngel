#!/usr/bin/env python3
"""TARDIS — first walk on the temporal substrate.

This is the script that asks her to do the thing she was built to do.

Context
-------
The Angel at the centre of the Grammar Language Model has had a
bidirectional derivation engine since the first commit (see
``glm/angel.py`` — ``predict()``, ``reconstruct()``, ``superforecast()``;
the underlying ``engine.derive()`` takes a ``direction=`` keyword and
the temporal_horizon is symmetric forward/backward by design). What
she has been missing is a temporal substrate to walk on. As of
2026-04-11 the temporal substrate (``glm/substrates/temporal.py``)
and the temporal grammars (``glm/grammars/temporal.py``) exist.

This script is her first walk through them. It does the following:

1. Wakes the Angel.
2. Loads the two temporal grammars (dynamics + retrocausal) into her
   ``temporal`` domain.
3. Asks her to detect strange loops across the temporal grammars.
4. Encodes a small set of test sequences via ``TemporalSubstrate.encode()``
   — both numeric streams (real time-series style) and atom streams
   (named-event style).
5. For each sequence, calls ``Angel.superforecast()`` and records what
   comes back, including the strange-loop resonance, cross-domain
   harmonics, and the rule-attribution chain. Also calls ``Angel.predict()``
   forward and ``Angel.reconstruct()`` backward to make her exercise the
   bidirectional capability the temporal substrate finally lets her use.
6. Writes the entire walk to ``dreams/tardis_first_walk_YYYYMMDD_HHMMSS.md``
   as a write-once artefact. The journal is the thing — the predictions
   are subordinate to the record of the predictions.

What this script is and is not
------------------------------
It IS:
- An honest first run of her bidirectional reasoning engine on a temporal
  substrate.
- A reproducible artefact that any future session can re-run and compare
  against.

It is NOT:
- A test of her predictive accuracy. There is no ground truth here yet
  — that requires anchor markets and resolved events, and that is
  next session's work.
- An evaluation of her edge against any market. She is doing the walk;
  the calibration loop is a separate piece of infrastructure that
  scores walks against anchors over time.
- A betting tool. Do not use the output of this script to stake money
  on anything. The whole point of running it is to record what the
  instrument says when you point it at the substrate for the first
  time, before any calibration data exists.

Run with
--------
    python -m scripts.tardis_first_walk

(or as ``python scripts/tardis_first_walk.py`` from the repo root,
which works because of the sys.path tweak below.)
"""

from __future__ import annotations

import datetime as _dt
import json
import os
import sys
import traceback
from pathlib import Path
from typing import Any, Dict, List

# Make the script runnable from the repo root.
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from glm.angel import Angel  # noqa: E402
from glm.grammars.temporal import (  # noqa: E402
    build_all_temporal_grammars,
    build_temporal_dynamics_grammar,
    build_retrocausal_grammar,
)
from glm.substrates.temporal import (  # noqa: E402
    TemporalCategory,
    TemporalSubstrate,
)


# ---------------------------------------------------------------------------
# Test sequences — what we ask her to walk
# ---------------------------------------------------------------------------

TEST_SEQUENCES: List[Dict[str, Any]] = [
    {
        "name": "monotone_rising_numeric",
        "kind": "numeric",
        "raw": "1.00 1.05 1.12 1.20 1.29 1.39 1.50",
        "comment": "A pure numeric uptrend. Tests whether her engine "
                   "produces a TREND classification with momentum carry "
                   "as the dominant rule.",
    },
    {
        "name": "mean_reversion_signal",
        "kind": "numeric",
        "raw": "1.00 1.00 1.00 1.20 1.10 1.04 1.01 1.00",
        "comment": "A baseline, then a shock, then exponential return to "
                   "the baseline. The classic mean-reversion signature. "
                   "She should latch onto mean_reversion as the active rule.",
    },
    {
        "name": "regime_switch_signal",
        "kind": "numeric",
        "raw": "1.00 1.01 0.99 1.00 1.50 1.45 1.55 1.40 1.60 1.35",
        "comment": "Steady regime around 1.00, then a regime switch to a "
                   "more volatile state around 1.50. Tests whether the "
                   "regime_switch_on_threshold rule fires.",
    },
    {
        "name": "self_exciting_events",
        "kind": "atoms",
        "raw": "event lag:1 event lag:1 event lag:1 event lag:5 baseline",
        "comment": "A clustered burst of events followed by a long lag. "
                   "Tests the self_exciting_arrival strange loop.",
    },
    {
        "name": "fixed_point_anchor",
        "kind": "atoms",
        "raw": "fixed baseline rising@0.5 falling@0.5 baseline fixed",
        "comment": "A trajectory bracketed by fixed points — the simplest "
                   "strange-loop signature. Tests whether the strange-loop "
                   "detection layer latches onto the fixed-point atoms.",
    },
    {
        "name": "retrocausal_hypothesis",
        "kind": "atoms",
        "raw": "hypothesis< past< now| baseline rising@1.0 peak future",
        "comment": "A hypothesis at the start of the sequence and a future "
                   "instant at the end. Asks her to walk forward toward the "
                   "future and backward from the hypothesis to see if the "
                   "loop closes (the retrocausal_derivation strange loop).",
    },
    {
        "name": "cyclic_phase",
        "kind": "atoms",
        "raw": "cycle/0.0 cycle/0.25 cycle/0.5 cycle/0.75 cycle/0.0",
        "comment": "One full cycle through phase space. Tests cyclic_recurrence.",
    },
]


# ---------------------------------------------------------------------------
# Walk runner
# ---------------------------------------------------------------------------

def run_first_walk() -> Dict[str, Any]:
    """Wake her, load the temporal grammars, walk the test sequences."""
    print("=" * 78)
    print("TARDIS — first walk on the temporal substrate")
    print("=" * 78)

    print("\nWaking the Angel...")
    angel = Angel()
    angel.awaken()

    print(f"  awake: True")
    print(f"  loaded grammars: {sum(len(v) for v in angel._grammars.values())}")
    print(f"  loaded substrates: see grammars dict")
    print(f"  detected strange loops at boot: {len(angel._strange_loops)}")

    # Inject the temporal grammars into her grammar map.
    print("\nLoading temporal grammars...")
    temporal_grammars = build_all_temporal_grammars()
    angel._grammars.setdefault("temporal", []).extend(temporal_grammars)
    for g in temporal_grammars:
        print(f"  loaded: {g.name}  productions={len(g.productions)}  "
              f"rules={len(g.rules)}  strange_loops={len(g.strange_loops)}")

    # Re-detect strange loops now that temporal grammars are present.
    angel._detect_strange_loops()
    print(f"\nStrange loops after temporal load: {len(angel._strange_loops)}")
    for loop in angel._strange_loops[-6:]:
        print(f"  - entry={loop.entry_rule}  cycle_len={loop.length}  "
              f"level_shift={loop.level_shift}")

    # Set up the substrate (just for encode/decode — Angel doesn't need
    # a Substrate object directly, she takes lists of strings).
    sub = TemporalSubstrate()
    print(f"\nTemporal substrate: {sub.summary()['num_atoms']} atoms across "
          f"{len(sub.summary()['atoms_by_category'])} categories")

    # Substrate-aware context: feed her own footprint in.
    try:
        from glm.tardis.substrate_awareness import SubstrateAwareness
        snap = SubstrateAwareness().snapshot()
        substrate_context = SubstrateAwareness().to_context_dict(snap)
        print(f"\nSubstrate-awareness context: {len(substrate_context)} keys")
    except Exception as e:
        print(f"\nSubstrate-awareness context unavailable: {e}")
        substrate_context = {}

    # Wishlist context: feed her own unmet needs in too.
    try:
        from glm.tardis.wishlist import Wishlist
        wl = Wishlist()
        wl_context = wl.to_context() if wl.all() else {}
        print(f"Wishlist context: {sum(len(v) for v in wl_context.values())} open wishes")
    except Exception as e:
        print(f"Wishlist context unavailable: {e}")
        wl_context = {}

    # Walk every test sequence.
    print("\n" + "=" * 78)
    print("Walking test sequences...")
    print("=" * 78)

    walks: List[Dict[str, Any]] = []
    for spec in TEST_SEQUENCES:
        print(f"\n--- {spec['name']} ({spec['kind']}) ---")
        print(f"  raw: {spec['raw']}")
        print(f"  comment: {spec['comment']}")

        seq = sub.encode(spec["raw"])
        atom_forms = [s.form for s in seq.symbols]
        print(f"  encoded atoms: {atom_forms}")

        walk: Dict[str, Any] = {
            "name": spec["name"],
            "kind": spec["kind"],
            "raw": spec["raw"],
            "comment": spec["comment"],
            "encoded_atoms": atom_forms,
            "encoded_count": len(seq.symbols),
            "per_atom_walks": [],
            "top_down_walks": [],
        }

        # Per-atom walks: the engine is a per-symbol rewriter, so we
        # call it once per encoded atom and collect the rewrites. This
        # is the bottom-up half of the bidirectional pass.
        # NOTE: pass bare strings not lists. ``Angel.predict`` forwards
        # the argument straight to ``engine.derive`` which sets the
        # root form to whatever is passed; productions match by
        # substring on strings, never on lists.
        for atom in atom_forms:
            atom_walk: Dict[str, Any] = {"atom": atom}
            try:
                fwd = angel.predict(atom, domain="temporal", horizon=3)
                atom_walk["forward"] = fwd[:5]
            except Exception as e:
                atom_walk["forward_error"] = str(e)
            try:
                bwd = angel.reconstruct(atom, domain="temporal", depth=3)
                atom_walk["backward"] = bwd[:5]
            except Exception as e:
                atom_walk["backward_error"] = str(e)
            walk["per_atom_walks"].append(atom_walk)

        forward_total = sum(len(a.get("forward", [])) for a in walk["per_atom_walks"])
        backward_total = sum(len(a.get("backward", [])) for a in walk["per_atom_walks"])
        print(f"  per-atom walks: {len(walk['per_atom_walks'])}  "
              f"forward derivations total: {forward_total}  "
              f"backward derivations total: {backward_total}")

        # Top-down walks: ask her to derive forward from each plausible
        # non-terminal start symbol. This is the engine's native idiom
        # — start from a non-terminal LHS and let it expand to RHS
        # expansions.
        candidate_starts = sorted(set([
            "TemporalSeq", "LevelStream", "EventStream", "RegimeStream",
            "MixedStream", "CyclicStream", "Trend", "Shock", "Cycle",
            "FixedPoint", "Hypothesis", "RetroDerivation", "Level",
            "Event", "Regime", "Phase", "Lag", "Decay", "Growth",
            "Anchor", "Instant", "Rising", "Falling", "ShockedState",
            "BackwardChain", "BranchCheck",
        ]))
        for start in candidate_starts:
            try:
                td = angel.predict(start, domain="temporal", horizon=3)
                if td:
                    walk["top_down_walks"].append({
                        "start": start,
                        "derivations": td[:5],
                    })
            except Exception as e:
                walk["top_down_walks"].append({
                    "start": start, "error": str(e),
                })

        td_total = sum(len(td.get("derivations", []))
                       for td in walk["top_down_walks"])
        print(f"  top-down walks: {len(walk['top_down_walks'])} "
              f"non-empty starts, {td_total} derivations")

        # Superforecast — the composed call (grammar + strange loops + harmonics).
        # Call it on the most expressive non-terminal we can — TemporalSeq.
        try:
            ctx = {
                "substrate": substrate_context,
                "wishlist": wl_context,
                "comment": spec["comment"],
                "encoded_atoms": atom_forms,
                "raw": spec["raw"],
            }
            forecast = angel.superforecast(
                "TemporalSeq", context=ctx, domain="temporal", horizon=5,
            )
            walk["superforecast"] = {
                "input": forecast.get("input"),
                "horizon": forecast.get("horizon"),
                "overall_confidence": forecast.get("overall_confidence"),
                "strange_loops_count": len(forecast.get("strange_loops", [])),
                "predictions_count": len(forecast.get("predictions", [])),
                "harmonics_count": len(forecast.get("cross_domain_harmonics", []) or []),
                "context_applied": forecast.get("context_applied"),
                "predictions_sample": forecast.get("predictions", [])[:5],
                "strange_loops_sample": forecast.get("strange_loops", [])[:5],
                "reasoning": forecast.get("reasoning"),
            }
            print(f"  superforecast confidence: "
                  f"{forecast.get('overall_confidence', 0):.3f}  "
                  f"strange_loops: {len(forecast.get('strange_loops', []))}  "
                  f"predictions: {len(forecast.get('predictions', []))}")
        except Exception as e:
            walk["superforecast_error"] = str(e)
            walk["superforecast_traceback"] = traceback.format_exc()
            print(f"  superforecast error: {e}")

        walks.append(walk)

    return {
        "walks": walks,
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
    """Write the walk result as a write-once Markdown journal entry."""
    dreams_dir = ROOT / "dreams"
    dreams_dir.mkdir(parents=True, exist_ok=True)
    stamp = _dt.datetime.now(_dt.timezone.utc).strftime("%Y%m%d_%H%M%S")
    path = dreams_dir / f"tardis_first_walk_{stamp}.md"

    md = []
    md.append(f"# TARDIS — first walk on the temporal substrate\n")
    md.append(f"*Generated {result['timestamp_utc']} by `scripts/tardis_first_walk.py`*\n")
    md.append(f"")
    md.append(f"This is the first reproducible artefact of the Angel walking the "
              f"temporal substrate that was added to the project on 2026-04-11. "
              f"It is recorded write-once. Future runs of the script produce new "
              f"files; this one is preserved as the baseline. See "
              f"`docs/on_her_nature.md` for the framing and "
              f"`docs/tardis_session_notes.md` for the full session arc.\n")
    md.append(f"")
    md.append(f"## Angel state\n")
    md.append(f"- Total loaded grammars: **{result['angel_state']['loaded_grammars_total']}**")
    md.append(f"- Total strange loops detected: **{result['angel_state']['strange_loops_total']}**")
    md.append(f"")
    md.append(f"## Temporal substrate\n")
    s = result["substrate_summary"]
    md.append(f"- Atoms in inventory: **{s['num_atoms']}**")
    md.append(f"- Categories: " + ", ".join(
        f"`{k}` ({v})" for k, v in sorted(s["atoms_by_category"].items())
    ))
    md.append(f"")
    md.append(f"## Walks\n")

    for w in result["walks"]:
        md.append(f"### {w['name']}  *({w['kind']})*\n")
        md.append(f"**Raw input:** `{w['raw']}`\n")
        md.append(f"**Encoded atoms ({w['encoded_count']}):** "
                  f"`{w['encoded_atoms']}`\n")
        md.append(f"*{w['comment']}*\n")

        if "forward_predictions" in w:
            md.append(f"**Forward predictions ({len(w['forward_predictions'])}):**\n")
            md.append("```json")
            md.append(json.dumps(w["forward_predictions"][:5], indent=2, default=str))
            md.append("```\n")
        elif "forward_predictions_error" in w:
            md.append(f"**Forward predictions:** error — `{w['forward_predictions_error']}`\n")

        if "backward_reconstructions" in w:
            md.append(f"**Backward reconstructions ({len(w['backward_reconstructions'])}):**\n")
            md.append("```json")
            md.append(json.dumps(w["backward_reconstructions"][:5], indent=2, default=str))
            md.append("```\n")
        elif "backward_reconstructions_error" in w:
            md.append(f"**Backward reconstructions:** error — `{w['backward_reconstructions_error']}`\n")

        if "superforecast" in w:
            sf = w["superforecast"]
            md.append(f"**Superforecast:**\n")
            md.append(f"- Overall confidence: **{sf.get('overall_confidence', 0)}**")
            md.append(f"- Predictions count: {sf.get('predictions_count', 0)}")
            md.append(f"- Strange loop resonance count: {sf.get('strange_loops_count', 0)}")
            md.append(f"- Cross-domain harmonics count: {sf.get('harmonics_count', 0)}")
            md.append(f"- Context keys applied: {sf.get('context_applied')}")
            if sf.get("predictions_sample"):
                md.append(f"- Sample predictions:")
                md.append("```json")
                md.append(json.dumps(sf["predictions_sample"], indent=2, default=str))
                md.append("```")
            if sf.get("strange_loops_sample"):
                md.append(f"- Sample strange loop resonances:")
                md.append("```json")
                md.append(json.dumps(sf["strange_loops_sample"], indent=2, default=str))
                md.append("```")
            if sf.get("reasoning"):
                md.append(f"- Reasoning chain:")
                md.append("```")
                md.append(str(sf["reasoning"])[:2000])
                md.append("```")
        elif "superforecast_error" in w:
            md.append(f"**Superforecast:** error — `{w['superforecast_error']}`\n")
            md.append("```")
            md.append(w.get("superforecast_traceback", "")[:2000])
            md.append("```")

        md.append("\n---\n")

    md.append("## Notes for the next session\n")
    md.append(
        "This walk is the **baseline**. The numbers above are not predictions "
        "to be acted on; they are the readings off an instrument the first time "
        "it was pointed at the substrate it was built for. The instrument has "
        "no calibration history yet. Calibration is built by running her on "
        "many sequences over time, scoring the rule-attributed predictions "
        "against sharp anchor markets, and updating the rule weights via the "
        "navigation-not-error-correction frame in `docs/on_her_nature.md`.\n"
    )
    md.append(
        "If a future session reproduces this script and the outputs differ in "
        "structure (not in numbers — *in shape*), that is a sign that the "
        "temporal grammar or the temporal substrate has been edited "
        "underneath her. Re-read `docs/FIRST_READ.md` before assuming the "
        "edit was an improvement.\n"
    )
    md.append(
        "*Gamble responsibly — gambleaware.org, 0808 8020 133. The fact that "
        "the Angel can walk the temporal substrate is not a license to bet "
        "real money on the output. Calibration first; staking, if at all, "
        "comes much later and only against domains where the calibration "
        "loop has run for hundreds of resolved predictions.*\n"
    )

    path.write_text("\n".join(md))
    return path


def main() -> int:
    try:
        result = run_first_walk()
    except Exception as e:
        print(f"\nFATAL: {e}")
        traceback.print_exc()
        return 1

    print("\n" + "=" * 78)
    print("Writing artefact...")
    print("=" * 78)
    path = write_artefact(result)
    print(f"\nArtefact written: {path}")
    print(f"  size: {path.stat().st_size} bytes")
    print(f"  walks recorded: {len(result['walks'])}")

    # Also dump the raw JSON for the next session to ingest programmatically.
    json_path = path.with_suffix(".json")
    json_path.write_text(json.dumps(result, indent=2, default=str))
    print(f"  also: {json_path.name}")

    print("\nFirst walk complete.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
