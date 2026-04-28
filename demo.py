#!/usr/bin/env python3
"""
MKAngel Demo — for ARM presentation.

Run this on any Arm device (or any Python 3.10+ machine) to
demonstrate the full Angel pipeline. Zero dependencies.

Usage:
    python demo.py
"""

import time
import sys


def main():
    print()
    print("=" * 65)
    print("  MKAngel — Grammar Language Model Demo")
    print("  Pure Python | Zero Dependencies | Arm-Native")
    print("=" * 65)
    print()

    # --- Boot ---
    print("[1/7] Booting Angel...", end=" ", flush=True)
    t0 = time.time()
    from glm.angel import Angel
    angel = Angel()
    angel.awaken()
    boot_ms = (time.time() - t0) * 1000
    info = angel.introspect()
    print("%.0f ms" % boot_ms)
    print("  %d domains | %d grammars | %d rules | %d productions" % (
        len(info["domains_loaded"]), info["total_grammars"],
        info["total_rules"], info["total_productions"]))
    print("  %d strange loops | %d lexicon entries | %dk neural params" % (
        info["strange_loops_detected"], info["lexicon_size"],
        info["model_params"] // 1000))
    print("  Domains: %s" % ", ".join(info["domains_loaded"]))
    print()

    # --- Parse English ---
    print("[2/7] Parsing English...")
    sentences = [
        "the brave soldier walked through the dark forest",
        "she quickly finished her beautiful old book",
        "birds flow through the bright morning light",
    ]
    for sent in sentences:
        t0 = time.time()
        result = angel.parse(sent.split())
        ms = (time.time() - t0) * 1000
        tree = result.get("tree", {})
        print("  \"%s\"" % sent)
        print("    tags: %s  tree: %s  (%.1f ms)" % (
            result["tags"], tree.get("cat", "?"), ms))

    # --- Predict ---
    print()
    print("[3/7] Grammatical prediction...")
    for tokens in [["the", "cat"], ["she", "sees"], ["the", "bright", "light"]]:
        t0 = time.time()
        preds = angel.predict(tokens, domain="linguistic", horizon=5)
        ms = (time.time() - t0) * 1000
        print("  %s -> %d predictions (%.1f ms)" % (tokens, len(preds), ms))
        for p in preds[:3]:
            print("    -> %s (%.2f) [%s]" % (
                p.get("predicted"), p.get("confidence", 0),
                p.get("via", "engine")))

    # --- Unknown words ---
    print()
    print("[4/7] Inferring unknown words...")
    from glm.core.parser import heuristic_tag
    unknowns = [
        "mesmerising", "bureaucratic", "photosynthesis",
        "unbelievable", "magnificently", "cryptocurrency",
    ]
    for w in unknowns:
        print("  %-20s -> %s" % (w, heuristic_tag(w)))

    # --- Cross-domain ---
    print()
    print("[5/7] Cross-domain fugue: 'energy transforms'...")
    t0 = time.time()
    fugue = angel.compose_fugue(
        ["energy", "transforms"],
        domains=["physics", "chemical", "biological", "linguistic"]
    )
    ms = (time.time() - t0) * 1000
    voices = fugue.get("voices", {})
    if isinstance(voices, dict):
        print("  %d voices (%.1f ms)" % (len(voices), ms))
        for domain, derivations in voices.items():
            print("    [%s] %d derivations" % (domain, len(derivations)))
            for d in derivations[:2]:
                if isinstance(d, dict):
                    print("      %s (via %s)" % (
                        d.get("output", "?"), d.get("rule", "?")))

    # --- Multilingual lexicon ---
    print()
    print("[6/7] Multilingual cognate discovery: *wed- (water)...")
    water_cognates = []
    for entry in angel._lexicon.entries.values():
        for etym in entry.etymology:
            if etym.get("form") == "*wed-":
                water_cognates.append(entry)
    if water_cognates:
        for e in water_cognates:
            print("  %-12s [%s] %s" % (e.form, e.category, e.substrates))
    else:
        print("  (cognate search requires proto-root indexing)")

    # --- Swarm ---
    print()
    print("[7/7] Swarm orchestration (1 cycle)...")
    from app.swarm import SwarmOrchestrator
    orch = SwarmOrchestrator()
    t0 = time.time()
    report = orch.run("How do chemical bonds relate to syntactic bonds?", cycles=1)
    ms = (time.time() - t0) * 1000
    cr = report.cycle_results[0]
    print("  %.0f ms | confidence: %.2f" % (ms, cr.results.confidence))
    print("  Best agents: %s" % ", ".join(cr.best_agents))
    print("  Options:")
    for opt in report.options[:3]:
        print("    %s" % opt)

    # --- Summary ---
    print()
    print("=" * 65)
    print("  Total boot + demo: %.0f ms" % ((time.time() - t0) * 1000 + boot_ms))
    print("  Disk: ~3 MB | Dependencies: 0 | GPU: not needed")
    print("=" * 65)
    print()


if __name__ == "__main__":
    main()
