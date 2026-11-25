"""
10_run_jazz.py
==============
Dense epsilon sweep on Weimar Jazz Database (200 melodies).

Jazz solos are truncated to 30 notes from the first notes of each transcription.
Mean chromaticity ~38% — the highest of all corpora, driven by jazz's use of
chromatic passing tones, blue notes, and non-C-major scales.

Result (see paper): the effect reverses on this corpus. Under the
retrieval-based attackers the M-Scale/M-Random ratio is below 1 at every
epsilon, because M-Scale maps almost every (chromatic) melody into the same
C-major distribution.

Results saved to: results/jazz_eps_sweep.json

Run from experiments_extended/ directory:
    python scripts/10_run_jazz.py
"""

from __future__ import annotations

import json
import sys
import time
from math import log, exp
from pathlib import Path

SCRIPT_DIR = Path(__file__).parent
BASE_DIR = SCRIPT_DIR.parent
sys.path.insert(0, str(BASE_DIR))

from src.evaluation import run_experiment, run_experiment_aware
from src.metrics import chromatic_fraction
from src.utils_extended import load_dataset

EPS_SWEEP = [
    log(1.5), log(2), log(3), log(4), log(5),
    log(6),   log(7), log(8), log(9), log(10),
    log(12),  log(14), log(16), log(20), log(25),
]

NUM_TRIALS = 30


def print_row(epsilon: float, agnostic: dict, aware: dict) -> None:
    ag_s = agnostic["acc1_scale"]
    ag_r = agnostic["acc1_random"]
    aw_s = aware["aware_acc1_scale"]
    aw_r = aware["aware_acc1_random"]
    ag_ratio = ag_s / ag_r if ag_r > 1e-9 else float("inf")
    aw_ratio = aw_s / aw_r if aw_r > 1e-9 else float("inf")
    lo, hi = aware["aware_ci1_scale"]
    print(
        f"  eps=ln({exp(epsilon):.1f}):  "
        f"agnostic Scale={ag_s*100:5.1f}% Rnd={ag_r*100:5.1f}% ratio={ag_ratio:.2f}x  |  "
        f"aware Scale={aw_s*100:5.1f}% [{lo*100:.1f},{hi*100:.1f}] "
        f"Rnd={aw_r*100:5.1f}% ratio={aw_ratio:.2f}x"
    )


def main():
    processed_dir = BASE_DIR / "data" / "processed"
    jazz_path = processed_dir / "jazz_200.json"

    if not jazz_path.exists():
        print(f"ERROR: {jazz_path} not found.")
        print("Run:  python scripts/09_prepare_jazz.py  first.")
        sys.exit(1)

    dataset = load_dataset(jazz_path)
    print(f"Loaded {len(dataset)} jazz melodies (first 30 notes of each solo)")

    # Dataset statistics
    chrom_fracs = [chromatic_fraction(m) for m in dataset]
    near_diat   = [m for m in dataset if chromatic_fraction(m) < 0.05]
    high_chrom  = [m for m in dataset if chromatic_fraction(m) >= 0.40]
    import statistics
    print(f"\nChromatic fraction statistics (C-major definition):")
    print(f"  Mean:  {statistics.mean(chrom_fracs)*100:.1f}%")
    print(f"  Std:   {statistics.stdev(chrom_fracs)*100:.1f}%")
    print(f"  Near-diatonic (chi<0.05):  {len(near_diat)} melodies")
    print(f"  High-chromatic (chi>=0.40): {len(high_chrom)} melodies")
    print("  (High chromaticity expected: jazz uses blue notes, chromatic runs, non-C scales)")

    print("\n" + "="*70)
    print("Epsilon sweep on Jazz 200 (15 values)")
    print("="*70)

    all_results = []

    for i, eps in enumerate(EPS_SWEEP):
        t0 = time.time()
        print(f"\n[{i+1}/{len(EPS_SWEEP)}] eps = ln({exp(eps):.2f}) = {eps:.4f}")

        res_ag = run_experiment(dataset, eps, num_trials=NUM_TRIALS)
        res_aw = run_experiment_aware(dataset, eps, num_trials=NUM_TRIALS)
        elapsed = time.time() - t0

        print_row(eps, res_ag, res_aw)

        # Near-diatonic subset (very small for jazz — only ~8 melodies)
        nd_entry = None
        if len(near_diat) >= 5:
            nd_ag = run_experiment(near_diat, eps, num_trials=NUM_TRIALS)
            nd_aw = run_experiment_aware(near_diat, eps, num_trials=NUM_TRIALS)
            nd_s = nd_aw["aware_acc1_scale"]
            nd_r = nd_aw["aware_acc1_random"]
            nd_ratio = nd_s / nd_r if nd_r > 1e-9 else float("inf")
            print(f"    near-diatonic (N={len(near_diat)}): "
                  f"aware Scale={nd_s*100:.1f}% Rnd={nd_r*100:.1f}% ratio={nd_ratio:.2f}x")
            nd_entry = {
                "agnostic": {k: v for k, v in nd_ag.items() if k != "raw"},
                "aware": {k: v for k, v in nd_aw.items() if k != "raw"},
                "aware_ratio": nd_ratio,
            }
        else:
            print(f"    near-diatonic N={len(near_diat)} (too small for reliable stats, skipping)")

        print(f"    elapsed: {elapsed:.1f}s")

        aw_s = res_aw["aware_acc1_scale"]
        aw_r = res_aw["aware_acc1_random"]
        ag_s = res_ag["acc1_scale"]
        ag_r = res_ag["acc1_random"]
        aw_ratio = aw_s / aw_r if aw_r > 1e-9 else float("inf")
        ag_ratio = ag_s / ag_r if ag_r > 1e-9 else float("inf")

        entry = {
            "epsilon": eps,
            "exp_epsilon": exp(eps),
            "agnostic": {k: v for k, v in res_ag.items() if k != "raw"},
            "aware": {k: v for k, v in res_aw.items() if k != "raw"},
            "agnostic_ratio": ag_ratio,
            "aware_ratio": aw_ratio,
            "near_diatonic_N": len(near_diat),
        }
        if nd_entry:
            entry["near_diatonic"] = nd_entry

        all_results.append(entry)

    # Save
    results_dir = BASE_DIR / "results"
    results_dir.mkdir(parents=True, exist_ok=True)
    out_path = results_dir / "jazz_eps_sweep.json"

    output = {
        "dataset": f"Weimar Jazz Database v1.2 (N={len(dataset)})",
        "N": len(dataset),
        "near_diatonic_N": len(near_diat),
        "high_chromatic_N": len(high_chrom),
        "num_trials": NUM_TRIALS,
        "note": "First 30 notes of each solo transcription. Pitch-only (no rhythm/dynamics).",
        "epsilon_sweep": all_results,
    }

    with open(out_path, "w") as fh:
        json.dump(output, fh, indent=2)
    print(f"\nResults saved to {out_path}")

    # Summary table
    print("\n" + "="*70)
    print("FINAL SUMMARY TABLE — Jazz 200")
    print(f"{'eps':>8}  {'Ag-Scale':>9}  {'Ag-Rnd':>8}  {'Ag-Ratio':>9}  "
          f"{'Aw-Scale':>9}  {'Aw-Rnd':>8}  {'Aw-Ratio':>9}")
    print("-"*70)
    for r in all_results:
        eps = r["epsilon"]
        ag = r["agnostic"]
        aw = r["aware"]
        ag_s, ag_r = ag["acc1_scale"]*100, ag["acc1_random"]*100
        aw_s, aw_r = aw["aware_acc1_scale"]*100, aw["aware_acc1_random"]*100
        marker = " *" if abs(exp(eps) - 7) < 0.1 else ""
        print(f"ln({exp(eps):4.1f})  {ag_s:8.1f}%  {ag_r:7.1f}%  {r['agnostic_ratio']:8.2f}x  "
              f"{aw_s:8.1f}%  {aw_r:7.1f}%  {r['aware_ratio']:8.2f}x{marker}")
    print("-"*70)
    print("* marks ln(7) reference point")


if __name__ == "__main__":
    main()
