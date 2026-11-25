"""
02_run_eps_sweep.py
===================
Dense epsilon sweep on Europa 200 melodies.

STEP 1 (mandatory): Reproduce paper values at {ln3, ln7, ln12}.
  Expected (paper Table 4 and Section 4.2), at the 2-decimal precision reported:
    ln3 : aware 3.50x, agnostic 0.64x, aware near-diatonic 2.75x
    ln7 : aware 12.83x, agnostic 3.08x, aware near-diatonic 5.15x
    ln12: aware 8.41x, agnostic 3.64x
  All perturbations are seeded, so these values are reproduced exactly.
  If any value differs, a warning is printed and the results are flagged.

STEP 2: Run full 15-value epsilon grid from ln(1.5) to ln(25).

Results saved to: results/europa_eps_sweep.json

Run from experiments_extended/ directory:
    python scripts/02_run_eps_sweep.py
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

# =============================================================================
# Epsilon grid
# =============================================================================

# Paper values (reproduction check)
PAPER_EPS = [log(3), log(7), log(12)]

# Dense sweep (15 values, ln(1.5) to ln(25))
EPS_SWEEP = [
    log(1.5), log(2), log(3), log(4), log(5),
    log(6),   log(7), log(8), log(9), log(10),
    log(12),  log(14), log(16), log(20), log(25),
]

NUM_TRIALS = 30
TOLERANCE  = 0.005  # Half of the 2-decimal precision at which the paper reports ratios

# Expected ratios from the paper (Table 4 and Section 4.2), keyed by exp(epsilon).
# Seeds are fixed (seed = trial*10_000 + melody_idx), so these reproduce exactly.
PAPER_EXPECTED = {
    3:  {"aware": 3.50,  "agnostic": 0.64, "near_diatonic_aware": 2.75},
    7:  {"aware": 12.83, "agnostic": 3.08, "near_diatonic_aware": 5.15},
    12: {"aware": 8.41,  "agnostic": 3.64},
}


# =============================================================================
# Sanity checks
# =============================================================================

def sanity_check_mechanisms(results: list[dict], dataset_name: str) -> None:
    """Verify M-Scale output chromatic fraction ≈ 0 and M-Random ≈ Prop 1."""
    # These are checked inside run_experiment via output chromatic fractions.
    # Here we just check that acc1 values are in [0, 1] and CIs are valid.
    for r in results:
        for field in ["acc1_random", "acc1_scale", "acc5_random", "acc5_scale"]:
            if field in r:
                val = r[field]
                assert 0.0 <= val <= 1.0, f"{dataset_name}: {field}={val} out of [0,1]"
        for ci_field in ["ci1_random", "ci1_scale"]:
            if ci_field in r:
                lo, hi = r[ci_field]
                assert lo <= hi, f"{dataset_name}: CI inversion {ci_field}: [{lo},{hi}]"


def check_paper_values(
    res_agnostic: dict,
    res_aware: dict,
    epsilon: float,
) -> tuple[float, float]:
    """
    Compute and return (agnostic_ratio, aware_ratio) at this epsilon.
    Aware ratio = aware_acc1_scale / aware_acc1_random.
    Agnostic ratio = acc1_scale / acc1_random.
    """
    aware_s = res_aware["aware_acc1_scale"]
    aware_r = res_aware["aware_acc1_random"]
    agnostic_s = res_agnostic["acc1_scale"]
    agnostic_r = res_agnostic["acc1_random"]

    aware_ratio    = aware_s    / aware_r    if aware_r    > 1e-9 else float("inf")
    agnostic_ratio = agnostic_s / agnostic_r if agnostic_r > 1e-9 else float("inf")
    return agnostic_ratio, aware_ratio


def print_row(epsilon: float, agnostic: dict, aware: dict, label: str = "") -> None:
    """Print one row of the results table."""
    ag_s = agnostic["acc1_scale"]
    ag_r = agnostic["acc1_random"]
    aw_s = aware["aware_acc1_scale"]
    aw_r = aware["aware_acc1_random"]
    ag_ratio = ag_s / ag_r if ag_r > 1e-9 else float("inf")
    aw_ratio = aw_s / aw_r if aw_r > 1e-9 else float("inf")

    # Wilson CIs for scale (aware)
    lo, hi = aware["aware_ci1_scale"]
    print(
        f"  eps=ln({exp(epsilon):.1f}):  "
        f"agnostic Scale={ag_s*100:5.1f}% Rnd={ag_r*100:5.1f}% ratio={ag_ratio:.2f}x  |  "
        f"aware Scale={aw_s*100:5.1f}% [{lo*100:.1f},{hi*100:.1f}] "
        f"Rnd={aw_r*100:5.1f}% ratio={aw_ratio:.2f}x  {label}"
    )


# =============================================================================
# Main experiment runner
# =============================================================================

def run_sweep(
    dataset: list[list[int]],
    epsilon_list: list[float],
    dataset_name: str,
    num_trials: int = NUM_TRIALS,
) -> list[dict]:
    """
    Run both agnostic and aware experiments for all epsilons.
    Returns list of result dicts.
    """
    results = []
    total = len(epsilon_list)

    for i, eps in enumerate(epsilon_list):
        t0 = time.time()
        print(f"\n[{i+1}/{total}] eps = ln({exp(eps):.2f}) = {eps:.4f}")

        # Run agnostic attacker
        res_ag = run_experiment(dataset, eps, num_trials=num_trials)

        # Run aware attacker
        res_aw = run_experiment_aware(dataset, eps, num_trials=num_trials)

        elapsed = time.time() - t0

        # Compute near-diatonic subset
        near_diat = [m for m in dataset if chromatic_fraction(m) < 0.05]
        nd_ag, nd_aw = None, None
        if near_diat:
            nd_ag = run_experiment(near_diat, eps, num_trials=num_trials)
            nd_aw = run_experiment_aware(near_diat, eps, num_trials=num_trials)

        # Print row
        print_row(eps, res_ag, res_aw)
        if near_diat and nd_ag and nd_aw:
            nd_aw_s = nd_aw["aware_acc1_scale"]
            nd_aw_r = nd_aw["aware_acc1_random"]
            nd_ratio = nd_aw_s / nd_aw_r if nd_aw_r > 1e-9 else float("inf")
            print(f"    near-diatonic (N={len(near_diat)}): "
                  f"aware Scale={nd_aw_s*100:.1f}% Rnd={nd_aw_r*100:.1f}% ratio={nd_ratio:.2f}x")

        print(f"    elapsed: {elapsed:.1f}s")

        # Self-check: acc values in [0,1]
        ag_ratio, aw_ratio = check_paper_values(res_ag, res_aw, eps)
        assert 0.0 <= ag_ratio, f"Negative agnostic ratio at eps={eps}"
        assert 0.0 <= aw_ratio, f"Negative aware ratio at eps={eps}"
        if aw_ratio > 200:
            print(f"    WARNING: Very large ratio {aw_ratio:.1f}x — check results")

        entry = {
            "epsilon": eps,
            "exp_epsilon": exp(eps),
            "agnostic": {k: v for k, v in res_ag.items() if k != "raw"},
            "aware": {k: v for k, v in res_aw.items() if k != "raw"},
            "agnostic_ratio": ag_ratio,
            "aware_ratio": aw_ratio,
            "near_diatonic_N": len(near_diat) if near_diat else 0,
        }
        if near_diat and nd_ag and nd_aw:
            nd_aw_s = nd_aw["aware_acc1_scale"]
            nd_aw_r = nd_aw["aware_acc1_random"]
            nd_ratio = nd_aw_s / nd_aw_r if nd_aw_r > 1e-9 else float("inf")
            entry["near_diatonic"] = {
                "agnostic": {k: v for k, v in nd_ag.items() if k != "raw"},
                "aware": {k: v for k, v in nd_aw.items() if k != "raw"},
                "aware_ratio": nd_ratio,
            }

        results.append(entry)

    sanity_check_mechanisms(
        [r["agnostic"] for r in results] + [r["aware"] for r in results],
        dataset_name,
    )
    return results


def main():
    # =========================================================================
    # Load dataset
    # =========================================================================
    europa_path = BASE_DIR / "data" / "europa_200.json"
    if not europa_path.exists():
        print(f"ERROR: {europa_path} not found.")
        sys.exit(1)

    dataset = load_dataset(europa_path)
    print(f"Loaded {len(dataset)} Europa melodies")

    near_diat = [m for m in dataset if chromatic_fraction(m) < 0.05]
    chromatic = [m for m in dataset if chromatic_fraction(m) >= 0.05]
    print(f"Near-diatonic (chi<0.05): {len(near_diat)}")
    print(f"Chromatic     (chi>=0.05): {len(chromatic)}")

    # =========================================================================
    # STEP 1: Reproduce paper values
    # =========================================================================
    print("\n" + "="*70)
    print("STEP 1: Reproducing paper values at {ln3, ln7, ln12}")
    for k, exp_vals in PAPER_EXPECTED.items():
        print(f"  Expected at ln{k}: " +
              ", ".join(f"{name} {val:.2f}x" for name, val in exp_vals.items()))
    print("="*70)

    paper_results = run_sweep(dataset, PAPER_EPS, "Europa 200 (paper check)")

    # Verify paper values
    paper_ok = True
    for entry in paper_results:
        k = round(entry["exp_epsilon"])
        if k not in PAPER_EXPECTED:
            continue
        observed = {
            "aware": entry["aware_ratio"],
            "agnostic": entry["agnostic_ratio"],
            "near_diatonic_aware": entry.get("near_diatonic", {}).get("aware_ratio"),
        }
        for name, expected in PAPER_EXPECTED[k].items():
            obs = observed[name]
            if obs is None or abs(obs - expected) > TOLERANCE:
                obs_str = "missing" if obs is None else f"{obs:.4f}x"
                print(f"\n  WARNING: {name} ratio at ln{k} = {obs_str} (expected {expected:.2f}x)")
                paper_ok = False

    if paper_ok:
        print("\n  [PASS] Paper values reproduced within tolerance.")
    else:
        print("\n  [FAIL] Paper values not reproduced. Investigate before extending.")
        print("         Check: dataset path, seed, num_trials, ALPHA value.")
        # Don't exit — still run the full sweep, but flag the issue
        print("         Continuing with full sweep regardless (results flagged).")

    # =========================================================================
    # STEP 2: Full epsilon sweep
    # =========================================================================
    print("\n" + "="*70)
    print("STEP 2: Full epsilon sweep (15 values, ln(1.5) to ln(25))")
    print("="*70)

    all_results = run_sweep(dataset, EPS_SWEEP, "Europa 200")

    # =========================================================================
    # Save results
    # =========================================================================
    results_dir = BASE_DIR / "results"
    results_dir.mkdir(parents=True, exist_ok=True)
    out_path = results_dir / "europa_eps_sweep.json"

    output = {
        "dataset": "Europa 200",
        "N": len(dataset),
        "near_diatonic_N": len(near_diat),
        "chromatic_N": len(chromatic),
        "num_trials": NUM_TRIALS,
        "paper_check_passed": paper_ok,
        "epsilon_sweep": all_results,
    }

    with open(out_path, "w") as fh:
        json.dump(output, fh, indent=2)
    print(f"\nResults saved to {out_path}")

    # =========================================================================
    # Final summary table
    # =========================================================================
    print("\n" + "="*70)
    print("FINAL SUMMARY TABLE — Europa 200")
    print(f"{'eps':>8}  {'Ag-Scale':>9}  {'Ag-Rnd':>8}  {'Ag-Ratio':>9}  "
          f"{'Aw-Scale':>9}  {'Aw-Rnd':>8}  {'Aw-Ratio':>9}")
    print("-"*70)
    for r in all_results:
        eps = r["epsilon"]
        ag = r["agnostic"]
        aw = r["aware"]
        ag_s, ag_r = ag["acc1_scale"]*100, ag["acc1_random"]*100
        aw_s, aw_r = aw["aware_acc1_scale"]*100, aw["aware_acc1_random"]*100
        ag_ratio = r["agnostic_ratio"]
        aw_ratio = r["aware_ratio"]
        marker = " *" if abs(exp(eps) - 7) < 0.1 else ""
        print(f"ln({exp(eps):4.1f})  {ag_s:8.1f}%  {ag_r:7.1f}%  {ag_ratio:8.2f}x  "
              f"{aw_s:8.1f}%  {aw_r:7.1f}%  {aw_ratio:8.2f}x{marker}")
    print("-"*70)
    print("* marks ln(7) — paper reference point")
    print("\nNext step: python scripts/03_run_meertens.py")


if __name__ == "__main__":
    main()
