"""
07_run_bayesian.py
==================
Bayesian (information-theoretic optimal) attacker on Europa 200.

Run at eps in {ln3, ln7, ln12}, 30 trials each.

Compares three attacker tiers:
  - Agnostic   (existing results, reloaded from europa_eps_sweep.json)
  - Aware      (existing results)
  - Bayesian   (this script)

For each trial, the Bayesian attacker:
  1. Applies the mechanism to each melody independently to produce a perturbed copy.
  2. Ranks all 200 database entries by descending log-likelihood under the
     true mechanism.  The true original is the query melody itself (index = trial melody).
  3. Records Acc@1 (rank == 1) and Acc@5 (rank <= 5).

Expected results
----------------
  Bayesian Acc@1(M-Scale)  >= Aware Acc@1(M-Scale)  for all eps
  Bayesian Acc@1(M-Random) >= Aware Acc@1(M-Random) for all eps
  Bayesian ratio >= Aware ratio (or discuss if not)

Approx runtime
--------------
  Per eps per trial per melody: O(N * L) scale lookups = O(200 * 20) = 4000 ops
  For M-Random: marginalisation cached after first call, so similar speed.
  Total: 3 eps * 30 trials * 200 melodies = 18000 perturbed queries
         Each query: 200 * 20 = 4000 note likelihoods
         => ~72M likelihood evaluations — fast, no simulation needed.

Run from experiments_extended/ directory:
    python scripts/07_run_bayesian.py
"""

from __future__ import annotations

import json
import math
import random
import sys
import time
from math import exp, log
from pathlib import Path

SCRIPT_DIR = Path(__file__).parent
BASE_DIR   = SCRIPT_DIR.parent
sys.path.insert(0, str(BASE_DIR))

from src.bayesian_attacker import melody_log_likelihood
from src.utils_extended import load_dataset

# ── Configuration ─────────────────────────────────────────────────────────────

# Full 15-point epsilon grid (same as eps sweep scripts)
EPS_SWEEP = [
    log(1.5), log(2), log(3), log(4), log(5),
    log(6),   log(7), log(8), log(9), log(10),
    log(12),  log(14), log(16), log(20), log(25),
]
PAPER_EPS  = EPS_SWEEP   # run full grid
NUM_TRIALS = 30
SEED       = 42

# M-Scale fixed alphabet
SCALE = frozenset({0, 2, 4, 5, 7, 9, 11})


# ── Mechanism simulators ───────────────────────────────────────────────────────

def apply_mscale(melody: list[int], epsilon: float, rng: random.Random) -> list[int]:
    """Apply M-Scale: randomised response on C-major alphabet."""
    e = math.exp(epsilon)
    p = e / (e + 6.0)
    scale_list = sorted(SCALE)
    out = []
    for note in melody:
        pc = note % 12
        if pc in SCALE:
            if rng.random() < p:
                out.append(note)  # retain
            else:
                alts = [x for x in scale_list if x != pc]
                new_pc = rng.choice(alts)
                out.append((note // 12) * 12 + new_pc)
        else:
            # chromatic: replace with uniform draw from scale
            new_pc = rng.choice(scale_list)
            out.append((note // 12) * 12 + new_pc)
    return out


def apply_mrandom(melody: list[int], epsilon: float, rng: random.Random) -> list[int]:
    """Apply M-Random: random 7-pc alphabet per melody."""
    from itertools import combinations as _comb
    e = math.exp(epsilon)
    p = e / (e + 6.0)
    # Sample random 7-element alphabet
    alpha = sorted(rng.sample(range(12), 7))
    alpha_set = set(alpha)
    out = []
    for note in melody:
        pc = note % 12
        if pc in alpha_set:
            if rng.random() < p:
                out.append(note)
            else:
                alts = [x for x in alpha if x != pc]
                new_pc = rng.choice(alts)
                out.append((note // 12) * 12 + new_pc)
        else:
            # outside alphabet: replace uniformly over alpha
            new_pc = rng.choice(alpha)
            out.append((note // 12) * 12 + new_pc)
    return out


# ── Bayesian ranking ───────────────────────────────────────────────────────────

def bayesian_rank_query(
    perturbed: list[int],
    database: list[list[int]],
    true_idx: int,
    mechanism: str,
    epsilon: float,
) -> int:
    """1-based rank of true_idx by descending log-likelihood."""
    true_ll = melody_log_likelihood(perturbed, database[true_idx], mechanism, epsilon)
    rank = 1
    for j, cand in enumerate(database):
        if j == true_idx:
            continue
        ll = melody_log_likelihood(perturbed, cand, mechanism, epsilon)
        if ll > true_ll:
            rank += 1
    return rank


# ── Run one epsilon value ──────────────────────────────────────────────────────

def run_bayesian_eps(
    dataset: list[list[int]],
    epsilon: float,
    num_trials: int = NUM_TRIALS,
    seed: int = SEED,
) -> dict:
    """
    Run Bayesian attacker for both mechanisms at one epsilon value.

    Returns dict with acc1_scale, acc1_random, acc5_scale, acc5_random
    (and corresponding Wilson CIs).
    """
    from statsmodels.stats.proportion import proportion_confint

    N = len(dataset)
    rng = random.Random(seed)

    hits1_scale = 0
    hits5_scale = 0
    hits1_rand  = 0
    hits5_rand  = 0
    total       = 0

    for trial in range(num_trials):
        for i, melody in enumerate(dataset):
            # M-Scale
            pert_s = apply_mscale(melody, epsilon, rng)
            rank_s = bayesian_rank_query(pert_s, dataset, i, "scale", epsilon)
            if rank_s == 1:
                hits1_scale += 1
            if rank_s <= 5:
                hits5_scale += 1

            # M-Random
            pert_r = apply_mrandom(melody, epsilon, rng)
            rank_r = bayesian_rank_query(pert_r, dataset, i, "random", epsilon)
            if rank_r == 1:
                hits1_rand += 1
            if rank_r <= 5:
                hits5_rand += 1

            total += 1

        if (trial + 1) % 5 == 0:
            print(f"    trial {trial+1}/{num_trials}: "
                  f"Scale hits1={hits1_scale}, Rand hits1={hits1_rand} "
                  f"(of {(trial+1)*N} queries each)")

    n = num_trials * N  # total queries per mechanism

    def acc_and_ci(hits):
        a = hits / n
        lo, hi = proportion_confint(hits, n, alpha=0.05, method="wilson")
        return a, lo, hi

    a1s, lo1s, hi1s = acc_and_ci(hits1_scale)
    a5s, lo5s, hi5s = acc_and_ci(hits5_scale)
    a1r, lo1r, hi1r = acc_and_ci(hits1_rand)
    a5r, lo5r, hi5r = acc_and_ci(hits5_rand)

    return {
        "acc1_scale": a1s, "ci1_scale": [lo1s, hi1s],
        "acc5_scale": a5s, "ci5_scale": [lo5s, hi5s],
        "acc1_random": a1r, "ci1_random": [lo1r, hi1r],
        "acc5_random": a5r, "ci5_random": [lo5r, hi5r],
    }


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    europa_path = BASE_DIR / "data" / "europa_200.json"
    if not europa_path.exists():
        print(f"ERROR: {europa_path} not found.")
        sys.exit(1)

    dataset = load_dataset(europa_path)
    print(f"Loaded {len(dataset)} Europa melodies")

    # Load existing agnostic + aware results for comparison
    existing_path = BASE_DIR / "results" / "europa_eps_sweep.json"
    existing = {}
    if existing_path.exists():
        with open(existing_path) as f:
            data = json.load(f)
        for entry in data["epsilon_sweep"]:
            existing[round(entry["exp_epsilon"], 4)] = entry
        print(f"Loaded existing results for comparison")

    print("\n" + "="*70)
    print("Bayesian attacker: Europa 200, {ln3, ln7, ln12}, 30 trials")
    print("="*70)

    results = []

    for eps in PAPER_EPS:
        exp_eps = exp(eps)
        print(f"\n[eps = ln({exp_eps:.0f}) = {eps:.4f}]")
        t0 = time.time()

        bres = run_bayesian_eps(dataset, eps, num_trials=NUM_TRIALS, seed=SEED)

        elapsed = time.time() - t0
        print(f"  elapsed: {elapsed:.1f}s")

        # Ratios
        b_ratio_s = bres["acc1_scale"] / bres["acc1_random"] if bres["acc1_random"] > 1e-9 else float("inf")

        # Lookup existing for comparison
        key = round(exp_eps, 4)
        ag_s = ag_r = aw_s = aw_r = None
        if key in existing:
            e = existing[key]
            ag_s = e["agnostic"]["acc1_scale"]
            ag_r = e["agnostic"]["acc1_random"]
            aw_s = e["aware"]["aware_acc1_scale"]
            aw_r = e["aware"]["aware_acc1_random"]

        print(f"\n  Attacker comparison at eps=ln({exp_eps:.0f}):")
        print(f"  {'Attacker':<12} {'M-Scale':>9} {'M-Random':>9} {'Ratio':>8}")
        print(f"  {'-'*42}")
        if ag_s is not None:
            ag_ratio = ag_s / ag_r if ag_r > 1e-9 else float("inf")
            print(f"  {'Agnostic':<12} {ag_s*100:8.2f}% {ag_r*100:8.2f}% {ag_ratio:7.2f}x")
        if aw_s is not None:
            aw_ratio = aw_s / aw_r if aw_r > 1e-9 else float("inf")
            print(f"  {'Aware':<12} {aw_s*100:8.2f}% {aw_r*100:8.2f}% {aw_ratio:7.2f}x")
        print(f"  {'Bayesian':<12} {bres['acc1_scale']*100:8.2f}% {bres['acc1_random']*100:8.2f}% {b_ratio_s:7.2f}x")

        # Verify ordering
        if aw_s is not None:
            scale_ok  = bres["acc1_scale"]  >= aw_s  - 1e-9
            random_ok = bres["acc1_random"] >= aw_r  - 1e-9
            print(f"\n  Bayesian >= Aware (M-Scale):  {'PASS' if scale_ok  else 'FAIL'}")
            print(f"  Bayesian >= Aware (M-Random): {'PASS' if random_ok else 'FAIL'}")

        entry = {
            "epsilon": eps,
            "exp_epsilon": exp_eps,
            "bayesian": bres,
            "agnostic": {"acc1_scale": ag_s, "acc1_random": ag_r} if ag_s is not None else None,
            "aware":    {"acc1_scale": aw_s, "acc1_random": aw_r} if aw_s is not None else None,
            "bayesian_ratio": b_ratio_s,
            "aware_ratio": (aw_s / aw_r if aw_r and aw_r > 1e-9 else None),
        }
        results.append(entry)

    # Save
    results_dir = BASE_DIR / "results"
    results_dir.mkdir(parents=True, exist_ok=True)
    out_path = results_dir / "bayesian_europa.json"
    with open(out_path, "w") as fh:
        json.dump({
            "dataset": "Europa 200",
            "N": len(dataset),
            "num_trials": NUM_TRIALS,
            "epsilon_values": PAPER_EPS,
            "results": results,
        }, fh, indent=2)
    print(f"\nResults saved to {out_path}")

    # Summary
    print("\n" + "="*70)
    print("SUMMARY: Bayesian vs. Aware vs. Agnostic (Acc@1)")
    print(f"{'eps':<10} {'Ag-S':>7} {'Ag-R':>7} {'Aw-S':>7} {'Aw-R':>7} "
          f"{'Bay-S':>7} {'Bay-R':>7} {'Aw-Ratio':>9} {'Bay-Ratio':>10}")
    print("-"*80)
    for r in results:
        e = r["exp_epsilon"]
        b = r["bayesian"]
        ag = r.get("agnostic") or {}
        aw = r.get("aware") or {}
        ag_s = (ag.get("acc1_scale") or 0) * 100
        ag_r = (ag.get("acc1_random") or 0) * 100
        aw_s = (aw.get("acc1_scale") or 0) * 100
        aw_r = (aw.get("acc1_random") or 0) * 100
        b_s  = b["acc1_scale"] * 100
        b_r  = b["acc1_random"] * 100
        aw_rat = r.get("aware_ratio") or 0
        b_rat  = r.get("bayesian_ratio") or 0
        print(f"ln({e:4.0f})  {ag_s:6.2f}% {ag_r:6.2f}% {aw_s:6.2f}% {aw_r:6.2f}% "
              f"{b_s:6.2f}% {b_r:6.2f}% {aw_rat:8.2f}x {b_rat:9.2f}x")

    print("\nNext step: python scripts/08_plot_bayesian.py")


if __name__ == "__main__":
    main()
