"""
12_run_bayesian_all.py
======================
Bayesian (information-theoretically optimal) attacker on all four datasets:
  - Europa 200   (data/europa_200.json)
  - Meertens 500 (data/processed/meertens_500.json)
  - SymbTr 200   (data/processed/symbtr_200.json)
  - Jazz 200     (data/processed/jazz_200.json)

Full 15-point epsilon grid: ln(1.5), ln(2), ..., ln(25)
30 trials each.

Results saved to:
  results/bayesian_europa.json
  results/bayesian_meertens.json
  results/bayesian_symbtr.json
  results/bayesian_jazz.json

Run from experiments_extended/ directory:
    python scripts/12_run_bayesian_all.py
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

# ── Configuration ──────────────────────────────────────────────────────────────

EPS_SWEEP = [
    log(1.5), log(2),  log(3),  log(4),  log(5),
    log(6),   log(7),  log(8),  log(9),  log(10),
    log(12),  log(14), log(16), log(20), log(25),
]
NUM_TRIALS = 30
SEED       = 42

SCALE = frozenset({0, 2, 4, 5, 7, 9, 11})

DATASETS = [
    ("Europa 200",   BASE_DIR / "data" / "europa_200.json",            "bayesian_europa"),
    ("Meertens 500", BASE_DIR / "data" / "processed" / "meertens_500.json", "bayesian_meertens"),
    ("SymbTr 200",   BASE_DIR / "data" / "processed" / "symbtr_200.json",   "bayesian_symbtr"),
    ("Jazz 200",     BASE_DIR / "data" / "processed" / "jazz_200.json",     "bayesian_jazz"),
]


# ── Mechanism simulators ───────────────────────────────────────────────────────

def apply_mscale(melody: list[int], epsilon: float, rng: random.Random) -> list[int]:
    e = math.exp(epsilon)
    p = e / (e + 6.0)
    scale_list = sorted(SCALE)
    out = []
    for note in melody:
        pc = note % 12
        if pc in SCALE:
            if rng.random() < p:
                out.append(note)
            else:
                alts = [x for x in scale_list if x != pc]
                new_pc = rng.choice(alts)
                out.append((note // 12) * 12 + new_pc)
        else:
            new_pc = rng.choice(scale_list)
            out.append((note // 12) * 12 + new_pc)
    return out


def apply_mrandom(melody: list[int], epsilon: float, rng: random.Random) -> list[int]:
    e = math.exp(epsilon)
    p = e / (e + 6.0)
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
    true_ll = melody_log_likelihood(perturbed, database[true_idx], mechanism, epsilon)
    rank = 1
    for j, cand in enumerate(database):
        if j == true_idx:
            continue
        ll = melody_log_likelihood(perturbed, cand, mechanism, epsilon)
        if ll > true_ll:
            rank += 1
    return rank


# ── Run one epsilon on one dataset ─────────────────────────────────────────────

def run_bayesian_eps(
    dataset: list[list[int]],
    epsilon: float,
    num_trials: int = NUM_TRIALS,
    seed: int = SEED,
) -> dict:
    from statsmodels.stats.proportion import proportion_confint

    N = len(dataset)
    rng = random.Random(seed)

    hits1_scale = hits5_scale = 0
    hits1_rand  = hits5_rand  = 0

    for trial in range(num_trials):
        for i, melody in enumerate(dataset):
            pert_s = apply_mscale(melody, epsilon, rng)
            rank_s = bayesian_rank_query(pert_s, dataset, i, "scale", epsilon)
            if rank_s == 1: hits1_scale += 1
            if rank_s <= 5: hits5_scale += 1

            pert_r = apply_mrandom(melody, epsilon, rng)
            rank_r = bayesian_rank_query(pert_r, dataset, i, "random", epsilon)
            if rank_r == 1: hits1_rand += 1
            if rank_r <= 5: hits5_rand += 1

        if (trial + 1) % 5 == 0:
            print(f"    trial {trial+1}/{num_trials}: "
                  f"Scale hits1={hits1_scale}, Rand hits1={hits1_rand} "
                  f"(of {(trial+1)*N} each)")

    n = num_trials * N

    def acc_and_ci(hits):
        a = hits / n
        lo, hi = proportion_confint(hits, n, alpha=0.05, method="wilson")
        return a, lo, hi

    a1s, lo1s, hi1s = acc_and_ci(hits1_scale)
    a5s, lo5s, hi5s = acc_and_ci(hits5_scale)
    a1r, lo1r, hi1r = acc_and_ci(hits1_rand)
    a5r, lo5r, hi5r = acc_and_ci(hits5_rand)

    return {
        "acc1_scale":  a1s, "ci1_scale":  [lo1s, hi1s],
        "acc5_scale":  a5s, "ci5_scale":  [lo5s, hi5s],
        "acc1_random": a1r, "ci1_random": [lo1r, hi1r],
        "acc5_random": a5r, "ci5_random": [lo5r, hi5r],
    }


# ── Load existing sweep for comparison ────────────────────────────────────────

def load_existing(name: str) -> dict:
    """Load eps_sweep JSON for a dataset, keyed by rounded exp_epsilon."""
    path = BASE_DIR / "results" / f"{name}_eps_sweep.json"
    if not path.exists():
        return {}
    with open(path) as f:
        data = json.load(f)
    out = {}
    for entry in data["epsilon_sweep"]:
        key = round(entry["exp_epsilon"], 4)
        out[key] = entry
    return out


# ── Run one full dataset ───────────────────────────────────────────────────────

def run_dataset(label: str, data_path: Path, out_name: str) -> None:
    print(f"\n{'='*70}")
    print(f"Dataset: {label}")
    print(f"{'='*70}")

    dataset = load_dataset(data_path)
    N = len(dataset)
    print(f"Loaded {N} melodies from {data_path.name}")

    # Map short name for existing results lookup
    name_map = {
        "bayesian_europa":   "europa",
        "bayesian_meertens": "meertens",
        "bayesian_symbtr":   "symbtr",
        "bayesian_jazz":     "jazz",
    }
    existing = load_existing(name_map.get(out_name, ""))

    results = []
    for eps in EPS_SWEEP:
        exp_eps = exp(eps)
        print(f"\n  [eps = ln({exp_eps:.1f}) = {eps:.4f}]")
        t0 = time.time()

        bres = run_bayesian_eps(dataset, eps, num_trials=NUM_TRIALS, seed=SEED)

        elapsed = time.time() - t0
        print(f"  elapsed: {elapsed:.1f}s")

        b_ratio = (bres["acc1_scale"] / bres["acc1_random"]
                   if bres["acc1_random"] > 1e-9 else float("inf"))

        key = round(exp_eps, 4)
        ag_s = ag_r = aw_s = aw_r = None
        if key in existing:
            e = existing[key]
            ag = e.get("agnostic", {})
            aw = e.get("aware", {})
            ag_s = ag.get("acc1_scale")
            ag_r = ag.get("acc1_random")
            aw_s = aw.get("aware_acc1_scale")
            aw_r = aw.get("aware_acc1_random")

        print(f"  {'Attacker':<12} {'M-Scale':>9} {'M-Random':>9} {'Ratio':>8}")
        print(f"  {'-'*42}")
        if ag_s is not None:
            ag_ratio = ag_s / ag_r if ag_r and ag_r > 1e-9 else float("inf")
            print(f"  {'Agnostic':<12} {ag_s*100:8.2f}% {ag_r*100:8.2f}% {ag_ratio:7.2f}x")
        if aw_s is not None:
            aw_ratio = aw_s / aw_r if aw_r and aw_r > 1e-9 else float("inf")
            print(f"  {'Aware':<12} {aw_s*100:8.2f}% {aw_r*100:8.2f}% {aw_ratio:7.2f}x")
        print(f"  {'Bayesian':<12} {bres['acc1_scale']*100:8.2f}% {bres['acc1_random']*100:8.2f}% {b_ratio:7.2f}x")

        entry = {
            "epsilon":      eps,
            "exp_epsilon":  exp_eps,
            "bayesian":     bres,
            "agnostic":     {"acc1_scale": ag_s, "acc1_random": ag_r} if ag_s is not None else None,
            "aware":        {"acc1_scale": aw_s, "acc1_random": aw_r} if aw_s is not None else None,
            "bayesian_ratio": b_ratio,
            "aware_ratio":  (aw_s / aw_r if aw_r and aw_r > 1e-9 else None),
        }
        results.append(entry)

    # Save
    results_dir = BASE_DIR / "results"
    results_dir.mkdir(parents=True, exist_ok=True)
    out_path = results_dir / f"{out_name}.json"
    with open(out_path, "w") as fh:
        json.dump({
            "dataset":        label,
            "N":              N,
            "num_trials":     NUM_TRIALS,
            "epsilon_values": EPS_SWEEP,
            "results":        results,
        }, fh, indent=2)
    print(f"\n  Saved: {out_path}")

    # Summary table
    print(f"\n  SUMMARY: {label}")
    print(f"  {'eps':<10} {'Aw-S':>7} {'Aw-R':>7} {'Bay-S':>7} {'Bay-R':>7} {'Aw-Ratio':>9} {'Bay-Ratio':>10}")
    print(f"  {'-'*65}")
    for r in results:
        ee = r["exp_epsilon"]
        b  = r["bayesian"]
        aw = r.get("aware") or {}
        aw_s = (aw.get("acc1_scale") or 0) * 100
        aw_r = (aw.get("acc1_random") or 0) * 100
        b_s  = b["acc1_scale"]  * 100
        b_r  = b["acc1_random"] * 100
        aw_rat = r.get("aware_ratio") or 0
        b_rat  = r.get("bayesian_ratio") or 0
        print(f"  ln({ee:4.1f})  {aw_s:6.2f}% {aw_r:6.2f}% {b_s:6.2f}% {b_r:6.2f}% {aw_rat:8.2f}x {b_rat:9.2f}x")


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    print("Bayesian attacker — all four datasets, 15-epsilon grid, 30 trials")
    print("NOTE: M-Random marginalisation caches after first call (lru_cache).")
    print("      First epsilon may be slower; subsequent eps are faster.\n")

    for label, data_path, out_name in DATASETS:
        if not data_path.exists():
            print(f"WARNING: {data_path} not found — skipping {label}")
            continue
        run_dataset(label, data_path, out_name)

    print(f"\n{'='*70}")
    print("All done. Next step: python scripts/13_plot_bayesian_all.py")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
