"""
run_adaptive.py
===============
Run the epsilon-adaptive aware attacker on all four corpora and save results.

Compares:
  - Original heuristic aware attacker (fixed ln7 constants)
  - Epsilon-adaptive aware attacker (correct formula at each eps)

Run from the adaptive_aware/ directory:
    python run_adaptive.py

Results saved to results/adaptive_<corpus>.json
"""

from __future__ import annotations

import json
import sys
import time
from math import exp, log
from pathlib import Path

BASE_DIR = Path(__file__).parent
EXT_DIR = BASE_DIR.parent / "experiments_extended"

# experiments_extended/src for shared utilities
sys.path.insert(0, str(EXT_DIR))

from src.evaluation import run_experiment_aware        # original fixed-heuristic
from src.metrics import chromatic_fraction
from src.utils_extended import load_dataset

# Import adaptive module directly by file path to avoid src/ name collision
import importlib.util as _ilu
_spec = _ilu.spec_from_file_location(
    "adaptive_evaluation",
    BASE_DIR / "src" / "adaptive_evaluation.py"
)
_mod = _ilu.module_from_spec(_spec)
_spec.loader.exec_module(_mod)
run_experiment_adaptive = _mod.run_experiment_adaptive

EPS_SWEEP = [
    log(1.5), log(2), log(3), log(4), log(5),
    log(6),   log(7), log(8), log(9), log(10),
    log(12),  log(14), log(16), log(20), log(25),
]

NUM_TRIALS = 30

CORPORA = {
    "europa_200":    EXT_DIR / "data" / "europa_200.json",
    "meertens_500":  EXT_DIR / "data" / "processed" / "meertens_500.json",
    "symbtr_200":    EXT_DIR / "data" / "processed" / "symbtr_200.json",
    "jazz_200":      EXT_DIR / "data" / "processed" / "jazz_200.json",
}


def run_corpus(name: str, path: Path) -> list[dict]:
    dataset = load_dataset(path)
    print(f"\n{'='*60}")
    print(f"Corpus: {name}  (N={len(dataset)})")
    mean_chi = sum(chromatic_fraction(m) for m in dataset) / len(dataset)
    print(f"Mean chi: {mean_chi:.3f}")
    print(f"{'='*60}")

    results = []
    for i, eps in enumerate(EPS_SWEEP):
        t0 = time.time()
        print(f"  [{i+1}/{len(EPS_SWEEP)}] eps=ln({exp(eps):.2f})", end="  ", flush=True)

        # Original fixed-heuristic aware
        orig = run_experiment_aware(dataset, eps, num_trials=NUM_TRIALS)
        orig_ratio = (orig["aware_acc1_scale"] / orig["aware_acc1_random"]
                      if orig["aware_acc1_random"] > 1e-9 else float("inf"))

        # Epsilon-adaptive aware
        adapt = run_experiment_adaptive(dataset, eps, num_trials=NUM_TRIALS)

        elapsed = time.time() - t0
        print(f"heuristic={orig_ratio:.2f}x  adaptive={adapt['adaptive_ratio']:.2f}x  "
              f"({elapsed:.1f}s)")

        results.append({
            "epsilon": eps,
            "exp_epsilon": exp(eps),
            "heuristic": {
                "acc1_scale":  orig["aware_acc1_scale"],
                "acc1_random": orig["aware_acc1_random"],
                "ratio":       orig_ratio,
                "ci1_scale":   orig["aware_ci1_scale"],
                "ci1_random":  orig["aware_ci1_random"],
            },
            "adaptive": {
                "acc1_scale":  adapt["adaptive_acc1_scale"],
                "acc1_random": adapt["adaptive_acc1_random"],
                "ratio":       adapt["adaptive_ratio"],
                "ci1_scale":   adapt["adaptive_ci1_scale"],
                "ci1_random":  adapt["adaptive_ci1_random"],
                "intercept":   adapt["intercept"],
                "slope":       adapt["slope"],
                "p":           adapt["p"],
            },
        })

    out_path = BASE_DIR / "results" / f"adaptive_{name}.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as fh:
        json.dump({
            "corpus": name,
            "N": len(dataset),
            "mean_chi": mean_chi,
            "num_trials": NUM_TRIALS,
            "epsilon_sweep": results,
        }, fh, indent=2)
    print(f"  Saved: {out_path}")
    return results


def main():
    print("Epsilon-adaptive aware attacker experiment")
    print("Comparison: heuristic (fixed ln7) vs adaptive (correct eps)")

    all_results = {}
    for name, path in CORPORA.items():
        if not path.exists():
            print(f"  SKIP {name}: {path} not found")
            continue
        all_results[name] = run_corpus(name, path)

    print("\n" + "="*60)
    print("SUMMARY — adaptive_ratio vs heuristic_ratio at peak eps")
    print("="*60)
    for name, results in all_results.items():
        peak_h = max(r["heuristic"]["ratio"] for r in results
                     if r["heuristic"]["ratio"] != float("inf"))
        peak_a = max(r["adaptive"]["ratio"] for r in results
                     if r["adaptive"]["ratio"] != float("inf"))
        eps_h = next(r["exp_epsilon"] for r in results
                     if r["heuristic"]["ratio"] == peak_h)
        eps_a = next(r["exp_epsilon"] for r in results
                     if r["adaptive"]["ratio"] == peak_a)
        print(f"  {name:20s}  heuristic peak={peak_h:.2f}x @ln{eps_h:.0f}"
              f"  adaptive peak={peak_a:.2f}x @ln{eps_a:.0f}")


if __name__ == "__main__":
    main()
