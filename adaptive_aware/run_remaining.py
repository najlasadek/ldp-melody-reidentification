"""
run_remaining.py
================
Run adaptive vs heuristic for Meertens, SymbTr, Jazz with 10 trials.
Europa is already done (30 trials). Skips if result file already exists.
"""

from __future__ import annotations

import json
import sys
import time
import importlib.util as _ilu
from math import exp, log
from pathlib import Path

BASE_DIR = Path(__file__).parent
EXT_DIR = BASE_DIR.parent / "experiments_extended"

sys.path.insert(0, str(EXT_DIR))
from src.evaluation import run_experiment_aware
from src.metrics import chromatic_fraction
from src.utils_extended import load_dataset

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

NUM_TRIALS = 10

CORPORA = {
    "meertens_500": EXT_DIR / "data" / "processed" / "meertens_500.json",
    "symbtr_200":   EXT_DIR / "data" / "processed" / "symbtr_200.json",
    "jazz_200":     EXT_DIR / "data" / "processed" / "jazz_200.json",
}


def run_corpus(name: str, path: Path) -> None:
    out_path = BASE_DIR / "results" / f"adaptive_{name}.json"
    if out_path.exists():
        print(f"  SKIP {name}: already done")
        return

    dataset = load_dataset(path)
    mean_chi = sum(chromatic_fraction(m) for m in dataset) / len(dataset)
    print(f"\n{'='*60}")
    print(f"Corpus: {name}  N={len(dataset)}  chi_mean={mean_chi:.3f}  trials={NUM_TRIALS}")
    print(f"{'='*60}")

    results = []
    for i, eps in enumerate(EPS_SWEEP):
        t0 = time.time()
        print(f"  [{i+1}/{len(EPS_SWEEP)}] eps=ln({exp(eps):.2f})", end="  ", flush=True)

        orig = run_experiment_aware(dataset, eps, num_trials=NUM_TRIALS)
        orig_ratio = (orig["aware_acc1_scale"] / orig["aware_acc1_random"]
                      if orig["aware_acc1_random"] > 1e-9 else float("inf"))

        adapt = run_experiment_adaptive(dataset, eps, num_trials=NUM_TRIALS)

        elapsed = time.time() - t0
        print(f"heuristic={orig_ratio:.2f}x  adaptive={adapt['adaptive_ratio']:.2f}x  ({elapsed:.1f}s)")

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


def main():
    print("Running remaining corpora with 10 trials (Europa already done with 30)")
    for name, path in CORPORA.items():
        if not path.exists():
            print(f"  SKIP {name}: {path} not found")
            continue
        run_corpus(name, path)
    print("\nAll done. Run plot_adaptive.py to generate figures.")


if __name__ == "__main__":
    main()
