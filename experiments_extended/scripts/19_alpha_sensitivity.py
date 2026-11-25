"""
19_alpha_sensitivity.py
=======================
Sensitivity of the length-normalised results to the distance weight alpha
(paper Appendix A.5). Re-runs the agnostic and aware heuristic attackers with
alpha in {1.2, 1.9} (the paper uses 1.66) at eps in {ln3, ln7, ln16} on all four
corpora, using the evaluation code of 17_run_normalised_sweeps.py.

Usage (from experiments_extended/):
    python scripts/19_alpha_sensitivity.py

Results saved to: results_normalised/alpha_sensitivity.json
"""
from __future__ import annotations

import importlib.util
import json
from math import log
from multiprocessing import Pool
from pathlib import Path

HERE = Path(__file__).resolve().parent
ALPHAS = [1.2, 1.9]
EXP_EPS = [3, 7, 16]
ORDER = ["meertens", "europa", "symbtr", "jazz"]


def load_sweeps():
    spec = importlib.util.spec_from_file_location("sweeps", HERE / "17_run_normalised_sweeps.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def task(args):
    corpus, k, alpha = args
    sweeps = load_sweeps()
    sweeps.ALPHA = alpha  # top1() reads the module-level ALPHA
    data = json.load(open(sweeps.BASE_DIR / "data" / sweeps.CORPORA[corpus][0]))
    h = sweeps.evaluate(data, log(k), normalise=True, with_adaptive=False)
    return corpus, k, alpha, h


def main():
    tasks = [(c, k, a) for c in ORDER for k in EXP_EPS for a in ALPHAS]
    rows = []
    with Pool(11) as pool:
        for corpus, k, alpha, h in pool.imap_unordered(task, tasks):
            row = {"corpus": corpus, "exp_epsilon": k, "alpha": alpha, **h,
                   "aware_ratio": h["aw_s"] / h["aw_r"], "agnostic_ratio": h["ag_s"] / h["ag_r"]}
            rows.append(row)
            print(f"{corpus:8s} ln{k:<2d} alpha={alpha}: aware {h['aw_s']} vs {h['aw_r']} "
                  f"-> {row['aware_ratio']:.2f}x  agnostic -> {row['agnostic_ratio']:.2f}x", flush=True)
    sweeps = load_sweeps()
    out = sweeps.OUT_DIR / "alpha_sensitivity.json"
    rows.sort(key=lambda r: (ORDER.index(r["corpus"]), r["exp_epsilon"], r["alpha"]))
    json.dump({"alphas": ALPHAS, "exp_eps": EXP_EPS, "distance": "length-normalised: d / (n-1)",
               "results": rows}, open(out, "w"), indent=2)
    print(f"Saved {out}")


if __name__ == "__main__":
    main()
