"""
21_europa_chromatic_subset.py
=============================
Aware heuristic M-Scale/M-Random ratio on the chromatic Europa melodies
(chi >= 0.05, N = 127), evaluated as their own candidate database, with the
length-normalised distance (paper Section 4.2). Uses the evaluation code of
17_run_normalised_sweeps.py.

Usage (from experiments_extended/):
    python scripts/21_europa_chromatic_subset.py

Results saved to: results_normalised/europa_chromatic_subset.json
"""
from __future__ import annotations

import importlib.util
import json
from math import log
from multiprocessing import Pool
from pathlib import Path

HERE = Path(__file__).resolve().parent
EXP_EPS = [3, 4, 5, 6, 7, 8, 9, 10, 12, 14, 16]


def load17():
    spec = importlib.util.spec_from_file_location("s17", HERE / "17_run_normalised_sweeps.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def job(k):
    m = load17()
    data = json.load(open(m.BASE_DIR / "data" / "europa_200.json"))
    chrom = [x for x in data if m.chromatic_fraction(x) >= 0.05]
    h = m.evaluate(chrom, log(k), normalise=True, with_adaptive=False)
    return k, len(chrom), h


def main():
    with Pool(11) as pool:
        res = sorted(pool.map(job, EXP_EPS))
    out = {k: {"N": n, **h, "aware_ratio": h["aw_s"] / h["aw_r"]} for k, n, h in res}
    path = HERE.parent / "results_normalised" / "europa_chromatic_subset.json"
    json.dump(out, open(path, "w"), indent=2)
    for k, v in out.items():
        print(f"ln{k}: N={v['N']} aware {v['aw_s']} vs {v['aw_r']} -> {v['aware_ratio']:.2f}x")
    print(f"Saved {path}")


if __name__ == "__main__":
    main()
