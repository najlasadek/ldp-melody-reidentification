"""
20_degeneracy_unnormalised.py
=============================
Retrieval degeneracy with the ORIGINAL (un-normalised) distance at eps = ln 8,
for the comparison table in paper Appendix A.7. Records, for M-Random queries
under the aware heuristic attacker, the number of distinct candidates ever
returned and the share of predictions taken by the most frequent candidate.
(The length-normalised counterparts come from 16_run_length_normalised.py.)

Usage (from experiments_extended/):
    python scripts/20_degeneracy_unnormalised.py

Results saved to: results_normalised/degeneracy_unnormalised_ln8.json
"""
from __future__ import annotations

import importlib.util
import json
from math import log
from multiprocessing import Pool
from pathlib import Path

HERE = Path(__file__).resolve().parent


def load16():
    spec = importlib.util.spec_from_file_location("ln16", HERE / "16_run_length_normalised.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def job(corpus):
    m = load16()
    r = m.run(m.load(corpus), log(8), normalise=False)
    return corpus, {k: r[k] for k in ["aware_random", "aware_scale", "agnostic_random"]}


def main():
    with Pool(4) as pool:
        res = dict(pool.map(job, ["meertens", "europa", "symbtr", "jazz"]))
    out = HERE.parent / "results_normalised" / "degeneracy_unnormalised_ln8.json"
    out.parent.mkdir(exist_ok=True)
    json.dump(res, open(out, "w"), indent=2)
    for c, v in res.items():
        ar = v["aware_random"]
        print(f"{c}: M-Random hits {ar['hits']}, distinct {ar['distinct_predicted']}, "
              f"top share {100 * ar['top_candidate_share']:.1f}%")
    print(f"Saved {out}")


if __name__ == "__main__":
    main()
