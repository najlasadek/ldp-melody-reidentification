"""
18_run_normalised_ablation.py
=============================
Key-balanced ablation (paper Appendix A.7) with the LENGTH-NORMALISED distance.

Uses the same evaluation code as 17_run_normalised_sweeps.py on
../data/melodies_synthetic_balanced.json (200 diatonic melodies whose keys are
spread evenly over all 12 pitch classes), at eps in {ln3, ln7, ln12}.

Usage (from experiments_extended/):
    python scripts/18_run_normalised_ablation.py

Results saved to: results_normalised/ablation_balanced.json
"""
from __future__ import annotations

import importlib.util
import json
from math import log
from pathlib import Path

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("sweeps", HERE / "17_run_normalised_sweeps.py")
sweeps = importlib.util.module_from_spec(spec)
spec.loader.exec_module(sweeps)

DATA = HERE.parent.parent / "data" / "melodies_synthetic_balanced.json"


def main():
    data = json.load(open(DATA))
    rows = []
    print(f"Key-balanced dataset: {len(data)} melodies")
    for k in [3, 7, 12]:
        h = sweeps.evaluate(data, log(k), normalise=True, with_adaptive=False)
        row = {"exp_epsilon": k, **h,
               "agnostic_ratio": sweeps.ratio(h["ag_s"], h["ag_r"]),
               "aware_ratio": sweeps.ratio(h["aw_s"], h["aw_r"])}
        rows.append(row)
        print(f"  ln{k}: aware {h['aw_s']} vs {h['aw_r']} -> {row['aware_ratio']:.2f}x   "
              f"agnostic {h['ag_s']} vs {h['ag_r']} -> {row['agnostic_ratio']:.2f}x", flush=True)
    sweeps.OUT_DIR.mkdir(exist_ok=True)
    out = sweeps.OUT_DIR / "ablation_balanced.json"
    json.dump({"dataset": "synthetic key-balanced", "N": len(data), "num_trials": sweeps.NUM_TRIALS,
               "distance": "length-normalised: d / (n-1)", "results": rows}, open(out, "w"), indent=2)
    print(f"Saved {out}")


if __name__ == "__main__":
    main()
