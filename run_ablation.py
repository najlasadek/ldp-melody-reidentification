"""
run_ablation.py
===============
Ablation study: key-balanced synthetic dataset.

Loads the key-balanced synthetic dataset (data/melodies_synthetic_balanced.json:
200 diatonic melodies whose keys are spread evenly over all 12 pitch classes)
and runs both the agnostic and the mechanism-aware (heuristic) attacker at
eps in {ln3, ln7, ln12}, 30 trials each.

Compare the ratios with Europa 200 at the same eps and attacker
(paper Table 4): agnostic 0.64x / 3.08x / 3.64x, aware 3.50x / 12.83x / 8.41x.

Usage
-----
  # from the repository root
  python run_ablation.py
"""
from __future__ import annotations

import sys
from math import log
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.evaluation import run_experiment, run_experiment_aware
from src.utils import load_dataset
from src.metrics import ALPHA

# ── load key-balanced dataset ─────────────────────────────────────────────────
DATASET_PATH = PROJECT_ROOT / "data" / "melodies_synthetic_balanced.json"
dataset = load_dataset(str(DATASET_PATH))

print(f"Key-balanced dataset: {len(dataset)} melodies (keys spread evenly over all 12 pitch classes)")
print(f"Alpha: {ALPHA}")
print()

# ── run agnostic attacker ─────────────────────────────────────────────────────
EPS_VALUES = [log(3), log(7), log(12)]
EPS_LABELS = ["ln3",  "ln7",  "ln12"]
NUM_TRIALS = 30

print("Running agnostic attacker on key-balanced dataset ...")
print()

results = []
for lbl, eps in zip(EPS_LABELS, EPS_VALUES):
    print(f"  eps={lbl}", end=" ... ", flush=True)
    res = run_experiment(dataset, eps, num_trials=NUM_TRIALS)
    results.append(res)
    r = res["acc1_random"]
    s = res["acc1_scale"]
    ratio = s / max(r, 1e-9)
    print(f"M-Random={r*100:.1f}%  M-Scale={s*100:.1f}%  Ratio={ratio:.2f}x")

# ── Summary ───────────────────────────────────────────────────────────────────
W = 75
print()
print("=" * W)
print("ABLATION TABLE  (agnostic attacker, key-balanced synthetic dataset)")
print("=" * W)
header = f"{'eps':<6} {'M-RANDOM Acc@1':<20} {'M-SCALE Acc@1':<20} {'Ratio'}"
print(header)
print("-" * W)

for lbl, res in zip(EPS_LABELS, results):
    r    = res["acc1_random"];  lo_r, hi_r = res["ci1_random"]
    s    = res["acc1_scale"];   lo_s, hi_s = res["ci1_scale"]
    ratio = s / max(r, 1e-9)
    r_str = f"{r*100:.1f} [{lo_r*100:.2f}, {hi_r*100:.2f}]"
    s_str = f"{s*100:.1f} [{lo_s*100:.2f}, {hi_s*100:.2f}]"
    print(f"{lbl:<6} {r_str:<20} {s_str:<20} {ratio:.2f}x")

print("=" * W)

# ── run mechanism-aware (heuristic) attacker ─────────────────────────────────
print()
print("Running mechanism-aware (heuristic) attacker on key-balanced dataset ...")
print()
print("=" * W)
print("ABLATION TABLE  (aware heuristic attacker, key-balanced synthetic dataset)")
print("=" * W)
print(header)
print("-" * W)
for lbl, eps in zip(EPS_LABELS, EPS_VALUES):
    res = run_experiment_aware(dataset, eps, num_trials=NUM_TRIALS)
    r    = res["aware_acc1_random"];  lo_r, hi_r = res["aware_ci1_random"]
    s    = res["aware_acc1_scale"];   lo_s, hi_s = res["aware_ci1_scale"]
    ratio = s / max(r, 1e-9)
    r_str = f"{r*100:.1f} [{lo_r*100:.2f}, {hi_r*100:.2f}]"
    s_str = f"{s*100:.1f} [{lo_s*100:.2f}, {hi_s*100:.2f}]"
    print(f"{lbl:<6} {r_str:<20} {s_str:<20} {ratio:.2f}x   "
          f"(M-Scale {round(s*len(dataset)*NUM_TRIALS)} vs "
          f"M-Random {round(r*len(dataset)*NUM_TRIALS)} correct)")
print("=" * W)
print()
print("Europa 200 at the same eps (paper Table 4):")
print("  agnostic 0.64x / 3.08x / 3.64x,  aware heuristic 3.50x / 12.83x / 8.41x")
