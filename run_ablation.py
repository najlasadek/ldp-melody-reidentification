"""
run_ablation.py
===============
Ablation study: key-balanced synthetic dataset.

Constructs a balanced dataset of 240 melodies:
  20 seed melodies (diatonic, from C-major scale) x 12 transpositions.

On this dataset, M-Scale and M-Random should achieve nearly identical
re-identification accuracy (ratio ≈ 1×), confirming that M-Scale's excess
leakage on the Europa corpus is driven by key-distribution skew, not
by mechanism structure alone.

Usage
-----
  cd code/
  python run_ablation.py
"""
from __future__ import annotations

import sys
from math import log
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.evaluation import run_experiment
from src.utils import load_dataset
from src.metrics import ALPHA

# ── load key-balanced dataset ─────────────────────────────────────────────────
DATASET_PATH = PROJECT_ROOT / "data" / "melodies_synthetic_balanced.json"
dataset = load_dataset(str(DATASET_PATH))

print(f"Key-balanced dataset: {len(dataset)} melodies (20 seed melodies x 12 transpositions)")
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
print("Expected: ratio ~1x for all epsilon values")
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
print()
print("Interpretation: ratios near 1x confirm that the gap observed on the")
print("Europa corpus (up to 8.44x) is due to key-distribution skew, not")
print("to the mechanism structure itself.")
