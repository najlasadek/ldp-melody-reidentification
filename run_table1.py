"""
run_table1.py
=============
Reproduces Table 1 from the paper.

Produces:
  - Agnostic attacker, all 200 melodies
  - Aware attacker, all 200 melodies
  - Aware attacker, near-diatonic subset (chi < 0.05, n=73)
  - Aware attacker, non-near-diatonic  (chi >= 0.05, n=127)

epsilon in {ln3, ln7, ln12}, 30 trials, 95% Wilson score CIs.

Usage
-----
  cd code/
  python run_table1.py

Requires data/melodies_europa.json (see README.md for download instructions).
"""
from __future__ import annotations

import sys
from math import exp, log
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.evaluation import run_experiment, run_experiment_aware
from src.utils import load_dataset
from src.metrics import chromatic_fraction, ALPHA

# ── dataset ─────────────────────────────────────────────────────────────────
DATASET_PATH = PROJECT_ROOT / "data" / "melodies_europa.json"
dataset = load_dataset(str(DATASET_PATH))[:200]

C_MAJOR_IDX     = [i for i, m in enumerate(dataset) if chromatic_fraction(m) < 0.05]
NON_C_MAJOR_IDX = [i for i, m in enumerate(dataset) if chromatic_fraction(m) >= 0.05]
c_major_mel     = [dataset[i] for i in C_MAJOR_IDX]
non_c_major_mel = [dataset[i] for i in NON_C_MAJOR_IDX]

print(f"Dataset: {len(dataset)} melodies")
print(f"Near-diatonic (chi<0.05): n={len(c_major_mel)}")
print(f"Non-near-diatonic:        n={len(non_c_major_mel)}")
print(f"Alpha (variance-matched): {ALPHA}")
print()

# ── parameters ──────────────────────────────────────────────────────────────
EPS_VALUES = [log(3), log(7), log(12)]
EPS_LABELS = ["ln3",  "ln7",  "ln12"]
NUM_TRIALS = 30
K          = 7

print("Retention probabilities p = e^eps / (e^eps + k-1), k=7:")
for lbl, eps in zip(EPS_LABELS, EPS_VALUES):
    e = exp(eps)
    p = e / (e + K - 1)
    print(f"  eps={lbl}: p={p:.6f}")
print()

# ── helpers ──────────────────────────────────────────────────────────────────
def fmt(mean, ci_lo, ci_hi):
    return f"{mean*100:.1f} [{ci_lo*100:.2f}, {ci_hi*100:.2f}]"

def ratio_str(r):
    return f"{r:.2f}x"

# ── run all conditions ────────────────────────────────────────────────────────
print("Running experiments (this takes a few minutes) ...")
print()

# 1. Agnostic — All
print("Agnostic / All 200 ...")
ag_all = []
for lbl, eps in zip(EPS_LABELS, EPS_VALUES):
    print(f"  eps={lbl}", end=" ... ", flush=True)
    res = run_experiment(dataset, eps, num_trials=NUM_TRIALS)
    ag_all.append(res)
    print(f"M-Random={res['acc1_random']*100:.1f}%  M-Scale={res['acc1_scale']*100:.1f}%")

# 2. Aware — All
print()
print("Aware / All 200 ...")
aw_all = []
for lbl, eps in zip(EPS_LABELS, EPS_VALUES):
    print(f"  eps={lbl}", end=" ... ", flush=True)
    res = run_experiment_aware(dataset, eps, num_trials=NUM_TRIALS)
    aw_all.append(res)
    print(f"M-Random={res['aware_acc1_random']*100:.1f}%  M-Scale={res['aware_acc1_scale']*100:.1f}%")

# 3. Aware — Near-diatonic
print()
print(f"Aware / Near-diatonic (n={len(c_major_mel)}) ...")
aw_cmaj = []
for lbl, eps in zip(EPS_LABELS, EPS_VALUES):
    print(f"  eps={lbl}", end=" ... ", flush=True)
    res = run_experiment_aware(c_major_mel, eps, num_trials=NUM_TRIALS)
    aw_cmaj.append(res)
    print(f"M-Random={res['aware_acc1_random']*100:.1f}%  M-Scale={res['aware_acc1_scale']*100:.1f}%")

# 4. Aware — Non-near-diatonic
print()
print(f"Aware / Non-near-diatonic (n={len(non_c_major_mel)}) ...")
aw_noncmaj = []
for lbl, eps in zip(EPS_LABELS, EPS_VALUES):
    print(f"  eps={lbl}", end=" ... ", flush=True)
    res = run_experiment_aware(non_c_major_mel, eps, num_trials=NUM_TRIALS)
    aw_noncmaj.append(res)
    print(f"M-Random={res['aware_acc1_random']*100:.1f}%  M-Scale={res['aware_acc1_scale']*100:.1f}%")

# ── Table 1 ──────────────────────────────────────────────────────────────────
W = 95
print()
print("=" * W)
print("TABLE 1  (alpha=1.66, Wilson 95% CI, 30 trials)")
print("Acc@1 format: point_estimate [ci_low, ci_high]")
print("=" * W)
header = (f"{'Attacker':<10} {'Subset':<24} {'eps':<5} "
          f"{'M-RANDOM Acc@1':<28} {'M-SCALE Acc@1':<28} {'Ratio'}")
print(header)
print("-" * W)

# Agnostic / All
for lbl, res in zip(EPS_LABELS, ag_all):
    r    = res["acc1_random"];  lo_r, hi_r = res["ci1_random"]
    s    = res["acc1_scale"];   lo_s, hi_s = res["ci1_scale"]
    ratio = s / max(r, 1e-9)
    print(f"{'Agnostic':<10} {'All (200)':<24} {lbl:<5} "
          f"{fmt(r, lo_r, hi_r):<28} {fmt(s, lo_s, hi_s):<28} {ratio_str(ratio)}")

print()

# Aware / All
for lbl, res in zip(EPS_LABELS, aw_all):
    r    = res["aware_acc1_random"];  lo_r, hi_r = res["aware_ci1_random"]
    s    = res["aware_acc1_scale"];   lo_s, hi_s = res["aware_ci1_scale"]
    ratio = s / max(r, 1e-9)
    print(f"{'Aware':<10} {'All (200)':<24} {lbl:<5} "
          f"{fmt(r, lo_r, hi_r):<28} {fmt(s, lo_s, hi_s):<28} {ratio_str(ratio)}")

print()

# Aware / Near-diatonic
nc = len(c_major_mel)
for lbl, res in zip(EPS_LABELS, aw_cmaj):
    r    = res["aware_acc1_random"];  lo_r, hi_r = res["aware_ci1_random"]
    s    = res["aware_acc1_scale"];   lo_s, hi_s = res["aware_ci1_scale"]
    ratio = s / max(r, 1e-9)
    print(f"{'Aware':<10} {f'Near-diat. (n={nc})':<24} {lbl:<5} "
          f"{fmt(r, lo_r, hi_r):<28} {fmt(s, lo_s, hi_s):<28} {ratio_str(ratio)}")

print()

# Aware / Non-near-diatonic
nn = len(non_c_major_mel)
for lbl, res in zip(EPS_LABELS, aw_noncmaj):
    r    = res["aware_acc1_random"];  lo_r, hi_r = res["aware_ci1_random"]
    s    = res["aware_acc1_scale"];   lo_s, hi_s = res["aware_ci1_scale"]
    ratio = s / max(r, 1e-9)
    print(f"{'Aware':<10} {f'Non-diat. (n={nn})':<24} {lbl:<5} "
          f"{fmt(r, lo_r, hi_r):<28} {fmt(s, lo_s, hi_s):<28} {ratio_str(ratio)}")

print("=" * W)
print(f"Chance baseline = {100/len(dataset):.1f}%   (1 / {len(dataset)})")
print()

# ── Acc@5 supplement ─────────────────────────────────────────────────────────
print("=" * W)
print("ACC@5 SUPPLEMENT")
print("=" * W)
header5 = (f"{'Attacker':<10} {'Subset':<24} {'eps':<5} "
           f"{'M-RANDOM Acc@5':<28} {'M-SCALE Acc@5':<28} {'Ratio'}")
print(header5)
print("-" * W)

for lbl, res in zip(EPS_LABELS, ag_all):
    r = res["acc5_random"]; lo_r, hi_r = res["ci5_random"]
    s = res["acc5_scale"];  lo_s, hi_s = res["ci5_scale"]
    print(f"{'Agnostic':<10} {'All (200)':<24} {lbl:<5} "
          f"{fmt(r, lo_r, hi_r):<28} {fmt(s, lo_s, hi_s):<28} {ratio_str(s/max(r,1e-9))}")

print()
for lbl, res in zip(EPS_LABELS, aw_all):
    r = res["aware_acc5_random"]; lo_r, hi_r = res["aware_ci5_random"]
    s = res["aware_acc5_scale"];  lo_s, hi_s = res["aware_ci5_scale"]
    print(f"{'Aware':<10} {'All (200)':<24} {lbl:<5} "
          f"{fmt(r, lo_r, hi_r):<28} {fmt(s, lo_s, hi_s):<28} {ratio_str(s/max(r,1e-9))}")

print("=" * W)
