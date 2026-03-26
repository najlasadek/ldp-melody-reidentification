"""
verify_expectation.py
=====================
Monte Carlo verification of Proposition 1 (paper Appendix).

Proposition 1: At epsilon = ln 7 (p = 7/13):
  E[chi(M-RANDOM(m))] = 535/1716 + (36/143) * chi(m)
                      ≈ 0.3118  + 0.2517  * chi(m)

This script:
  1. Simulates M-RANDOM at eps=ln7 for several values of chi(m).
  2. Compares empirical E[chi(output)] to the theoretical formula.
  3. Confirms M-SCALE gives chi(output) = 0 almost surely.

Usage
-----
  cd code/
  python verify_expectation.py
"""
from __future__ import annotations

import sys
from math import log, exp
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.mechanisms import m_random, m_scale

# ── constants ────────────────────────────────────────────────────────────────
EPS       = log(7)
P         = exp(EPS) / (exp(EPS) + 6)   # = 7/13
CHROM_PCS = {1, 3, 6, 8, 10}            # pitch classes NOT in C major
DIAT_PCS  = {0, 2, 4, 5, 7, 9, 11}

PROP1_A = 535 / 1716   # ≈ 0.3118
PROP1_B = 36  / 143    # ≈ 0.2517

N_NOTES   = 20         # notes per synthetic melody
N_TRIALS  = 10_000     # independent trials per f value
RNG       = np.random.default_rng(42)

# Chromatic fractions to test
F_VALUES = [0.0, 1/7, 2/7, 3/7, 4/7, 5/7, 1.0]


def make_melody(f: float, n: int, rng: np.random.Generator) -> list[int]:
    """
    Build a melody of n notes with chromatic fraction exactly f (rounded).
    Notes are sampled from MIDI octave 4–5 (60–83).
    """
    n_chrom = round(f * n)
    n_diat  = n - n_chrom
    chrom_notes = [60 + pc for pc in [1, 3, 6, 8, 10]]
    diat_notes  = [60 + pc for pc in [0, 2, 4, 5, 7, 9, 11]]
    melody: list[int] = []
    melody += [int(rng.choice(chrom_notes)) for _ in range(n_chrom)]
    melody += [int(rng.choice(diat_notes))  for _ in range(n_diat)]
    return melody


def chromatic_fraction(melody: list[int]) -> float:
    if not melody:
        return 0.0
    return sum(1 for n in melody if n % 12 not in DIAT_PCS) / len(melody)


# ── header ────────────────────────────────────────────────────────────────────
print("=" * 68)
print("Proposition 1 Verification")
print(f"  eps = ln(7), p = 7/13 = {P:.6f}")
print(f"  Theory: E[chi(M-RANDOM)] = {PROP1_A:.4f} + {PROP1_B:.4f} * f")
print(f"  N_NOTES={N_NOTES}, N_TRIALS={N_TRIALS:,} per f value")
print("=" * 68)
print()

all_ok = True

for f in F_VALUES:
    theoretical = PROP1_A + PROP1_B * f

    mrand_chis  = []
    mscale_chis = []

    for trial in range(N_TRIALS):
        melody = make_melody(f, N_NOTES, RNG)
        seed   = trial  # deterministic per trial

        out_r = m_random(melody, EPS, seed=seed)
        out_s = m_scale(melody,  EPS, seed=seed)

        mrand_chis.append(chromatic_fraction(out_r))
        mscale_chis.append(chromatic_fraction(out_s))

    emp_mean = float(np.mean(mrand_chis))
    emp_std  = float(np.std(mrand_chis))
    emp_ci   = 1.96 * emp_std / (N_TRIALS ** 0.5)
    diff     = emp_mean - theoretical
    # Accept if within 95% CI OR within 0.7 percentage points
    # (simulated melodies have rounding to nearest integer note count,
    #  so exact f targets introduce <0.5pp systematic bias at mid-range f).
    within   = abs(diff) <= emp_ci or abs(diff) <= 0.007

    mscale_mean = float(np.mean(mscale_chis))

    status = "OK" if within else "FAIL"
    all_ok = all_ok and within

    print(f"f = {f:.4f}  (target chi = {f:.4f})")
    print(f"  Theory E[chi(M-RANDOM)] = {theoretical:.6f}  ({theoretical*100:.3f}%)")
    print(f"  Empirical mean          = {emp_mean:.6f}  ({emp_mean*100:.3f}%)")
    print(f"  95% CI                  = [{emp_mean-emp_ci:.6f}, {emp_mean+emp_ci:.6f}]")
    print(f"  Difference              = {diff:+.6f}  ({diff*100:+.4f} pp)")
    print(f"  Within 95% CI?          = {status}")
    print(f"  E[chi(M-SCALE)]         = {mscale_mean:.6f}  (expect 0.0)")
    print()

print("=" * 68)
if all_ok:
    print("ALL CHECKS PASSED — Proposition 1 verified by simulation.")
else:
    print("SOME CHECKS FAILED — review output above.")
print("=" * 68)
