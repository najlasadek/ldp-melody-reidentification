"""
adaptive_evaluation.py
======================
Epsilon-adaptive aware attacker for M-Random.

The original paper uses Proposition 1 constants (535/1716, 36/143) derived at
eps=ln7 as a fixed heuristic across all epsilon. This module replaces that with
the correct epsilon-adaptive formula.

Derivation of E[chi(M-Random(m))] as a function of p = e^eps/(e^eps+6):

For M-Random, the alphabet A is a random 7-note subset of {0,...,11}.
By linearity of expectation, we compute Pr(output note is chromatic)
conditioning on whether the input note is chromatic or diatonic.

C-major has 7 diatonic PCs and 5 chromatic PCs.
A random 7-note alphabet A is drawn from all 12 PCs.

Case 1: Input note x is chromatic (prob f = chi(m)).
  Fix chromatic x. There are 11 other PCs: 4 chromatic, 7 diatonic.
  - x in A (prob 7/12): retain x (chromatic) w.p. p; else replace uniform
    from A\{x} (6 notes: 4 chrom / 7-1=6 ??? we need to be careful).

  Actually, A is drawn fresh each time and contains 7 of 12 PCs uniformly.
  For a fixed input note x:

  P(x in A) = C(11,6)/C(12,7) = 462/792 = 7/12.
  P(x not in A) = 5/12.

  Given x in A (7-note alphabet containing x):
    Remaining 6 slots filled from 11 others: 4 chromatic, 7 diatonic.
    P(output chromatic | x in A) = p*1 + (1-p)*(E[# chrom in A\{x}]/6)
      where E[# chrom in A\{x}] = 6 * (4/11)  [hypergeometric mean]
    = p + (1-p)*(4/11)

  Given x not in A (7-note alphabet not containing x):
    x is replaced uniform from A. A has 7 notes from the remaining 11.
    E[# chromatic in A] = 7 * (4/11)
    P(output chromatic | x not in A) = 7*(4/11)/7 = 4/11

  P_cc(p) = (7/12)*(p + (1-p)*4/11) + (5/12)*(4/11)
           = (7/12)*p + (7/12)*(1-p)*(4/11) + (5/12)*(4/11)
           = (7/12)*p + (4/11)*[(7/12)*(1-p) + 5/12]
           = (7/12)*p + (4/11)*[(7 - 7p + 5)/12]
           = (7/12)*p + (4/11)*(12 - 7p)/12
           = p*(7/12 - 28/(12*11)) + 4*(12-7p)/(11*12)  ... simplify:
  P_cc(p) = (7p)/12 + (4/11)*( (12 - 7p)/12 )
           = [7p*11 + 4*(12-7p)] / (12*11)
           = [77p + 48 - 28p] / 132
           = (49p + 48) / 132

Case 2: Input note x is diatonic (prob 1-f).
  Fix diatonic x. There are 11 others: 5 chromatic, 6 diatonic.
  P(x in A) = 7/12, P(x not in A) = 5/12.

  Given x in A:
    A\{x} has 6 notes from remaining 11: 5 chrom, 6 diatonic.
    E[# chrom in A\{x}] = 6*(5/11)
    P(output chromatic | x in A) = p*0 + (1-p)*(5/11)
    = (1-p)*5/11

  Given x not in A:
    A has 7 notes from remaining 11: 5 chrom, 6 diatonic.
    E[# chrom in A] = 7*(5/11)
    P(output chromatic | x not in A) = 5/11

  P_dc(p) = (7/12)*(1-p)*(5/11) + (5/12)*(5/11)
           = (5/11)*[(7/12)*(1-p) + 5/12]
           = (5/11)*[(7 - 7p + 5)/12]
           = (5/11)*(12 - 7p)/12
           = 5*(12 - 7p) / (11*12)
           = (60 - 35p) / 132

Combining:
  E[chi(m')] = P_dc(p) + (P_cc(p) - P_dc(p)) * f
             = (60-35p)/132 + [(49p+48)/132 - (60-35p)/132] * f
             = (60-35p)/132 + (84p-12)/132 * f

Verification at p=7/13 (eps=ln7):
  P_dc = (60 - 35*7/13)/132 = (60 - 245/13)/132 = (780/13 - 245/13)/132
       = (535/13)/132 = 535/1716  ✓
  P_cc - P_dc = (84*(7/13) - 12)/132 = (588/13 - 156/13)/132
              = (432/13)/132 = 432/1716 = 36/143  ✓
"""

from __future__ import annotations

import sys
from math import exp
from pathlib import Path

# Allow importing from experiments_extended/src
_EXT_DIR = Path(__file__).parent.parent.parent / "experiments_extended"
sys.path.insert(0, str(_EXT_DIR))

from src.mechanisms import m_random, m_scale
from src.metrics import chromatic_fraction, interval_hamming, ALPHA


def _keep_prob(epsilon: float, k: int = 7) -> float:
    e_eps = exp(epsilon)
    return e_eps / (e_eps + k - 1)


def _adaptive_chi_coeffs(epsilon: float) -> tuple[float, float]:
    """
    Return (intercept, slope) for E[chi(M-Random(m))] = intercept + slope*f
    at the given epsilon, using the epsilon-adaptive formula derived above.

    intercept = P_dc(p) = (60 - 35p) / 132
    slope     = P_cc(p) - P_dc(p) = (84p - 12) / 132
    """
    p = _keep_prob(epsilon)
    intercept = (60 - 35 * p) / 132
    slope = (84 * p - 12) / 132
    return intercept, slope


def _wilson_ci(successes: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return 0.0, 0.0
    p_hat = successes / n
    denom = 1 + z * z / n
    centre = (p_hat + z * z / (2 * n)) / denom
    margin = z * ((p_hat * (1 - p_hat) / n + z * z / (4 * n * n)) ** 0.5) / denom
    return max(0.0, centre - margin), min(1.0, centre + margin)


def _aware_dist_scale(perturbed: list[int], candidate: list[int], alpha: float) -> float:
    """Aware distance for M-Scale output: chi(m')=0 so penalise by chi(c)."""
    iv_dist = interval_hamming(perturbed, candidate)
    n = min(len(perturbed), len(candidate))
    return iv_dist + alpha * (n - 1) * chromatic_fraction(candidate)


def _aware_dist_random_adaptive(
    perturbed: list[int],
    candidate: list[int],
    alpha: float,
    intercept: float,
    slope: float,
) -> float:
    """
    Aware distance for M-Random output using epsilon-adaptive formula.
    predicted = intercept + slope * chi(candidate)
    d = d_IH + alpha*(n-1)*|chi(m') - predicted|
    """
    iv_dist = interval_hamming(perturbed, candidate)
    predicted = intercept + slope * chromatic_fraction(candidate)
    chrom_err = abs(chromatic_fraction(perturbed) - predicted)
    n = min(len(perturbed), len(candidate))
    return iv_dist + alpha * (n - 1) * chrom_err


def run_experiment_adaptive(
    dataset: list[list[int]],
    epsilon: float,
    num_trials: int = 30,
    alpha: float = ALPHA,
) -> dict:
    """
    Epsilon-adaptive aware attacker:
    - M-Scale: same as original (chi(m')=0 exactly, no eps dependence)
    - M-Random: uses epsilon-adaptive E[chi] formula instead of fixed ln7 constants

    Seeds identical to original experiments for comparability.
    """
    intercept, slope = _adaptive_chi_coeffs(epsilon)

    ranks_random: list[int] = []
    ranks_scale: list[int] = []

    for trial in range(num_trials):
        for melody_idx, melody in enumerate(dataset):
            seed_r = trial * 10_000 + melody_idx
            seed_s = trial * 10_000 + melody_idx + 5_000

            # M-Random with adaptive aware attacker
            perturbed_r = m_random(melody, epsilon, seed=seed_r)
            dists_r = [
                (i, _aware_dist_random_adaptive(perturbed_r, orig, alpha, intercept, slope))
                for i, orig in enumerate(dataset)
            ]
            dists_r.sort(key=lambda x: x[1])
            rank_r = [idx for idx, _ in dists_r].index(melody_idx)
            ranks_random.append(rank_r)

            # M-Scale (unchanged)
            perturbed_s = m_scale(melody, epsilon, seed=seed_s)
            dists_s = [
                (i, _aware_dist_scale(perturbed_s, orig, alpha))
                for i, orig in enumerate(dataset)
            ]
            dists_s.sort(key=lambda x: x[1])
            rank_s = [idx for idx, _ in dists_s].index(melody_idx)
            ranks_scale.append(rank_s)

    n = len(ranks_random)
    hits1_r = sum(1 for r in ranks_random if r == 0)
    hits1_s = sum(1 for r in ranks_scale if r == 0)

    ci1_r = _wilson_ci(hits1_r, n)
    ci1_s = _wilson_ci(hits1_s, n)

    ratio = hits1_s / hits1_r if hits1_r > 0 else float("inf")

    return {
        "epsilon": epsilon,
        "exp_epsilon": exp(epsilon),
        "adaptive_acc1_random": hits1_r / n,
        "adaptive_acc1_scale": hits1_s / n,
        "adaptive_ci1_random": ci1_r,
        "adaptive_ci1_scale": ci1_s,
        "adaptive_ratio": ratio,
        "intercept": intercept,
        "slope": slope,
        "p": _keep_prob(epsilon),
    }
