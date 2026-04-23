"""
evaluation.py
=============
Retrieval-attack evaluation for LDP melody perturbation.

Two attacker variants
---------------------
1. Mechanism-agnostic (run_experiment):
   Uses combined_distance for BOTH mechanisms identically.
   combined_distance = interval_hamming + alpha*(n-1)*|chrom(a)-chrom(b)|

2. Mechanism-aware (run_experiment_aware):
   The attacker knows which mechanism was used and exploits it.

   Case A — M-Scale output (paper eq. after Prop. 1):
     M-Scale replacements always land on C-major notes, so chi(output) = 0
     almost surely. The attacker therefore uses chi(m') = 0 in the combined
     distance, which reduces the chromatic term to alpha*(n-1)*chi(candidate):
       d_scale(m', c) = d_IH(m', c) + alpha*(n-1)*chi(c)

   Case B — M-Random output (paper eq. after Prop. 1):
     Proposition 1 gives E[chi(M-Random(m))] = 535/1716 + (36/143)*chi(c).
     The attacker predicts this expected output chromatic fraction for each
     candidate c and compares it to the observed chi(m'):
       d_random(m', c) = d_IH(m', c)
                         + alpha*(n-1)*|chi(m') - (535/1716 + (36/143)*chi(c))|

Accuracy metrics
----------------
Acc@1 : true original is the nearest neighbour (rank 0)
Acc@5 : true original is in the top-5 nearest neighbours (rank < 5)

30 independent trials, 95% Wilson score confidence intervals.
"""

from __future__ import annotations

from math import exp

import numpy as np

from .mechanisms import m_random, m_scale
from .metrics import chromatic_fraction, combined_distance, interval_hamming, ALPHA

# Proposition 1 constants (E[chi(M-Random)] = A + B*f at eps=ln7)
_PROP1_A: float = 535 / 1716   # ≈ 0.3118
_PROP1_B: float = 36 / 143     # ≈ 0.2517


def _keep_prob(epsilon: float, k: int = 7) -> float:
    """Return the retention probability p = e^eps / (e^eps + k - 1)."""
    e_eps = exp(epsilon)
    return e_eps / (e_eps + k - 1)


def _wilson_ci(successes: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """
    Wilson score confidence interval for a proportion.
    Returns (lower, upper) bounds.
    """
    if n == 0:
        return 0.0, 0.0
    p_hat = successes / n
    denom = 1 + z * z / n
    centre = (p_hat + z * z / (2 * n)) / denom
    margin = z * ((p_hat * (1 - p_hat) / n + z * z / (4 * n * n)) ** 0.5) / denom
    return max(0.0, centre - margin), min(1.0, centre + margin)


# ---------------------------------------------------------------------------
# Core attack — mechanism-agnostic
# ---------------------------------------------------------------------------

def retrieval_rank(
    perturbed: list[int],
    database: list[list[int]],
    true_idx: int,
    alpha: float = ALPHA,
) -> int:
    """
    Return the 0-based rank of the true original in the distance-sorted list.
    Uses combined_distance (interval-Hamming + alpha*(n-1)*|chi(a)-chi(b)|).
    Rank 0 = nearest neighbour = perfect re-identification.
    """
    distances = [
        (i, combined_distance(perturbed, orig, alpha=alpha))
        for i, orig in enumerate(database)
    ]
    distances.sort(key=lambda x: x[1])
    return [idx for idx, _ in distances].index(true_idx)


# ---------------------------------------------------------------------------
# Mechanism-aware distance functions (paper Section 3.5)
# ---------------------------------------------------------------------------

def _aware_dist_scale(
    perturbed: list[int],
    candidate: list[int],
    alpha: float,
) -> float:
    """
    Case A: mechanism-aware distance for M-Scale output.

    Since chi(M-Scale output) = 0 almost surely, substitute chi(m') = 0:
      d_scale(m', c) = d_IH(m', c) + alpha*(n-1)*chi(c)
    """
    iv_dist = interval_hamming(perturbed, candidate)
    n = min(len(perturbed), len(candidate))
    return iv_dist + alpha * (n - 1) * chromatic_fraction(candidate)


def _aware_dist_random(
    perturbed: list[int],
    candidate: list[int],
    alpha: float,
) -> float:
    """
    Case B: mechanism-aware distance for M-Random output.

    Uses Proposition 1 to predict E[chi(output)] for each candidate:
      predicted = 535/1716 + (36/143)*chi(candidate)
      d_random(m', c) = d_IH(m', c) + alpha*(n-1)*|chi(m') - predicted|

    In practice the chromatic term adds limited signal for M-RANDOM outputs
    because the random alphabet makes the output chromatic fraction noisy.
    The dominant term is interval-Hamming.
    """
    iv_dist   = interval_hamming(perturbed, candidate)
    predicted = _PROP1_A + _PROP1_B * chromatic_fraction(candidate)
    chrom_err = abs(chromatic_fraction(perturbed) - predicted)
    n = min(len(perturbed), len(candidate))
    return iv_dist + alpha * (n - 1) * chrom_err


def retrieval_rank_aware_scale(
    perturbed: list[int],
    database: list[list[int]],
    true_idx: int,
    alpha: float = ALPHA,
) -> int:
    """Mechanism-aware rank for M-Scale output (Case A)."""
    distances = [
        (i, _aware_dist_scale(perturbed, orig, alpha=alpha))
        for i, orig in enumerate(database)
    ]
    distances.sort(key=lambda x: x[1])
    return [idx for idx, _ in distances].index(true_idx)


def retrieval_rank_aware_random(
    perturbed: list[int],
    database: list[list[int]],
    true_idx: int,
    alpha: float = ALPHA,
) -> int:
    """Mechanism-aware rank for M-Random output (Case B)."""
    distances = [
        (i, _aware_dist_random(perturbed, orig, alpha=alpha))
        for i, orig in enumerate(database)
    ]
    distances.sort(key=lambda x: x[1])
    return [idx for idx, _ in distances].index(true_idx)


# ---------------------------------------------------------------------------
# Experiment runners
# ---------------------------------------------------------------------------

def run_experiment(
    dataset: list[list[int]],
    epsilon: float,
    num_trials: int = 30,
    alpha: float = ALPHA,
) -> dict:
    """
    Mechanism-agnostic attacker: uses combined_distance for both mechanisms.

    Seeds: seed_random = trial*10_000 + melody_idx
           seed_scale  = trial*10_000 + melody_idx + 5_000

    Returns dict with acc1/acc5 for M-Random and M-Scale, plus Wilson CIs.
    """
    ranks_random: list[int] = []
    ranks_scale: list[int] = []

    for trial in range(num_trials):
        for melody_idx, melody in enumerate(dataset):
            seed_r = trial * 10_000 + melody_idx
            seed_s = trial * 10_000 + melody_idx + 5_000

            perturbed_r = m_random(melody, epsilon, seed=seed_r)
            ranks_random.append(retrieval_rank(perturbed_r, dataset, melody_idx, alpha=alpha))

            perturbed_s = m_scale(melody, epsilon, seed=seed_s)
            ranks_scale.append(retrieval_rank(perturbed_s, dataset, melody_idx, alpha=alpha))

    n = len(ranks_random)
    hits1_r = sum(1 for r in ranks_random if r == 0)
    hits1_s = sum(1 for r in ranks_scale  if r == 0)
    hits5_r = sum(1 for r in ranks_random if r < 5)
    hits5_s = sum(1 for r in ranks_scale  if r < 5)

    ci1_r = _wilson_ci(hits1_r, n)
    ci1_s = _wilson_ci(hits1_s, n)
    ci5_r = _wilson_ci(hits5_r, n)
    ci5_s = _wilson_ci(hits5_s, n)

    return {
        "epsilon":     epsilon,
        "acc1_random": hits1_r / n,
        "acc1_scale":  hits1_s / n,
        "acc5_random": hits5_r / n,
        "acc5_scale":  hits5_s / n,
        "ci1_random":  ci1_r,
        "ci1_scale":   ci1_s,
        "ci5_random":  ci5_r,
        "ci5_scale":   ci5_s,
        "raw": {"ranks_random": ranks_random, "ranks_scale": ranks_scale},
    }


def run_experiment_aware(
    dataset: list[list[int]],
    epsilon: float,
    num_trials: int = 30,
    alpha: float = ALPHA,
) -> dict:
    """
    Mechanism-aware attacker: uses Case A for M-Scale, Case B for M-Random.

    Returns dict with aware_acc1/aware_acc5 for M-Random and M-Scale,
    plus Wilson CIs.
    """
    ranks_random: list[int] = []
    ranks_scale: list[int] = []

    for trial in range(num_trials):
        for melody_idx, melody in enumerate(dataset):
            seed_r = trial * 10_000 + melody_idx
            seed_s = trial * 10_000 + melody_idx + 5_000

            perturbed_r = m_random(melody, epsilon, seed=seed_r)
            ranks_random.append(
                retrieval_rank_aware_random(perturbed_r, dataset, melody_idx, alpha=alpha)
            )

            perturbed_s = m_scale(melody, epsilon, seed=seed_s)
            ranks_scale.append(
                retrieval_rank_aware_scale(perturbed_s, dataset, melody_idx, alpha=alpha)
            )

    n = len(ranks_random)
    hits1_r = sum(1 for r in ranks_random if r == 0)
    hits1_s = sum(1 for r in ranks_scale  if r == 0)
    hits5_r = sum(1 for r in ranks_random if r < 5)
    hits5_s = sum(1 for r in ranks_scale  if r < 5)

    ci1_r = _wilson_ci(hits1_r, n)
    ci1_s = _wilson_ci(hits1_s, n)
    ci5_r = _wilson_ci(hits5_r, n)
    ci5_s = _wilson_ci(hits5_s, n)

    return {
        "epsilon":           epsilon,
        "aware_acc1_random": hits1_r / n,
        "aware_acc1_scale":  hits1_s / n,
        "aware_acc5_random": hits5_r / n,
        "aware_acc5_scale":  hits5_s / n,
        "aware_ci1_random":  ci1_r,
        "aware_ci1_scale":   ci1_s,
        "aware_ci5_random":  ci5_r,
        "aware_ci5_scale":   ci5_s,
        "raw": {"ranks_random": ranks_random, "ranks_scale": ranks_scale},
    }


# ---------------------------------------------------------------------------
# Quick sanity check (run: python -m src.evaluation)
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    from math import log

    toy = [
        [60, 64, 67, 72, 71, 67, 64, 60],
        [60, 62, 64, 65, 67, 69, 71, 72],
        [72, 71, 69, 67, 65, 64, 62, 60],
        [60, 67, 64, 72, 60, 67, 64, 72],
        [55, 60, 64, 67, 72, 67, 64, 60],
    ]

    eps = log(7)
    res  = run_experiment(toy, eps, num_trials=5)
    resa = run_experiment_aware(toy, eps, num_trials=5)

    print(f"eps = ln(7) ~ {eps:.3f}")
    print("--- Mechanism-agnostic attacker ---")
    print(f"M-Random  Acc@1 = {res['acc1_random']*100:.1f}%  Acc@5 = {res['acc5_random']*100:.1f}%")
    print(f"M-Scale   Acc@1 = {res['acc1_scale']*100:.1f}%  Acc@5 = {res['acc5_scale']*100:.1f}%")
    print("--- Mechanism-aware attacker ---")
    print(f"M-Random  Acc@1 = {resa['aware_acc1_random']*100:.1f}%  Acc@5 = {resa['aware_acc5_random']*100:.1f}%")
    print(f"M-Scale   Acc@1 = {resa['aware_acc1_scale']*100:.1f}%  Acc@5 = {resa['aware_acc5_scale']*100:.1f}%")
