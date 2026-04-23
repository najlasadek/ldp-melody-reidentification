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
   The attacker knows which mechanism was used and exploits it:

   For M-Scale output:
     The attacker knows M-Scale replacements always land on C-major notes.
     Expected chromatic%(output) = original_chromatic% * p.
     Distance uses the PREDICTION ERROR on this fingerprint:
       aware_dist_scale(output, candidate) =
           interval_hamming(output, candidate)
           + alpha*(n-1)*|chrom(output) - chrom(candidate)*p|

   For M-Random output:
     The attacker knows M-Random replacements land on ~42% chromatic notes.
     M-Random destroys key info -> chromatic term adds no signal.
     Distance uses ONLY interval-Hamming:
       aware_dist_random(output, candidate) = interval_hamming(output, candidate)

   The gap between the two attacker accuracies shows how much M-Scale's
   structure contributes to leakage beyond what M-Random leaks.

Accuracy metrics
----------------
SUCCESS @1 : true original is the nearest neighbour (rank 0)
SUCCESS @5 : true original is in the top-5 nearest neighbours (rank < 5)

We run `num_trials` independent repetitions and average over all
(trial, melody) pairs to get stable accuracy estimates.

95 % confidence intervals are computed via the normal approximation
(standard error of the mean), which is fine for n >= 30 * 200 = 6000.
"""

from __future__ import annotations

from math import exp

import numpy as np

from .mechanisms import m_random, m_scale
from .metrics import chromatic_fraction, combined_distance, interval_hamming


def _keep_prob(epsilon: float, k: int = 7) -> float:
    """Return the retention probability p = e^eps / (e^eps + k - 1).
    Standard k-RR formula, valid with the corrected mechanism that never
    retains out-of-alphabet notes.
    """
    e_eps = exp(epsilon)
    return e_eps / (e_eps + k - 1)


# ---------------------------------------------------------------------------
# Core attack
# ---------------------------------------------------------------------------

def retrieval_rank(
    perturbed: list[int],
    database: list[list[int]],
    true_idx: int,
    alpha: float = 2.53,
) -> int:
    """
    Return the 0-based rank of the true original in the distance-sorted list.

    Rank 0  -> true original is the nearest neighbour (perfect attack).
    Rank k  -> k melodies are closer than the true original.

    Distance metric: combined_distance (interval-Hamming + chromatic term).

    Parameters
    ----------
    perturbed : perturbed melody
    database  : list of all original melodies
    true_idx  : index into database of the true original
    alpha     : weight for the chromatic-fraction term (default 2.53)

    Returns
    -------
    integer rank (0 = success @1)
    """
    distances = [
        (i, combined_distance(perturbed, orig, alpha=alpha))
        for i, orig in enumerate(database)
    ]
    distances.sort(key=lambda x: x[1])
    ranked_indices = [idx for idx, _ in distances]
    return ranked_indices.index(true_idx)


# ---------------------------------------------------------------------------
# Mechanism-aware distance functions
# ---------------------------------------------------------------------------

def _aware_dist_scale(
    perturbed: list[int],
    candidate: list[int],
    p: float,
    alpha: float,
) -> float:
    """
    Mechanism-aware distance for M-Scale output.

    Exploits the fact that M-Scale replacements always land on C-major notes.
    The expected chromatic fraction of the output is:
        E[chrom(output)] = chrom(original) * p

    So the attacker predicts: if `candidate` is the true original, the output
    should have chromatic fraction ~ chrom(candidate) * p.

    Distance = interval_hamming(perturbed, candidate)
               + alpha * (n-1) * |chrom(perturbed) - chrom(candidate)*p|

    Parameters
    ----------
    perturbed : M-Scale perturbed melody
    candidate : candidate original from the database
    p         : keep probability for the current epsilon
    alpha     : weighting coefficient
    """
    iv_dist = interval_hamming(perturbed, candidate)
    predicted_chrom = chromatic_fraction(candidate) * p
    chrom_err = abs(chromatic_fraction(perturbed) - predicted_chrom)
    n = min(len(perturbed), len(candidate))
    return iv_dist + alpha * (n - 1) * chrom_err


def retrieval_rank_aware_scale(
    perturbed: list[int],
    database: list[list[int]],
    true_idx: int,
    p: float,
    alpha: float = 2.53,
) -> int:
    """
    Mechanism-aware rank for M-Scale output.

    Uses _aware_dist_scale which predicts the expected chromatic fingerprint
    for each candidate assuming M-Scale was applied.

    Parameters
    ----------
    perturbed : M-Scale perturbed melody
    database  : list of all original melodies
    true_idx  : index of the true original in database
    p         : keep probability = e^eps / (e^eps + 6)
    alpha     : weight for chromatic prediction error
    """
    distances = [
        (i, _aware_dist_scale(perturbed, orig, p=p, alpha=alpha))
        for i, orig in enumerate(database)
    ]
    distances.sort(key=lambda x: x[1])
    return [idx for idx, _ in distances].index(true_idx)


def retrieval_rank_aware_random(
    perturbed: list[int],
    database: list[list[int]],
    true_idx: int,
) -> int:
    """
    Mechanism-aware rank for M-Random output.

    M-Random destroys key-signature information (replacements land on any of
    12 notes, ~42% chromatic regardless of original key). The chromatic term
    therefore adds no signal -- use interval-Hamming only.

    Parameters
    ----------
    perturbed : M-Random perturbed melody
    database  : list of all original melodies
    true_idx  : index of the true original in database
    """
    distances = [
        (i, interval_hamming(perturbed, orig))
        for i, orig in enumerate(database)
    ]
    distances.sort(key=lambda x: x[1])
    return [idx for idx, _ in distances].index(true_idx)


# ---------------------------------------------------------------------------
# Experiment runner
# ---------------------------------------------------------------------------

def run_experiment(
    dataset: list[list[int]],
    epsilon: float,
    num_trials: int = 30,
    alpha: float = 2.53,
) -> dict:
    """
    Run the full re-identification experiment for one epsilon value.

    For each (trial, melody) pair we:
      - Perturb with M-Random  -> record rank of true original
      - Perturb with M-Scale   -> record rank of true original

    Uses deterministic seeds:
      seed_random = trial * 10_000 + melody_idx
      seed_scale  = trial * 10_000 + melody_idx + 5_000

    Parameters
    ----------
    dataset    : list of original melodies (MIDI note lists)
    epsilon    : per-note privacy parameter
    num_trials : number of independent repetitions
    alpha      : weight for chromatic-fraction term in combined_distance

    Returns
    -------
    dict with keys:
      epsilon, acc1_random, acc1_scale, acc5_random, acc5_scale,
      ci1_random, ci1_scale, ci5_random, ci5_scale, raw
    """
    ranks_random: list[int] = []
    ranks_scale: list[int] = []

    n = len(dataset)

    for trial in range(num_trials):
        for melody_idx, melody in enumerate(dataset):
            seed_r = trial * 10_000 + melody_idx
            seed_s = trial * 10_000 + melody_idx + 5_000

            # M-Random
            perturbed_r = m_random(melody, epsilon, seed=seed_r)
            ranks_random.append(retrieval_rank(perturbed_r, dataset, melody_idx, alpha=alpha))

            # M-Scale
            perturbed_s = m_scale(melody, epsilon, seed=seed_s)
            ranks_scale.append(retrieval_rank(perturbed_s, dataset, melody_idx, alpha=alpha))

    # Accuracy @1 and @5
    hits1_random = np.array([r == 0 for r in ranks_random], dtype=float)
    hits1_scale  = np.array([r == 0 for r in ranks_scale],  dtype=float)
    hits5_random = np.array([r < 5  for r in ranks_random], dtype=float)
    hits5_scale  = np.array([r < 5  for r in ranks_scale],  dtype=float)

    def _ci(arr: np.ndarray) -> float:
        """95 % CI half-width via normal approximation."""
        return 1.96 * np.std(arr) / np.sqrt(len(arr))

    return {
        "epsilon":     epsilon,
        "acc1_random": float(np.mean(hits1_random)),
        "acc1_scale":  float(np.mean(hits1_scale)),
        "acc5_random": float(np.mean(hits5_random)),
        "acc5_scale":  float(np.mean(hits5_scale)),
        "ci1_random":  float(_ci(hits1_random)),
        "ci1_scale":   float(_ci(hits1_scale)),
        "ci5_random":  float(_ci(hits5_random)),
        "ci5_scale":   float(_ci(hits5_scale)),
        "raw": {
            "ranks_random": ranks_random,
            "ranks_scale":  ranks_scale,
        },
    }


# ---------------------------------------------------------------------------
# Mechanism-aware experiment runner
# ---------------------------------------------------------------------------

def run_experiment_aware(
    dataset: list[list[int]],
    epsilon: float,
    num_trials: int = 30,
    alpha: float = 2.53,
) -> dict:
    """
    Run the re-identification experiment with MECHANISM-AWARE attackers.

    M-Scale attacker: uses predicted chromatic fingerprint chrom(candidate)*p
    M-Random attacker: uses interval-Hamming only (chromatic term is noise)

    This models an attacker who knows which mechanism was deployed and
    exploits that knowledge optimally.

    Parameters
    ----------
    dataset    : list of original melodies (MIDI note lists)
    epsilon    : per-note privacy parameter
    num_trials : number of independent repetitions
    alpha      : weight for chromatic prediction error (M-Scale attacker only)

    Returns
    -------
    dict with same keys as run_experiment(), prefixed with "aware_"
    """
    p = _keep_prob(epsilon)

    ranks_random: list[int] = []
    ranks_scale: list[int] = []

    for trial in range(num_trials):
        for melody_idx, melody in enumerate(dataset):
            seed_r = trial * 10_000 + melody_idx
            seed_s = trial * 10_000 + melody_idx + 5_000

            # M-Random with mechanism-aware attacker (interval-Hamming only)
            perturbed_r = m_random(melody, epsilon, seed=seed_r)
            ranks_random.append(
                retrieval_rank_aware_random(perturbed_r, dataset, melody_idx)
            )

            # M-Scale with mechanism-aware attacker (chromatic prediction)
            perturbed_s = m_scale(melody, epsilon, seed=seed_s)
            ranks_scale.append(
                retrieval_rank_aware_scale(perturbed_s, dataset, melody_idx, p=p, alpha=alpha)
            )

    hits1_random = np.array([r == 0 for r in ranks_random], dtype=float)
    hits1_scale  = np.array([r == 0 for r in ranks_scale],  dtype=float)
    hits5_random = np.array([r < 5  for r in ranks_random], dtype=float)
    hits5_scale  = np.array([r < 5  for r in ranks_scale],  dtype=float)

    def _ci(arr: np.ndarray) -> float:
        return 1.96 * np.std(arr) / np.sqrt(len(arr))

    return {
        "epsilon":           epsilon,
        "aware_acc1_random": float(np.mean(hits1_random)),
        "aware_acc1_scale":  float(np.mean(hits1_scale)),
        "aware_acc5_random": float(np.mean(hits5_random)),
        "aware_acc5_scale":  float(np.mean(hits5_scale)),
        "aware_ci1_random":  float(_ci(hits1_random)),
        "aware_ci1_scale":   float(_ci(hits1_scale)),
        "aware_ci5_random":  float(_ci(hits5_random)),
        "aware_ci5_scale":   float(_ci(hits5_scale)),
        "raw": {
            "ranks_random": ranks_random,
            "ranks_scale":  ranks_scale,
        },
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
