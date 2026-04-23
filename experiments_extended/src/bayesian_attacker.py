"""
bayesian_attacker.py
====================
Optimal (information-theoretic) Bayesian attacker for 1-NN re-identification
under both M-Scale and M-Random randomised-response mechanisms.

For each perturbed melody m', the Bayesian attacker picks the candidate c*
that maximises the likelihood P(output = m' | original = c).

Per-note model
--------------
Both mechanisms use randomised response on a pitch-class alphabet of size k=7.
Let p = e^eps / (e^eps + k - 1) be the retention probability.

M-Scale (fixed C-major alphabet A = {0,2,4,5,7,9,11}):
  - If c_pc in A:
      P(m'_pc = c_pc | c_pc) = p
      P(m'_pc = a   | c_pc) = (1-p)/6  for a in A, a != c_pc
      P(m'_pc = x   | c_pc) = 0         for x not in A
  - If c_pc not in A (chromatic input):
      P(m'_pc = a   | c_pc) = 1/7       for any a in A

M-Random (random per-melody alphabet, unknown to attacker):
  The attacker knows the mechanism but NOT the specific alphabet used.
  It marginalises over all C(12,7) = 792 possible 7-element subsets of {0..11}.
  For each possible alphabet A:
      P(m'_pc | c_pc, A) as above (same formula with that A)
  P(m'_pc | c_pc) = average over all 792 alphabets.

Melody likelihood
-----------------
Assuming conditional independence of notes given the original:
  log P(m' | c) = sum_i log P(m'_i_pc | c_i_pc, mechanism)

The Bayesian attacker ranks all database candidates by this log-likelihood
(descending) and returns the rank of the true original.

Usage
-----
    from src.bayesian_attacker import bayesian_attack_melody, precompute_mrand_table

    # Precompute M-Random marginalised probabilities (optional, speeds up repeat calls)
    table = precompute_mrand_table(epsilon)   # dict[(m_pc, c_pc)] -> log_prob

    rank = bayesian_attack_melody(
        perturbed, candidate_idx, database,
        mechanism="scale", epsilon=epsilon
    )
"""

from __future__ import annotations

import math
from functools import lru_cache
from itertools import combinations
from typing import Literal

import numpy as np

# Fixed C-major alphabet (pitch classes)
SCALE = frozenset({0, 2, 4, 5, 7, 9, 11})
_ALL_ALPHABETS: list[frozenset] | None = None  # lazy init


def _get_all_alphabets() -> list[frozenset]:
    """All C(12,7) = 792 pitch-class subsets of size 7."""
    global _ALL_ALPHABETS
    if _ALL_ALPHABETS is None:
        _ALL_ALPHABETS = [frozenset(c) for c in combinations(range(12), 7)]
    return _ALL_ALPHABETS


def _retention_prob(epsilon: float) -> float:
    """Retention probability p = e^eps / (e^eps + 6)."""
    e = math.exp(epsilon)
    return e / (e + 6.0)


# ── Per-note likelihood: M-Scale ─────────────────────────────────────────────

def note_likelihood_scale(
    perturbed_pc: int,
    candidate_pc: int,
    epsilon: float,
) -> float:
    """
    P(perturbed_pc | candidate_pc) under M-Scale (fixed C-major alphabet).

    Parameters
    ----------
    perturbed_pc  : output pitch class (0-11)
    candidate_pc  : original pitch class (0-11)
    epsilon       : privacy budget

    Returns
    -------
    Probability in [0, 1].
    """
    p = _retention_prob(epsilon)
    m = perturbed_pc % 12
    c = candidate_pc % 12

    if c in SCALE:
        if m == c:
            return p
        elif m in SCALE:
            return (1.0 - p) / 6.0
        else:
            return 0.0  # M-Scale never outputs chromatic notes
    else:
        # Chromatic input: replaced uniformly over scale
        return 1.0 / 7.0 if m in SCALE else 0.0


# ── Per-note likelihood: M-Random (marginalised) ─────────────────────────────

@lru_cache(maxsize=None)
def _mrand_pair_prob(perturbed_pc: int, candidate_pc: int, epsilon: float) -> float:
    """
    P(perturbed_pc | candidate_pc) under M-Random, averaged over all 792 alphabets.

    Cached by (perturbed_pc, candidate_pc, epsilon).
    """
    p = _retention_prob(epsilon)
    m = perturbed_pc % 12
    c = candidate_pc % 12
    alphabets = _get_all_alphabets()

    total = 0.0
    for alpha in alphabets:
        if c in alpha:
            if m == c:
                prob = p
            elif m in alpha:
                prob = (1.0 - p) / 6.0
            else:
                prob = 0.0
        else:
            # c not in alphabet: uniform replacement over alpha
            prob = 1.0 / 7.0 if m in alpha else 0.0
        total += prob

    return total / len(alphabets)  # 792


def note_likelihood_random(
    perturbed_pc: int,
    candidate_pc: int,
    epsilon: float,
) -> float:
    """
    P(perturbed_pc | candidate_pc) under M-Random, marginalised over all 792
    possible 7-element pitch-class alphabets.

    Parameters
    ----------
    perturbed_pc  : output pitch class (0-11)
    candidate_pc  : original pitch class (0-11)
    epsilon       : privacy budget

    Returns
    -------
    Probability in [0, 1].
    """
    return _mrand_pair_prob(perturbed_pc, candidate_pc, epsilon)


# ── Melody log-likelihood ────────────────────────────────────────────────────

_LOG_FLOOR = -50.0  # floor for log(0) to avoid -inf in sum


def melody_log_likelihood(
    perturbed: list[int],
    candidate: list[int],
    mechanism: Literal["scale", "random"],
    epsilon: float,
) -> float:
    """
    log P(perturbed | candidate) = sum_i log P(perturbed[i]_pc | candidate[i]_pc).

    Notes at positions beyond the shorter melody are ignored (truncates to
    min(len(perturbed), len(candidate))).

    Parameters
    ----------
    perturbed  : perturbed melody (MIDI pitches 0-127)
    candidate  : candidate original melody (MIDI pitches 0-127)
    mechanism  : "scale" for M-Scale, "random" for M-Random
    epsilon    : privacy budget

    Returns
    -------
    Log-likelihood (float, typically negative).
    """
    n = min(len(perturbed), len(candidate))
    log_lik = 0.0

    if mechanism == "scale":
        for i in range(n):
            p_pc = perturbed[i] % 12
            c_pc = candidate[i] % 12
            prob = note_likelihood_scale(p_pc, c_pc, epsilon)
            log_lik += math.log(prob) if prob > 1e-300 else _LOG_FLOOR
    else:  # random
        for i in range(n):
            p_pc = perturbed[i] % 12
            c_pc = candidate[i] % 12
            prob = _mrand_pair_prob(p_pc, c_pc, epsilon)
            log_lik += math.log(prob) if prob > 1e-300 else _LOG_FLOOR

    return log_lik


# ── Bayesian rank for one perturbed melody ───────────────────────────────────

def bayesian_rank(
    perturbed: list[int],
    database: list[list[int]],
    true_idx: int,
    mechanism: Literal["scale", "random"],
    epsilon: float,
) -> int:
    """
    Rank the true original by descending log-likelihood among all database candidates.

    Parameters
    ----------
    perturbed  : the perturbed melody
    database   : list of all candidate originals (the full database)
    true_idx   : index of the true original in database
    mechanism  : "scale" or "random"
    epsilon    : privacy budget

    Returns
    -------
    1-based rank of the true original (1 = best match).
    """
    true_ll = melody_log_likelihood(perturbed, database[true_idx], mechanism, epsilon)

    rank = 1
    for j, cand in enumerate(database):
        if j == true_idx:
            continue
        ll = melody_log_likelihood(perturbed, cand, mechanism, epsilon)
        if ll > true_ll:
            rank += 1

    return rank


# ── Precompute all pairwise log-likelihoods for efficiency ───────────────────

def compute_log_likelihood_matrix(
    database: list[list[int]],
    mechanism: Literal["scale", "random"],
    epsilon: float,
) -> np.ndarray:
    """
    Compute the N x N log-likelihood matrix LL[i,j] = log P(database[i] | database[j]).

    This pre-computation allows fast ranking without recomputing likelihoods per trial.

    Parameters
    ----------
    database  : N candidate melodies
    mechanism : "scale" or "random"
    epsilon   : privacy budget

    Returns
    -------
    N x N float array.
    """
    N = len(database)
    LL = np.zeros((N, N), dtype=np.float64)

    for i in range(N):
        perturbed = database[i]
        for j in range(N):
            LL[i, j] = melody_log_likelihood(perturbed, database[j], mechanism, epsilon)

    return LL


def rank_from_ll_matrix(
    ll_matrix: np.ndarray,
    perturbed_idx: int,
    true_idx: int,
) -> int:
    """
    Given a precomputed LL matrix (from compute_log_likelihood_matrix or a
    per-trial matrix), return the 1-based rank of true_idx for row perturbed_idx.
    """
    row = ll_matrix[perturbed_idx]
    true_ll = row[true_idx]
    # Rank = 1 + number of candidates with strictly higher likelihood
    rank = 1 + int(np.sum(row > true_ll)) - (1 if row[true_idx] > true_ll else 0)
    # More careful: count j != perturbed_idx if desired, but for 1-NN re-id
    # the perturbed query IS the true original (noise applied), so all N are candidates
    return 1 + int(np.sum(row > true_ll))
