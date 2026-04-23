"""
mechanisms.py
=============
LDP perturbation mechanisms for melodic sequences.

Two mechanisms, both satisfying ε-LDP with alphabet size k=7:
  - M-Random : structure-agnostic  (7 randomly chosen pitch classes)
  - M-Scale  : structure-aware     (7 diatonic pitch classes, C major)

Privacy guarantee (randomized response):
  p = e^ε / (e^ε + k)          [corrected formula for mixed in/out-of-alphabet inputs]
  With prob p  → keep original note  (only if note is IN the alphabet)
  If note NOT in alphabet → always replace uniformly from all k alphabet members
  With prob 1-p (note in alphabet) → replace uniformly from the other k-1 members
"""

import random
from math import exp


# C major scale pitch classes (always fixed for M-Scale)
C_MAJOR_SCALE: list[int] = [0, 2, 4, 5, 7, 9, 11]


def _keep_prob(epsilon: float, k: int = 7) -> float:
    """Return the retention probability p = e^eps / (e^eps + k - 1).
    Standard k-RR formula.  Valid because the corrected mechanism never
    retains out-of-alphabet notes, so the worst-case LDP ratio is
    p*(k-1)/(1-p) = e^eps, achieved by the both-in-alphabet case.
    """
    e_eps = exp(epsilon)
    return e_eps / (e_eps + k - 1)


def m_random(melody: list[int], epsilon: float, seed: int | None = None) -> list[int]:
    """
    Structure-Agnostic Perturbation (M-Random).

    For each note:
      - With probability p: keep the original note.
      - With probability 1-p: replace pitch-class with one drawn
        uniformly from 7 *randomly chosen* pitch classes.

    The random alphabet is re-sampled once per melody call, so it
    varies across melodies (and across calls with different seeds).

    Parameters
    ----------
    melody  : list of MIDI note numbers (integers 0-127)
    epsilon : privacy parameter (per-note)
    seed    : optional RNG seed for reproducibility

    Returns
    -------
    perturbed melody as a list of MIDI note numbers
    """
    rng = random.Random(seed)
    k = 7
    p = _keep_prob(epsilon, k)

    # Pick 7 random pitch classes (0-11) for this melody
    alphabet = rng.sample(range(12), k)

    result: list[int] = []
    for note in melody:
        pitch_class = note % 12
        octave = note // 12

        if pitch_class in alphabet:
            # Note is in alphabet: retain with prob p, else replace from k-1 others
            if rng.random() < p:
                result.append(note)
            else:
                choices = [x for x in alphabet if x != pitch_class]
                result.append(octave * 12 + rng.choice(choices))
        else:
            # Note is NOT in alphabet: always replace uniformly from all k members
            # (never retain — required for eps-LDP across cross-alphabet pairs)
            result.append(octave * 12 + rng.choice(alphabet))

    return result


def m_scale(melody: list[int], epsilon: float, seed: int | None = None) -> list[int]:
    """
    Structure-Aware Perturbation (M-Scale).

    For each note:
      - With probability p: keep the original note.
      - With probability 1-p: replace pitch-class with one drawn
        uniformly from the 7 *diatonic* C-major pitch classes.

    The alphabet {0,2,4,5,7,9,11} is fixed for every call.

    Parameters
    ----------
    melody  : list of MIDI note numbers (integers 0-127)
    epsilon : privacy parameter (per-note)
    seed    : optional RNG seed for reproducibility

    Returns
    -------
    perturbed melody as a list of MIDI note numbers
    """
    rng = random.Random(seed)
    k = 7
    p = _keep_prob(epsilon, k)
    scale = C_MAJOR_SCALE  # {C, D, E, F, G, A, B}

    result: list[int] = []
    for note in melody:
        pitch_class = note % 12
        octave = note // 12

        if pitch_class in scale:
            # Note is diatonic: retain with prob p, else replace from k-1 others
            if rng.random() < p:
                result.append(note)
            else:
                choices = [x for x in scale if x != pitch_class]
                result.append(octave * 12 + rng.choice(choices))
        else:
            # Chromatic note: always replace uniformly from all k C-major members
            # (never retain — required for eps-LDP across cross-alphabet pairs)
            result.append(octave * 12 + rng.choice(scale))

    return result


# ---------------------------------------------------------------------------
# Quick sanity check (run: python -m src.mechanisms)
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    from math import log

    melody = [60, 64, 67, 72, 71, 67, 64, 60]  # C-E-G-C arpeggio
    eps = log(7)

    print(f"Original  : {melody}")
    print(f"M-Random  : {m_random(melody, eps, seed=42)}")
    print(f"M-Scale   : {m_scale(melody, eps,  seed=42)}")
    print(f"Keep prob : {_keep_prob(eps):.4f}  (expect 7/13 ≈ 0.5385)")
