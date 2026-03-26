"""
metrics.py
==========
Distance metrics for melodic sequences, as used in the paper.

Combined distance:
    d(a, b) = d_IH(a, b) + alpha * (n-1) * |chi(a) - chi(b)|
where n = min(|a|, |b|) and alpha = 1.66 (variance-matched).
"""
from __future__ import annotations

C_MAJOR_PCS: frozenset[int] = frozenset({0, 2, 4, 5, 7, 9, 11})
ALPHA: float = 1.66  # variance-matched weight (paper Section 3.3)


def chromatic_fraction(melody: list[int]) -> float:
    """Fraction of notes whose pitch class is NOT in C major {0,2,4,5,7,9,11}."""
    if not melody:
        return 0.0
    return sum(1 for n in melody if n % 12 not in C_MAJOR_PCS) / len(melody)


def interval_hamming(melody_a: list[int], melody_b: list[int]) -> int:
    """
    Transposition-invariant Hamming distance on interval sequences.
    Truncates to the shorter melody before computing intervals.
    """
    min_len = min(len(melody_a), len(melody_b))
    a = melody_a[:min_len]
    b = melody_b[:min_len]
    if min_len < 2:
        return 0
    intervals_a = [a[i + 1] - a[i] for i in range(min_len - 1)]
    intervals_b = [b[i + 1] - b[i] for i in range(min_len - 1)]
    return sum(1 for x, y in zip(intervals_a, intervals_b) if x != y)


def combined_distance(
    melody_a: list[int],
    melody_b: list[int],
    alpha: float = ALPHA,
) -> float:
    """
    Combined interval-Hamming + chromatic-fraction distance (paper eq. 3).
    alpha = 1.66 chosen by variance matching on the Europa dataset.
    """
    iv_dist = interval_hamming(melody_a, melody_b)
    chrom_diff = abs(chromatic_fraction(melody_a) - chromatic_fraction(melody_b))
    n = min(len(melody_a), len(melody_b))
    return iv_dist + alpha * (n - 1) * chrom_diff
