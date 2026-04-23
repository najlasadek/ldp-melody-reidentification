"""
metrics.py
==========
Similarity / distance metrics for melodic sequences.

Interval-Sequence Hamming Distance
-----------------------------------
Compare two melodies via their interval sequences rather than raw pitches.
This makes the metric transposition-invariant: shifting every note in a
melody by the same semitone offset leaves the distance unchanged.

  intervals(m) = [m[1]-m[0], m[2]-m[1], ..., m[n-1]-m[n-2]]
  dist(a, b)   = |{i : intervals_a[i] != intervals_b[i]}|

If the two sequences differ in length, we truncate to the shorter one
before computing intervals (so distance is over min-1 interval positions).

Chromatic Fraction
------------------
Fraction of notes whose pitch class falls OUTSIDE C major {0,2,4,5,7,9,11}.
M-Scale replacements always land on C-major notes (chromatic fraction -> 0),
while M-Random replacements land on any of 12 pitch classes (~42% chromatic
on average).  This creates a detectable signature when the original melody
is NOT already in C major.

Combined Distance
-----------------
combined_distance(a, b, alpha) = interval_hamming(a, b)
                                 + alpha * (n-1) * |chrom(a) - chrom(b)|

where n = min(len(a), len(b)).  The (n-1) factor scales the chromatic term
to the same order of magnitude as the interval-Hamming term.

Empirically derived alpha = 2.53 (see FINDINGS_AND_APPROACH.md).
"""

from __future__ import annotations


# Pitch classes in C major
C_MAJOR_PCS: frozenset[int] = frozenset({0, 2, 4, 5, 7, 9, 11})


def chromatic_fraction(melody: list[int]) -> float:
    """
    Fraction of notes whose pitch class is NOT in C major.

    Parameters
    ----------
    melody : list of MIDI note numbers

    Returns
    -------
    Float in [0, 1].  0.0 = all notes diatonic in C major.
                      1.0 = all notes chromatic.
    """
    if not melody:
        return 0.0
    return sum(1 for n in melody if n % 12 not in C_MAJOR_PCS) / len(melody)


def interval_hamming(melody_a: list[int], melody_b: list[int]) -> int:
    """
    Transposition-invariant Hamming distance on interval sequences.

    Parameters
    ----------
    melody_a, melody_b : lists of MIDI note numbers

    Returns
    -------
    Non-negative integer distance.
    0 means the melodies share identical interval sequences (up to the
    shorter length).  Larger values mean more intervals differ.
    """
    # Truncate to the same length
    min_len = min(len(melody_a), len(melody_b))
    a = melody_a[:min_len]
    b = melody_b[:min_len]

    if min_len < 2:
        # Cannot compute any intervals
        return 0

    intervals_a = [a[i + 1] - a[i] for i in range(min_len - 1)]
    intervals_b = [b[i + 1] - b[i] for i in range(min_len - 1)]

    return sum(1 for x, y in zip(intervals_a, intervals_b) if x != y)


def combined_distance(
    melody_a: list[int],
    melody_b: list[int],
    alpha: float = 2.53,
) -> float:
    """
    Combined interval-Hamming + chromatic-fraction distance.

    combined_distance(a, b) = interval_hamming(a, b)
                              + alpha * (n-1) * |chrom(a) - chrom(b)|

    The (n-1) factor makes the chromatic term comparable in scale to the
    interval-Hamming term (which also ranges over n-1 interval positions).

    alpha = 2.53 is empirically derived as:
        Var(interval_hamming) / Var((n-1) * chromatic_diff)

    Parameters
    ----------
    melody_a, melody_b : lists of MIDI note numbers
    alpha              : weighting coefficient (default 2.53)

    Returns
    -------
    Non-negative float.
    """
    iv_dist = interval_hamming(melody_a, melody_b)
    chrom_diff = abs(chromatic_fraction(melody_a) - chromatic_fraction(melody_b))
    n = min(len(melody_a), len(melody_b))
    return iv_dist + alpha * (n - 1) * chrom_diff


# ---------------------------------------------------------------------------
# Quick sanity check (run: python -m src.metrics)
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    original   = [60, 64, 67, 72]   # C-E-G-C  (all C-major)
    transposed = [65, 69, 72, 77]   # F-A-C-F  (all C-major)
    scrambled  = [60, 68, 63, 72]   # C-G#-D#-C (chromatic notes)

    print(f"Original  vs Transposed  (interval_hamming) : {interval_hamming(original, transposed)}")   # 0
    print(f"Original  vs Scrambled   (interval_hamming) : {interval_hamming(original, scrambled)}")    # 3
    print(f"Identical vs Identical   (interval_hamming) : {interval_hamming(original, original)}")     # 0

    print()
    print(f"chrom_fraction(original)   = {chromatic_fraction(original):.3f}")    # 0.0
    print(f"chrom_fraction(scrambled)  = {chromatic_fraction(scrambled):.3f}")   # 0.5 (G#, D# are chromatic)

    print()
    print(f"combined_distance(original, transposed) = {combined_distance(original, transposed):.3f}")  # 0.0
    print(f"combined_distance(original, scrambled)  = {combined_distance(original, scrambled):.3f}")   # > 3
