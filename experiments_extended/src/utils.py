"""
utils.py
========
Data loading and preprocessing for the Essen Folksong Collection.

Pipeline
--------
1. load_essen_kern()   — parse .krn files downloaded from the Essen corpus.
2. transpose_to_c()    — transpose each melody so its first note is in C major
                         (i.e. offset all pitches so the tonic maps to C).
3. filter_melodies()   — keep only monophonic melodies with 10–30 notes.
4. build_dataset()     — combine the above into a single function that returns
                         a list of up to `max_melodies` pitch sequences.
5. save_dataset() /
   load_dataset()      — persist and reload processed data as JSON.

If the Essen corpus is not available, `make_synthetic_dataset()` generates
a stand-in dataset of random diatonic melodies for testing purposes.

Essen corpus download
---------------------
  git clone https://github.com/sharc-lab/essen-folksongs data/essen_raw/
Or download individual .krn files from:
  https://kern.humdrum.org/cgi-bin/browse?l=essen
"""

from __future__ import annotations

import json
import os
import re
import random
from pathlib import Path


# ---------------------------------------------------------------------------
# C major scale helpers
# ---------------------------------------------------------------------------

C_MAJOR_PCS: set[int] = {0, 2, 4, 5, 7, 9, 11}  # pitch classes in C major

# Map of note names to pitch-class numbers (used when parsing kern files)
_KERN_BASE: dict[str, int] = {
    "c": 0, "d": 2, "e": 4, "f": 5, "g": 7, "a": 9, "b": 11,
}


# ---------------------------------------------------------------------------
# Kern parser (minimal — handles monophonic .krn files)
# ---------------------------------------------------------------------------

def _kern_token_to_pitch(token: str) -> int | None:
    """
    Convert a single **kern pitch token to a MIDI note number.

    Returns None for rests or unparseable tokens.
    Handles sharps (#), flats (-), and octave encoding (upper/lower case).
    """
    token = token.strip()
    if not token or token.startswith("r") or token.startswith("="):
        return None

    # Strip leading duration digits and dots
    pitch_part = re.sub(r"^[\d.]+", "", token)
    if not pitch_part:
        return None

    # **kern octave: lowercase letters = octave 4+, uppercase = octave 3-
    # Repeated letters raise/lower by an octave each time.
    # E.g. "c" = C4 (MIDI 60), "cc" = C5 (MIDI 72), "C" = C3 (MIDI 48)
    base_char = pitch_part[0]
    if base_char.lower() not in _KERN_BASE:
        return None

    base_pc = _KERN_BASE[base_char.lower()]

    # Octave
    if base_char.islower():
        octave = 4 + (len(re.match(r"[a-g]+", pitch_part).group()) - 1)
    else:
        octave = 3 - (len(re.match(r"[A-G]+", pitch_part).group()) - 1)

    # Accidentals: '#' = sharp (+1), '-' = flat (-1)
    accidentals = pitch_part.replace(base_char * len(re.match(r"[a-zA-Z]+", pitch_part).group()), "")
    semitone_offset = accidentals.count("#") - accidentals.count("-")

    midi = octave * 12 + base_pc + semitone_offset
    return midi


def parse_kern_file(path: str | Path) -> list[int] | None:
    """
    Parse a monophonic **kern file and return a list of MIDI note numbers.

    Returns None if the file cannot be parsed or contains no notes.
    """
    notes: list[int] = []
    try:
        with open(path, "r", encoding="utf-8", errors="ignore") as fh:
            for line in fh:
                line = line.strip()
                if not line or line.startswith("!") or line.startswith("*"):
                    continue
                # Each column separated by tab — take first spine
                token = line.split("\t")[0]
                pitch = _kern_token_to_pitch(token)
                if pitch is not None:
                    notes.append(pitch)
    except OSError:
        return None

    return notes if notes else None


# ---------------------------------------------------------------------------
# Transposition to C major
# ---------------------------------------------------------------------------

def transpose_to_c(melody: list[int]) -> list[int]:
    """
    Transpose a melody so that the majority of its pitch-classes fall in
    C major (pitch classes {0,2,4,5,7,9,11}).

    Strategy: try all 12 semitone offsets, pick the one that maximises
    the count of notes whose pitch-class is in C major.

    Parameters
    ----------
    melody : list of MIDI note numbers

    Returns
    -------
    Transposed melody as a list of MIDI note numbers.
    """
    if not melody:
        return melody

    best_offset = 0
    best_score = -1

    for offset in range(12):
        shifted = [(n + offset) % 12 for n in melody]
        score = sum(1 for pc in shifted if pc in C_MAJOR_PCS)
        if score > best_score:
            best_score = score
            best_offset = offset

    return [n + best_offset for n in melody]


# ---------------------------------------------------------------------------
# Filtering
# ---------------------------------------------------------------------------

def filter_melodies(
    melodies: list[list[int]],
    min_notes: int = 10,
    max_notes: int = 30,
) -> list[list[int]]:
    """
    Keep only monophonic melodies with between min_notes and max_notes notes.
    Also clamps notes to valid MIDI range [0, 127].
    """
    filtered = []
    for m in melodies:
        m = [max(0, min(127, n)) for n in m]
        if min_notes <= len(m) <= max_notes:
            filtered.append(m)
    return filtered


# ---------------------------------------------------------------------------
# Main dataset builder
# ---------------------------------------------------------------------------

def build_dataset(
    essen_dir: str | Path,
    max_melodies: int = 200,
    min_notes: int = 10,
    max_notes: int = 30,
    seed: int = 0,
    transpose: bool = False,
) -> list[list[int]]:
    """
    Build the experiment dataset from Essen .krn files.

    1. Scan `essen_dir` recursively for *.krn files.
    2. Parse each file to a pitch sequence.
    3. Optionally transpose each melody to C major (default: OFF).
    4. Filter to melodies with min_notes-max_notes notes.
    5. Shuffle and return up to max_melodies melodies.

    NOTE: transpose=False (default) is correct for the privacy experiment.
    Transposing all melodies to C major collapses the key-signature variance
    that the chromatic-fraction distance term relies on, destroying the
    discriminating signal between M-Scale and M-Random outputs.

    Parameters
    ----------
    essen_dir    : path to directory containing .krn files
    max_melodies : maximum number of melodies to return
    min_notes    : minimum melody length (inclusive)
    max_notes    : maximum melody length (inclusive)
    seed         : random seed for shuffling
    transpose    : if True, transpose every melody to C major before filtering.
                   Use False (default) for the privacy/retrieval experiment.

    Returns
    -------
    List of up to max_melodies melody sequences.
    """
    essen_dir = Path(essen_dir)
    kern_files = sorted(essen_dir.rglob("*.krn"))

    raw: list[list[int]] = []
    for kf in kern_files:
        notes = parse_kern_file(kf)
        if notes:
            raw.append(notes)

    if transpose:
        raw = [transpose_to_c(m) for m in raw]

    melodies = filter_melodies(raw, min_notes, max_notes)

    rng = random.Random(seed)
    rng.shuffle(melodies)
    return melodies[:max_melodies]


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------

def save_dataset(melodies: list[list[int]], path: str | Path) -> None:
    """Save processed dataset to a JSON file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as fh:
        json.dump(melodies, fh)
    print(f"Saved {len(melodies)} melodies to {path}")


def load_dataset(path: str | Path) -> list[list[int]]:
    """Load processed dataset from a JSON file."""
    with open(path, "r") as fh:
        data = json.load(fh)
    print(f"Loaded {len(data)} melodies from {path}")
    return data


# ---------------------------------------------------------------------------
# Synthetic dataset fallback (for testing without the corpus)
# ---------------------------------------------------------------------------

def make_synthetic_dataset(
    n: int = 200,
    min_notes: int = 10,
    max_notes: int = 30,
    seed: int = 42,
) -> list[list[int]]:
    """
    Generate n synthetic diatonic melodies for testing.

    Each melody consists of notes drawn randomly from C major across
    octaves 4–5, with a random walk contour so they are not all the same.

    Parameters
    ----------
    n         : number of melodies
    min_notes : minimum length
    max_notes : maximum length
    seed      : RNG seed

    Returns
    -------
    List of n melody sequences.
    """
    rng = random.Random(seed)
    scale_notes = [60, 62, 64, 65, 67, 69, 71, 72, 74, 76, 77, 79, 81, 83]  # C4–B5
    melodies: list[list[int]] = []

    for _ in range(n):
        length = rng.randint(min_notes, max_notes)
        # Start at a random scale degree
        idx = rng.randint(0, len(scale_notes) - 1)
        melody: list[int] = [scale_notes[idx]]
        for _ in range(length - 1):
            # Step by -2, -1, 0, +1, or +2 scale degrees
            step = rng.choice([-2, -1, -1, 0, 1, 1, 2])
            idx = max(0, min(len(scale_notes) - 1, idx + step))
            melody.append(scale_notes[idx])
        melodies.append(melody)

    return melodies


# ---------------------------------------------------------------------------
# Quick sanity check (run: python -m src.utils)
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    dataset = make_synthetic_dataset(n=10, seed=0)
    print(f"Generated {len(dataset)} synthetic melodies")
    for i, m in enumerate(dataset[:3]):
        print(f"  Melody {i}: length={len(m)}, notes={m}")

    # Test transpose_to_c
    test = [61, 65, 68, 73]  # C#4-F4-G#4-C#5
    t = transpose_to_c(test)
    print(f"\nTranspose test: {test} → {t}")
    pcs = [n % 12 for n in t]
    in_scale = sum(1 for pc in pcs if pc in C_MAJOR_PCS)
    print(f"  Pitch classes: {pcs}, {in_scale}/{len(pcs)} in C major")
