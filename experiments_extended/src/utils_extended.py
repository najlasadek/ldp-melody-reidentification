"""
utils_extended.py
=================
Dataset parsers for the extended experiments.

Two new datasets:
  1. Meertens Tune Collections (MTC-FS-INST-2.0)
     JSON Lines format — pitch sequences already extracted as MIDI integers.
     Source: https://zenodo.org/records/3551003

  2. SymbTr — Turkish Makam Music (12-TET semitone representation)
     MIDI files — read ONLY note_on pitch integers (0-127).
     Pitch-bend is NEVER read or used at any point.
     Source: https://github.com/MTG/SymbTr  (or Zenodo 15470412)
     Result: pure semitone pitch sequences, identical format to Europa data.
"""

from __future__ import annotations

import json
import random
from pathlib import Path


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def filter_melodies(
    melodies: list[list[int]],
    min_notes: int = 10,
    max_notes: int = 30,
) -> list[list[int]]:
    """Keep melodies with between min_notes and max_notes notes (inclusive)."""
    out = []
    for m in melodies:
        m = [max(0, min(127, n)) for n in m]
        if min_notes <= len(m) <= max_notes:
            out.append(m)
    return out


def save_dataset(melodies: list[list[int]], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as fh:
        json.dump(melodies, fh)
    print(f"Saved {len(melodies)} melodies to {path}")


def load_dataset(path: str | Path) -> list[list[int]]:
    with open(path, "r") as fh:
        data = json.load(fh)
    print(f"Loaded {len(data)} melodies from {path}")
    return data


def print_dataset_stats(melodies: list[list[int]], name: str) -> None:
    """Print N, mean/std length, chromatic fraction distribution."""
    import statistics

    import sys
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from src.metrics import chromatic_fraction

    lengths = [len(m) for m in melodies]
    chrom_fracs = [chromatic_fraction(m) for m in melodies]
    near_diat = sum(1 for f in chrom_fracs if f < 0.05)

    print(f"\n=== Dataset: {name} ===")
    print(f"  N                : {len(melodies)}")
    print(f"  Length mean      : {statistics.mean(lengths):.1f}")
    print(f"  Length std       : {statistics.stdev(lengths):.1f}")
    print(f"  Length range     : {min(lengths)}–{max(lengths)}")
    print(f"  Chi mean         : {statistics.mean(chrom_fracs)*100:.1f}%")
    print(f"  Chi std          : {statistics.stdev(chrom_fracs)*100:.1f}%")
    print(f"  Near-diatonic    : {near_diat} (chi < 0.05)")
    print(f"  Chromatic        : {len(melodies) - near_diat} (chi >= 0.05)")


# ---------------------------------------------------------------------------
# Meertens Tune Collections (MTC-FS-INST-2.0) parser
# ---------------------------------------------------------------------------

def parse_mtc_jsonl(
    jsonl_path: str | Path,
    max_melodies: int = 500,
    min_notes: int = 10,
    max_notes: int = 30,
    seed: int = 42,
) -> list[list[int]]:
    """
    Parse MTC-FS-INST-2.0 JSON Lines file.

    The MTC-FS format stores each melody as a JSON object per line.
    Relevant fields (vary by MTC version — we handle both):
      - "pitch"     : list of MIDI pitch integers (primary)
      - "midipitch" : alternative field name
      - "notes"     : list of dicts with "pitch" key

    Parameters
    ----------
    jsonl_path   : path to the .jsonl or .json file
    max_melodies : maximum melodies to return after filtering
    min_notes    : minimum melody length
    max_notes    : maximum melody length
    seed         : random seed for shuffling

    Returns
    -------
    List of pitch sequences (MIDI integers)
    """
    raw: list[list[int]] = []
    path = Path(jsonl_path)

    with open(path, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue

            # Try different MTC field names for pitch sequences
            pitches = None

            # Option 1: MTC-FS-INST-2.0 format — features['midipitch']
            # This is the primary format for MTC-FS-INST-2.0_sequences.jsonl
            if "features" in obj and isinstance(obj["features"], dict):
                feat = obj["features"]
                for key in ("midipitch", "pitch40"):
                    if key in feat and isinstance(feat[key], list):
                        candidate = feat[key]
                        if candidate and isinstance(candidate[0], (int, float)):
                            pitches = [int(x) for x in candidate]
                            break

            # Option 2: direct list under "pitch" or "midipitch"
            if pitches is None:
                for key in ("midipitch", "midi_pitch", "pitches"):
                    if key in obj and isinstance(obj[key], list):
                        candidate = obj[key]
                        if candidate and isinstance(candidate[0], (int, float)):
                            pitches = [int(x) for x in candidate]
                            break

            # Option 3: "notes" is a list of dicts
            if pitches is None and "notes" in obj and isinstance(obj["notes"], list):
                notes_list = obj["notes"]
                if notes_list and isinstance(notes_list[0], dict):
                    extracted = []
                    for note in notes_list:
                        for k in ("pitch", "midi", "midipitch"):
                            if k in note and isinstance(note[k], (int, float)):
                                extracted.append(int(note[k]))
                                break
                    if extracted:
                        pitches = extracted

            # Option 4: "melody" field
            if pitches is None and "melody" in obj and isinstance(obj["melody"], list):
                candidate = obj["melody"]
                if candidate and isinstance(candidate[0], (int, float)):
                    pitches = [int(x) for x in candidate]

            if pitches and len(pitches) >= min_notes:
                raw.append(pitches)

    # Filter and sample
    filtered = filter_melodies(raw, min_notes, max_notes)
    rng = random.Random(seed)
    rng.shuffle(filtered)
    result = filtered[:max_melodies]
    print(f"MTC parser: {len(raw)} raw, {len(filtered)} filtered (10-30 notes), {len(result)} selected")
    return result


def parse_mtc_directory(
    data_dir: str | Path,
    max_melodies: int = 500,
    min_notes: int = 10,
    max_notes: int = 30,
    seed: int = 42,
) -> list[list[int]]:
    """
    Parse MTC data from a directory containing JSON/JSONL files.
    Tries each file in sorted order, accumulates all parseable melodies,
    then samples max_melodies.
    """
    data_dir = Path(data_dir)
    raw: list[list[int]] = []

    # Look for jsonl, json, txt files
    files = sorted(data_dir.glob("**/*.jsonl")) + \
            sorted(data_dir.glob("**/*.json")) + \
            sorted(data_dir.glob("**/*.txt"))

    if not files:
        print(f"No JSON/JSONL files found in {data_dir}")
        return []

    for fp in files:
        try:
            partial = parse_mtc_jsonl(fp, max_melodies=99999,
                                       min_notes=min_notes, max_notes=max_notes,
                                       seed=seed)
            raw.extend(partial)
        except Exception as e:
            print(f"  Skipping {fp.name}: {e}")
        if len(raw) >= max_melodies * 10:
            break  # Enough candidates

    rng = random.Random(seed)
    rng.shuffle(raw)
    result = raw[:max_melodies]
    print(f"MTC directory: {len(raw)} total, {len(result)} selected")
    return result


# ---------------------------------------------------------------------------
# SymbTr MIDI parser (semitone-only, pitch-bend NEVER used)
# ---------------------------------------------------------------------------

def parse_symbtr_midi(
    midi_path: str | Path,
    min_notes: int = 10,
    max_notes: int = 30,
) -> list[int] | None:
    """
    Parse a SymbTr MIDI file and return a pure-semitone pitch sequence.

    IMPORTANT: Only note_on messages with velocity > 0 are read.
    Pitch-bend messages are completely ignored — never loaded, never stored.
    The result is a list of integers in [0, 127], identical in format to
    all other datasets in this study.

    For polyphonic tracks, the highest simultaneous pitch is taken (melody
    line heuristic). Tracks where more than 20% of events are simultaneous
    notes are flagged as likely accompaniment and skipped.

    Parameters
    ----------
    midi_path : path to MIDI file
    min_notes : minimum acceptable melody length
    max_notes : maximum acceptable melody length

    Returns
    -------
    List of MIDI pitch integers, or None if parsing fails / filters out.
    """
    try:
        import mido
    except ImportError:
        raise ImportError("mido is required for SymbTr parsing: pip install mido")

    midi_path = Path(midi_path)
    try:
        mid = mido.MidiFile(midi_path)
    except Exception as e:
        return None

    best_track_notes = None
    best_track_len = -1

    for track in mid.tracks:
        # Extract note_on events only (velocity > 0 = note on; velocity 0 = note off)
        # Pitch-bend messages are simply not extracted — this loop ignores them.
        time_events: list[tuple[float, int]] = []  # (absolute_time, pitch)
        abs_time = 0.0

        for msg in track:
            abs_time += msg.time
            # Only note_on with velocity > 0
            if msg.type == "note_on" and msg.velocity > 0:
                time_events.append((abs_time, msg.note))

        if len(time_events) < min_notes:
            continue

        # Check for polyphony: group notes by time, detect simultaneity
        from collections import defaultdict
        time_groups: dict = defaultdict(list)
        for t, p in time_events:
            time_groups[t].append(p)

        n_events = len(time_events)
        n_simultaneous = sum(len(g) - 1 for g in time_groups.values() if len(g) > 1)
        polyphony_ratio = n_simultaneous / n_events if n_events > 0 else 0

        if polyphony_ratio > 0.20:
            # More than 20% simultaneous notes — likely accompaniment, skip
            continue

        # Take highest pitch when simultaneous notes exist (melody heuristic)
        pitches = [max(g) for t, g in sorted(time_groups.items())]

        # Clamp to valid MIDI range
        pitches = [max(0, min(127, p)) for p in pitches]

        if len(pitches) > best_track_len:
            best_track_len = len(pitches)
            best_track_notes = pitches

    if best_track_notes is None:
        return None

    # Apply length filter
    if not (min_notes <= len(best_track_notes) <= max_notes):
        # Try truncation if too long
        if len(best_track_notes) > max_notes:
            best_track_notes = best_track_notes[:max_notes]
        else:
            return None

    return best_track_notes


def parse_symbtr_directory(
    symbtr_dir: str | Path,
    max_melodies: int = 200,
    min_notes: int = 10,
    max_notes: int = 30,
    seed: int = 42,
) -> list[list[int]]:
    """
    Parse all MIDI files in a SymbTr directory.

    Scans recursively for *.mid / *.midi files.
    Returns up to max_melodies pitch sequences (semitone-only).

    Parameters
    ----------
    symbtr_dir   : directory containing SymbTr MIDI files
    max_melodies : maximum number to return
    min_notes    : minimum melody length
    max_notes    : maximum melody length
    seed         : random seed for shuffling

    Returns
    -------
    List of pitch sequences (MIDI integers, 0-127)
    """
    symbtr_dir = Path(symbtr_dir)
    midi_files = sorted(symbtr_dir.glob("**/*.mid")) + \
                 sorted(symbtr_dir.glob("**/*.midi"))

    if not midi_files:
        print(f"No MIDI files found in {symbtr_dir}")
        return []

    print(f"Found {len(midi_files)} MIDI files in {symbtr_dir}")

    raw: list[list[int]] = []
    failed = 0

    for mf in midi_files:
        result = parse_symbtr_midi(mf, min_notes=min_notes, max_notes=max_notes)
        if result is not None:
            raw.append(result)
        else:
            failed += 1

        # Stop scanning if we have enough candidates
        if len(raw) >= max_melodies * 5:
            break

    print(f"SymbTr: parsed {len(raw)} valid tracks, {failed} failed/filtered")

    rng = random.Random(seed)
    rng.shuffle(raw)
    result = raw[:max_melodies]
    print(f"SymbTr: selected {len(result)} melodies")
    return result


# ---------------------------------------------------------------------------
# Quick sanity check
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import sys
    print("utils_extended.py loaded OK")
    print("Available parsers:")
    print("  parse_mtc_jsonl(path)       — single MTC JSON Lines file")
    print("  parse_mtc_directory(dir)    — directory of MTC files")
    print("  parse_symbtr_midi(path)     — single SymbTr MIDI (semitone-only)")
    print("  parse_symbtr_directory(dir) — directory of SymbTr MIDI files")
