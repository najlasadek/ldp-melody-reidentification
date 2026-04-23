"""
09_prepare_jazz.py
==================
Prepare the Weimar Jazz Database for the LDP experiment.

Source: Weimar Jazz Database v1.2 (WJazzD)
        https://jazzomat.hfm-weimar.de/
        score CSV data: github.com/jazzomat/raw_data/wjazzd_v_1_2/score/score_csv.zip

The score CSVs contain one note per row: pitch (MIDI float), onset, duration.
Each file = one solo transcription (299 solos total).

Solos average 427 notes — none fall naturally in 10-30 notes, so we truncate
each solo to the first 30 notes (consistent with the length cap used for all
other corpora) and keep those with >= 10 notes.

Jazz characteristics (C-major definition):
  - Chi mean ~38%: much higher than SymbTr (21.5%), Meertens (8.8%), Europa (12.4%)
  - Only ~10/299 solos are near-diatonic (chi < 0.05)
  - Expected: largest M-Scale/M-Random re-identification gap of all corpora

Output: data/processed/jazz_200.json
        data/jazz_raw/score_csv.zip  (cached raw)

Run from experiments_extended/ directory:
    python scripts/09_prepare_jazz.py
"""

from __future__ import annotations

import json
import random
import statistics
import sys
import urllib.request
import zipfile
from pathlib import Path

SCRIPT_DIR = Path(__file__).parent
BASE_DIR   = SCRIPT_DIR.parent
sys.path.insert(0, str(BASE_DIR))

from src.utils_extended import save_dataset

# ── Configuration ──────────────────────────────────────────────────────────────

CSV_ZIP_URL = (
    "https://raw.githubusercontent.com/jazzomat/raw_data/master/"
    "wjazzd_v_1_2/score/score_csv.zip"
)
RAW_DIR  = BASE_DIR / "data" / "jazz_raw"
ZIP_PATH = RAW_DIR / "score_csv.zip"
OUT_PATH = BASE_DIR / "data" / "processed" / "jazz_200.json"

MAX_MELODIES = 200
MIN_NOTES    = 10
MAX_NOTES    = 30   # truncate longer solos
SEED         = 42

SCALE = frozenset({0, 2, 4, 5, 7, 9, 11})  # C-major


def chromatic_fraction(melody: list[int]) -> float:
    if not melody:
        return 0.0
    return sum(1 for n in melody if (n % 12) not in SCALE) / len(melody)


def download_zip() -> None:
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    if ZIP_PATH.exists():
        print(f"Score CSV zip already exists at {ZIP_PATH} ({ZIP_PATH.stat().st_size // 1024} KB)")
        return
    print(f"Downloading WJazzD score CSVs from GitHub...")
    print(f"  {CSV_ZIP_URL}")
    try:
        urllib.request.urlretrieve(CSV_ZIP_URL, ZIP_PATH)
        print(f"  Downloaded: {ZIP_PATH.stat().st_size // 1024} KB")
    except Exception as e:
        print(f"  Download failed: {e}")
        print("\nManual download:")
        print(f"  URL: {CSV_ZIP_URL}")
        print(f"  Save as: {ZIP_PATH}")
        sys.exit(1)


def parse_score_csvs(zip_path: Path) -> list[list[int]]:
    """
    Parse all solo CSVs from the WJazzD score_csv.zip.

    Each CSV row: pitch_float, onset, duration (no header).
    Pitch is MIDI pitch as float (e.g. 65.0 = F4).

    Returns list of pitch sequences (one per solo), truncated to MAX_NOTES.
    Solos with fewer than MIN_NOTES notes are dropped.
    """
    solos: list[list[int]] = []
    with zipfile.ZipFile(zip_path, "r") as zf:
        names = sorted(zf.namelist())
        print(f"  Found {len(names)} CSV files in zip")
        for name in names:
            if not name.endswith(".csv"):
                continue
            try:
                data = zf.read(name).decode("utf-8", errors="replace")
            except Exception:
                continue

            pitches: list[int] = []
            for line in data.strip().split("\n"):
                parts = line.strip().split(",")
                if not parts or not parts[0].strip():
                    continue
                try:
                    p = int(float(parts[0]))
                    p = max(0, min(127, p))
                    pitches.append(p)
                except (ValueError, IndexError):
                    continue

            # Truncate to MAX_NOTES
            pitches = pitches[:MAX_NOTES]

            if len(pitches) >= MIN_NOTES:
                solos.append(pitches)

    return solos


def main():
    print("=" * 60)
    print("Weimar Jazz Database — data preparation")
    print("=" * 60)

    # Step 1: Download
    download_zip()

    # Step 2: Parse
    print("\nParsing score CSVs...")
    raw_solos = parse_score_csvs(ZIP_PATH)
    print(f"Parsed {len(raw_solos)} valid solos (>= {MIN_NOTES} notes after truncation)")

    if len(raw_solos) < MAX_MELODIES:
        print(f"WARNING: only {len(raw_solos)} solos available (need {MAX_MELODIES})")
        max_actual = len(raw_solos)
    else:
        max_actual = MAX_MELODIES

    # Step 3: Random sample
    rng = random.Random(SEED)
    rng.shuffle(raw_solos)
    dataset = raw_solos[:max_actual]

    # Step 4: Stats
    chrom_fracs = [chromatic_fraction(m) for m in dataset]
    near_diat   = sum(1 for f in chrom_fracs if f < 0.05)
    lengths     = [len(m) for m in dataset]

    print(f"\n=== Jazz 200 dataset stats ===")
    print(f"  N                         : {len(dataset)}")
    print(f"  Length mean               : {statistics.mean(lengths):.1f}")
    print(f"  Length std                : {statistics.stdev(lengths):.1f}")
    print(f"  Length range              : {min(lengths)}–{max(lengths)}")
    print(f"  Chi mean (C-major def)    : {statistics.mean(chrom_fracs)*100:.1f}%")
    print(f"  Chi std                   : {statistics.stdev(chrom_fracs)*100:.1f}%")
    print(f"  Near-diatonic (chi<0.05)  : {near_diat}")
    print(f"  Chromatic (chi>=0.05)     : {len(dataset) - near_diat}")

    # Pipeline table
    raw_total = len(parse_score_csvs(ZIP_PATH))  # already computed but reuse
    print(f"\n--- Pipeline survival ---")
    print(f"  Total solos in WJazzD v1.2   : {len(raw_solos)}")
    print(f"  After length filter (>= 10)  : {len(raw_solos)}")
    print(f"  (all solos truncated to 30)  : all pass")
    print(f"  Selected (seed={SEED})       : {len(dataset)}")

    # Step 5: Save
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    save_dataset(dataset, OUT_PATH)
    print(f"\nSaved to {OUT_PATH}")


if __name__ == "__main__":
    main()
