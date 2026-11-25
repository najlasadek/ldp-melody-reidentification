"""
01_prepare_datasets.py
======================
Parse and filter all three datasets, save processed JSON files.

Datasets:
  1. Europa 200      — already exists at data/europa_200.json (just print stats)
  2. Meertens 500    — parsed from data/meertens_raw/ → data/processed/meertens_500.json
  3. SymbTr 200      — parsed from data/symbtr_raw/   → data/processed/symbtr_200.json

Run from experiments_extended/ directory:
    python scripts/01_prepare_datasets.py

Requirements:
    pip install mido  (for SymbTr MIDI parsing)
"""

from __future__ import annotations

import json
import statistics
import sys
from pathlib import Path

# Add src/ to path so we can import our modules
SCRIPT_DIR = Path(__file__).parent
BASE_DIR = SCRIPT_DIR.parent
sys.path.insert(0, str(BASE_DIR))

from src.metrics import chromatic_fraction
from src.utils_extended import (
    parse_mtc_directory,
    parse_mtc_jsonl,
    parse_symbtr_directory,
    save_dataset,
    load_dataset,
    filter_melodies,
    print_dataset_stats,
)


def dataset_stats(melodies: list[list[int]], name: str) -> dict:
    """Compute and print statistics, return dict of key numbers."""
    lengths = [len(m) for m in melodies]
    chrom_fracs = [chromatic_fraction(m) for m in melodies]
    near_diat = sum(1 for f in chrom_fracs if f < 0.05)

    stats = {
        "name": name,
        "N": len(melodies),
        "length_mean": statistics.mean(lengths),
        "length_std": statistics.stdev(lengths) if len(lengths) > 1 else 0,
        "length_min": min(lengths),
        "length_max": max(lengths),
        "chi_mean": statistics.mean(chrom_fracs),
        "chi_std": statistics.stdev(chrom_fracs) if len(chrom_fracs) > 1 else 0,
        "near_diatonic": near_diat,
        "chromatic": len(melodies) - near_diat,
    }

    print(f"\n{'='*50}")
    print(f"Dataset: {name}")
    print(f"{'='*50}")
    print(f"  N                : {stats['N']}")
    print(f"  Length mean±std  : {stats['length_mean']:.1f} ± {stats['length_std']:.1f}")
    print(f"  Length range     : {stats['length_min']}–{stats['length_max']}")
    print(f"  Chi mean±std     : {stats['chi_mean']*100:.1f}% ± {stats['chi_std']*100:.1f}%")
    print(f"  Near-diatonic    : {stats['near_diatonic']} (chi < 0.05)")
    print(f"  Chromatic        : {stats['chromatic']} (chi >= 0.05)")

    return stats


def main():
    data_dir = BASE_DIR / "data"
    processed_dir = data_dir / "processed"
    processed_dir.mkdir(parents=True, exist_ok=True)

    all_stats = {}

    # =========================================================================
    # Dataset 1: Europa 200 (existing)
    # =========================================================================
    print("\n" + "="*60)
    print("DATASET 1: Europa 200 (existing)")
    print("="*60)

    europa_path = data_dir / "europa_200.json"
    if not europa_path.exists():
        print(f"ERROR: {europa_path} not found. Copy it from ../data/melodies_europa.json")
        sys.exit(1)

    europa = load_dataset(europa_path)
    stats_europa = dataset_stats(europa, "Europa 200")
    all_stats["europa"] = stats_europa

    # =========================================================================
    # Dataset 2: Meertens 500
    # =========================================================================
    print("\n" + "="*60)
    print("DATASET 2: Meertens Tune Collections — 500 melodies")
    print("="*60)

    meertens_out = processed_dir / "meertens_500.json"

    if meertens_out.exists():
        print(f"Found existing {meertens_out} — loading (delete to re-parse)")
        meertens = load_dataset(meertens_out)
    else:
        meertens_raw_dir = data_dir / "meertens_raw"
        if not meertens_raw_dir.exists() or not any(meertens_raw_dir.iterdir()):
            print(f"ERROR: {meertens_raw_dir} is empty or missing.")
            print("Run:  bash scripts/00_download_data.sh  first.")
            print("Skipping Meertens dataset.")
            meertens = []
        else:
            print(f"Parsing Meertens from {meertens_raw_dir} ...")
            meertens = parse_mtc_directory(
                meertens_raw_dir,
                max_melodies=500,
                min_notes=10,
                max_notes=30,
                seed=42,
            )

            if len(meertens) < 10:
                print(f"WARNING: Only {len(meertens)} Meertens melodies found.")
                print("Check the data directory structure.")
            else:
                save_dataset(meertens, meertens_out)

    if meertens:
        stats_meertens = dataset_stats(meertens, f"Meertens (N={len(meertens)})")
        all_stats["meertens"] = stats_meertens
    else:
        print("Meertens dataset unavailable — skipping.")
        all_stats["meertens"] = {"N": 0}

    # =========================================================================
    # Dataset 3: SymbTr 200 (Turkish Makam, semitone-only)
    # =========================================================================
    print("\n" + "="*60)
    print("DATASET 3: SymbTr Turkish Makam Music — 200 melodies")
    print("(semitone representation: note_on pitch integers only)")
    print("="*60)

    symbtr_out = processed_dir / "symbtr_200.json"

    if symbtr_out.exists():
        print(f"Found existing {symbtr_out} — loading (delete to re-parse)")
        symbtr = load_dataset(symbtr_out)
    else:
        symbtr_raw_dir = data_dir / "symbtr_raw"
        midi_files = list(symbtr_raw_dir.glob("**/*.mid")) + \
                     list(symbtr_raw_dir.glob("**/*.midi"))

        if not midi_files:
            print(f"ERROR: No MIDI files in {symbtr_raw_dir}")
            print("Run:  bash scripts/00_download_data.sh  first.")
            print("Skipping SymbTr dataset.")
            symbtr = []
        else:
            print(f"Parsing {len(midi_files)} SymbTr MIDI files ...")
            print("Note: pitch-bend is NEVER read. Only note_on pitch integers used.")
            symbtr = parse_symbtr_directory(
                symbtr_raw_dir,
                max_melodies=200,
                min_notes=10,
                max_notes=30,
                seed=42,
            )

            if len(symbtr) < 10:
                print(f"WARNING: Only {len(symbtr)} SymbTr melodies found.")
            else:
                save_dataset(symbtr, symbtr_out)

    if symbtr:
        stats_symbtr = dataset_stats(symbtr, f"SymbTr (N={len(symbtr)})")
        all_stats["symbtr"] = stats_symbtr

        # Extra check: verify pitch-bend was never used by checking no quarter-tones
        # All values should be integers in [0, 127]
        all_valid = all(
            isinstance(n, int) and 0 <= n <= 127
            for melody in symbtr for n in melody
        )
        print(f"\n  Semitone sanity check: all notes are int in [0,127]: {all_valid}")
        assert all_valid, "SymbTr contains invalid pitch values!"

    else:
        print("SymbTr dataset unavailable — skipping.")
        all_stats["symbtr"] = {"N": 0}

    # =========================================================================
    # Summary
    # =========================================================================
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    for name, s in all_stats.items():
        n = s.get("N", 0)
        nd = s.get("near_diatonic", "N/A")
        chi = s.get("chi_mean", 0)
        print(f"  {name:12s}  N={n:4d}  chi_mean={chi*100:.1f}%  near_diat={nd}")

    print("\nNext step: python scripts/02_run_eps_sweep.py")
    print("           (reproduces paper values on Europa 200 first)")


if __name__ == "__main__":
    main()
