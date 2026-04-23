# Extended Experiments: LDP Re-identification Across Datasets

Extended version of the experiments from the paper:
**"Equal ε, Unequal Exposure: Re-identification Disparities in Structure-Aware LDP for Music"**

This folder runs the same experiments on:
1. A **dense epsilon grid** (15 values from ln(1.5) to ln(25))
2. A **larger Western dataset**: Meertens Tune Collections (500 melodies)
3. A **non-Western dataset**: SymbTr Turkish Makam Music (200 melodies, semitone-only)

The original `code/` folder is untouched. All extended experiments live here.

---

## Folder Structure

```
experiments_extended/
├── src/
│   ├── mechanisms.py        (copy from code/src/ — unchanged)
│   ├── metrics.py           (copy from code/src/ — unchanged)
│   ├── evaluation.py        (copy from code/src/ — unchanged)
│   ├── utils.py             (copy from code/src/ — unchanged)
│   └── utils_extended.py    (new: MTC JSON Lines parser + SymbTr MIDI parser)
├── data/
│   ├── europa_200.json      (copy of existing 200-melody dataset)
│   ├── meertens_raw/        (raw MTC-FS download)
│   ├── symbtr_raw/          (raw SymbTr MIDI files)
│   └── processed/
│       ├── meertens_500.json
│       └── symbtr_200.json
├── scripts/
│   ├── 00_download_data.sh  (download MTC-FS and SymbTr)
│   ├── 01_prepare_datasets.py
│   ├── 02_run_eps_sweep.py  (Europa 200, dense epsilon grid)
│   ├── 03_run_meertens.py
│   ├── 04_run_symbtr.py
│   └── 05_plot_all.py
├── results/
│   ├── europa_eps_sweep.json
│   ├── meertens_eps_sweep.json
│   └── symbtr_eps_sweep.json
└── figures/
    ├── fig_eps_sweep_europa.pdf
    ├── fig_eps_sweep_meertens.pdf
    ├── fig_eps_sweep_symbtr.pdf
    ├── fig_comparison_all.pdf
    └── fig_comparison_nd.pdf
```

---

## Setup

```bash
pip install -r requirements.txt
```

Dependencies: `numpy`, `matplotlib`, `mido` (for SymbTr MIDI parsing).

---

## Execution Order

All scripts run from the `experiments_extended/` directory.

### Step 0: Download data
```bash
bash scripts/00_download_data.sh
```
Downloads:
- Meertens Tune Collections (MTC-FS-INST-2.0) from Zenodo record 3551003
- SymbTr Turkish Makam Music from GitHub (MTG/SymbTr)

### Step 1: Prepare datasets
```bash
python scripts/01_prepare_datasets.py
```
- Parses and filters all datasets to 10–30 notes
- Prints statistics: N, length distribution, chromatic fraction
- Saves `data/processed/meertens_500.json` and `data/processed/symbtr_200.json`

### Step 2: Europa 200 sweep (MUST run first)
```bash
python scripts/02_run_eps_sweep.py
```
- **First reproduces paper values** at {ln3, ln7, ln12} and verifies within ±0.5x
- Then runs full 15-epsilon sweep
- Saves `results/europa_eps_sweep.json`

### Step 3: Meertens sweep
```bash
python scripts/03_run_meertens.py
```

### Step 4: SymbTr sweep
```bash
python scripts/04_run_symbtr.py
```

### Step 5: Generate figures
```bash
python scripts/05_plot_all.py
```
Generates 5 PDF figures in `figures/`.

---

## Epsilon Grid

15 values from ln(1.5) ≈ 0.41 to ln(25) ≈ 3.22:

| ε | exp(ε) | Retention prob p |
|---|--------|-----------------|
| ln(1.5) | 1.5 | 0.300 |
| ln(2)   | 2   | 0.333 |
| ln(3)   | 3   | 0.375 |
| ln(4)   | 4   | 0.400 |
| ln(5)   | 5   | 0.417 |
| ln(6)   | 6   | 0.429 |
| **ln(7)** | **7** | **0.538** ← paper reference |
| ln(8)   | 8   | 0.533 |
| ln(9)   | 9   | 0.600 |
| ln(10)  | 10  | 0.625 |
| ln(12)  | 12  | 0.667 |
| ln(14)  | 14  | 0.700 |
| ln(16)  | 16  | 0.727 |
| ln(20)  | 20  | 0.769 |
| ln(25)  | 25  | 0.806 |

---

## Datasets

### Dataset 1: Europa 200 (baseline)
- Essen Folksong Collection, Europa subset
- 200 monophonic melodies, 10–30 notes, natural keys (no transposition)
- Chromatic fraction mean = 12.4%, std = 12.5%
- Near-diatonic (χ < 0.05): 73 melodies

### Dataset 2: Meertens Tune Collections (MTC-FS-INST-2.0)
- Source: https://zenodo.org/records/3551003
- Dutch and Western European folk tunes
- 500 melodies selected (filtered from ~18,000)
- Pitch sequences already extracted as MIDI integers in the MTC format

### Dataset 3: SymbTr Turkish Makam Music (semitone-only)
- Source: https://github.com/MTG/SymbTr
- 200 melodies from a corpus of 2,200 pieces across 155 makams
- **Semitone representation**: only `note_on` MIDI pitch integers (0–127) are used
- Pitch-bend messages are **never loaded, never stored, never used**
- Result: pure semitone pitch sequences identical in format to Europa/Meertens
- Turkish makam music rarely aligns with C-major → expect high chromatic fraction (χ ≈ 0.4–0.7)
- Scientific motivation: M-Scale forces all outputs to χ = 0; the gap from input χ is larger,
  potentially amplifying the fingerprint effect studied in the paper
- ISMIR 2026 is in Abu Dhabi, UAE, themed "Crossroads" — Turkish makam directly represents
  the East-West musical intersection

---

## Rigorous Checks

1. **Paper reproduction**: Script 02 verifies the paper's {ln3, ln7, ln12} results within ±0.5x
   before running any new epsilon values.

2. **Mechanism sanity**: M-Scale output chromatic fraction should be ≈ 0 (verified in evaluation.py).

3. **SymbTr semitone check**: Script 04 asserts all pitch values are `int` in [0, 127] before any computation.

4. **Wilson CI sanity**: Asserted lower ≤ point estimate ≤ upper for all cells.

5. **Ratio sanity**: Negative ratios flagged as assertion errors; very large ratios (>200x) emit warnings.

---

## Key Results (fill in after running)

| Dataset | ε | Agnostic ratio | Aware ratio |
|---------|---|---------------|------------|
| Europa 200 | ln(7) | ~3.32x | ~8.44x |
| Meertens   | ln(7) | TBD    | TBD    |
| SymbTr     | ln(7) | TBD    | TBD    |

---

## Notes

- All scripts are run-from-directory scripts: `python scripts/XX_....py` from `experiments_extended/`
- Results are saved as JSON with full raw numbers; nothing is lost if plotting fails
- Figures use `tableau-colorblind10` palette (ISMIR accessibility guideline)
- The original `code/` folder is never modified
