# ldp-melody-reidentification

Code, data and results for the paper:

**Equal ε, Unequal Exposure: Re-identification Disparities in LDP for Music**
Najla Sadek (University of Oxford) and Razane Tajeddine (American University of Beirut)

<!-- TODO: add venue, paper link and BibTeX once the camera-ready version is published. -->

The paper compares two randomised-response mechanisms for melodies that satisfy the
same per-note ε-LDP guarantee over pitch classes: **M-Scale** (replacements restricted
to the C-major scale) and **M-Random** (a fresh random 7-note replacement set per
melody). It shows that their empirical re-identification risk under a closed-world
1-nearest-neighbour retrieval attack can differ substantially, and that the direction
of the difference depends on the corpus and on the attacker.

The paper's appendix contains the proofs of ε-LDP and Proposition 1, pseudocode, the
variance-matching derivation for α, the full results table, the balanced-key ablation
and the likelihood-based attacker.

---

## Repository structure

```
src/                          # Core implementation (Europa experiments)
├── mechanisms.py             # M-Random and M-Scale LDP perturbation
├── metrics.py                # Interval-Hamming + combined distance (alpha = 1.66)
├── evaluation.py             # Agnostic and aware attacker experiments, Wilson CIs
└── utils.py                  # Dataset loading, Essen parser, synthetic data

data/
├── melodies_europa.json             # Europa 200 (10-30 notes)
└── melodies_synthetic_balanced.json # Key-balanced ablation dataset

figures/
└── fig3_chromatic_fingerprint.py  # Paper Figure 1

experiments_extended/         # Four-corpus experiments
├── src/                      # Same core modules + utils_extended.py (MTC/SymbTr parsers)
├── scripts/                  # Numbered scripts: data preparation, sweeps, likelihood attacker, plots
├── data/                     # europa_200.json, processed/ (meertens_500, symbtr_200,
│                             # jazz_200), symbtr_raw/ (SymbTr MIDI), jazz_raw/ (WJazzD CSV)
├── results/                  # JSON results: *_eps_sweep.json, bayesian_*.json
└── figures/                  # Paper Figures 2, 3 and 4

adaptive_aware/               # Heuristic vs epsilon-adaptive aware attacker
├── src/adaptive_evaluation.py
├── run_adaptive.py           # All four corpora, 30 trials
└── results/                  # adaptive_<corpus>.json

run_table1.py                 # Europa ln3/ln7/ln12 rows (agnostic and aware heuristic)
run_ablation.py               # Key-balanced ablation, un-normalised distance (submission version)
verify_expectation.py         # Monte Carlo verification of Proposition 1
```

---

## Setup

Python 3.10 or later is required.

```bash
pip install -r requirements.txt
```

---

## Datasets

| Dataset | N | χ_mean | Length handling | Source |
|---------|---|--------|-----------------|--------|
| Europa 200 | 200 | 12.4% | filtered to 10–30 notes | Essen Folksong Collection (Europa subset) |
| Meertens 500 | 500 | 8.8% | filtered to 10–30 notes | MTC-FS-INST-2.0 (Zenodo record 3551003) |
| SymbTr 200 | 200 | 21.5% | truncated to the first 30 notes | MTG/SymbTr (GitHub) |
| Jazz 200 | 200 | 37.9% | truncated to the first 30 notes | Weimar Jazz Database v1.2 |

χ is the fraction of notes outside the C-major scale. The processed datasets are
included, so no download is needed to run the experiments. To rebuild them from the
raw data, the SymbTr MIDI files and the WJazzD score CSVs are included; the Meertens
data are downloaded by:

```bash
cd experiments_extended/
bash scripts/00_download_data.sh      # Meertens (and SymbTr if missing)
python scripts/01_prepare_datasets.py # Meertens 500, SymbTr 200
python scripts/09_prepare_jazz.py     # Jazz 200
```

Each corpus remains subject to the licence and citation requirements of its original
source; please cite the original datasets if you use them.

---

## Reproducing the results

All perturbations use fixed seeds (per trial and per melody), so the results are
deterministic.

### Main results: length-normalised sweeps (all four corpora)
The paper's results use the length-normalised distance `d / (n-1)`.
```bash
cd experiments_extended/
python scripts/17_run_normalised_sweeps.py --validate  # un-normalised run; must reproduce results/ exactly
python scripts/17_run_normalised_sweeps.py             # writes results_normalised/
python scripts/18_run_normalised_ablation.py           # key-balanced ablation (Appendix A.8)
python scripts/16_run_length_normalised.py all         # degeneracy diagnostics (Appendix A.7)
python scripts/19_alpha_sensitivity.py                # sensitivity to alpha (Appendix A.5)
python scripts/20_degeneracy_unnormalised.py          # un-normalised degeneracy (Appendix A.7)
python scripts/21_europa_chromatic_subset.py          # Europa chromatic subset (Section 4.2)
python scripts/07_plot_four_corpora.py --normalised    # Figure 2
python scripts/08_plot_acc_curves.py --normalised      # Figure 3
```

### Original un-normalised sweeps (submission version; compared in Appendix A.7)
```bash
cd experiments_extended/
python scripts/02_run_eps_sweep.py    # Europa 200 (also checks values against the paper)
python scripts/03_run_meertens.py     # Meertens 500
python scripts/04_run_symbtr.py       # SymbTr 200
python scripts/10_run_jazz.py         # Jazz 200
```

### Heuristic vs epsilon-adaptive aware attacker (all four corpora)
```bash
cd adaptive_aware/
python run_adaptive.py
```

### Likelihood-based attacker (all four corpora)
```bash
cd experiments_extended/
python scripts/12_run_bayesian_all.py
```

### Figure 1
```bash
python figures/fig3_chromatic_fingerprint.py   # from the repository root; saved to output/
```

Without `--normalised`, the plotting scripts reproduce the submission-version figures
from `results/` and `../adaptive_aware/results/`.

### Other checks (from the repository root)
```bash
python run_table1.py          # Europa rows at ln3, ln7, ln12
python run_ablation.py        # Key-balanced ablation
python verify_expectation.py  # Monte Carlo check of Proposition 1
```

---

## Attackers

1. **Agnostic**: does not know the mechanism; uses the same distance for both.
2. **Aware heuristic**: knows the mechanism but not ε. For M-Random it uses
   Proposition 1 fixed at ε = ln 7:
   `E[chi(m')] = 535/1716 + (36/143)*chi(c)`
3. **Aware ε-adaptive**: knows the mechanism and ε, and recomputes the expectation
   with `p = exp(eps)/(exp(eps)+6)`:
   `E[chi(m')] = (60-35p)/132 + (84p-12)/132 * chi(c)`
4. **Likelihood-based**: ranks candidates by per-note log-likelihood over pitch classes,
   marginalising over all C(12,7) = 792 possible M-Random alphabets. It sums over the
   shorter melody and ignores octave and length, so it is not optimal and is only
   interpretable on fixed-length corpora (Jazz, SymbTr); see the paper's Appendix A.9.

---

## Key results (aware heuristic attacker, length-normalised distance, ε from ln 3 to ln 16)

| Corpus | M-Scale / M-Random ratio | Correct retrievals (M-Scale vs M-Random) |
|--------|--------------------------|------------------------------------------|
| Europa 200 | 2.93× at ln 4 (maximum), 1.58× at ln 16 | 572 vs 195 of 6,000 at ln 4 |
| Meertens 500 | 3.38× at ln 7 (maximum), 2.45× at ln 16 | 3,314 vs 981 of 15,000 at ln 7 |
| SymbTr 200 | 2.10× at ln 3, falling to 0.93× at ln 16 | 1,743 vs 1,878 of 6,000 at ln 16 |
| Jazz 200 | 0.76× at ln 3, falling to 0.36× at ln 16 | 707 vs 1,954 of 6,000 at ln 16 |

On Jazz the ratio is below 1 at every ε under the distance-based attackers, but the
likelihood-based attacker gives a ratio slightly above 1.

---

## Implementation notes

- **α = 1.66**: variance-matched weight for the combined distance (paper Section 3.3, Appendix A.5).
- **Aware attacker (M-Scale)**: `d(m', c) = d_IH(m', c)/(n-1) + alpha*chi(c)`
- **Aware attacker (M-Random)**: `d(m', c) = d_IH(m', c)/(n-1) + alpha*|chi(m') - E[chi(m')]|`
- **Privacy unit**: the ε-LDP guarantee is per note and over pitch classes. The octave
  and the melody length are released unperturbed and are not covered by the guarantee.
- **Distance**: the paper uses the length-normalised distance (`n = min(|m'|, |c|)`).
  `src/` keeps the original un-normalised distance so the submission results stay
  reproducible; the normalised runs are in `experiments_extended/scripts/16-18`.
- **Wilson CIs**: 95% Wilson score intervals are used throughout.
- **Results range**: the paper reports ε from ln 3 to ln 16; the un-normalised JSON files
  in `results/` also contain ln 1.5, ln 2, ln 20 and ln 25.
