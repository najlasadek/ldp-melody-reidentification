# ldp-melody-reidentification

Code for the paper:
**Equal ε, Unequal Exposure: Re-identification Disparities in Structure-Aware LDP for Music**

---

## Repository structure

```
src/
├── mechanisms.py          # M-Random and M-Scale LDP perturbation
├── metrics.py             # Interval-Hamming + combined distance (alpha=1.66)
├── evaluation.py          # Agnostic and aware attacker experiments with Wilson CIs
└── utils.py               # Dataset loading, Essen corpus parser, synthetic data

data/
├── melodies_europa.json             # 200 Europa melodies (10-30 notes)
└── melodies_synthetic_balanced.json # Key-balanced ablation dataset

experiments_extended/
├── src/utils_extended.py  # MTC and SymbTr dataset parsers
├── scripts/               # 00_download_data.sh through 06_plot_paper_style.py
├── data/processed/        # meertens_500.json, symbtr_200.json, jazz_200.json
└── results/               # JSON results for Europa, Meertens, SymbTr, Jazz

adaptive_aware/
├── src/adaptive_evaluation.py  # epsilon-adaptive aware attacker
├── run_adaptive.py             # Europa 30-trial sweep (heuristic + adaptive)
├── run_remaining.py            # Meertens/SymbTr/Jazz 10-trial sweep
├── plot_paper_style.py         # Paper-style figures (Fig. 8 and 9)
├── plot_adaptive.py            # Exploratory plots
├── results/                    # JSON results for all four corpora
└── figures/                    # Generated PDF/PNG figures

run_table1.py              # Reproduce Table 1
run_ablation.py            # Key-balanced ablation
verify_expectation.py      # Monte Carlo verification of Proposition 1
requirements.txt
```

---

## Setup

```bash
pip install -r requirements.txt
```

---

## Datasets

| Dataset | N | chi_mean | Source |
|---------|---|----------|--------|
| Europa 200 | 200 | 12.4% | Essen Folksong Collection (Europa subset) |
| Meertens 500 | 500 | 8.8% | MTC-FS-INST-2.0 (Zenodo 3551003) |
| SymbTr 200 | 200 | 21.5% | MTG/SymbTr (GitHub) |
| Jazz 200 | 200 | 37.9% | Weimar Jazz Database v1.2 |

Raw data files for Meertens and SymbTr are excluded due to size. Download via:
```bash
bash experiments_extended/scripts/00_download_data.sh
```

---

## Reproducing main results

### Table 1 (Europa 200, aware attacker)
```bash
python run_table1.py
```

### Extended epsilon sweep (all four corpora)
```bash
cd experiments_extended/
python scripts/02_run_eps_sweep.py    # Europa 200
python scripts/03_run_meertens.py     # Meertens 500
python scripts/04_run_symbtr.py       # SymbTr 200
python scripts/05_run_jazz.py         # Jazz 200
python scripts/06_plot_paper_style.py # Generate figures
```

### Heuristic vs epsilon-adaptive aware attacker
```bash
cd adaptive_aware/
python run_adaptive.py     # Europa (30 trials)
python run_remaining.py    # Meertens, SymbTr, Jazz (10 trials each)
python plot_paper_style.py # Generates Fig. 8 and Fig. 9
```

---

## Attacker hierarchy

Three levels of increasing strength (paper Section 4.5, Appendix 8.10):

1. **Heuristic aware**: knows the mechanism but not epsilon. Uses Proposition 1
   fixed at epsilon = ln 7:
   `E[chi(m')] = 535/1716 + (36/143)*chi(c)`

2. **epsilon-adaptive aware**: knows both mechanism and epsilon. Recomputes at
   each epsilon using `p = exp(e)/(exp(e)+6)`:
   `E[chi(m')] = (60-35p)/132 + (84p-12)/132 * chi(c)`

3. **Bayesian**: exact per-note log-likelihoods, marginalising over all
   C(12,7)=792 possible M-Random alphabets (Europa only; Appendix 8.9).

---

## Key results

| Corpus | Peak ratio | at epsilon | Notes |
|--------|-----------|------------|-------|
| Europa 200 (heuristic) | 14.38x | ln 8 | |
| Europa 200 (adaptive) | 13.50x vs 4.50x heuristic | ln 4 | adaptive >> heuristic at low eps |
| Meertens 500 | 21.3x | ln 16 | |
| SymbTr 200 | 47.7x | ln 5 | high chromaticity amplifies fingerprint |
| Jazz 200 | 0.36x min | ln 16 | complete reversal |

---

## Verifying Proposition 1

```bash
python verify_expectation.py
```
Monte Carlo check of `E[chi(M-Random(m))] = 535/1716 + (36/143)*chi(m)` at epsilon = ln 7.

## Ablation

```bash
python run_ablation.py
```
Key-balanced dataset (20 melodies x 12 keys). Ratio drops to ~1.7x, confirming
the fingerprint exploits corpus key-skew rather than mechanism structure alone.

---

## Implementation notes

- **Alpha = 1.66**: variance-matched weight for combined distance (paper Section 3.3)
- **Aware attacker (M-Scale)**: `d(m', c) = d_IH(m', c) + alpha*(n-1)*chi(c)`
- **Aware attacker (M-Random)**: `d(m', c) = d_IH(m', c) + alpha*(n-1)*|chi(m') - E[chi(m')]|`
- **Wilson CIs**: 95% Wilson score intervals used throughout
- **Results cutoff**: main paper reports up to epsilon = ln 16; full sweep to ln 25 is in the JSON files
