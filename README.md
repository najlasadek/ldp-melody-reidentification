# ldp-melody-reidentification

Code for the paper:
**Equal ε, Unequal Exposure: Re-identification Disparities in Structure-Aware LDP for Music**

## Repository structure

```
code/
├── src/
│   ├── mechanisms.py      # M-RANDOM and M-SCALE LDP perturbation mechanisms
│   ├── metrics.py         # Distance metrics: interval-Hamming, combined distance
│   ├── evaluation.py      # Attacker models and experiment runners
│   └── utils.py           # Dataset loading, Essen corpus parser, synthetic data
├── figures/
│   ├── fig1_acc_vs_eps.py          # Fig. 1: Acc@1 vs. epsilon
│   ├── fig2_ratio_vs_eps.py        # Fig. 2: Ratio M-Scale/M-Random vs. epsilon
│   └── fig3_chromatic_fingerprint.py # Fig. 3: Key-signature fingerprint
├── data/
│   └── melodies_europa.json        # 200 Europa melodies (see below)
├── output/                         # Generated figures (created automatically)
├── run_table1.py          # Reproduce Table 1 (main experiment)
├── run_ablation.py        # Ablation: key-balanced synthetic dataset
├── verify_expectation.py  # Verify Proposition 1 via Monte Carlo
└── requirements.txt
```

## Setup

```bash
pip install -r requirements.txt
```

## Data

The experiment uses 200 monophonic melodies from the **Europa subset** of the
[Essen Folksong Collection](https://kern.humdrum.org/cgi-bin/browse?l=essen),
filtered to lengths 10–30 notes with a fixed random seed.

To prepare the data:
```bash
# Download the Essen corpus
git clone https://github.com/sharc-lab/essen-folksongs data/essen_raw/

# Then run (from the code/ directory):
python -c "
from src.utils import build_dataset, save_dataset
d = build_dataset('data/essen_raw/europa', max_melodies=200, seed=0, transpose=False)
save_dataset(d, 'data/melodies_europa.json')
"
```

The processed JSON file `data/melodies_europa.json` is included in the
supplementary material.

## Reproducing Table 1

```bash
cd code/
python run_table1.py
```

Expected key values at ε = ln 7, mechanism-aware attacker:
- M-Random Acc@1 ≈ 0.1%,  M-Scale Acc@1 ≈ 1.3%,  Ratio ≈ 8.44×
- Near-diatonic (χ < 0.05): M-Scale Acc@1 ≈ 4.8%,  Ratio ≈ 7.10×

## Generating figures

```bash
cd code/
python figures/fig1_acc_vs_eps.py
python figures/fig2_ratio_vs_eps.py
python figures/fig3_chromatic_fingerprint.py
```

Figures are saved as PDF in `output/`.

## Ablation study

```bash
cd code/
python run_ablation.py
```

Runs the agnostic attacker on a key-balanced dataset (20 diatonic seed
melodies × 12 transpositions = 240 melodies). Ratios drop to 1.74–2.02×
compared to up to 8.44× on the Europa corpus, confirming that M-Scale's
large gap is driven by key-distribution skew in the corpus rather than
mechanism structure alone.

## Verifying Proposition 1

```bash
cd code/
python verify_expectation.py
```

Checks by Monte Carlo simulation that
E[χ(M-RANDOM(m))] = 535/1716 + (36/143)·χ(m)  at ε = ln 7.

## Key implementation notes

- **Alpha = 1.66**: the variance-matched weight for the combined distance metric
  (paper Section 3.4). Earlier drafts used 2.53 — use 1.66 for paper results.
- **Aware attacker formulas** (paper Section 3.5):
  - M-Scale: `d(m', c) = d_IH(m', c) + α(n-1)·χ(c)`  (exploits χ(output)=0)
  - M-Random: `d(m', c) = d_IH(m', c) + α(n-1)·|χ(m') − (535/1716 + 36/143·χ(c))|`
- **Wilson CIs**: used instead of normal approximation for accuracy values near 0%.
