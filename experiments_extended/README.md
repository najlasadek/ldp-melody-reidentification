# Four-corpus experiments

Experiments for the paper **"Equal ε, Unequal Exposure: Re-identification Disparities
in LDP for Music"** on four corpora: Europa 200, Meertens 500, SymbTr 200 and
Jazz 200. See the [main README](../README.md) for setup, datasets and the full list
of commands.

All scripts are run from this directory: `python scripts/XX_....py`.

---

## Scripts

| Script | Purpose |
|--------|---------|
| `00_download_data.sh` | Download Meertens (MTC-FS-INST-2.0) and, if missing, SymbTr |
| `01_prepare_datasets.py` | Build `data/processed/meertens_500.json` and `symbtr_200.json` |
| `02_run_eps_sweep.py` | Europa 200 sweep; first checks ln3/ln7/ln12 against the paper |
| `03_run_meertens.py` | Meertens 500 sweep |
| `04_run_symbtr.py` | SymbTr 200 sweep |
| `09_prepare_jazz.py` | Build `data/processed/jazz_200.json` from the WJazzD score CSVs |
| `10_run_jazz.py` | Jazz 200 sweep |
| `07_run_bayesian.py` | Likelihood-based attacker, Europa only |
| `12_run_bayesian_all.py` | Likelihood-based attacker, all four corpora (paper Appendix A.9) |
| `16_run_length_normalised.py` | Degeneracy diagnostics, normalised vs. un-normalised (Appendix A.7) |
| `17_run_normalised_sweeps.py` | Main results: length-normalised sweeps, all corpora and attackers; `--validate` checks the un-normalised path against `results/` |
| `18_run_normalised_ablation.py` | Key-balanced ablation, length-normalised (Appendix A.8) |
| `19_alpha_sensitivity.py` | Sensitivity of the normalised results to alpha (Appendix A.5) |
| `20_degeneracy_unnormalised.py` | Retrieval degeneracy with the un-normalised distance at ln 8 (Appendix A.7) |
| `21_europa_chromatic_subset.py` | Ratio on the chromatic Europa melodies, normalised (Section 4.2) |
| `07_plot_four_corpora.py` | Paper Figure 2 (ratio vs ε); use `--normalised` |
| `08_plot_acc_curves.py` | Paper Figure 3 (Acc@1 vs ε); use `--normalised` |
| `08_plot_bayesian.py` | Likelihood-attacker comparison on Europa (submission version; not in the camera-ready paper) |

The remaining plotting scripts produce exploratory figures that are not in the paper.

Each sweep script writes `results/<corpus>_eps_sweep.json` with agnostic and aware
heuristic results for all melodies and for the near-diatonic subset (χ < 0.05).

---

## Epsilon grid

15 values from ln(1.5) to ln(25). The paper reports ln 3 to ln 16.
Retention probability: p = e^ε / (e^ε + 6).

| exp(ε) | 1.5 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 | 12 | 14 | 16 | 20 | 25 |
|--------|-----|---|---|---|---|---|---|---|---|----|----|----|----|----|----|
| p | 0.200 | 0.250 | 0.333 | 0.400 | 0.455 | 0.500 | 0.538 | 0.571 | 0.600 | 0.625 | 0.667 | 0.700 | 0.727 | 0.769 | 0.806 |

---

## Notes

- SymbTr: only `note_on` MIDI pitch integers (0–127) are used; pitch-bend messages are
  ignored, so pitches are semitone-quantised.
- Jazz and SymbTr melodies are truncated to their first 30 notes; Europa and Meertens
  are filtered to 10–30 notes.
- Results are saved as JSON with all accuracies and 95% Wilson CIs.
