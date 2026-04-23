"""
plot_paper_style.py
===================
Paper-style figures comparing heuristic vs epsilon-adaptive aware attacker.

Style matches existing paper figures (06_plot_paper_style.py):
  - figsize 5x3.5 per panel, tight_layout
  - M-Scale/heuristic : tomato, s-
  - M-Scale/adaptive  : tomato, s- solid (darker/thicker)
  - Heuristic curve   : tomato dashed
  - Adaptive curve    : tomato solid
  - Bayesian curve    : purple, D- (where available)
  - No-advantage line : black dotted y=1.0
  - Reference epsilons {ln3,ln7,ln12} : lightgray dotted verticals
  - x-axis labels: ln(1.5), ln(2), ln(3)... every other tick

Produces:
  fig_adaptive_europa.pdf       -- Europa: heuristic vs adaptive vs Bayesian ratio
  fig_adaptive_all_corpora.pdf  -- 4-panel: all corpora, heuristic vs adaptive
"""

from __future__ import annotations

import json
import sys
from math import exp, log
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

BASE_DIR    = Path(__file__).parent
EXT_DIR     = BASE_DIR.parent / "experiments_extended"
RESULTS_DIR = BASE_DIR / "results"
# Output goes directly into the ISMIR paper figures folder
PAPER_FIG_DIR = (BASE_DIR.parent.parent / "ISMIR-extended-red" / "figures")
LOCAL_FIG_DIR = BASE_DIR / "figures"
LOCAL_FIG_DIR.mkdir(exist_ok=True)

# ── Style constants (matching paper) ──────────────────────────────────────────
C_HEURISTIC = "tomato"
C_ADAPTIVE  = "#8B0000"    # dark red — same family as tomato, clearly distinct
C_BAYESIAN  = "#6a0dad"    # purple
C_AG        = "gray"

REF_EPS = [log(3), log(7), log(12)]   # vertical reference lines

def x_labels(exp_eps_vals):
    out = []
    for v in exp_eps_vals:
        if abs(v - 1.5) < 0.1:   out.append("ln(1.5)")
        elif abs(v - round(v)) < 0.05: out.append(f"ln({int(round(v))})")
        else:                     out.append(f"ln({v:.1f})")
    return out

def load(fname):
    p = RESULTS_DIR / fname
    if not p.exists():
        return None
    with open(p) as f:
        return json.load(f)

def safe_ratio(r, key="heuristic"):
    v = r[key]["ratio"]
    return np.nan if v == float("inf") or v != v else v

def save(fig, stem):
    for d in [LOCAL_FIG_DIR, PAPER_FIG_DIR]:
        if d.exists():
            fig.savefig(d / f"{stem}.pdf", bbox_inches="tight")
            fig.savefig(d / f"{stem}.png", dpi=150, bbox_inches="tight")
    print(f"Saved: {stem}.pdf/.png")


# ── Figure 1: Europa — heuristic vs adaptive vs Bayesian ─────────────────────

def plot_europa_three_way():
    data = load("adaptive_europa_200.json")
    if data is None:
        print("Missing adaptive_europa_200.json"); return

    # Load Bayesian Europa results
    bay_path = EXT_DIR / "results" / "bayesian_europa.json"
    bay_data = None
    if bay_path.exists():
        with open(bay_path) as f:
            bay_data = json.load(f)

    sweep = data["epsilon_sweep"]
    exp_e = np.array([r["exp_epsilon"] for r in sweep])
    h_rat = np.array([safe_ratio(r, "heuristic") for r in sweep])
    a_rat = np.array([safe_ratio(r, "adaptive")  for r in sweep])

    fig, ax = plt.subplots(figsize=(5, 3.5))

    # Reference verticals
    for ref in REF_EPS:
        ax.axvline(exp(ref), color="lightgray", linestyle=":", linewidth=0.7, zorder=0)

    # No-advantage line
    ax.axhline(1.0, color="black", linestyle=":", linewidth=0.8, zorder=1)

    # Heuristic (fixed ln7)
    ax.plot(exp_e, h_rat, "s--", color=C_HEURISTIC, linewidth=1.5, markersize=4,
            label=r"Aware (fixed $\varepsilon{=}\ln 7$ heuristic)", zorder=3)

    # Adaptive (correct eps)
    ax.plot(exp_e, a_rat, "s-", color=C_ADAPTIVE, linewidth=2.0, markersize=5,
            label=r"Aware ($\varepsilon$-adaptive formula)", zorder=4)

    # Bayesian
    if bay_data is not None:
        bay_sweep = bay_data.get("epsilon_sweep", bay_data.get("results", []))
        b_exp_e = np.array([r.get("exp_epsilon", exp(r["epsilon"])) for r in bay_sweep])
        b_rat   = np.array([r.get("bayesian_ratio", r.get("ratio", np.nan))
                            for r in bay_sweep], dtype=float)
        b_rat[b_rat == float("inf")] = np.nan
        ax.plot(b_exp_e, b_rat, "D-", color=C_BAYESIAN, linewidth=1.5, markersize=4,
                label="Bayesian (exact likelihood)", zorder=5)

    ax.set_xticks(exp_e[::2])
    ax.set_xticklabels(x_labels(exp_e[::2]), fontsize=7, rotation=45)
    ax.set_xlabel(r"Privacy budget $\varepsilon$", fontsize=9)
    ax.set_ylabel(r"Acc@1 ratio \textsc{M-Scale}/\textsc{M-Random}", fontsize=8)
    ax.set_title("Europa 200 — attacker hierarchy", fontsize=10, fontweight="bold")
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.2)

    plt.tight_layout()
    save(fig, "fig_adaptive_europa")
    plt.close()


# ── Figure 2: All four corpora — heuristic vs adaptive ───────────────────────

CORPUS_FILES = [
    ("adaptive_europa_200.json",   "Europa 200",   C_HEURISTIC),
    ("adaptive_meertens_500.json", "Meertens 500", "#e08040"),
    ("adaptive_symbtr_200.json",   "SymbTr 200",   "seagreen"),
    ("adaptive_jazz_200.json",     "Jazz 200",     "steelblue"),
]

def plot_all_corpora():
    available = [(f, lab, col) for f, lab, col in CORPUS_FILES
                 if (RESULTS_DIR / f).exists()]
    if not available:
        print("No result files found."); return

    n = len(available)
    ncols = 2
    nrows = (n + 1) // 2
    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 3.5 * nrows),
                             squeeze=False)
    axes_flat = [ax for row in axes for ax in row]

    for idx, (fname, label, col) in enumerate(available):
        data = load(fname)
        ax   = axes_flat[idx]
        sweep = data["epsilon_sweep"]
        exp_e = np.array([r["exp_epsilon"] for r in sweep])
        h_rat = np.array([safe_ratio(r, "heuristic") for r in sweep])
        a_rat = np.array([safe_ratio(r, "adaptive")  for r in sweep])

        # Darker version of col for adaptive
        for ref in REF_EPS:
            ax.axvline(exp(ref), color="lightgray", linestyle=":", linewidth=0.7, zorder=0)
        ax.axhline(1.0, color="black", linestyle=":", linewidth=0.8, zorder=1)

        ax.plot(exp_e, h_rat, "s--", color=col, linewidth=1.5, markersize=4,
                alpha=0.65, label=r"Heuristic (fixed $\varepsilon{=}\ln 7$)", zorder=3)
        ax.plot(exp_e, a_rat, "s-",  color=col, linewidth=2.2, markersize=5,
                alpha=1.00, label=r"Adaptive (correct $\varepsilon$)", zorder=4)

        ax.set_xticks(exp_e[::2])
        ax.set_xticklabels(x_labels(exp_e[::2]), fontsize=7, rotation=45)
        ax.set_xlabel(r"Privacy budget $\varepsilon$", fontsize=9)
        ax.set_ylabel(r"Acc@1 ratio \textsc{M-Scale}/\textsc{M-Random}", fontsize=8)
        chi = data["mean_chi"]
        chi_pct = f"{chi*100:.1f}\\%"
        ax.set_title(f"{label}  ($\\chi_{{\\mathrm{{mean}}}}={chi_pct}$)",
                     fontsize=10, fontweight="bold")
        ax.legend(fontsize=7)
        ax.grid(True, alpha=0.2)

    for ax in axes_flat[n:]:
        ax.set_visible(False)

    fig.suptitle(
        r"Heuristic vs $\varepsilon$-adaptive aware attacker"
        "\nM-Scale / M-Random re-identification ratio",
        fontsize=11, fontweight="bold", y=1.01
    )
    plt.tight_layout()
    save(fig, "fig_adaptive_all_corpora")
    plt.close()


if __name__ == "__main__":
    plot_europa_three_way()
    plot_all_corpora()
    print("Done.")
