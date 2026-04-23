"""
plot_adaptive.py
================
Plot heuristic vs adaptive aware attacker ratio curves for all four corpora.

Run after run_adaptive.py:
    python plot_adaptive.py
"""

from __future__ import annotations

import json
from math import exp
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np

BASE_DIR = Path(__file__).parent
RESULTS_DIR = BASE_DIR / "results"
FIGURES_DIR = BASE_DIR / "figures"
FIGURES_DIR.mkdir(exist_ok=True)

CORPORA = [
    ("adaptive_europa_200.json",   "Europa 200",   "#2c7bb6"),
    ("adaptive_meertens_500.json", "Meertens 500", "#d7191c"),
    ("adaptive_symbtr_200.json",   "SymbTr 200",   "#1a9641"),
    ("adaptive_jazz_200.json",     "Jazz 200",     "#f46d43"),
]


def load(fname: str) -> dict | None:
    path = RESULTS_DIR / fname
    if not path.exists():
        print(f"  Missing: {path}")
        return None
    with open(path) as fh:
        return json.load(fh)


def plot_all():
    available = [(f, label, col) for f, label, col in CORPORA
                 if (RESULTS_DIR / f).exists()]
    if not available:
        print("No result files found. Run run_adaptive.py first.")
        return

    ncols = min(2, len(available))
    nrows = (len(available) + 1) // 2
    fig, axes = plt.subplots(nrows, ncols, figsize=(6 * ncols, 4 * nrows),
                             squeeze=False)
    axes_flat = [ax for row in axes for ax in row]

    for ax_idx, (fname, label, color) in enumerate(available):
        data = load(fname)
        ax = axes_flat[ax_idx]

        eps_vals = [r["exp_epsilon"] for r in data["epsilon_sweep"]]
        heuristic = [r["heuristic"]["ratio"] if r["heuristic"]["ratio"] != float("inf")
                     else np.nan for r in data["epsilon_sweep"]]
        adaptive  = [r["adaptive"]["ratio"] if r["adaptive"]["ratio"] != float("inf")
                     else np.nan for r in data["epsilon_sweep"]]

        ax.plot(eps_vals, heuristic, "o--", color=color, alpha=0.7, linewidth=1.5,
                markersize=4, label="Heuristic (fixed ln7)")
        ax.plot(eps_vals, adaptive,  "s-",  color=color, alpha=1.0, linewidth=2.0,
                markersize=5, label="Adaptive (correct ε)")

        ax.axhline(1.0, color="gray", linewidth=0.8, linestyle=":")
        ax.set_xscale("log")
        ax.set_xlabel("$e^\\varepsilon$", fontsize=11)
        ax.set_ylabel("M-Scale / M-Random ratio", fontsize=10)
        ax.set_title(f"{label}  (N={data['N']}, "
                     f"$\\chi_{{\\mathrm{{mean}}}}$={data['mean_chi']:.1%})", fontsize=11)
        ax.legend(fontsize=9)
        ax.xaxis.set_major_formatter(ticker.FuncFormatter(
            lambda x, _: f"ln({x:.0f})" if x >= 2 else f"ln({x:.1f})"
        ))
        ax.grid(True, alpha=0.3)

    # Hide unused axes
    for ax in axes_flat[len(available):]:
        ax.set_visible(False)

    fig.suptitle("Heuristic vs Epsilon-Adaptive Aware Attacker\n"
                 "M-Scale / M-Random Re-identification Ratio",
                 fontsize=13, fontweight="bold")
    plt.tight_layout()
    out = FIGURES_DIR / "adaptive_vs_heuristic_all.pdf"
    plt.savefig(out, bbox_inches="tight")
    plt.savefig(str(out).replace(".pdf", ".png"), dpi=150, bbox_inches="tight")
    print(f"Saved: {out}")
    plt.close()


def plot_formula_comparison():
    """Plot the chi formula intercept and slope as a function of epsilon."""
    from math import log

    eps_range = np.linspace(log(1.5), log(25), 200)
    p_vals = np.array([exp(e) / (exp(e) + 6) for e in eps_range])
    intercepts = (60 - 35 * p_vals) / 132
    slopes = (84 * p_vals - 12) / 132

    # Fixed ln7 values
    p_ln7 = exp(log(7)) / (exp(log(7)) + 6)
    fixed_intercept = (60 - 35 * p_ln7) / 132
    fixed_slope = (84 * p_ln7 - 12) / 132

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))

    exp_eps = np.exp(eps_range)
    ax1.plot(exp_eps, intercepts, "b-", linewidth=2, label="Adaptive $P_{dc}(p)$")
    ax1.axhline(fixed_intercept, color="b", linestyle="--", linewidth=1.2,
                label=f"Fixed ln7: {fixed_intercept:.4f}")
    ax1.set_xscale("log")
    ax1.set_xlabel("$e^\\varepsilon$", fontsize=11)
    ax1.set_ylabel("Intercept", fontsize=11)
    ax1.set_title("$\\mathbb{E}[\\chi(m\\')] = $ intercept $+ $ slope $\\cdot f$\n"
                  "Intercept = $P_{dc}(p)$", fontsize=10)
    ax1.legend(fontsize=9)
    ax1.grid(True, alpha=0.3)

    ax2.plot(exp_eps, slopes, "r-", linewidth=2, label="Adaptive slope $(P_{cc}-P_{dc})$")
    ax2.axhline(fixed_slope, color="r", linestyle="--", linewidth=1.2,
                label=f"Fixed ln7: {fixed_slope:.4f}")
    ax2.set_xscale("log")
    ax2.set_xlabel("$e^\\varepsilon$", fontsize=11)
    ax2.set_ylabel("Slope", fontsize=11)
    ax2.set_title("Slope = $P_{cc}(p) - P_{dc}(p)$", fontsize=10)
    ax2.legend(fontsize=9)
    ax2.grid(True, alpha=0.3)

    plt.suptitle("Epsilon-adaptive vs fixed-ln7 formula coefficients", fontsize=12)
    plt.tight_layout()
    out = FIGURES_DIR / "adaptive_formula_coeffs.pdf"
    plt.savefig(out, bbox_inches="tight")
    plt.savefig(str(out).replace(".pdf", ".png"), dpi=150, bbox_inches="tight")
    print(f"Saved: {out}")
    plt.close()


if __name__ == "__main__":
    plot_formula_comparison()
    plot_all()
