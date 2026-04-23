"""
08_plot_bayesian.py
===================
Plot Bayesian attacker comparison figure for ISMIR_extended.

Generates:
  fig_bayesian_comparison.pdf  -- two-panel figure:
    Left:  M-Scale Acc@1 for all three attackers (Agnostic, Aware, Bayesian)
    Right: M-Random Acc@1 for all three attackers

Style matches paper exactly (steelblue/tomato/seagreen, 5x3.5 per panel).

Run from experiments_extended/ directory:
    python scripts/08_plot_bayesian.py
"""

from __future__ import annotations

import json
from math import log, exp
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

SCRIPT_DIR  = Path(__file__).parent
BASE_DIR    = SCRIPT_DIR.parent
RESULTS_DIR = BASE_DIR / "results"
FIGURES_DIR = BASE_DIR / "figures"
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

# Paper colours
C_AGNOSTIC = "gray"
C_AWARE    = "tomato"
C_BAYES    = "steelblue"

REF_EPS = [log(3), log(7), log(12)]
REF_LABELS = [r"$\ln 3$", r"$\ln 7$", r"$\ln 12$"]


def main():
    bpath = RESULTS_DIR / "bayesian_europa.json"
    if not bpath.exists():
        print(f"ERROR: {bpath} not found. Run 07_run_bayesian.py first.")
        return

    with open(bpath) as f:
        data = json.load(f)

    results = data["results"]

    eps_vals    = np.array([r["epsilon"]     for r in results])
    exp_eps     = np.array([r["exp_epsilon"] for r in results])

    # Bayesian
    bay_s = np.array([r["bayesian"]["acc1_scale"]  for r in results]) * 100
    bay_r = np.array([r["bayesian"]["acc1_random"] for r in results]) * 100
    bay_s_lo = np.array([r["bayesian"]["ci1_scale"][0]  for r in results]) * 100
    bay_s_hi = np.array([r["bayesian"]["ci1_scale"][1]  for r in results]) * 100
    bay_r_lo = np.array([r["bayesian"]["ci1_random"][0] for r in results]) * 100
    bay_r_hi = np.array([r["bayesian"]["ci1_random"][1] for r in results]) * 100

    # Aware and Agnostic (may be partial / None)
    aw_s  = np.array([(r["aware"]["acc1_scale"]  if r["aware"]   else np.nan) for r in results]) * 100
    aw_r  = np.array([(r["aware"]["acc1_random"] if r["aware"]   else np.nan) for r in results]) * 100
    ag_s  = np.array([(r["agnostic"]["acc1_scale"]  if r["agnostic"] else np.nan) for r in results]) * 100
    ag_r  = np.array([(r["agnostic"]["acc1_random"] if r["agnostic"] else np.nan) for r in results]) * 100

    x = eps_vals

    # Build x-tick labels showing every-other value to avoid crowding
    def xlabels(exp_e, stride=2):
        labels = []
        for i, e in enumerate(exp_e):
            if i % stride == 0:
                v = round(e)
                labels.append(rf"$\ln {v}$")
            else:
                labels.append("")
        return labels

    xtick_labels = xlabels(exp_eps, stride=2)

    # ── Figure 1: Acc@1 per mechanism (2 panels) ─────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.5))

    for ax, bay_y, bay_lo, bay_hi, aw_y, ag_y, mech_label in [
        (axes[0], bay_s, bay_s_lo, bay_s_hi, aw_s, ag_s, "M-Scale"),
        (axes[1], bay_r, bay_r_lo, bay_r_hi, aw_r, ag_r, "M-Random"),
    ]:
        for ref in REF_EPS:
            ax.axvline(ref, color="lightgray", linestyle=":", linewidth=0.7, zorder=0)

        mask = ~np.isnan(ag_y)
        if mask.any():
            ax.plot(x[mask], ag_y[mask], "o--", color=C_AGNOSTIC,
                    label="Agnostic", markersize=5, linewidth=1.2)

        mask = ~np.isnan(aw_y)
        if mask.any():
            ax.plot(x[mask], aw_y[mask], "s-", color=C_AWARE,
                    label="Aware", markersize=5, linewidth=1.2)

        mask = ~np.isnan(bay_y)
        ax.errorbar(
            x[mask], bay_y[mask],
            yerr=[
                np.maximum(0, bay_y[mask] - bay_lo[mask]),
                np.maximum(0, bay_hi[mask] - bay_y[mask]),
            ],
            fmt="^-", color=C_BAYES, label="Bayesian (optimal)",
            capsize=4, elinewidth=0.8, markersize=5, linewidth=1.2,
        )
        ax.fill_between(
            x[mask], bay_lo[mask], bay_hi[mask],
            color=C_BAYES, alpha=0.15,
        )

        ax.set_xticks(x)
        ax.set_xticklabels(xtick_labels, fontsize=7, rotation=45)
        ax.set_xlabel(r"Privacy budget $\varepsilon$", fontsize=9)
        ax.set_ylabel("Acc@1 (%)", fontsize=9)
        ax.set_title(mech_label, fontsize=10, fontweight="bold")
        ax.legend(fontsize=8)
        ax.set_ylim(bottom=-0.2)

    fig.suptitle(
        r"Bayesian vs.\ Aware vs.\ Agnostic attacker: Europa 200",
        fontsize=11, y=1.01,
    )
    plt.tight_layout()
    out = FIGURES_DIR / "fig_bayesian_comparison.pdf"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out}")

    # ── Figure 2: M-Scale/M-Random ratio (non-monotone) ───────────────────────
    # Compute ratios (use 1e-9 floor to avoid div-by-zero)
    def safe_ratio(s, r):
        return np.where(r > 1e-9 / 100, s / r, np.nan)

    ag_ratio  = safe_ratio(ag_s,  ag_r)
    aw_ratio  = safe_ratio(aw_s,  aw_r)
    bay_ratio = safe_ratio(bay_s, bay_r)

    fig2, ax2 = plt.subplots(figsize=(5, 3.5))

    for ref in REF_EPS:
        ax2.axvline(ref, color="lightgray", linestyle=":", linewidth=0.7, zorder=0)
    ax2.axhline(1.0, color="black", linestyle=":", linewidth=0.8,
                label=r"No advantage ($1\times$)")

    mask = ~np.isnan(ag_ratio)
    ax2.plot(x[mask], ag_ratio[mask], "o--", color=C_AGNOSTIC,
             label="Agnostic", markersize=5, linewidth=1.2)

    mask = ~np.isnan(aw_ratio)
    ax2.plot(x[mask], aw_ratio[mask], "s-", color=C_AWARE,
             label="Aware", markersize=5, linewidth=1.2)

    mask = ~np.isnan(bay_ratio)
    ax2.plot(x[mask], bay_ratio[mask], "^-", color=C_BAYES,
             label="Bayesian (optimal)", markersize=5, linewidth=1.2)

    ax2.set_xticks(REF_EPS)
    ax2.set_xticklabels(REF_LABELS, fontsize=9)
    ax2.set_xlabel(r"Privacy budget $\varepsilon$", fontsize=9)
    ax2.set_ylabel(r"Acc@1 ratio \textsc{M-Scale}/\textsc{M-Random}", fontsize=8)
    ax2.set_title("Re-identification ratio: Europa 200\n(non-monotone peak regime)",
                  fontsize=10)
    ax2.legend(fontsize=8)
    ax2.set_ylim(bottom=0)
    plt.tight_layout()
    out2 = FIGURES_DIR / "fig_bayesian_ratio.pdf"
    fig2.savefig(out2, bbox_inches="tight")
    plt.close(fig2)
    print(f"Saved: {out2}")

    # Also copy to ISMIR_extended/figures/
    ismir_fig_dir = BASE_DIR.parent.parent / "ISMIR_extended" / "figures"
    if ismir_fig_dir.exists():
        import shutil
        shutil.copy(out, ismir_fig_dir / "fig_bayesian_comparison.pdf")
        print(f"Copied to: {ismir_fig_dir / 'fig_bayesian_comparison.pdf'}")
    else:
        print(f"Note: ISMIR_extended/figures/ not found at {ismir_fig_dir}")


if __name__ == "__main__":
    main()
