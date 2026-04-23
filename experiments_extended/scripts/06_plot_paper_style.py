"""
06_plot_paper_style.py
======================
Reproduce extended-experiment figures in the exact visual style of the
original paper (fig1/fig2 in code/figures/).

Style spec:
  - figsize 5x3.5 (per panel), tight_layout
  - M-Random : steelblue, o--
  - M-Scale  : tomato,    s-
  - Near-diatonic (aware) : seagreen, ^-
  - Agnostic (all)        : gray, o--
  - Error bars : Wilson 95% CI, capsize=4, elinewidth=0.8
  - CI shading : fill_between, alpha=0.15
  - Chance baseline : gray dotted, lw=0.8
  - No-advantage    : black dotted y=1.0, lw=0.8
  - Reference epsilons {ln3,ln7,ln12} : light gray dotted verticals

Figures produced:
  A. fig_ext_acc_vs_eps.pdf  -- 3-panel side-by-side (one per dataset)
                                Acc@1 vs eps, aware attacker
  B. fig_ext_ratio_vs_eps.pdf -- 3-panel: ratio curves for each dataset
                                 (Agnostic-all gray, Aware-all tomato, Aware-ND seagreen)
  C. fig_ext_comparison.pdf  -- single plot: aware-all ratio, all 3 datasets overlaid

Run from experiments_extended/ directory:
    python scripts/06_plot_paper_style.py
"""

from __future__ import annotations

import json
import sys
from math import exp, log
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

SCRIPT_DIR  = Path(__file__).parent
BASE_DIR    = SCRIPT_DIR.parent
RESULTS_DIR = BASE_DIR / "results"
FIGURES_DIR = BASE_DIR / "figures"
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

# ── colours / markers (paper spec) ──────────────────────────────────────────
C_MRAND  = "steelblue"
C_MSCALE = "tomato"
C_ND     = "seagreen"
C_AG     = "gray"

# Reference epsilon values from the original paper
REF_EPS = [log(3), log(7), log(12)]

# ── helpers ──────────────────────────────────────────────────────────────────

def load(name: str) -> dict | None:
    p = RESULTS_DIR / f"{name}_eps_sweep.json"
    if not p.exists():
        print(f"  Not found: {p}")
        return None
    with open(p) as f:
        return json.load(f)


def arrays(sweep: list[dict]) -> dict:
    """Extract numpy arrays from a sweep result list."""
    eps    = np.array([r["epsilon"]        for r in sweep])
    exp_e  = np.array([r["exp_epsilon"]    for r in sweep])

    ag_s   = np.array([r["agnostic"]["acc1_scale"]           for r in sweep])
    ag_r   = np.array([r["agnostic"]["acc1_random"]          for r in sweep])
    aw_s   = np.array([r["aware"]["aware_acc1_scale"]        for r in sweep])
    aw_r   = np.array([r["aware"]["aware_acc1_random"]       for r in sweep])

    ag_s_lo = np.array([r["agnostic"]["ci1_scale"][0]        for r in sweep])
    ag_s_hi = np.array([r["agnostic"]["ci1_scale"][1]        for r in sweep])
    ag_r_lo = np.array([r["agnostic"]["ci1_random"][0]       for r in sweep])
    ag_r_hi = np.array([r["agnostic"]["ci1_random"][1]       for r in sweep])
    aw_s_lo = np.array([r["aware"]["aware_ci1_scale"][0]     for r in sweep])
    aw_s_hi = np.array([r["aware"]["aware_ci1_scale"][1]     for r in sweep])
    aw_r_lo = np.array([r["aware"]["aware_ci1_random"][0]    for r in sweep])
    aw_r_hi = np.array([r["aware"]["aware_ci1_random"][1]    for r in sweep])

    ag_ratio = np.where(ag_r > 1e-9, ag_s / ag_r, np.nan)
    aw_ratio = np.where(aw_r > 1e-9, aw_s / aw_r, np.nan)

    # Near-diatonic
    nd_aw_s, nd_aw_r, nd_aw_ratio = [], [], []
    for r in sweep:
        nd = r.get("near_diatonic")
        if nd:
            s = nd["aware"]["aware_acc1_scale"]
            rv = nd["aware"]["aware_acc1_random"]
            nd_aw_s.append(s)
            nd_aw_r.append(rv)
            nd_aw_ratio.append(s / rv if rv > 1e-9 else np.nan)
        else:
            nd_aw_s.append(np.nan)
            nd_aw_r.append(np.nan)
            nd_aw_ratio.append(np.nan)

    return dict(
        eps=eps, exp_e=exp_e,
        ag_s=ag_s, ag_r=ag_r, aw_s=aw_s, aw_r=aw_r,
        ag_s_lo=ag_s_lo, ag_s_hi=ag_s_hi,
        ag_r_lo=ag_r_lo, ag_r_hi=ag_r_hi,
        aw_s_lo=aw_s_lo, aw_s_hi=aw_s_hi,
        aw_r_lo=aw_r_lo, aw_r_hi=aw_r_hi,
        ag_ratio=ag_ratio, aw_ratio=aw_ratio,
        nd_aw_s=np.array(nd_aw_s),
        nd_aw_r=np.array(nd_aw_r),
        nd_aw_ratio=np.array(nd_aw_ratio),
    )


def x_labels(exp_e: np.ndarray) -> list[str]:
    """Compact LaTeX x-tick labels: ln(3), ln(7), etc."""
    labels = []
    for e in exp_e:
        v = round(e)
        if abs(e - v) < 0.15:
            labels.append(rf"$\ln {v}$")
        else:
            labels.append(rf"$\ln {e:.1f}$")
    return labels


def add_ref_verticals(ax):
    for ref in REF_EPS:
        ax.axvline(ref, color="lightgray", linestyle=":", linewidth=0.7, zorder=0)


def eb(ax, x, y, lo, hi, fmt, color, label, alpha_fill=0.15):
    """Plot errorbar + shaded CI band."""
    mask = ~np.isnan(y)
    ax.errorbar(
        x[mask], y[mask] * 100,
        yerr=[
            np.maximum(0, (y[mask] - lo[mask]) * 100),
            np.maximum(0, (hi[mask] - y[mask]) * 100),
        ],
        fmt=fmt, color=color, label=label,
        capsize=4, elinewidth=0.8, markersize=5,
    )
    ax.fill_between(
        x[mask],
        lo[mask] * 100,
        hi[mask] * 100,
        color=color, alpha=alpha_fill,
    )


# ── Figure A: Acc@1 vs eps, aware attacker, 3 panels ────────────────────────

def fig_A(datasets: dict[str, dict]) -> None:
    """3-panel side-by-side: one per dataset, Acc@1 aware attacker."""
    names   = list(datasets.keys())
    n_panel = len(names)
    fig, axes = plt.subplots(1, n_panel, figsize=(5 * n_panel, 3.5), sharey=False)
    if n_panel == 1:
        axes = [axes]

    for ax, name in zip(axes, names):
        data = datasets[name]
        sw   = data["epsilon_sweep"]
        a    = arrays(sw)
        x    = a["eps"]
        N    = data["N"]
        chance = 1 / N  # random baseline

        add_ref_verticals(ax)
        eb(ax, x, a["aw_s"], a["aw_s_lo"], a["aw_s_hi"],
           "s-",  C_MSCALE, r"\textsc{M-Scale}")
        eb(ax, x, a["aw_r"], a["aw_r_lo"], a["aw_r_hi"],
           "o--", C_MRAND,  r"\textsc{M-Random}")
        ax.axhline(chance * 100, color="gray", linestyle=":", linewidth=0.8,
                   label=f"Random ({chance*100:.1f}\\%)")

        ticks = x[::2]
        ax.set_xticks(ticks)
        ax.set_xticklabels(x_labels(a["exp_e"][::2]), fontsize=7, rotation=45)
        ax.set_xlabel(r"Privacy budget $\varepsilon$", fontsize=9)
        ax.set_ylabel("Acc@1 (\\%)", fontsize=9)
        ax.set_title(name, fontsize=10, fontweight="bold")
        ax.legend(fontsize=7)
        ax.set_ylim(bottom=-0.2)

    fig.suptitle(
        r"Re-identification accuracy vs.\ $\varepsilon$ (mechanism-aware attacker)",
        fontsize=11, y=1.01,
    )
    plt.tight_layout()
    out = FIGURES_DIR / "fig_ext_acc_vs_eps.pdf"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out}")


# ── Figure B: Ratio vs eps, 3 panels ────────────────────────────────────────

def fig_B(datasets: dict[str, dict]) -> None:
    """3-panel: ratio curves per dataset (Agnostic-all, Aware-all, Aware-ND)."""
    names   = list(datasets.keys())
    n_panel = len(names)
    fig, axes = plt.subplots(1, n_panel, figsize=(5 * n_panel, 3.5), sharey=False)
    if n_panel == 1:
        axes = [axes]

    for ax, name in zip(axes, names):
        data = datasets[name]
        sw   = data["epsilon_sweep"]
        a    = arrays(sw)
        x    = a["eps"]
        nd_N = data.get("near_diatonic_N", 0)

        add_ref_verticals(ax)
        ax.axhline(1.0, color="black", linestyle=":", linewidth=0.8,
                   label=r"No advantage ($1\times$)")

        ax.plot(x, a["ag_ratio"], "o--", color=C_AG,
                label="Agnostic (all)", markersize=5)
        ax.plot(x, a["aw_ratio"], "s-",  color=C_MSCALE,
                label="Aware (all)", markersize=5)

        nd_mask = ~np.isnan(a["nd_aw_ratio"])
        if nd_mask.any():
            ax.plot(x[nd_mask], a["nd_aw_ratio"][nd_mask], "^-", color=C_ND,
                    label=rf"Aware (near-diatonic, $n={nd_N}$)", markersize=5)

        ticks = x[::2]
        ax.set_xticks(ticks)
        ax.set_xticklabels(x_labels(a["exp_e"][::2]), fontsize=7, rotation=45)
        ax.set_xlabel(r"Privacy budget $\varepsilon$", fontsize=9)
        ax.set_ylabel(r"Acc@1 ratio \textsc{M-Scale}/\textsc{M-Random}", fontsize=8)
        ax.set_title(name, fontsize=10, fontweight="bold")
        ax.legend(fontsize=7)
        ax.set_ylim(bottom=0)

    fig.suptitle(
        r"Re-identification ratio \textsc{M-Scale}/\textsc{M-Random} vs.\ $\varepsilon$",
        fontsize=11, y=1.01,
    )
    plt.tight_layout()
    out = FIGURES_DIR / "fig_ext_ratio_vs_eps.pdf"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out}")


# ── Figure C: Cross-dataset comparison (aware-all ratio) ────────────────────

def fig_C(datasets: dict[str, dict]) -> None:
    """Single plot: aware-all ratio vs eps for all datasets."""
    # Assign paper-style colors to datasets
    dataset_styles = {
        "Europa 200":    (C_MSCALE,  "s-"),
        "Meertens 500":  (C_MRAND,   "o--"),
        "SymbTr 200":    (C_ND,      "^-"),
    }

    fig, ax = plt.subplots(figsize=(5, 3.5))
    add_ref_verticals(ax)
    ax.axhline(1.0, color="black", linestyle=":", linewidth=0.8,
               label=r"No advantage ($1\times$)")

    for name, data in datasets.items():
        sw  = data["epsilon_sweep"]
        a   = arrays(sw)
        x   = a["eps"]
        col, fmt = dataset_styles.get(name, ("black", "x-"))
        ax.plot(x, a["aw_ratio"], fmt, color=col,
                label=name, markersize=5)

    # Use ticks from the first dataset (all share same eps grid)
    first_sw = next(iter(datasets.values()))["epsilon_sweep"]
    first_a  = arrays(first_sw)
    ticks    = first_a["eps"][::2]
    ax.set_xticks(ticks)
    ax.set_xticklabels(x_labels(first_a["exp_e"][::2]), fontsize=7, rotation=45)

    ax.set_xlabel(r"Privacy budget $\varepsilon$", fontsize=9)
    ax.set_ylabel(r"Acc@1 ratio \textsc{M-Scale}/\textsc{M-Random}", fontsize=9)
    ax.set_title(
        r"Re-identification ratio: cross-dataset comparison" + "\n"
        r"(mechanism-aware attacker, all melodies)",
        fontsize=10,
    )
    ax.legend(fontsize=8)
    ax.set_ylim(bottom=0)
    plt.tight_layout()
    out = FIGURES_DIR / "fig_ext_comparison.pdf"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out}")


# ── main ─────────────────────────────────────────────────────────────────────

def main():
    raw = {
        "Europa 200":   load("europa"),
        "Meertens 500": load("meertens"),
        "SymbTr 200":   load("symbtr"),
    }
    datasets = {k: v for k, v in raw.items() if v is not None}

    if not datasets:
        print("No results found. Run scripts 02-04 first.")
        sys.exit(1)

    print(f"Loaded {len(datasets)} datasets: {list(datasets.keys())}")
    print("\nGenerating paper-style figures...")

    fig_A(datasets)
    fig_B(datasets)
    fig_C(datasets)

    print(f"\nAll figures saved to {FIGURES_DIR}/")
    print("  fig_ext_acc_vs_eps.pdf")
    print("  fig_ext_ratio_vs_eps.pdf")
    print("  fig_ext_comparison.pdf")


if __name__ == "__main__":
    main()
