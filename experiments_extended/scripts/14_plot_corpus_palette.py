"""
14_plot_corpus_palette.py
=========================
Regenerate all extended figures with a per-corpus colour palette:

  Europa   — red/orange family  : M-Scale #d62728 (dark red)
                                   M-Random #ff7f0e (orange)
                                   Near-diatonic #e377c2 (pink-red)

  Meertens — blue family        : M-Scale #1f77b4 (dark blue)
                                   M-Random #aec7e8 (light blue)
                                   Near-diatonic #17becf (teal)

  SymbTr   — green family       : M-Scale #2ca02c (dark green)
                                   M-Random #98df8a (light green)
                                   Near-diatonic #bcbd22 (yellow-green)

  Jazz     — purple family      : M-Scale #9467bd (purple)
                                   M-Random #c5b0d5 (light purple)
                                   Near-diatonic #7f7f7f (gray — very few ND)

Cross-dataset comparison uses one representative colour per corpus
(the M-Scale shade of each).

Figures produced:
  fig_europa_ratio_only.pdf        (replaces old)
  fig_europa_all_eps.pdf           (replaces old)
  fig_ext_ratio_vs_eps.pdf         (replaces old — 3-panel Europa/Meertens/SymbTr)
  fig_ext_acc_vs_eps.pdf           (replaces old)
  fig_jazz_ratio_vs_eps.pdf        (replaces old)
  fig_jazz_acc_vs_eps.pdf          (replaces old)
  fig_ext_comparison_jazz.pdf      (replaces old — all 4 datasets)

Run from experiments_extended/ directory:
    python scripts/14_plot_corpus_palette.py
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
matplotlib.rcParams.update({"text.usetex": False})
import matplotlib.pyplot as plt
import numpy as np

SCRIPT_DIR  = Path(__file__).parent
BASE_DIR    = SCRIPT_DIR.parent
RESULTS_DIR = BASE_DIR / "results"
FIGURES_DIR = BASE_DIR / "figures"
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

# ── Per-corpus colour palettes ────────────────────────────────────────────────
PALETTES = {
    "europa": {
        "mscale":  "#d62728",   # dark red   — M-Scale line
        "mrandom": "#ff7f0e",   # orange     — M-Random line
        "nd":      "#c7003a",   # deep red   — near-diatonic
        "agnostic":"#fdae6b",   # pale orange — agnostic
        "repr":    "#d62728",   # representative for cross-dataset
        "label":   "Europa 200",
    },
    "meertens": {
        "mscale":  "#1f77b4",   # dark blue
        "mrandom": "#6baed6",   # mid blue
        "nd":      "#08519c",   # deep blue
        "agnostic":"#bdd7e7",   # pale blue
        "repr":    "#1f77b4",
        "label":   "Meertens 500",
    },
    "symbtr": {
        "mscale":  "#2ca02c",   # dark green
        "mrandom": "#74c476",   # mid green
        "nd":      "#006d2c",   # deep green
        "agnostic":"#c7e9c0",   # pale green
        "repr":    "#2ca02c",
        "label":   "SymbTr 200",
    },
    "jazz": {
        "mscale":  "#6a3d9a",   # dark purple — M-Scale
        "mrandom": "#cab2d6",   # light purple — M-Random
        "nd":      "#9467bd",   # mid purple  — near-diatonic
        "agnostic":"#e0d4f0",   # pale purple  — agnostic
        "repr":    "#6a3d9a",
        "label":   "Jazz 200",
    },
}

FIGSIZE = (5, 3.5)
REF_EPS_VALS = [math.log(3), math.log(7), math.log(12)]


# ── Data helpers ─────────────────────────────────────────────────────────────

def load(name: str) -> dict:
    p = RESULTS_DIR / f"{name}_eps_sweep.json"
    with open(p) as f:
        return json.load(f)


def get_eps_labels(sweep: list[dict]) -> tuple[np.ndarray, list[str]]:
    exp_e = np.array([r["exp_epsilon"] for r in sweep])
    labels = []
    for e in exp_e:
        v = round(e)
        labels.append(f"ln({v})" if abs(e - v) < 0.15 else f"ln({e:.1f})")
    return exp_e, labels


def add_ref_lines(ax, x_eps: np.ndarray) -> None:
    for ref in REF_EPS_VALS:
        # find closest index
        idx = int(np.argmin(np.abs(x_eps - ref)))
        ax.axvline(idx, color="lightgray", linestyle=":", linewidth=0.7, zorder=0)


def wilson_ci(k: int, n: int, z: float = 1.96):
    if n == 0:
        return 0.0, 0.0
    p = k / n
    denom = 1 + z**2 / n
    center = (p + z**2 / (2 * n)) / denom
    margin = z * math.sqrt(p * (1-p)/n + z**2/(4*n**2)) / denom
    return max(0.0, center - margin), min(1.0, center + margin)


def errorbar_with_ci(ax, x, y_pct, lo_pct, hi_pct, color, fmt, label):
    yerr = np.vstack([
        np.maximum(0, y_pct - lo_pct),
        np.maximum(0, hi_pct - y_pct),
    ])
    ax.errorbar(x, y_pct, yerr=yerr, fmt=fmt, color=color,
                capsize=4, elinewidth=0.8, markersize=5, label=label)
    ax.fill_between(x, lo_pct, hi_pct, color=color, alpha=0.15)


# ── Figure: Ratio-only plot for one corpus (Europa style) ────────────────────

def plot_ratio_only(name: str, data: dict, out_path: Path) -> None:
    pal   = PALETTES[name]
    sweep = data["epsilon_sweep"]
    exp_e, xlabels = get_eps_labels(sweep)
    x     = np.arange(len(sweep))
    nd_N  = data.get("near_diatonic_N", 0)
    N     = data["N"]

    ag_ratio  = np.array([r.get("agnostic_ratio", float("nan")) for r in sweep])
    aw_ratio  = np.array([r.get("aware_ratio",    float("nan")) for r in sweep])
    nd_ratios = []
    for r in sweep:
        nd = r.get("near_diatonic")
        nd_ratios.append(nd["aware_ratio"] if nd and math.isfinite(nd.get("aware_ratio", float("nan"))) else float("nan"))
    nd_ratio = np.array(nd_ratios)

    fig, ax = plt.subplots(figsize=FIGSIZE)
    add_ref_lines(ax, exp_e)
    ax.axhline(1.0, color="black", linestyle=":", linewidth=0.8,
               label="No advantage (1×)")

    ax.plot(x, ag_ratio, "o--", color=pal["agnostic"],
            markersize=5, label=f"Agnostic (all, N={N})")
    ax.plot(x, aw_ratio, "s-",  color=pal["mscale"],
            markersize=5, label=f"Aware (all, N={N})")

    nd_mask = np.isfinite(nd_ratio)
    if nd_mask.any():
        ax.plot(x[nd_mask], nd_ratio[nd_mask], "^-", color=pal["nd"],
                markersize=5, label=f"Aware (near-diatonic, n={nd_N})")

    sparse = [l if i % 2 == 0 else "" for i, l in enumerate(xlabels)]
    ax.set_xticks(x); ax.set_xticklabels(sparse, fontsize=7, rotation=45, ha="right")
    ax.set_xlabel("Privacy budget  ε", fontsize=9)
    ax.set_ylabel("Acc@1 ratio  M-Scale / M-Random", fontsize=9)
    ax.set_title(f"{pal['label']} — Re-identification Ratio", fontsize=10)
    ax.set_ylim(bottom=0)
    ax.legend(fontsize=7)
    plt.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out_path}")


# ── Figure: Acc@1 curves for one corpus ──────────────────────────────────────

def plot_acc(name: str, data: dict, out_path: Path) -> None:
    pal   = PALETTES[name]
    sweep = data["epsilon_sweep"]
    exp_e, xlabels = get_eps_labels(sweep)
    x     = np.arange(len(sweep))
    N     = data["N"]
    n_obs = N * data.get("num_trials", 30)
    chance = 100.0 / N

    aw_s  = np.array([r["aware"]["aware_acc1_scale"]  for r in sweep])
    aw_r  = np.array([r["aware"]["aware_acc1_random"] for r in sweep])
    aw_s_lo = np.array([r["aware"]["aware_ci1_scale"][0]  for r in sweep])
    aw_s_hi = np.array([r["aware"]["aware_ci1_scale"][1]  for r in sweep])
    aw_r_lo = np.array([r["aware"]["aware_ci1_random"][0] for r in sweep])
    aw_r_hi = np.array([r["aware"]["aware_ci1_random"][1] for r in sweep])

    fig, ax = plt.subplots(figsize=FIGSIZE)
    add_ref_lines(ax, exp_e)

    errorbar_with_ci(ax, x, aw_s*100, aw_s_lo*100, aw_s_hi*100,
                     pal["mscale"],  "s-",  "M-Scale")
    errorbar_with_ci(ax, x, aw_r*100, aw_r_lo*100, aw_r_hi*100,
                     pal["mrandom"], "o--", "M-Random")
    ax.axhline(chance, color="gray", linestyle=":", linewidth=0.8,
               label=f"Chance ({chance:.1f}%)")

    sparse = [l if i % 2 == 0 else "" for i, l in enumerate(xlabels)]
    ax.set_xticks(x); ax.set_xticklabels(sparse, fontsize=7, rotation=45, ha="right")
    ax.set_xlabel("Privacy budget  ε", fontsize=9)
    ax.set_ylabel("Acc@1 (%)", fontsize=9)
    ax.set_title(f"{pal['label']} — Mechanism-Aware Attacker", fontsize=10)
    ax.legend(fontsize=7)
    plt.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out_path}")


# ── Figure: 3-panel ratio (Europa / Meertens / SymbTr) ───────────────────────

def plot_3panel_ratio(all_data: dict[str, dict], out_path: Path) -> None:
    names = list(all_data.keys())
    fig, axes = plt.subplots(1, len(names), figsize=(5 * len(names), 3.5), sharey=False)
    if len(names) == 1:
        axes = [axes]

    for ax, name in zip(axes, names):
        pal   = PALETTES[name]
        data  = all_data[name]
        sweep = data["epsilon_sweep"]
        exp_e, xlabels = get_eps_labels(sweep)
        x     = np.arange(len(sweep))
        nd_N  = data.get("near_diatonic_N", 0)

        ag_ratio = np.array([r.get("agnostic_ratio", float("nan")) for r in sweep])
        aw_ratio = np.array([r.get("aware_ratio",    float("nan")) for r in sweep])
        nd_ratio = np.array([
            r["near_diatonic"]["aware_ratio"]
            if r.get("near_diatonic") and math.isfinite(r["near_diatonic"].get("aware_ratio", float("nan")))
            else float("nan")
            for r in sweep
        ])

        add_ref_lines(ax, exp_e)
        ax.axhline(1.0, color="black", linestyle=":", linewidth=0.8,
                   label="No advantage (1×)")
        ax.plot(x, ag_ratio, "o--", color=pal["agnostic"], markersize=4,
                label="Agnostic (all)")
        ax.plot(x, aw_ratio, "s-",  color=pal["mscale"],   markersize=4,
                label="Aware (all)")
        nd_mask = np.isfinite(nd_ratio)
        if nd_mask.any():
            ax.plot(x[nd_mask], nd_ratio[nd_mask], "^-", color=pal["nd"],
                    markersize=4, label=f"Aware (near-diatonic, n={nd_N})")

        sparse = [l if i % 2 == 0 else "" for i, l in enumerate(xlabels)]
        ax.set_xticks(x); ax.set_xticklabels(sparse, fontsize=7, rotation=45, ha="right")
        ax.set_xlabel("Privacy budget  ε", fontsize=9)
        ax.set_ylabel("Acc@1 ratio  M-Scale / M-Random", fontsize=8)
        ax.set_title(pal["label"], fontsize=10, fontweight="bold")
        ax.legend(fontsize=7)
        ax.set_ylim(bottom=0)

    fig.suptitle(
        "Re-identification ratio  M-Scale / M-Random  vs.  ε",
        fontsize=11, y=1.01,
    )
    plt.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out_path}")


# ── Figure: 3-panel Acc@1 (Europa / Meertens / SymbTr) ───────────────────────

def plot_3panel_acc(all_data: dict[str, dict], out_path: Path) -> None:
    names = list(all_data.keys())
    fig, axes = plt.subplots(1, len(names), figsize=(5 * len(names), 3.5), sharey=False)
    if len(names) == 1:
        axes = [axes]

    for ax, name in zip(axes, names):
        pal   = PALETTES[name]
        data  = all_data[name]
        sweep = data["epsilon_sweep"]
        exp_e, xlabels = get_eps_labels(sweep)
        x     = np.arange(len(sweep))
        N     = data["N"]
        chance = 100.0 / N

        aw_s    = np.array([r["aware"]["aware_acc1_scale"]     for r in sweep])
        aw_r    = np.array([r["aware"]["aware_acc1_random"]    for r in sweep])
        aw_s_lo = np.array([r["aware"]["aware_ci1_scale"][0]   for r in sweep])
        aw_s_hi = np.array([r["aware"]["aware_ci1_scale"][1]   for r in sweep])
        aw_r_lo = np.array([r["aware"]["aware_ci1_random"][0]  for r in sweep])
        aw_r_hi = np.array([r["aware"]["aware_ci1_random"][1]  for r in sweep])

        add_ref_lines(ax, exp_e)
        errorbar_with_ci(ax, x, aw_s*100, aw_s_lo*100, aw_s_hi*100,
                         pal["mscale"],  "s-",  "M-Scale")
        errorbar_with_ci(ax, x, aw_r*100, aw_r_lo*100, aw_r_hi*100,
                         pal["mrandom"], "o--", "M-Random")
        ax.axhline(chance, color="gray", linestyle=":", linewidth=0.8,
                   label=f"Chance ({chance:.1f}%)")

        sparse = [l if i % 2 == 0 else "" for i, l in enumerate(xlabels)]
        ax.set_xticks(x); ax.set_xticklabels(sparse, fontsize=7, rotation=45, ha="right")
        ax.set_xlabel("Privacy budget  ε", fontsize=9)
        ax.set_ylabel("Acc@1 (%)", fontsize=9)
        ax.set_title(pal["label"], fontsize=10, fontweight="bold")
        ax.legend(fontsize=7)
        ax.set_ylim(bottom=-0.2)

    fig.suptitle(
        "Re-identification accuracy vs. ε  (mechanism-aware attacker)",
        fontsize=11, y=1.01,
    )
    plt.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out_path}")


# ── Figure: Cross-dataset comparison (all 4 corpora, representative colours) ─

def plot_cross_dataset(all_data: dict[str, dict], out_path: Path) -> None:
    """One line per corpus, each in its representative colour."""
    fmts = {
        "europa":   "s-",
        "meertens": "o--",
        "symbtr":   "^-",
        "jazz":     "D:",
    }

    fig, ax = plt.subplots(figsize=FIGSIZE)

    # Use Europa x-axis as reference
    ref_sweep = all_data["europa"]["epsilon_sweep"]
    exp_e_ref, xlabels_ref = get_eps_labels(ref_sweep)
    x_ref = np.arange(len(ref_sweep))

    add_ref_lines(ax, exp_e_ref)
    ax.axhline(1.0, color="black", linestyle=":", linewidth=0.8,
               label="No advantage (1×)")

    for name, data in all_data.items():
        pal   = PALETTES[name]
        sweep = data["epsilon_sweep"]
        exp_e, _ = get_eps_labels(sweep)
        x     = np.arange(len(sweep))
        ratios = np.array([
            r.get("aware_ratio", float("nan")) for r in sweep
        ])
        ax.plot(x, ratios, fmts[name], color=pal["repr"],
                markersize=5, linewidth=1.5, label=pal["label"])

    sparse = [l if i % 2 == 0 else "" for i, l in enumerate(xlabels_ref)]
    ax.set_xticks(x_ref)
    ax.set_xticklabels(sparse, fontsize=7, rotation=45, ha="right")
    ax.set_xlabel("Privacy budget  ε", fontsize=9)
    ax.set_ylabel("Acc@1 ratio  M-Scale / M-Random", fontsize=9)
    ax.set_title(
        "Cross-dataset: Aware Attacker Ratio\n"
        "(Europa=red, Meertens=blue, SymbTr=green, Jazz=purple)",
        fontsize=9,
    )
    ax.legend(fontsize=8)
    ax.set_ylim(bottom=0)
    plt.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out_path}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    print("Loading results...")
    europa   = load("europa")
    meertens = load("meertens")
    symbtr   = load("symbtr")
    jazz     = load("jazz")

    diatonic3 = {"europa": europa, "meertens": meertens, "symbtr": symbtr}
    all4      = {"europa": europa, "meertens": meertens, "symbtr": symbtr, "jazz": jazz}

    print("\nPer-corpus ratio plots...")
    plot_ratio_only("europa",   europa,   FIGURES_DIR / "fig_europa_ratio_only.pdf")
    plot_ratio_only("meertens", meertens, FIGURES_DIR / "fig_meertens_ratio_only.pdf")
    plot_ratio_only("symbtr",   symbtr,   FIGURES_DIR / "fig_symbtr_ratio_only.pdf")
    plot_ratio_only("jazz",     jazz,     FIGURES_DIR / "fig_jazz_ratio_vs_eps.pdf")

    print("\nPer-corpus Acc@1 plots...")
    plot_acc("europa",   europa,   FIGURES_DIR / "fig_europa_all_eps.pdf")
    plot_acc("meertens", meertens, FIGURES_DIR / "fig_meertens_all_eps.pdf")
    plot_acc("symbtr",   symbtr,   FIGURES_DIR / "fig_symbtr_all_eps.pdf")
    plot_acc("jazz",     jazz,     FIGURES_DIR / "fig_jazz_acc_vs_eps.pdf")

    print("\n3-panel figures...")
    plot_3panel_ratio(diatonic3, FIGURES_DIR / "fig_ext_ratio_vs_eps.pdf")
    plot_3panel_acc(diatonic3,   FIGURES_DIR / "fig_ext_acc_vs_eps.pdf")

    print("\nCross-dataset comparison (all 4)...")
    plot_cross_dataset(all4, FIGURES_DIR / "fig_ext_comparison_jazz.pdf")

    print(f"\nDone. All figures saved to {FIGURES_DIR}/")


if __name__ == "__main__":
    main()
