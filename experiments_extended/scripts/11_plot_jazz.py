"""
11_plot_jazz.py
===============
Plot results for Jazz 200 in paper style, and add Jazz to the
cross-dataset comparison figure.

Figures generated:
  fig_jazz_acc_vs_eps.pdf      -- Acc@1 vs eps (M-Scale tomato, M-Random steelblue)
  fig_jazz_ratio_vs_eps.pdf    -- Ratio curves (Agnostic gray, Aware-all tomato, Aware-ND seagreen)
  fig_ext_comparison_jazz.pdf  -- Updated cross-dataset comparison:
                                   Europa (tomato), Meertens (steelblue), SymbTr (seagreen),
                                   Jazz (darkorange)

Run from experiments_extended/ directory:
    python scripts/11_plot_jazz.py
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

SCRIPT_DIR = Path(__file__).parent
BASE_DIR   = SCRIPT_DIR.parent
sys.path.insert(0, str(BASE_DIR))

RESULTS_DIR = BASE_DIR / "results"
FIGURES_DIR = BASE_DIR / "figures"
FIGURES_DIR.mkdir(exist_ok=True)

# ── Style ──────────────────────────────────────────────────────────────────────
C_MRAND  = "steelblue"
C_MSCALE = "tomato"
C_ND     = "seagreen"
C_AG     = "gray"

C_EUROPA   = "tomato"
C_MEERTENS = "steelblue"
C_SYMBTR   = "seagreen"
C_JAZZ     = "darkorange"

FIGSIZE = (5, 3.5)

# ── Wilson CI ─────────────────────────────────────────────────────────────────

def wilson_ci(k: int, n: int, z: float = 1.96):
    if n == 0:
        return 0.0, 0.0
    p = k / n
    denom = 1 + z**2 / n
    center = (p + z**2 / (2 * n)) / denom
    margin = z * math.sqrt(p * (1 - p) / n + z**2 / (4 * n**2)) / denom
    return max(0.0, center - margin), min(1.0, center + margin)


def load_sweep(name: str) -> dict:
    path = RESULTS_DIR / f"{name}_eps_sweep.json"
    with open(path) as f:
        return json.load(f)


def extract_series(sweep: dict, key_scale: str, key_random: str, n_trials_key: str = "num_trials"):
    eps_vals, s_vals, r_vals, s_lo, s_hi, r_lo, r_hi = [], [], [], [], [], [], []
    N = sweep.get("num_trials", 30)
    total = sweep.get("N", 200)
    n_obs = N * total

    for entry in sweep["epsilon_sweep"]:
        eps_vals.append(entry["exp_epsilon"])
        sv = entry[key_scale]
        rv = entry[key_random]
        s_vals.append(sv * 100)
        r_vals.append(rv * 100)
        k_s = round(sv * n_obs)
        k_r = round(rv * n_obs)
        lo_s, hi_s = wilson_ci(k_s, n_obs)
        lo_r, hi_r = wilson_ci(k_r, n_obs)
        s_lo.append(lo_s * 100); s_hi.append(hi_s * 100)
        r_lo.append(lo_r * 100); r_hi.append(hi_r * 100)

    return (np.array(eps_vals), np.array(s_vals), np.array(r_vals),
            np.array(s_lo), np.array(s_hi), np.array(r_lo), np.array(r_hi))


def extract_ratio_series(sweep: dict, ratio_key: str):
    eps_vals, ratios = [], []
    for entry in sweep["epsilon_sweep"]:
        eps_vals.append(entry["exp_epsilon"])
        r = entry.get(ratio_key, None)
        if r is None or not math.isfinite(r):
            r = float("nan")
        ratios.append(r)
    return np.array(eps_vals), np.array(ratios)


def x_ticks(eps_vals):
    labels = [f"ln({v:.0f})" if v == int(v) else f"ln({v:.1f})" for v in eps_vals]
    # Show every other label to avoid crowding
    sparse = [l if i % 2 == 0 else "" for i, l in enumerate(labels)]
    return np.arange(len(eps_vals)), sparse


# ── Figure 1: Jazz Acc@1 vs eps ────────────────────────────────────────────────

def plot_jazz_acc(jazz: dict) -> None:
    eps_vals, s_vals, r_vals, slo, shi, rlo, rhi = [], [], [], [], [], [], []
    N_total = jazz.get("N", 200) * jazz.get("num_trials", 30)
    for entry in jazz["epsilon_sweep"]:
        eps_vals.append(entry["exp_epsilon"])
        sv = entry["aware"]["aware_acc1_scale"]
        rv = entry["aware"]["aware_acc1_random"]
        s_vals.append(sv * 100); r_vals.append(rv * 100)
        k_s = round(sv * N_total); k_r = round(rv * N_total)
        lo_s, hi_s = wilson_ci(k_s, N_total)
        lo_r, hi_r = wilson_ci(k_r, N_total)
        slo.append(lo_s * 100); shi.append(hi_s * 100)
        rlo.append(lo_r * 100); rhi.append(hi_r * 100)

    eps_arr = np.array(eps_vals)
    s_arr = np.array(s_vals); r_arr = np.array(r_vals)
    slo_arr = np.array(slo); shi_arr = np.array(shi)
    rlo_arr = np.array(rlo); rhi_arr = np.array(rhi)
    x = np.arange(len(eps_arr))
    xlabels = [f"ln({v:.0f})" if v == int(v) else f"ln({v:.1f})" for v in eps_vals]
    xlabels_sparse = [l if i % 2 == 0 else "" for i, l in enumerate(xlabels)]

    chance = 100.0 / jazz["N"]

    fig, ax = plt.subplots(figsize=FIGSIZE)

    # M-Scale
    s_yerr = np.vstack([
        np.maximum(0, s_arr - slo_arr),
        np.maximum(0, shi_arr - s_arr),
    ])
    ax.errorbar(x, s_arr, yerr=s_yerr, fmt="s-", color=C_MSCALE,
                capsize=4, elinewidth=0.8, label=r"\textsc{M-Scale}")
    ax.fill_between(x, slo_arr, shi_arr, color=C_MSCALE, alpha=0.15)

    # M-Random
    r_yerr = np.vstack([
        np.maximum(0, r_arr - rlo_arr),
        np.maximum(0, rhi_arr - r_arr),
    ])
    ax.errorbar(x, r_arr, yerr=r_yerr, fmt="o--", color=C_MRAND,
                capsize=4, elinewidth=0.8, label=r"\textsc{M-Random}")
    ax.fill_between(x, rlo_arr, rhi_arr, color=C_MRAND, alpha=0.15)

    ax.axhline(chance, color="gray", linestyle=":", linewidth=0.8, label="Chance")

    # Reference verticals
    for ref in [3, 7, 12]:
        idx = min(range(len(eps_vals)), key=lambda i: abs(eps_vals[i] - ref))
        ax.axvline(idx, color="lightgray", linestyle=":", linewidth=0.6)

    ax.set_xticks(x)
    ax.set_xticklabels(xlabels_sparse, fontsize=7, rotation=45, ha="right")
    ax.set_xlabel(r"$\varepsilon$")
    ax.set_ylabel("Acc@1 (%)")
    ax.set_title("Jazz 200 — Mechanism-Aware Attacker")
    ax.legend(fontsize=8)
    plt.tight_layout()
    out = FIGURES_DIR / "fig_jazz_acc_vs_eps.pdf"
    plt.savefig(out)
    plt.close()
    print(f"Saved {out}")


# ── Figure 2: Jazz ratio curves ────────────────────────────────────────────────

def plot_jazz_ratio(jazz: dict) -> None:
    eps_vals = [e["exp_epsilon"] for e in jazz["epsilon_sweep"]]
    x = np.arange(len(eps_vals))
    xlabels = [f"ln({v:.0f})" if v == int(v) else f"ln({v:.1f})" for v in eps_vals]
    xlabels_sparse = [l if i % 2 == 0 else "" for i, l in enumerate(xlabels)]

    ag_ratios = []
    aw_ratios = []
    nd_ratios = []

    for entry in jazz["epsilon_sweep"]:
        ag_ratios.append(entry.get("agnostic_ratio", float("nan")))
        aw_ratios.append(entry.get("aware_ratio", float("nan")))
        if "near_diatonic" in entry:
            nd_ratios.append(entry["near_diatonic"].get("aware_ratio", float("nan")))
        else:
            nd_ratios.append(float("nan"))

    def safe(arr):
        return np.array([x if math.isfinite(x) else float("nan") for x in arr])

    fig, ax = plt.subplots(figsize=FIGSIZE)
    ax.plot(x, safe(ag_ratios), "o--", color=C_AG, label="Agnostic (all)")
    ax.plot(x, safe(aw_ratios), "s-",  color=C_MSCALE, label="Aware (all)")
    if any(math.isfinite(v) for v in nd_ratios):
        ax.plot(x, safe(nd_ratios), "^-", color=C_ND, label="Aware (near-diatonic)")

    ax.axhline(1.0, color="black", linestyle=":", linewidth=0.8)
    for ref in [3, 7, 12]:
        idx = min(range(len(eps_vals)), key=lambda i: abs(eps_vals[i] - ref))
        ax.axvline(idx, color="lightgray", linestyle=":", linewidth=0.6)

    ax.set_xticks(x)
    ax.set_xticklabels(xlabels_sparse, fontsize=7, rotation=45, ha="right")
    ax.set_xlabel(r"$\varepsilon$")
    ax.set_ylabel(r"\textsc{M-Scale}/\textsc{M-Random} Acc@1 ratio")
    ax.set_title("Jazz 200 — Re-identification Ratio")
    ax.legend(fontsize=8)
    plt.tight_layout()
    out = FIGURES_DIR / "fig_jazz_ratio_vs_eps.pdf"
    plt.savefig(out)
    plt.close()
    print(f"Saved {out}")


# ── Figure 3: Cross-dataset comparison (all 4 datasets) ───────────────────────

def plot_cross_dataset_4(europa: dict, meertens: dict, symbtr: dict, jazz: dict) -> None:
    datasets = [
        (europa,   "Europa 200",   C_EUROPA,   "s-"),
        (meertens, "Meertens 500", C_MEERTENS, "o--"),
        (symbtr,   "SymbTr 200",   C_SYMBTR,   "^-"),
        (jazz,     "Jazz 200",     C_JAZZ,     "D:"),
    ]

    fig, ax = plt.subplots(figsize=FIGSIZE)

    for sweep, label, color, fmt in datasets:
        eps_vals = [e["exp_epsilon"] for e in sweep["epsilon_sweep"]]
        x = np.arange(len(eps_vals))
        ratios = [e.get("aware_ratio", float("nan")) for e in sweep["epsilon_sweep"]]
        ratios = [r if math.isfinite(r) else float("nan") for r in ratios]
        ax.plot(x, ratios, fmt, color=color, label=label)

    # Use Jazz x-axis (all have 15 points)
    eps_vals = [e["exp_epsilon"] for e in jazz["epsilon_sweep"]]
    xlabels = [f"ln({v:.0f})" if v == int(v) else f"ln({v:.1f})" for v in eps_vals]
    xlabels_sparse = [l if i % 2 == 0 else "" for i, l in enumerate(xlabels)]
    ax.set_xticks(np.arange(len(eps_vals)))
    ax.set_xticklabels(xlabels_sparse, fontsize=7, rotation=45, ha="right")

    ax.axhline(1.0, color="black", linestyle=":", linewidth=0.8, label="No advantage")
    for ref in [3, 7, 12]:
        idx = min(range(len(eps_vals)), key=lambda i: abs(eps_vals[i] - ref))
        ax.axvline(idx, color="lightgray", linestyle=":", linewidth=0.6)

    ax.set_xlabel(r"$\varepsilon$")
    ax.set_ylabel(r"\textsc{M-Scale}/\textsc{M-Random} Acc@1 ratio")
    ax.set_title("Cross-Dataset: Aware Attacker Ratio")
    ax.legend(fontsize=8)
    plt.tight_layout()
    out = FIGURES_DIR / "fig_ext_comparison_jazz.pdf"
    plt.savefig(out)
    plt.close()
    print(f"Saved {out}")


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    jazz_path = RESULTS_DIR / "jazz_eps_sweep.json"
    if not jazz_path.exists():
        print(f"ERROR: {jazz_path} not found.")
        print("Run:  python scripts/10_run_jazz.py  first.")
        sys.exit(1)

    with open(jazz_path) as f:
        jazz = json.load(f)

    with open(RESULTS_DIR / "europa_eps_sweep.json") as f:
        europa = json.load(f)
    with open(RESULTS_DIR / "meertens_eps_sweep.json") as f:
        meertens = json.load(f)
    with open(RESULTS_DIR / "symbtr_eps_sweep.json") as f:
        symbtr = json.load(f)

    print("Generating Jazz figures...")

    # Print summary
    print("\n--- Jazz 200 peak results ---")
    best_aw = max(jazz["epsilon_sweep"], key=lambda e: e.get("aware_ratio", 0))
    print(f"  Peak aware ratio: {best_aw['aware_ratio']:.1f}x at eps=ln({best_aw['exp_epsilon']:.1f})")
    best_ag = max(jazz["epsilon_sweep"], key=lambda e: e.get("agnostic_ratio", 0))
    print(f"  Peak agnostic ratio: {best_ag['agnostic_ratio']:.1f}x at eps=ln({best_ag['exp_epsilon']:.1f})")

    plot_jazz_acc(jazz)
    plot_jazz_ratio(jazz)
    plot_cross_dataset_4(europa, meertens, symbtr, jazz)

    print("\nAll figures saved to", FIGURES_DIR)


if __name__ == "__main__":
    main()
