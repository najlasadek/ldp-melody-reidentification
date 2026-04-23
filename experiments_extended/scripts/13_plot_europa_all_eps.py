"""
13_plot_europa_all_eps.py
=========================
Generate fig_europa_all_eps.pdf: a 2-panel figure for Europa 200 only,
showing all 15 epsilon values.

Left panel:  Acc@1 vs epsilon (aware attacker), M-Scale and M-Random with CIs
Right panel: M-Scale/M-Random ratio vs epsilon (Agnostic-all, Aware-all, Aware near-diatonic)
             — ratio points where M-Random Acc@1 == 0 are dropped (avoids infinity spikes)

Run from experiments_extended/ directory:
    python scripts/13_plot_europa_all_eps.py
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

SCRIPT_DIR  = Path(__file__).parent
BASE_DIR    = SCRIPT_DIR.parent
RESULTS_DIR = BASE_DIR / "results"
FIGURES_DIR = BASE_DIR / "figures"
FIGURES_DIR.mkdir(exist_ok=True)

# Paper style
C_MRAND  = "steelblue"
C_MSCALE = "tomato"
C_ND     = "seagreen"
C_AG     = "gray"

REF_EPS = [math.log(3), math.log(7), math.log(12)]


def wilson_ci(k: int, n: int, z: float = 1.96):
    if n == 0:
        return 0.0, 0.0
    p = k / n
    denom = 1 + z**2 / n
    center = (p + z**2 / (2 * n)) / denom
    margin = z * math.sqrt(p * (1 - p) / n + z**2 / (4 * n**2)) / denom
    return max(0.0, center - margin), min(1.0, center + margin)


def safe_ratio(s, r, floor=1e-6):
    """Return s/r, or NaN if r is below floor (avoids infinity)."""
    return s / r if r > floor else float("nan")


def main():
    path = RESULTS_DIR / "europa_eps_sweep.json"
    if not path.exists():
        print(f"ERROR: {path} not found.")
        sys.exit(1)

    with open(path) as f:
        data = json.load(f)

    sweep = data["epsilon_sweep"]
    N     = data["N"]          # 200
    n_trials = data.get("num_trials", 30)
    n_obs = N * n_trials

    # ── Extract series ────────────────────────────────────────────────────────
    eps_vals  = []
    exp_vals  = []

    aw_s_vals = []; aw_r_vals = []
    aw_s_lo   = []; aw_s_hi   = []
    aw_r_lo   = []; aw_r_hi   = []

    ag_ratio_vals = []
    aw_ratio_vals = []
    nd_ratio_vals = []

    for entry in sweep:
        eps_vals.append(entry["epsilon"])
        exp_vals.append(entry["exp_epsilon"])

        aw = entry["aware"]
        ag = entry["agnostic"]

        aw_s = aw["aware_acc1_scale"]
        aw_r = aw["aware_acc1_random"]
        ag_s = ag["acc1_scale"]
        ag_r = ag["acc1_random"]

        aw_s_vals.append(aw_s)
        aw_r_vals.append(aw_r)

        # CIs from stored values if available, else recompute
        if "aware_ci1_scale" in aw:
            aw_s_lo.append(aw["aware_ci1_scale"][0])
            aw_s_hi.append(aw["aware_ci1_scale"][1])
        else:
            lo, hi = wilson_ci(round(aw_s * n_obs), n_obs)
            aw_s_lo.append(lo); aw_s_hi.append(hi)

        if "aware_ci1_random" in aw:
            aw_r_lo.append(aw["aware_ci1_random"][0])
            aw_r_hi.append(aw["aware_ci1_random"][1])
        else:
            lo, hi = wilson_ci(round(aw_r * n_obs), n_obs)
            aw_r_lo.append(lo); aw_r_hi.append(hi)

        ag_ratio_vals.append(safe_ratio(ag_s, ag_r))
        aw_ratio_vals.append(safe_ratio(aw_s, aw_r))

        nd = entry.get("near_diatonic")
        if nd:
            nd_aw = nd.get("aware", nd)
            nd_s = nd_aw.get("aware_acc1_scale", nd_aw.get("acc1_scale", float("nan")))
            nd_r = nd_aw.get("aware_acc1_random", nd_aw.get("acc1_random", float("nan")))
            nd_ratio_vals.append(safe_ratio(nd_s, nd_r))
        else:
            nd_ratio_vals.append(float("nan"))

    eps_arr = np.array(eps_vals)
    exp_arr = np.array(exp_vals)

    aw_s_arr = np.array(aw_s_vals)
    aw_r_arr = np.array(aw_r_vals)
    aw_s_lo_arr = np.array(aw_s_lo)
    aw_s_hi_arr = np.array(aw_s_hi)
    aw_r_lo_arr = np.array(aw_r_lo)
    aw_r_hi_arr = np.array(aw_r_hi)

    ag_ratio_arr = np.array(ag_ratio_vals)
    aw_ratio_arr = np.array(aw_ratio_vals)
    nd_ratio_arr = np.array(nd_ratio_vals)

    # x-tick labels: every other label to avoid crowding
    xlabels_all = []
    for i, e in enumerate(exp_arr):
        v = round(e)
        lbl = f"ln({v})" if abs(e - v) < 0.15 else f"ln({e:.1f})"
        xlabels_all.append(lbl if i % 2 == 0 else "")
    x = np.arange(len(eps_arr))

    # ── Build figure ──────────────────────────────────────────────────────────
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 3.5))

    # Reference verticals
    for ref in REF_EPS:
        idx = min(range(len(eps_vals)), key=lambda i: abs(eps_vals[i] - ref))
        ax1.axvline(idx, color="lightgray", linestyle=":", linewidth=0.7, zorder=0)
        ax2.axvline(idx, color="lightgray", linestyle=":", linewidth=0.7, zorder=0)

    # ── Left: Acc@1 vs eps ────────────────────────────────────────────────────
    chance = 100.0 / N

    # M-Scale
    s_yerr = np.vstack([
        np.maximum(0, aw_s_arr - aw_s_lo_arr),
        np.maximum(0, aw_s_hi_arr - aw_s_arr),
    ]) * 100
    ax1.errorbar(x, aw_s_arr * 100, yerr=s_yerr,
                 fmt="s-", color=C_MSCALE, capsize=4, elinewidth=0.8,
                 markersize=5, label=r"$\mathcal{M}$-Scale")
    ax1.fill_between(x, aw_s_lo_arr * 100, aw_s_hi_arr * 100,
                     color=C_MSCALE, alpha=0.15)

    # M-Random
    r_yerr = np.vstack([
        np.maximum(0, aw_r_arr - aw_r_lo_arr),
        np.maximum(0, aw_r_hi_arr - aw_r_arr),
    ]) * 100
    ax1.errorbar(x, aw_r_arr * 100, yerr=r_yerr,
                 fmt="o--", color=C_MRAND, capsize=4, elinewidth=0.8,
                 markersize=5, label=r"$\mathcal{M}$-Random")
    ax1.fill_between(x, aw_r_lo_arr * 100, aw_r_hi_arr * 100,
                     color=C_MRAND, alpha=0.15)

    ax1.axhline(chance, color="gray", linestyle=":", linewidth=0.8,
                label=f"Chance ({chance:.1f}%)")
    ax1.set_xticks(x)
    ax1.set_xticklabels(xlabels_all, fontsize=7, rotation=45, ha="right")
    ax1.set_xlabel(r"$\varepsilon$")
    ax1.set_ylabel("Acc@1 (%)")
    ax1.set_title("Europa 200 — Mechanism-Aware Attacker")
    ax1.legend(fontsize=8)
    ax1.set_ylim(bottom=-0.2)

    # ── Right: Ratio vs eps ───────────────────────────────────────────────────
    ax2.axhline(1.0, color="black", linestyle=":", linewidth=0.8,
                label=r"No advantage ($1\times$)")

    # Agnostic
    mask = np.isfinite(ag_ratio_arr)
    ax2.plot(x[mask], ag_ratio_arr[mask], "o--", color=C_AG,
             markersize=5, label="Agnostic (all)")

    # Aware all
    mask = np.isfinite(aw_ratio_arr)
    ax2.plot(x[mask], aw_ratio_arr[mask], "s-", color=C_MSCALE,
             markersize=5, label="Aware (all)")

    # Near-diatonic
    mask = np.isfinite(nd_ratio_arr)
    if mask.any():
        ax2.plot(x[mask], nd_ratio_arr[mask], "^-", color=C_ND,
                 markersize=5, label="Aware (near-diatonic)")

    ax2.set_xticks(x)
    ax2.set_xticklabels(xlabels_all, fontsize=7, rotation=45, ha="right")
    ax2.set_xlabel(r"$\varepsilon$")
    ax2.set_ylabel(r"M-Scale / M-Random Acc@1 ratio")
    ax2.set_title("Europa 200 — Re-identification Ratio")
    ax2.legend(fontsize=8)
    ax2.set_ylim(bottom=0)

    plt.tight_layout()
    out = FIGURES_DIR / "fig_europa_all_eps.pdf"
    plt.savefig(out, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out}")

    # ── Ratio-only figure (for main text) ─────────────────────────────────────
    fig2, ax = plt.subplots(figsize=(5, 3.5))

    for ref in REF_EPS:
        idx = min(range(len(eps_vals)), key=lambda i: abs(eps_vals[i] - ref))
        ax.axvline(idx, color="lightgray", linestyle=":", linewidth=0.7, zorder=0)

    ax.axhline(1.0, color="black", linestyle=":", linewidth=0.8,
               label=r"No advantage ($1\times$)")

    mask = np.isfinite(ag_ratio_arr)
    ax.plot(x[mask], ag_ratio_arr[mask], "o--", color=C_AG,
            markersize=5, label="Agnostic (all)")

    mask = np.isfinite(aw_ratio_arr)
    ax.plot(x[mask], aw_ratio_arr[mask], "s-", color=C_MSCALE,
            markersize=5, label="Aware (all)")

    mask = np.isfinite(nd_ratio_arr)
    if mask.any():
        ax.plot(x[mask], nd_ratio_arr[mask], "^-", color=C_ND,
                markersize=5, label="Aware (near-diatonic)")

    ax.set_xticks(x)
    ax.set_xticklabels(xlabels_all, fontsize=7, rotation=45, ha="right")
    ax.set_xlabel(r"$\varepsilon$")
    ax.set_ylabel(r"M-Scale / M-Random Acc@1 ratio")
    ax.set_title("Europa 200 — Re-identification Ratio")
    ax.legend(fontsize=8)
    ax.set_ylim(bottom=0)

    plt.tight_layout()
    out_ratio = FIGURES_DIR / "fig_europa_ratio_only.pdf"
    plt.savefig(out_ratio, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out_ratio}")

    # Copy both to ISMIR-extended-red figures
    import shutil
    ismir_figs = BASE_DIR.parent.parent / "ISMIR-extended-red" / "figures"
    if ismir_figs.exists():
        shutil.copy(out, ismir_figs / "fig_europa_all_eps.pdf")
        shutil.copy(out_ratio, ismir_figs / "fig_europa_ratio_only.pdf")
        print(f"Copied to: {ismir_figs}")


if __name__ == "__main__":
    main()
