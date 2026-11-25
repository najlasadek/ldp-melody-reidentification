"""
08_plot_acc_curves.py
=====================
2x2 Acc@1 accuracy curves for all four corpora.
Colors match the ratio plot (fig_four_corpora_ratio) so that
M-Random curve colors here equal the ratio curve colors there.

  Gray   (#888888) : M-Scale (numerator — neutral)
  Purple (#6a0dad) : M-Random, aware heuristic  ← ratio plot aware heuristic color
  Teal   (#008080) : M-Random, aware adaptive   ← ratio plot aware adaptive color
  Orange (#e07b00) : M-Random, agnostic         ← ratio plot agnostic color
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

BASE_DIR      = Path(__file__).parent.parent
RESULTS_DIR   = BASE_DIR / "results"
ADAPTIVE_DIR  = BASE_DIR.parent / "adaptive_aware" / "results"

# --normalised: plot the length-normalised results (results_normalised/,
# written by 17_run_normalised_sweeps.py) instead of the original results.
NORMALISED = "--normalised" in sys.argv
if NORMALISED:
    RESULTS_DIR = BASE_DIR / "results_normalised"
    ADAPTIVE_DIR = RESULTS_DIR
FIG_DIR       = BASE_DIR / "figures"
FIG_DIR.mkdir(exist_ok=True)
PAPER_FIG_DIR = BASE_DIR.parent.parent / "ISMIR-extended-red" / "figures"

C_SCALE    = "#444444"   # dark gray      — M-Scale (numerator, neutral)
C_SCALE_AD = "#bbbbbb"   # light gray    — M-Scale adaptive (overlaps, not in legend)
C_RAND     = "#6a0dad"   # purple        — M-Random, aware heuristic  (= ratio plot)
C_RAND_AD  = "#008080"   # teal          — M-Random, aware adaptive   (= ratio plot)
C_AG       = "#e07b00"   # orange        — M-Random, agnostic         (= ratio plot)

REF_EPS = [log(3), log(7), log(12)]
EPS_MIN = log(3) - 0.01
EPS_MAX = log(16) + 0.01

CORPORA = [
    ("europa",   "Europa 200",   r"$\chi_{\mathrm{mean}}=12.4\%$", "adaptive_europa_200.json"),
    ("meertens", "Meertens 500", r"$\chi_{\mathrm{mean}}=8.8\%$",  "adaptive_meertens_500.json"),
    ("symbtr",   "SymbTr 200",   r"$\chi_{\mathrm{mean}}=21.5\%$", "adaptive_symbtr_200.json"),
    ("jazz",     "Jazz 200",     r"$\chi_{\mathrm{mean}}=37.9\%$", "adaptive_jazz_200.json"),
]


def load(name):
    p = RESULTS_DIR / f"{name}_eps_sweep.json"
    if not p.exists():
        print(f"  Missing: {p}")
        return None
    with open(p) as f:
        return json.load(f)


def load_adaptive(fname):
    p = ADAPTIVE_DIR / fname
    if not p.exists():
        print(f"  Missing adaptive: {p}")
        return None
    with open(p) as f:
        return json.load(f)


def x_labels(exp_eps_vals):
    out = []
    for v in exp_eps_vals:
        if abs(v - 1.5) < 0.1:
            out.append(r"$\ln 1.5$")
        elif abs(v - round(v)) < 0.05:
            out.append(r"$\ln %d$" % int(round(v)))
        else:
            out.append(r"$\ln %.1f$" % v)
    return out


def plot():
    fig, axes = plt.subplots(2, 2, figsize=(10, 7.0))
    axes_flat = [axes[0, 0], axes[0, 1], axes[1, 0], axes[1, 1]]

    legend_handles = []

    for idx, (key, corpus_label, chi_label, adapt_fname) in enumerate(CORPORA):
        data  = load(key)
        adata = load_adaptive(adapt_fname)
        ax    = axes_flat[idx]

        if data is None:
            ax.set_visible(False)
            continue

        sweep = data["epsilon_sweep"]

        exp_e    = np.array([r["exp_epsilon"]                for r in sweep])
        sc_aw    = np.array([r["aware"]["aware_acc1_scale"]  for r in sweep]) * 100
        rd_aw    = np.array([r["aware"]["aware_acc1_random"] for r in sweep]) * 100
        rd_ag    = np.array([r["agnostic"]["acc1_random"]    for r in sweep]) * 100
        sc_aw_lo = np.array([r["aware"]["aware_ci1_scale"][0]  for r in sweep]) * 100
        sc_aw_hi = np.array([r["aware"]["aware_ci1_scale"][1]  for r in sweep]) * 100
        rd_aw_lo = np.array([r["aware"]["aware_ci1_random"][0] for r in sweep]) * 100
        rd_aw_hi = np.array([r["aware"]["aware_ci1_random"][1] for r in sweep]) * 100
        rd_ag_lo = np.array([r["agnostic"]["ci1_random"][0]    for r in sweep]) * 100
        rd_ag_hi = np.array([r["agnostic"]["ci1_random"][1]    for r in sweep]) * 100

        mask  = (exp_e >= exp(EPS_MIN)) & (exp_e <= exp(EPS_MAX))
        exp_e    = exp_e[mask]
        sc_aw    = sc_aw[mask];   sc_aw_lo = sc_aw_lo[mask];   sc_aw_hi = sc_aw_hi[mask]
        rd_aw    = rd_aw[mask];   rd_aw_lo = rd_aw_lo[mask];   rd_aw_hi = rd_aw_hi[mask]
        rd_ag    = rd_ag[mask];   rd_ag_lo = rd_ag_lo[mask];   rd_ag_hi = rd_ag_hi[mask]

        # adaptive arrays (aligned to main grid)
        sc_ad = rd_ad = sc_ad_lo = sc_ad_hi = rd_ad_lo = rd_ad_hi = None
        if adata is not None:
            asweep   = adata["epsilon_sweep"]
            a_exp_e  = np.array([r["exp_epsilon"]               for r in asweep])
            a_sc     = np.array([r["adaptive"]["acc1_scale"]    for r in asweep]) * 100
            a_rd     = np.array([r["adaptive"]["acc1_random"]   for r in asweep]) * 100
            a_sc_lo  = np.array([r["adaptive"]["ci1_scale"][0]  for r in asweep]) * 100
            a_sc_hi  = np.array([r["adaptive"]["ci1_scale"][1]  for r in asweep]) * 100
            a_rd_lo  = np.array([r["adaptive"]["ci1_random"][0] for r in asweep]) * 100
            a_rd_hi  = np.array([r["adaptive"]["ci1_random"][1] for r in asweep]) * 100
            amask    = (a_exp_e >= exp(EPS_MIN)) & (a_exp_e <= exp(EPS_MAX))
            a_exp_e  = a_exp_e[amask]
            sc_ad    = np.interp(exp_e, a_exp_e, a_sc[amask])
            rd_ad    = np.interp(exp_e, a_exp_e, a_rd[amask])
            sc_ad_lo = np.interp(exp_e, a_exp_e, a_sc_lo[amask])
            sc_ad_hi = np.interp(exp_e, a_exp_e, a_sc_hi[amask])
            rd_ad_lo = np.interp(exp_e, a_exp_e, a_rd_lo[amask])
            rd_ad_hi = np.interp(exp_e, a_exp_e, a_rd_hi[amask])

        # reference verticals
        for ref in REF_EPS:
            xv = exp(ref)
            if exp_e[0] - 0.1 <= xv <= exp_e[-1] + 0.1:
                ax.axvline(xv, color="lightgray", linestyle=":", linewidth=0.8, zorder=0)

        # chance baseline
        N = data["N"]
        ax.axhline(100 / N, color="gray", linestyle=":", linewidth=0.8, zorder=1)

        # ── M-Scale (gray) — plot first so it sits behind ────────────────────
        ax.fill_between(exp_e, sc_aw_lo, sc_aw_hi, color=C_SCALE, alpha=0.12, zorder=2)
        h_sc, = ax.plot(exp_e, sc_aw, "s-", color=C_SCALE, linewidth=2.0, markersize=5,
                        zorder=3,
                        label=r"M-Scale — aware heuristic / aware adaptive / agnostic"
                              r" (all equivalent: $\chi(m'){=}0$)")

        # ── M-Random aware heuristic — purple solid ───────────────────────────
        ax.fill_between(exp_e, rd_aw_lo, rd_aw_hi, color=C_RAND, alpha=0.12, zorder=3)
        h_rd, = ax.plot(exp_e, rd_aw, "o-", color=C_RAND, linewidth=2.0, markersize=5,
                        zorder=4,
                        label=r"M-Random — aware heuristic (fixed $\varepsilon{=}\ln 7$)")

        # ── M-Random agnostic — orange dashed ────────────────────────────────
        ax.fill_between(exp_e, rd_ag_lo, rd_ag_hi, color=C_AG, alpha=0.12, zorder=4)
        h_ag, = ax.plot(exp_e, rd_ag, "o--", color=C_AG, linewidth=1.5, markersize=4,
                        zorder=5, label=r"M-Random — agnostic")

        # ── Adaptive curves (dashed) ──────────────────────────────────────────
        if sc_ad is not None:
            # M-Scale adaptive: same gray, light dashed — no legend entry
            ax.fill_between(exp_e, sc_ad_lo, sc_ad_hi, color=C_SCALE_AD, alpha=0.10, zorder=5)
            ax.plot(exp_e, sc_ad, "s--", color=C_SCALE_AD, linewidth=1.6,
                    markersize=4, zorder=6)

            # M-Random adaptive — teal dashed
            ax.fill_between(exp_e, rd_ad_lo, rd_ad_hi, color=C_RAND_AD, alpha=0.10, zorder=6)
            h_rd_ad, = ax.plot(exp_e, rd_ad, "o--", color=C_RAND_AD, linewidth=1.6,
                               markersize=4, zorder=7,
                               label=r"M-Random — aware adaptive (knows $\varepsilon$)")

            if idx == 0:
                legend_handles = [h_sc, h_rd, h_rd_ad, h_ag]
        elif idx == 0:
            legend_handles = [h_sc, h_rd, h_ag]

        ax.set_xticks(exp_e[::2])
        ax.set_xticklabels(x_labels(exp_e[::2]), fontsize=9, rotation=45)
        ax.set_xlabel(r"Privacy budget $\varepsilon$", fontsize=11)
        ax.set_ylabel(r"Acc@1 (%)", fontsize=11)
        ax.set_title(f"{corpus_label}  ({chi_label})", fontsize=12, fontweight="bold")
        ax.grid(True, alpha=0.2)

    fig.legend(
        handles=legend_handles,
        loc="upper center",
        ncol=2,
        fontsize=11,
        frameon=True,
        bbox_to_anchor=(0.5, 1.03),
    )

    fig.suptitle(
        r"Acc@1 re-identification accuracy across four corpora"
        + (" (length-normalised distance)" if NORMALISED else "")
        + "\n"
        r"Solid: heuristic aware attacker (fixed $\varepsilon{=}\ln 7$, does not know $\varepsilon$). "
        r"Dashed: $\varepsilon$-adaptive aware attacker (knows $\varepsilon$). "
        r"Shaded bands: 95% Wilson CIs.",
        fontsize=11,
        y=1.09,
    )

    plt.tight_layout()

    stem = "fig_four_corpora_acc" + ("_normalised" if NORMALISED else "")
    for d in [FIG_DIR, PAPER_FIG_DIR]:
        if d.exists():
            fig.savefig(d / f"{stem}.pdf", bbox_inches="tight")
            fig.savefig(d / f"{stem}.png", dpi=150, bbox_inches="tight")
            print(f"Saved to {d / stem}.pdf")
    plt.close()


if __name__ == "__main__":
    plot()
    print("Done.")
