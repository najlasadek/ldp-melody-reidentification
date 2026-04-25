"""
07_plot_four_corpora.py
=======================
Single 2x2 figure: all four corpora, 4 lines per panel.
One shared legend. Consistent colors across all panels.
Starts at epsilon = ln(3), cutoff at ln(16).

Colors:
  Purple   (#6a0dad) : Aware attacker — all melodies (heuristic, fixed ln7)
  Hot pink (#ff1493) : Aware attacker — near-diatonic subset
  Teal     (#008080) : Aware attacker — epsilon-adaptive (knows epsilon)
  Orange   (#e07b00) : Agnostic attacker — all melodies
"""

from __future__ import annotations

import json
from math import exp, log
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

BASE_DIR        = Path(__file__).parent.parent
RESULTS_DIR     = BASE_DIR / "results"
ADAPTIVE_DIR    = BASE_DIR.parent / "adaptive_aware" / "results"
FIG_DIR         = BASE_DIR / "figures"
FIG_DIR.mkdir(exist_ok=True)
PAPER_FIG_DIR   = BASE_DIR.parent.parent / "ISMIR-extended-red" / "figures"

# ── Colors ────────────────────────────────────────────────────────────────────
C_AWARE    = "#6a0dad"   # purple   — aware, all melodies, fixed ln7 heuristic
C_AWARE_ND = "#ff1493"   # hot pink — aware, near-diatonic subset
C_ADAPTIVE = "#008080"   # teal     — aware, epsilon-adaptive
C_AGNOSTIC = "#e07b00"   # orange   — agnostic, all melodies

REF_EPS = [log(3), log(7), log(12)]
EPS_MIN = log(3) - 0.01
EPS_MAX = log(16) + 0.01

# corpus key, label, chi label, adaptive result filename
CORPORA = [
    ("europa",   "Europa 200",   r"$\chi_{\mathrm{mean}}=12.4\%$", "adaptive_europa_200.json"),
    ("meertens", "Meertens 500", r"$\chi_{\mathrm{mean}}=8.8\%$",  "adaptive_meertens_500.json"),
    ("symbtr",   "SymbTr 200",   r"$\chi_{\mathrm{mean}}=21.5\%$", "adaptive_symbtr_200.json"),
    ("jazz",     "Jazz 200",     r"$\chi_{\mathrm{mean}}=37.9\%$", "adaptive_jazz_200.json"),
]


def load_main(name):
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


def safe(v):
    return np.nan if v == float("inf") or v != v else v


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
        data   = load_main(key)
        adata  = load_adaptive(adapt_fname)
        ax     = axes_flat[idx]

        if data is None:
            ax.set_visible(False)
            continue

        sweep = data["epsilon_sweep"]

        # main arrays
        exp_e  = np.array([r["exp_epsilon"]          for r in sweep])
        ag_rat = np.array([safe(r["agnostic_ratio"]) for r in sweep])
        aw_rat = np.array([safe(r["aware_ratio"])     for r in sweep])

        mask   = (exp_e >= exp(EPS_MIN)) & (exp_e <= exp(EPS_MAX))
        exp_e  = exp_e[mask]
        ag_rat = ag_rat[mask]
        aw_rat = aw_rat[mask]

        # near-diatonic
        nd_raw = [safe(r["near_diatonic"]["aware_ratio"])
                  if "near_diatonic" in r and r["near_diatonic"] else np.nan
                  for r in sweep]
        nd_rat = np.array(nd_raw)[mask]

        # adaptive — align on exp_epsilon
        ad_rat = None
        if adata is not None:
            asweep  = adata["epsilon_sweep"]
            a_exp_e = np.array([r["exp_epsilon"]           for r in asweep])
            a_rat   = np.array([safe(r["adaptive"]["ratio"]) for r in asweep])
            amask   = (a_exp_e >= exp(EPS_MIN)) & (a_exp_e <= exp(EPS_MAX))
            a_exp_e = a_exp_e[amask]
            a_rat   = a_rat[amask]
            # interpolate onto main grid if needed
            ad_rat  = np.interp(exp_e, a_exp_e, a_rat)

        # reference verticals
        for ref in REF_EPS:
            xv = exp(ref)
            if exp_e[0] - 0.1 <= xv <= exp_e[-1] + 0.1:
                ax.axvline(xv, color="lightgray", linestyle=":", linewidth=0.8, zorder=0)

        ax.axhline(1.0, color="black", linestyle=":", linewidth=0.8, zorder=1)

        # agnostic — orange dashed circles
        h_ag, = ax.plot(exp_e, ag_rat, "o--", color=C_AGNOSTIC, linewidth=1.5,
                        markersize=4, zorder=3,
                        label=r"Agnostic — all melodies")

        # aware heuristic — purple solid squares
        h_aw, = ax.plot(exp_e, aw_rat, "s-", color=C_AWARE, linewidth=2.0,
                        markersize=5, zorder=4,
                        label=r"Aware heuristic — all melodies (fixed $\varepsilon{=}\ln 7$)")

        # aware near-diatonic — hot pink solid triangles
        h_nd, = ax.plot(exp_e, nd_rat, "^-", color=C_AWARE_ND, linewidth=1.8,
                        markersize=5, zorder=5,
                        label=r"Aware heuristic — near-diatonic ($\chi < 0.05$)")

        # adaptive — teal solid diamonds
        if ad_rat is not None:
            h_ad, = ax.plot(exp_e, ad_rat, "D-", color=C_ADAPTIVE, linewidth=1.8,
                            markersize=4, zorder=6,
                            label=r"Aware adaptive — all melodies (knows $\varepsilon$)")
            if idx == 0:
                legend_handles = [h_aw, h_nd, h_ad, h_ag]
        elif idx == 0:
            legend_handles = [h_aw, h_nd, h_ag]

        ax.set_xticks(exp_e[::2])
        ax.set_xticklabels(x_labels(exp_e[::2]), fontsize=9, rotation=45)
        ax.set_xlabel(r"Privacy budget $\varepsilon$", fontsize=11)
        ax.set_ylabel(r"Acc@1 ratio M-Scale / M-Random", fontsize=10)
        ax.set_title(f"{corpus_label}  ({chi_label})", fontsize=12, fontweight="bold")
        ax.grid(True, alpha=0.2)

    # single shared legend above panels
    fig.legend(
        handles=legend_handles,
        loc="upper center",
        ncol=2,
        fontsize=11,
        frameon=True,
        bbox_to_anchor=(0.5, 1.03),
    )

    fig.suptitle(
        r"M-Scale / M-Random re-identification ratio across four corpora",
        fontsize=13,
        y=1.09,
    )

    plt.tight_layout()

    stem = "fig_four_corpora_ratio"
    for d in [FIG_DIR, PAPER_FIG_DIR]:
        if d.exists():
            fig.savefig(d / f"{stem}.pdf", bbox_inches="tight")
            fig.savefig(d / f"{stem}.png", dpi=150, bbox_inches="tight")
            print(f"Saved to {d / stem}.pdf")
    plt.close()


if __name__ == "__main__":
    plot()
    print("Done.")
