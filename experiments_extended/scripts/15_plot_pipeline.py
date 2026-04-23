"""
15_plot_pipeline.py
===================
Interactive plotting pipeline.  Run with no arguments for a menu,
or pass plot IDs directly:

    python scripts/15_plot_pipeline.py 1 3 7
    python scripts/15_plot_pipeline.py all

Available plots
---------------
  === Per-corpus ratio plots (appendix) ===
  1   Europa  — ratio vs ε  (Agnostic / Aware-all / Aware-ND)
  2   Meertens— ratio vs ε
  3   SymbTr  — ratio vs ε
  4   Jazz    — ratio vs ε

  === Per-corpus Acc@1 plots (appendix) ===
  5   Europa  — Acc@1 vs ε  (M-Scale / M-Random, aware attacker)
  6   Meertens— Acc@1 vs ε
  7   SymbTr  — Acc@1 vs ε
  8   Jazz    — Acc@1 vs ε

  === Multi-corpus comparison (main paper) ===
  9   3-panel ratio : Europa / Meertens / SymbTr  (fig_ext_ratio_vs_eps)
  10  3-panel Acc@1 : Europa / Meertens / SymbTr  (fig_ext_acc_vs_eps)
  11  Cross-dataset ratio : all 4 corpora          (fig_ext_comparison_jazz)

  === Custom: choose any corpora on one figure ===
  12  Custom ratio  — you pick which corpora + attacker types to overlay
  13  Custom Acc@1  — you pick which corpora to overlay

Run from experiments_extended/ directory:
    python scripts/15_plot_pipeline.py
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
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

# ── Colour palettes ───────────────────────────────────────────────────────────
# Europa   = pink/hot-pink family
# Meertens = blue family
# SymbTr   = green family
# Jazz     = purple family
# Within each corpus: mscale (darkest), nd (deep), mrandom (mid), agnostic (medium-dark)
PALETTES = {
    # Europa = pink/hot-pink family
    # mscale: solid deep pink | mrandom: clearly lighter warm pink |
    # nd: darkest (almost magenta-maroon) | agnostic: salmon/coral — clearly ≠ mrandom
    "europa": {
        "mscale":  "#ad1457",   # deep cerise pink  — M-Scale (aware all)
        "mrandom": "#f48fb1",   # pale dusty pink   — M-Random (aware all)
        "nd":      "#560027",   # dark maroon-pink  — near-diatonic
        "agnostic":"#ff6f00",   # amber/orange      — agnostic (clearly different family)
        "repr":    "#ad1457",
        "label":   "Europa 200",
    },
    # Meertens = blue family
    # mscale: navy | mrandom: sky blue (clearly lighter) |
    # nd: darkest navy | agnostic: teal — clearly ≠ sky blue
    "meertens": {
        "mscale":  "#0d47a1",   # deep navy         — M-Scale
        "mrandom": "#90caf9",   # light sky blue    — M-Random
        "nd":      "#01002e",   # very dark navy    — near-diatonic
        "agnostic":"#00838f",   # dark cyan/teal    — agnostic (clearly ≠ sky blue)
        "repr":    "#0d47a1",
        "label":   "Meertens 500",
    },
    # SymbTr = green family
    # mscale: deep forest green | mrandom: lime green (clearly lighter) |
    # nd: darkest green | agnostic: olive — clearly ≠ lime
    "symbtr": {
        "mscale":  "#1b5e20",   # deep forest green — M-Scale
        "mrandom": "#a5d6a7",   # light mint green  — M-Random
        "nd":      "#003300",   # very dark green   — near-diatonic
        "agnostic":"#827717",   # dark olive        — agnostic (clearly ≠ mint)
        "repr":    "#1b5e20",
        "label":   "SymbTr 200",
    },
    # Jazz = purple family
    # mscale: deep indigo-purple | mrandom: lavender (clearly lighter) |
    # nd: darkest purple | agnostic: dark violet — clearly ≠ lavender
    "jazz": {
        "mscale":  "#4a148c",   # deep indigo       — M-Scale
        "mrandom": "#ce93d8",   # light lavender    — M-Random
        "nd":      "#1a0033",   # very dark violet  — near-diatonic
        "agnostic":"#1a237e",   # deep indigo-navy  — agnostic (clearly ≠ lavender, dark)
        "repr":    "#4a148c",
        "label":   "Jazz 200",
    },
}

CORPUS_NAMES = ["europa", "meertens", "symbtr", "jazz"]
FIGSIZE      = (5, 3.5)
REF_EPS_VALS = [math.log(3), math.log(7), math.log(12)]

PAPER_FIGURES = {
    "fig_ext_ratio_vs_eps":    "fig_ext_ratio_vs_eps.pdf",
    "fig_ext_acc_vs_eps":      "fig_ext_acc_vs_eps.pdf",
    "fig_ext_comparison_jazz": "fig_ext_comparison_jazz.pdf",
    "fig_jazz_ratio_vs_eps":   "fig_jazz_ratio_vs_eps.pdf",
    "fig_jazz_acc_vs_eps":     "fig_jazz_acc_vs_eps.pdf",
    "fig_europa_ratio_only":   "fig_europa_ratio_only.pdf",
    "fig_europa_all_eps":      "fig_europa_all_eps.pdf",
}


# ── Data loading ──────────────────────────────────────────────────────────────

_cache: dict[str, dict] = {}

def load(name: str) -> dict:
    if name not in _cache:
        p = RESULTS_DIR / f"{name}_eps_sweep.json"
        with open(p) as f:
            _cache[name] = json.load(f)
    return _cache[name]


def get_x(sweep: list[dict]):
    exp_e  = np.array([r["exp_epsilon"] for r in sweep])
    x      = np.arange(len(sweep))
    labels = [f"ln({round(e)})" if abs(e - round(e)) < 0.15 else f"ln({e:.1f})"
              for e in exp_e]
    sparse = [l if i % 2 == 0 else "" for i, l in enumerate(labels)]
    return x, exp_e, sparse


# ── Shared plot helpers ───────────────────────────────────────────────────────

def add_ref_lines(ax, exp_e):
    for ref in REF_EPS_VALS:
        idx = int(np.argmin(np.abs(exp_e - ref)))
        ax.axvline(idx, color="lightgray", linestyle=":", linewidth=0.7, zorder=0)


def plot_eb(ax, x, y_pct, lo_pct, hi_pct, color, fmt, label):
    yerr = np.vstack([np.maximum(0, y_pct - lo_pct),
                      np.maximum(0, hi_pct - y_pct)])
    ax.errorbar(x, y_pct, yerr=yerr, fmt=fmt, color=color,
                capsize=4, elinewidth=0.8, markersize=5, label=label)
    ax.fill_between(x, lo_pct, hi_pct, color=color, alpha=0.12)


def set_xticks(ax, x, sparse):
    ax.set_xticks(x)
    ax.set_xticklabels(sparse, fontsize=7, rotation=45, ha="right")


def finish(ax, xlabel=True):
    if xlabel:
        ax.set_xlabel("Privacy budget  ε", fontsize=9)
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.2)


def save(fig, name: str, also_paper=False):
    out = FIGURES_DIR / name
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out}")
    if also_paper:
        paper_dir = Path(__file__).parent.parent.parent.parent / \
                    "TEXT" / "ISMIR-extended-red" / "figures"
        if paper_dir.exists():
            import shutil
            shutil.copy(out, paper_dir / name)
            print(f"  Copied to paper: {paper_dir / name}")


# ── Plot type implementations ─────────────────────────────────────────────────

def plot_ratio_single(corpus: str, out_name: str | None = None) -> None:
    """Ratio plot for one corpus: Agnostic / Aware-all / Aware-ND."""
    data  = load(corpus)
    pal   = PALETTES[corpus]
    sweep = data["epsilon_sweep"]
    x, exp_e, sparse = get_x(sweep)
    nd_N  = data.get("near_diatonic_N", 0)
    N     = data["N"]

    ag_ratio = np.array([r.get("agnostic_ratio", float("nan")) for r in sweep])
    aw_ratio = np.array([r.get("aware_ratio",    float("nan")) for r in sweep])
    nd_ratio = np.array([
        r["near_diatonic"]["aware_ratio"]
        if r.get("near_diatonic") and math.isfinite(r["near_diatonic"].get("aware_ratio", float("nan")))
        else float("nan") for r in sweep
    ])

    fig, ax = plt.subplots(figsize=FIGSIZE)
    add_ref_lines(ax, exp_e)
    ax.axhline(1.0, color="black", linestyle=":", linewidth=0.8, label="No advantage (1×)")
    ax.plot(x, ag_ratio, "o--", color=pal["agnostic"], markersize=5,
            label=f"Agnostic (N={N})", linewidth=1.4)
    ax.plot(x, aw_ratio, "s-",  color=pal["mscale"],   markersize=5,
            label=f"Aware (N={N})", linewidth=1.6)
    nd_mask = np.isfinite(nd_ratio)
    if nd_mask.any():
        ax.plot(x[nd_mask], nd_ratio[nd_mask], "^-", color=pal["nd"],
                markersize=5, label=f"Aware near-diatonic (n={nd_N})", linewidth=1.4)

    set_xticks(ax, x, sparse)
    ax.set_ylabel("Acc@1 ratio  M-Scale / M-Random", fontsize=9)
    ax.set_title(f"{pal['label']} — Re-identification Ratio", fontsize=10)
    ax.set_ylim(bottom=0)
    finish(ax)
    plt.tight_layout()
    fname = out_name or f"fig_{corpus}_ratio_only.pdf"
    save(fig, fname, also_paper=True)


def plot_acc_single(corpus: str, out_name: str | None = None) -> None:
    """Acc@1 vs ε for one corpus (aware attacker, M-Scale + M-Random)."""
    data  = load(corpus)
    pal   = PALETTES[corpus]
    sweep = data["epsilon_sweep"]
    x, exp_e, sparse = get_x(sweep)
    N     = data["N"]
    chance = 100.0 / N

    aw_s    = np.array([r["aware"]["aware_acc1_scale"]    for r in sweep])
    aw_r    = np.array([r["aware"]["aware_acc1_random"]   for r in sweep])
    aw_s_lo = np.array([r["aware"]["aware_ci1_scale"][0]  for r in sweep])
    aw_s_hi = np.array([r["aware"]["aware_ci1_scale"][1]  for r in sweep])
    aw_r_lo = np.array([r["aware"]["aware_ci1_random"][0] for r in sweep])
    aw_r_hi = np.array([r["aware"]["aware_ci1_random"][1] for r in sweep])

    fig, ax = plt.subplots(figsize=FIGSIZE)
    add_ref_lines(ax, exp_e)
    plot_eb(ax, x, aw_s*100, aw_s_lo*100, aw_s_hi*100,
            pal["mscale"],  "s-",  "M-Scale")
    plot_eb(ax, x, aw_r*100, aw_r_lo*100, aw_r_hi*100,
            pal["mrandom"], "o--", "M-Random")
    ax.axhline(chance, color="gray", linestyle=":", linewidth=0.8,
               label=f"Chance ({chance:.1f}%)")

    set_xticks(ax, x, sparse)
    ax.set_ylabel("Acc@1 (%)", fontsize=9)
    ax.set_title(f"{pal['label']} — Aware Attacker Acc@1", fontsize=10)
    ax.set_ylim(bottom=-0.2)
    finish(ax)
    plt.tight_layout()
    fname = out_name or f"fig_{corpus}_all_eps.pdf"
    save(fig, fname, also_paper=True)


def plot_3panel_ratio(corpora: list[str], out_name: str) -> None:
    """Side-by-side ratio panels, one per corpus."""
    fig, axes = plt.subplots(1, len(corpora),
                             figsize=(5 * len(corpora), 3.5), sharey=False)
    if len(corpora) == 1:
        axes = [axes]

    for ax, corpus in zip(axes, corpora):
        pal   = PALETTES[corpus]
        data  = load(corpus)
        sweep = data["epsilon_sweep"]
        x, exp_e, sparse = get_x(sweep)
        nd_N  = data.get("near_diatonic_N", 0)
        N     = data["N"]

        ag_ratio = np.array([r.get("agnostic_ratio", float("nan")) for r in sweep])
        aw_ratio = np.array([r.get("aware_ratio",    float("nan")) for r in sweep])
        nd_ratio = np.array([
            r["near_diatonic"]["aware_ratio"]
            if r.get("near_diatonic") and math.isfinite(r["near_diatonic"].get("aware_ratio", float("nan")))
            else float("nan") for r in sweep
        ])

        add_ref_lines(ax, exp_e)
        ax.axhline(1.0, color="black", linestyle=":", linewidth=0.8,
                   label="No advantage (1×)")
        ax.plot(x, ag_ratio, "o--", color=pal["agnostic"], markersize=4,
                label="Agnostic (all)", linewidth=1.3)
        ax.plot(x, aw_ratio, "s-",  color=pal["mscale"],   markersize=4,
                label="Aware (all)", linewidth=1.6)
        nd_mask = np.isfinite(nd_ratio)
        if nd_mask.any():
            ax.plot(x[nd_mask], nd_ratio[nd_mask], "^-", color=pal["nd"],
                    markersize=4, label=f"Aware near-diatonic (n={nd_N})",
                    linewidth=1.3)

        set_xticks(ax, x, sparse)
        ax.set_xlabel("Privacy budget  ε", fontsize=9)
        ax.set_ylabel("Acc@1 ratio  M-Scale / M-Random", fontsize=8)
        ax.set_title(pal["label"], fontsize=10, fontweight="bold")
        ax.legend(fontsize=7)
        ax.set_ylim(bottom=0)
        ax.grid(True, alpha=0.2)

    fig.suptitle("Re-identification ratio  M-Scale / M-Random  vs.  ε",
                 fontsize=11, y=1.01)
    plt.tight_layout()
    save(fig, out_name, also_paper=True)


def plot_3panel_acc(corpora: list[str], out_name: str) -> None:
    """Side-by-side Acc@1 panels, one per corpus."""
    fig, axes = plt.subplots(1, len(corpora),
                             figsize=(5 * len(corpora), 3.5), sharey=False)
    if len(corpora) == 1:
        axes = [axes]

    for ax, corpus in zip(axes, corpora):
        pal   = PALETTES[corpus]
        data  = load(corpus)
        sweep = data["epsilon_sweep"]
        x, exp_e, sparse = get_x(sweep)
        N     = data["N"]
        chance = 100.0 / N

        aw_s    = np.array([r["aware"]["aware_acc1_scale"]    for r in sweep])
        aw_r    = np.array([r["aware"]["aware_acc1_random"]   for r in sweep])
        aw_s_lo = np.array([r["aware"]["aware_ci1_scale"][0]  for r in sweep])
        aw_s_hi = np.array([r["aware"]["aware_ci1_scale"][1]  for r in sweep])
        aw_r_lo = np.array([r["aware"]["aware_ci1_random"][0] for r in sweep])
        aw_r_hi = np.array([r["aware"]["aware_ci1_random"][1] for r in sweep])

        add_ref_lines(ax, exp_e)
        plot_eb(ax, x, aw_s*100, aw_s_lo*100, aw_s_hi*100,
                pal["mscale"],  "s-",  "M-Scale")
        plot_eb(ax, x, aw_r*100, aw_r_lo*100, aw_r_hi*100,
                pal["mrandom"], "o--", "M-Random")
        ax.axhline(chance, color="gray", linestyle=":", linewidth=0.8,
                   label=f"Chance ({chance:.1f}%)")

        set_xticks(ax, x, sparse)
        ax.set_xlabel("Privacy budget  ε", fontsize=9)
        ax.set_ylabel("Acc@1 (%)", fontsize=9)
        ax.set_title(pal["label"], fontsize=10, fontweight="bold")
        ax.legend(fontsize=7)
        ax.set_ylim(bottom=-0.2)
        ax.grid(True, alpha=0.2)

    fig.suptitle("Re-identification accuracy vs. ε  (mechanism-aware attacker)",
                 fontsize=11, y=1.01)
    plt.tight_layout()
    save(fig, out_name, also_paper=True)


def plot_cross_dataset(corpora: list[str], out_name: str,
                       series: str = "aware") -> None:
    """
    Overlay ratio curves for multiple corpora on one plot.
    series: 'aware' | 'agnostic' | 'both'
    """
    fmts = {"europa": "s-", "meertens": "o--", "symbtr": "^-", "jazz": "D:"}

    fig, ax = plt.subplots(figsize=FIGSIZE)

    ref_sweep = load(corpora[0])["epsilon_sweep"]
    _, exp_e_ref, sparse_ref = get_x(ref_sweep)
    x_ref = np.arange(len(ref_sweep))

    add_ref_lines(ax, exp_e_ref)
    ax.axhline(1.0, color="black", linestyle=":", linewidth=0.8,
               label="No advantage (1×)")

    for corpus in corpora:
        pal   = PALETTES[corpus]
        sweep = load(corpus)["epsilon_sweep"]
        x     = np.arange(len(sweep))

        if series in ("aware", "both"):
            aw = np.array([r.get("aware_ratio", float("nan")) for r in sweep])
            lbl = pal["label"] + (" (aware)" if series == "both" else "")
            ax.plot(x, aw, fmts[corpus], color=pal["repr"],
                    markersize=5, linewidth=1.5, label=lbl)

        if series in ("agnostic", "both"):
            ag = np.array([r.get("agnostic_ratio", float("nan")) for r in sweep])
            lbl = pal["label"] + " (agnostic)"
            ax.plot(x, ag, fmts[corpus], color=pal["agnostic"],
                    markersize=4, linewidth=1.2, linestyle="--", label=lbl)

    set_xticks(ax, x_ref, sparse_ref)
    ax.set_xlabel("Privacy budget  ε", fontsize=9)
    ax.set_ylabel("Acc@1 ratio  M-Scale / M-Random", fontsize=9)
    title_corpora = ", ".join(PALETTES[c]["label"] for c in corpora)
    ax.set_title(f"Cross-dataset ratio ({series} attacker)\n{title_corpora}",
                 fontsize=9)
    ax.legend(fontsize=8)
    ax.set_ylim(bottom=0)
    plt.tight_layout()
    save(fig, out_name, also_paper=True)


# ── Menu / CLI ────────────────────────────────────────────────────────────────

MENU = """
+--------------------------------------------------------------+
|        PLOTTING PIPELINE -- choose what to generate          |
+--------------------------------------------------------------+
|  Per-corpus ratio plots (appendix)                           |
|    1  Europa   -- ratio vs eps                               |
|    2  Meertens -- ratio vs eps                               |
|    3  SymbTr   -- ratio vs eps                               |
|    4  Jazz     -- ratio vs eps                               |
|                                                              |
|  Per-corpus Acc@1 plots (appendix)                           |
|    5  Europa   -- Acc@1 vs eps                               |
|    6  Meertens -- Acc@1 vs eps                               |
|    7  SymbTr   -- Acc@1 vs eps                               |
|    8  Jazz     -- Acc@1 vs eps                               |
|                                                              |
|  Multi-corpus panels (main paper)                            |
|    9  3-panel ratio : Europa / Meertens / SymbTr             |
|   10  3-panel Acc@1 : Europa / Meertens / SymbTr             |
|   11  Cross-dataset : all 4 corpora (aware ratio)            |
|                                                              |
|  Custom overlays                                             |
|   12  Custom ratio overlay  -- you choose corpora + series   |
|   13  Custom Acc@1 overlay  -- you choose corpora            |
|                                                              |
|   all  Generate everything                                   |
+--------------------------------------------------------------+
"""

CORPUS_MAP = {
    "1": ("europa",   "ratio"), "2": ("meertens", "ratio"),
    "3": ("symbtr",   "ratio"), "4": ("jazz",     "ratio"),
    "5": ("europa",   "acc"),   "6": ("meertens", "acc"),
    "7": ("symbtr",   "acc"),   "8": ("jazz",     "acc"),
}


def run_choice(choice: str) -> None:
    if choice in CORPUS_MAP:
        corpus, kind = CORPUS_MAP[choice]
        if kind == "ratio":
            plot_ratio_single(corpus)
        else:
            plot_acc_single(corpus)

    elif choice == "9":
        plot_3panel_ratio(["europa", "meertens", "symbtr"],
                          "fig_ext_ratio_vs_eps.pdf")

    elif choice == "10":
        plot_3panel_acc(["europa", "meertens", "symbtr"],
                        "fig_ext_acc_vs_eps.pdf")

    elif choice == "11":
        plot_cross_dataset(["europa", "meertens", "symbtr", "jazz"],
                           "fig_ext_comparison_jazz.pdf", series="aware")

    elif choice == "12":
        print("\nAvailable corpora: europa, meertens, symbtr, jazz")
        raw = input("Enter corpora (space-separated, e.g. europa jazz): ").strip()
        corpora = [c.strip() for c in raw.split() if c.strip() in CORPUS_MAP.values() or c.strip() in CORPUS_NAMES]
        if not corpora:
            print("  No valid corpora entered.")
            return
        print("Series options: aware / agnostic / both")
        series = input("Series [aware]: ").strip() or "aware"
        fname  = input("Output filename [fig_custom_ratio.pdf]: ").strip() or "fig_custom_ratio.pdf"
        plot_cross_dataset(corpora, fname, series=series)

    elif choice == "13":
        print("\nAvailable corpora: europa, meertens, symbtr, jazz")
        raw = input("Enter corpora (space-separated): ").strip()
        corpora = [c.strip() for c in raw.split() if c.strip() in CORPUS_NAMES]
        if not corpora:
            print("  No valid corpora entered.")
            return
        fname = input("Output filename [fig_custom_acc.pdf]: ").strip() or "fig_custom_acc.pdf"
        if len(corpora) == 1:
            plot_acc_single(corpora[0], fname)
        else:
            plot_3panel_acc(corpora, fname)

    else:
        print(f"  Unknown choice: {choice}")


def main():
    choices = sys.argv[1:]

    if not choices:
        print(MENU)
        raw = input("Enter plot numbers (space-separated) or 'all': ").strip()
        choices = raw.split()

    if "all" in choices:
        choices = [str(i) for i in range(1, 12)]  # 12/13 require interactive input

    print(f"\nGenerating {len(choices)} plot(s)...")
    for c in choices:
        print(f"\n> Plot {c}")
        run_choice(c)

    print("\nDone.")


if __name__ == "__main__":
    main()
