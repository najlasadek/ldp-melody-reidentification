"""
05_plot_all.py
==============
Generate all figures from the extended epsilon sweep results.

Figures produced:
  1. figures/fig_eps_sweep_europa.pdf    — Europa 200: Acc@1 vs ε (aware + agnostic)
  2. figures/fig_eps_sweep_meertens.pdf  — Meertens: same layout
  3. figures/fig_eps_sweep_symbtr.pdf    — SymbTr: same layout
  4. figures/fig_comparison_all.pdf      — M-Scale/M-Random aware ratio vs ε, all datasets

Each figure uses the 'tableau-colorblind10' palette (ISMIR guideline).
Error bars are Wilson 95% CIs.
Paper reference points (ln3, ln7, ln12) are marked with dotted vertical lines.

Run from experiments_extended/ directory:
    python scripts/05_plot_all.py
"""

from __future__ import annotations

import json
import sys
from math import exp, log
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np

# Use colorblind-friendly palette (ISMIR guideline)
try:
    plt.style.use("tableau-colorblind10")
except OSError:
    pass  # Fall back to default if not available

SCRIPT_DIR = Path(__file__).parent
BASE_DIR = SCRIPT_DIR.parent
RESULTS_DIR = BASE_DIR / "results"
FIGURES_DIR = BASE_DIR / "figures"
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

# Reference epsilon values (paper)
REF_EPS = [log(3), log(7), log(12)]
REF_LABELS = [r"$\ln 3$", r"$\ln 7$", r"$\ln 12$"]


# =============================================================================
# Load helpers
# =============================================================================

def load_results(path: Path) -> dict | None:
    if not path.exists():
        print(f"  Not found: {path} (skip)")
        return None
    with open(path, "r") as fh:
        return json.load(fh)


def extract_sweep_arrays(sweep: list[dict]) -> dict:
    """Extract numpy arrays for plotting from epsilon sweep results."""
    eps_vals = np.array([r["epsilon"] for r in sweep])
    exp_eps  = np.array([r["exp_epsilon"] for r in sweep])

    ag_scale  = np.array([r["agnostic"]["acc1_scale"]          for r in sweep])
    ag_rnd    = np.array([r["agnostic"]["acc1_random"]         for r in sweep])
    aw_scale  = np.array([r["aware"]["aware_acc1_scale"]       for r in sweep])
    aw_rnd    = np.array([r["aware"]["aware_acc1_random"]      for r in sweep])
    ag_ratio  = np.array([r["agnostic_ratio"]                  for r in sweep])
    aw_ratio  = np.array([r["aware_ratio"]                     for r in sweep])

    # Wilson CIs
    ag_scale_ci_lo = np.array([r["agnostic"]["ci1_scale"][0]  for r in sweep])
    ag_scale_ci_hi = np.array([r["agnostic"]["ci1_scale"][1]  for r in sweep])
    aw_scale_ci_lo = np.array([r["aware"]["aware_ci1_scale"][0] for r in sweep])
    aw_scale_ci_hi = np.array([r["aware"]["aware_ci1_scale"][1] for r in sweep])
    ag_rnd_ci_lo   = np.array([r["agnostic"]["ci1_random"][0] for r in sweep])
    ag_rnd_ci_hi   = np.array([r["agnostic"]["ci1_random"][1] for r in sweep])
    aw_rnd_ci_lo   = np.array([r["aware"]["aware_ci1_random"][0] for r in sweep])
    aw_rnd_ci_hi   = np.array([r["aware"]["aware_ci1_random"][1] for r in sweep])

    return {
        "eps": eps_vals,
        "exp_eps": exp_eps,
        "ag_scale": ag_scale, "ag_rnd": ag_rnd,
        "aw_scale": aw_scale, "aw_rnd": aw_rnd,
        "ag_ratio": ag_ratio, "aw_ratio": aw_ratio,
        "ag_scale_lo": ag_scale_ci_lo, "ag_scale_hi": ag_scale_ci_hi,
        "aw_scale_lo": aw_scale_ci_lo, "aw_scale_hi": aw_scale_ci_hi,
        "ag_rnd_lo":   ag_rnd_ci_lo,   "ag_rnd_hi":   ag_rnd_ci_hi,
        "aw_rnd_lo":   aw_rnd_ci_lo,   "aw_rnd_hi":   aw_rnd_ci_hi,
    }


# =============================================================================
# Per-dataset figure: Acc@1 vs ε (aware + agnostic, all + near-diatonic)
# =============================================================================

def plot_dataset_sweep(data: dict, title: str, out_path: Path) -> None:
    """
    4-panel figure:
      Top-left:  Agnostic — Acc@1 for M-Scale and M-Random
      Top-right: Aware    — Acc@1 for M-Scale and M-Random
      Bottom-left:  Agnostic ratio (Scale/Random)
      Bottom-right: Aware ratio (Scale/Random)
    """
    sweep = data.get("epsilon_sweep", [])
    if not sweep:
        print(f"  No sweep data for {title}")
        return

    arr = extract_sweep_arrays(sweep)
    x = arr["eps"]      # epsilon values for x-axis
    x_labels = [f"ln({e:.0f})" for e in arr["exp_eps"]]

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    fig.suptitle(title, fontsize=14, fontweight="bold")

    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    c_scale = colors[0]
    c_rnd   = colors[1]

    def add_ref_lines(ax):
        for ref in REF_EPS:
            ax.axvline(ref, color="gray", linestyle=":", linewidth=0.8, alpha=0.6)

    def add_ci(ax, x, y, lo, hi, color):
        ax.fill_between(x, lo, hi, alpha=0.15, color=color)

    # Top-left: Agnostic Acc@1
    ax = axes[0, 0]
    ax.plot(x, arr["ag_scale"] * 100, "o-", color=c_scale, label="M-Scale", markersize=4)
    ax.plot(x, arr["ag_rnd"]   * 100, "s--", color=c_rnd,  label="M-Random", markersize=4)
    add_ci(ax, x, arr["ag_scale"]*100, arr["ag_scale_lo"]*100, arr["ag_scale_hi"]*100, c_scale)
    add_ci(ax, x, arr["ag_rnd"]*100, arr["ag_rnd_lo"]*100, arr["ag_rnd_hi"]*100, c_rnd)
    add_ref_lines(ax)
    ax.set_title("Agnostic attacker — Acc@1")
    ax.set_ylabel("Acc@1 (%)")
    ax.legend(fontsize=9)
    ax.set_xticks(x[::2])
    ax.set_xticklabels(x_labels[::2], rotation=45, fontsize=7)
    ax.set_ylim(0, 100)
    ax.grid(True, alpha=0.3)

    # Top-right: Aware Acc@1
    ax = axes[0, 1]
    ax.plot(x, arr["aw_scale"] * 100, "o-", color=c_scale, label="M-Scale", markersize=4)
    ax.plot(x, arr["aw_rnd"]   * 100, "s--", color=c_rnd,  label="M-Random", markersize=4)
    add_ci(ax, x, arr["aw_scale"]*100, arr["aw_scale_lo"]*100, arr["aw_scale_hi"]*100, c_scale)
    add_ci(ax, x, arr["aw_rnd"]*100, arr["aw_rnd_lo"]*100, arr["aw_rnd_hi"]*100, c_rnd)
    add_ref_lines(ax)
    ax.set_title("Aware attacker — Acc@1")
    ax.set_ylabel("Acc@1 (%)")
    ax.legend(fontsize=9)
    ax.set_xticks(x[::2])
    ax.set_xticklabels(x_labels[::2], rotation=45, fontsize=7)
    ax.set_ylim(0, 100)
    ax.grid(True, alpha=0.3)

    # Bottom-left: Agnostic ratio
    ax = axes[1, 0]
    ax.plot(x, arr["ag_ratio"], "o-", color=colors[2], markersize=4)
    ax.axhline(1.0, color="black", linestyle="--", linewidth=0.8, alpha=0.7, label="No advantage")
    add_ref_lines(ax)
    ax.set_title("Agnostic ratio (M-Scale / M-Random)")
    ax.set_ylabel("Acc@1 ratio")
    ax.set_xlabel("Privacy budget ε")
    ax.legend(fontsize=9)
    ax.set_xticks(x[::2])
    ax.set_xticklabels(x_labels[::2], rotation=45, fontsize=7)
    ax.grid(True, alpha=0.3)

    # Bottom-right: Aware ratio
    ax = axes[1, 1]
    ax.plot(x, arr["aw_ratio"], "o-", color=colors[3], markersize=4)
    ax.axhline(1.0, color="black", linestyle="--", linewidth=0.8, alpha=0.7, label="No advantage")
    add_ref_lines(ax)
    ax.set_title("Aware ratio (M-Scale / M-Random)")
    ax.set_ylabel("Acc@1 ratio")
    ax.set_xlabel("Privacy budget ε")
    ax.legend(fontsize=9)
    ax.set_xticks(x[::2])
    ax.set_xticklabels(x_labels[::2], rotation=45, fontsize=7)
    ax.grid(True, alpha=0.3)

    # Reference line annotations
    for ax in axes.flat:
        for ref, lbl in zip(REF_EPS, REF_LABELS):
            ax.annotate(lbl, xy=(ref, ax.get_ylim()[0]),
                        xytext=(ref, ax.get_ylim()[0]),
                        fontsize=6, color="gray", ha="center", va="bottom")

    plt.tight_layout()
    fig.savefig(out_path, format="pdf", bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out_path}")


# =============================================================================
# Combined comparison figure: Aware ratio vs ε for all datasets
# =============================================================================

def plot_comparison(datasets: dict[str, dict | None], out_path: Path) -> None:
    """
    Single plot: M-Scale/M-Random aware Acc@1 ratio vs ε for all datasets.
    Datasets is a dict of {label: result_dict}.
    """
    fig, ax = plt.subplots(figsize=(9, 5))
    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]

    markers = ["o", "s", "^", "D"]
    linestyles = ["-", "--", "-."]

    plotted = 0
    for i, (label, data) in enumerate(datasets.items()):
        if data is None:
            continue
        sweep = data.get("epsilon_sweep", [])
        if not sweep:
            continue
        arr = extract_sweep_arrays(sweep)
        x = arr["eps"]
        y = arr["aw_ratio"]
        ax.plot(x, y,
                marker=markers[plotted % len(markers)],
                linestyle=linestyles[plotted % len(linestyles)],
                color=colors[plotted % len(colors)],
                label=label, markersize=5, linewidth=1.5)
        plotted += 1

    # Reference lines for paper epsilon values
    for ref, lbl in zip(REF_EPS, REF_LABELS):
        ax.axvline(ref, color="gray", linestyle=":", linewidth=0.8, alpha=0.6)
        ax.text(ref, ax.get_ylim()[1] if ax.get_ylim()[1] > 1 else 5,
                lbl, fontsize=8, color="gray", ha="center", va="top")

    ax.axhline(1.0, color="black", linestyle="--", linewidth=1.0, alpha=0.7,
               label="No advantage (ratio = 1)")

    ax.set_xlabel("Privacy budget ε", fontsize=12)
    ax.set_ylabel("Acc@1 ratio: M-Scale / M-Random (aware attacker)", fontsize=11)
    ax.set_title("Re-identification Advantage of Structure-Aware LDP\nAcross Datasets — Aware Attacker",
                 fontsize=12, fontweight="bold")
    ax.legend(fontsize=10, loc="upper left")
    ax.grid(True, alpha=0.3)

    # Custom x-tick labels
    all_eps = sorted(set(
        r["epsilon"]
        for data in datasets.values()
        if data
        for r in data.get("epsilon_sweep", [])
    ))
    ax.set_xticks(all_eps[::2])
    ax.set_xticklabels(
        [f"ln({exp(e):.0f})" for e in all_eps[::2]],
        rotation=45, fontsize=8
    )

    plt.tight_layout()
    fig.savefig(out_path, format="pdf", bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out_path}")


# =============================================================================
# Also plot near-diatonic results on a separate panel for the comparison
# =============================================================================

def plot_comparison_nd(datasets: dict[str, dict | None], out_path: Path) -> None:
    """
    Two-panel comparison: (left) all melodies, (right) near-diatonic subset.
    Only shows aware ratio.
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharey=False)
    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    markers = ["o", "s", "^", "D"]

    for panel, (ax, subset_key, panel_title) in enumerate(zip(
        axes,
        [None, "near_diatonic"],
        ["All melodies", "Near-diatonic (χ < 0.05)"],
    )):
        plotted = 0
        for i, (label, data) in enumerate(datasets.items()):
            if data is None:
                continue
            sweep = data.get("epsilon_sweep", [])
            if not sweep:
                continue

            if subset_key is None:
                x = np.array([r["epsilon"] for r in sweep])
                y = np.array([r["aware_ratio"] for r in sweep])
                nd_n = data.get("near_diatonic_N", "?")
                lbl  = f"{label} (N={data['N']})"
            else:
                # Near-diatonic subset
                x_list, y_list = [], []
                for r in sweep:
                    nd = r.get("near_diatonic")
                    if nd:
                        x_list.append(r["epsilon"])
                        y_list.append(nd["aware_ratio"])
                if not x_list:
                    continue
                x = np.array(x_list)
                y = np.array(y_list)
                nd_n = sweep[0].get("near_diatonic_N", "?") if sweep else "?"
                lbl  = f"{label} (N_nd={nd_n})"

            ax.plot(x, y,
                    marker=markers[plotted % len(markers)],
                    color=colors[plotted % len(colors)],
                    label=lbl, markersize=5, linewidth=1.5)
            plotted += 1

        for ref, lbl2 in zip(REF_EPS, REF_LABELS):
            ax.axvline(ref, color="gray", linestyle=":", linewidth=0.8, alpha=0.6)
        ax.axhline(1.0, color="black", linestyle="--", linewidth=1.0, alpha=0.7)
        ax.set_xlabel("Privacy budget ε", fontsize=11)
        ax.set_ylabel("Aware ratio (M-Scale / M-Random)", fontsize=10)
        ax.set_title(panel_title, fontsize=12)
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)

        # x-tick labels
        all_eps = sorted(set(
            r["epsilon"]
            for data in datasets.values()
            if data
            for r in data.get("epsilon_sweep", [])
        ))
        if all_eps:
            ax.set_xticks(all_eps[::2])
            ax.set_xticklabels(
                [f"ln({exp(e):.0f})" for e in all_eps[::2]],
                rotation=45, fontsize=8
            )

    fig.suptitle("Re-identification Advantage Across Datasets — Aware Attacker",
                 fontsize=13, fontweight="bold")
    plt.tight_layout()
    fig.savefig(out_path, format="pdf", bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out_path}")


# =============================================================================
# Main
# =============================================================================

def main():
    print("Loading results...")
    europa  = load_results(RESULTS_DIR / "europa_eps_sweep.json")
    meertens = load_results(RESULTS_DIR / "meertens_eps_sweep.json")
    symbtr  = load_results(RESULTS_DIR / "symbtr_eps_sweep.json")

    available = sum(1 for d in [europa, meertens, symbtr] if d is not None)
    print(f"Found {available}/3 result files")

    if available == 0:
        print("No results found. Run scripts 02, 03, 04 first.")
        sys.exit(1)

    print("\nGenerating per-dataset figures...")

    if europa:
        plot_dataset_sweep(europa, "Europa 200 (Western Folksongs)",
                           FIGURES_DIR / "fig_eps_sweep_europa.pdf")

    if meertens:
        n = meertens.get("N", "?")
        plot_dataset_sweep(meertens, f"Meertens Tune Collections (N={n})",
                           FIGURES_DIR / "fig_eps_sweep_meertens.pdf")

    if symbtr:
        n = symbtr.get("N", "?")
        plot_dataset_sweep(symbtr, f"SymbTr Turkish Makam (N={n}, semitone-only)",
                           FIGURES_DIR / "fig_eps_sweep_symbtr.pdf")

    print("\nGenerating comparison figures...")

    datasets = {}
    if europa:
        datasets["Europa 200"] = europa
    if meertens:
        n = meertens.get("N", "?")
        datasets[f"Meertens ({n})"] = meertens
    if symbtr:
        n = symbtr.get("N", "?")
        datasets[f"SymbTr ({n})"] = symbtr

    if len(datasets) >= 1:
        plot_comparison(datasets, FIGURES_DIR / "fig_comparison_all.pdf")
        plot_comparison_nd(datasets, FIGURES_DIR / "fig_comparison_nd.pdf")

    print(f"\nAll figures saved to {FIGURES_DIR}/")
    print("\nFigure list:")
    for f in sorted(FIGURES_DIR.glob("*.pdf")):
        print(f"  {f.name}")


if __name__ == "__main__":
    main()
