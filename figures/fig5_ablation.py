"""
fig5_ablation.py
================
Figure 5: Ablation — key-balanced synthetic dataset.
Compares M-RANDOM vs M-SCALE on a dataset with uniform key distribution.

Hardcoded from verified experimental results (run_ablation.py, 30 trials,
240 melodies = 20 seed melodies x 12 transpositions).

Usage
-----
  cd code/
  python figures/fig5_ablation.py
"""
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

OUT_DIR = Path(__file__).resolve().parent.parent / "output"
OUT_DIR.mkdir(exist_ok=True)

DPI = 300
C_RAND  = "#2166ac"
C_SCALE = "#d6604d"
EPS_LABELS = [r"$\ln\,3$", r"$\ln\,7$", r"$\ln\,12$"]

# Verified data from run_ablation.py (alpha=1.66, agnostic attacker)
SYN_RAND  = np.array([0.1, 0.4, 1.2])
SYN_SCALE = np.array([0.2, 1.2, 2.8])
SYN_CI_R  = np.array([0.1, 0.2, 0.3])
SYN_CI_S  = np.array([0.1, 0.3, 0.4])
SYN_RATIO = np.array([1.75, 2.02, 1.74])

X = np.arange(3)

fig, ax = plt.subplots(figsize=(7.0, 4.4))

ax.plot(X, SYN_RAND,  "o-", color=C_RAND,  lw=2.2, ms=7,
        label="M-Random", zorder=4)
ax.plot(X, SYN_SCALE, "s-", color=C_SCALE, lw=2.2, ms=7,
        label="M-Scale",  zorder=4)

ax.fill_between(X, SYN_RAND  - SYN_CI_R, SYN_RAND  + SYN_CI_R,
                color=C_RAND,  alpha=0.14, zorder=3)
ax.fill_between(X, SYN_SCALE - SYN_CI_S, SYN_SCALE + SYN_CI_S,
                color=C_SCALE, alpha=0.14, zorder=3)

for xi, r, s, ratio in zip(X, SYN_RAND, SYN_SCALE, SYN_RATIO):
    ax.annotate(
        f"{ratio:.2f}x",
        xy=(xi, max(r, s) + 0.5),
        ha="center", va="bottom",
        fontsize=9, color="#555555", fontweight="bold",
    )

ax.axhline(0.5, color="#aaaaaa", lw=1.0, ls=":", zorder=2,
           label="Chance (0.5%)")

ax.set_xticks(X)
ax.set_xticklabels(EPS_LABELS, fontsize=12)
ax.set_xlabel(r"Privacy parameter $\varepsilon$", fontsize=12)
ax.set_ylabel("Retrieval accuracy @1 (%)", fontsize=12)
ax.set_title(
    "Ablation: gap reduced on key-balanced synthetic dataset\n"
    r"20 diatonic melodies $\times$ 12 transpositions = 240 tracks  "
    "(30 trials; shaded = 95% CI)",
    fontsize=10.5, pad=8,
)
ax.legend(fontsize=10, loc="upper left")
ax.set_ylim(bottom=-0.3)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.yaxis.grid(True, alpha=0.22, zorder=0)

fig.text(
    0.5, -0.03,
    "Ratios 1.74-2.02x (vs. up to 8.44x on Europa), confirming M-Scale's large "
    "gap is driven by key-distribution skew in the corpus.",
    ha="center", fontsize=8.5, color="#444444",
    bbox=dict(boxstyle="round,pad=0.35", fc="#f9f9f9", ec="#cccccc", lw=0.8),
)

plt.tight_layout()

out = OUT_DIR / "fig5_ablation.pdf"
plt.savefig(out, bbox_inches="tight", dpi=DPI)
out2 = OUT_DIR / "fig5_ablation.png"
plt.savefig(out2, bbox_inches="tight", dpi=DPI)
plt.close()
print(f"Saved {out}")
