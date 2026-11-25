"""
fig3_chromatic_fingerprint.py
=============================
Figure 3: Key-signature fingerprint — theory vs. Monte Carlo simulation.

Theory (Proposition 1, paper Appendix):
  E[chi(M-RANDOM(m))] = 535/1716 + (36/143)*f   at eps = ln 7
  E[chi(M-SCALE(m))]  = 0  (exact, all output notes are diatonic)

Simulates both mechanisms at eps=ln7 for a range of input chromatic
fractions f and overlays the theoretical predictions.

Usage
-----
  # from the repository root
  python figures/fig3_chromatic_fingerprint.py
"""
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

OUT_DIR = Path(__file__).resolve().parent.parent / "output"
OUT_DIR.mkdir(exist_ok=True)

# ── constants ────────────────────────────────────────────────────────────────
EPS      = np.log(7)
P        = np.exp(EPS) / (np.exp(EPS) + 6)   # = 7/13
CHROM_PCS = {1, 3, 6, 8, 10}
DIAT_PCS  = {0, 2, 4, 5, 7, 9, 11}
N_NOTES  = 20
N_TRIALS = 8000
RNG      = np.random.default_rng(42)

f_pts = np.linspace(0, 1, 13)
sim_mrand  = []
sim_mscale = []

for f in f_pts:
    out_mrand  = []
    out_mscale = []
    for _ in range(N_TRIALS):
        # Draw melody with chromatic fraction f
        is_chrom = RNG.random(N_NOTES) < f
        notes = np.where(
            is_chrom,
            RNG.choice(list(CHROM_PCS), N_NOTES),
            RNG.choice(list(DIAT_PCS),  N_NOTES),
        )

        # M-RANDOM
        alphabet  = sorted(RNG.choice(12, 7, replace=False))
        alpha_set = set(alphabet)
        chrom_out = 0
        for pc in notes:
            pc = int(pc)
            if pc in alpha_set:
                if RNG.random() < P:
                    out_pc = pc
                else:
                    others = [x for x in alphabet if x != pc]
                    out_pc = int(RNG.choice(others))
            else:
                out_pc = int(RNG.choice(alphabet))
            chrom_out += out_pc in CHROM_PCS
        out_mrand.append(chrom_out / N_NOTES)

        # M-SCALE
        scale     = [0, 2, 4, 5, 7, 9, 11]
        chrom_out = 0
        for pc in notes:
            pc = int(pc)
            if pc in DIAT_PCS:
                if RNG.random() < P:
                    out_pc = pc
                else:
                    others = [x for x in scale if x != pc]
                    out_pc = int(RNG.choice(others))
            else:
                out_pc = int(RNG.choice(scale))
            chrom_out += out_pc in CHROM_PCS
        out_mscale.append(chrom_out / N_NOTES)

    sim_mrand.append(float(np.mean(out_mrand)))
    sim_mscale.append(float(np.mean(out_mscale)))

# Theory curves
f_theory      = np.linspace(0, 1, 200)
theory_mrand  = 535/1716 + (36/143) * f_theory
theory_mscale = np.zeros_like(f_theory)

# ── plot ─────────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(5, 3.5))

ax.plot(f_theory, theory_mrand, '-', color='tomato', linewidth=1.5,
        label=r'M-Random theory: $\frac{535}{1716}+\frac{36}{143}f$')
ax.plot(f_pts, sim_mrand, 'o', color='tomato', markersize=4,
        label='M-Random simulation')

ax.plot(f_theory, theory_mscale, '-', color='steelblue', linewidth=1.5,
        label=r'M-Scale theory: $0$ (exact)')
ax.plot(f_pts, sim_mscale, 's', color='steelblue', markersize=4,
        label='M-Scale simulation')

ax.set_xlabel(r'Input chromatic fraction $f = \chi(\mathbf{m})$')
ax.set_ylabel(r'$\mathbb{E}[\chi(\mathbf{m}^{\prime})]$')
ax.set_title(
    'Key-signature fingerprint: theory vs. simulation' + '\n'
    r'($\varepsilon = \ln 7$, 8000 trials per point)'
)
ax.legend(fontsize=8)
ax.set_xlim(0, 1)
ax.set_ylim(-0.02, 0.65)
plt.tight_layout()

out = OUT_DIR / "fig3_chromatic_fingerprint.pdf"
plt.savefig(out, bbox_inches='tight')
print(f"Saved {out}")
