"""
fig1_acc_vs_eps.py
==================
Figure 1: Re-identification accuracy vs. privacy budget.
Mechanism-aware attacker, all 200 melodies.

Hardcoded from verified experimental results (run_table1.py, 30 trials).

Usage
-----
  cd code/
  python figures/fig1_acc_vs_eps.py
"""
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

OUT_DIR = Path(__file__).resolve().parent.parent / "output"
OUT_DIR.mkdir(exist_ok=True)

# Verified Acc@1 values (%) — mechanism-aware attacker, all 200 melodies
# eps:      ln3    ln7    ln12
mrand  = [0.0,   0.1,   1.0]
mscale = [0.1,   1.3,   4.1]

# Wilson 95% CI bounds (%)
mrand_lo  = [0.00, 0.03, 0.63]
mrand_hi  = [0.20, 0.39, 1.58]
mscale_lo = [0.03, 0.74, 3.25]
mscale_hi = [0.38, 2.11, 5.15]

eps_labels = [r'$\ln 3$', r'$\ln 7$', r'$\ln 12$']
x_pos      = [0, 1, 2]

fig, ax = plt.subplots(figsize=(5, 3.5))

ax.errorbar(x_pos, mrand,
            yerr=[np.array(mrand) - np.array(mrand_lo),
                  np.array(mrand_hi) - np.array(mrand)],
            fmt='o--', color='steelblue', label='M-Random', capsize=4)
ax.errorbar(x_pos, mscale,
            yerr=[np.array(mscale) - np.array(mscale_lo),
                  np.array(mscale_hi) - np.array(mscale)],
            fmt='s-', color='tomato', label='M-Scale', capsize=4)

ax.axhline(0.5, color='gray', linestyle=':', linewidth=0.8,
           label='Random baseline (0.5%)')
ax.set_xticks(x_pos)
ax.set_xticklabels(eps_labels)
ax.set_xlabel(r'Privacy budget $\varepsilon$')
ax.set_ylabel('Acc@1 (%)')
ax.set_title(
    r'Re-identification accuracy vs.\ privacy budget' + '\n'
    r'(mechanism-aware attacker, all 200 melodies)'
)
ax.legend()
ax.set_ylim(-0.1, 6.5)
plt.tight_layout()

out = OUT_DIR / "fig1_acc_vs_eps.pdf"
plt.savefig(out, bbox_inches='tight')
print(f"Saved {out}")
