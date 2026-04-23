"""
fig2_ratio_vs_eps.py
====================
Figure 2: Acc@1 ratio M-Scale / M-Random vs. privacy budget.

Shows three series:
  - Agnostic attacker, all 200 melodies
  - Aware attacker, all 200 melodies
  - Aware attacker, near-diatonic subset (chi < 0.05, n=73)

Hardcoded from verified experimental results (run_table1.py, 30 trials).

Usage
-----
  cd code/
  python figures/fig2_ratio_vs_eps.py
"""
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

OUT_DIR = Path(__file__).resolve().parent.parent / "output"
OUT_DIR.mkdir(exist_ok=True)

# Verified ratio values (M-Scale Acc@1 / M-Random Acc@1)
# eps:           ln3    ln7    ln12
ratio_agnostic   = [0.58,  2.52,  3.32]
ratio_aware_all  = [7.00,  8.44,  4.07]
ratio_aware_nd   = [1.83,  4.79,  7.10]   # near-diatonic subset (n=73)

eps_labels = [r'$\ln 3$', r'$\ln 7$', r'$\ln 12$']
x_pos      = [0, 1, 2]

fig, ax = plt.subplots(figsize=(5, 3.5))

ax.plot(x_pos, ratio_agnostic,  'o--', color='gray',     label='Agnostic (all 200)')
ax.plot(x_pos, ratio_aware_all, 's-',  color='tomato',   label='Aware (all 200)')
ax.plot(x_pos, ratio_aware_nd,  '^-',  color='seagreen',
        label=r'Aware (near-diatonic, $n=73$)')
ax.axhline(1.0, color='black', linestyle=':', linewidth=0.8,
           label=r'No advantage ($1\times$)')

ax.set_xticks(x_pos)
ax.set_xticklabels(eps_labels)
ax.set_xlabel(r'Privacy budget $\varepsilon$')
ax.set_ylabel(r'Acc@1 ratio M-Scale / M-Random')
ax.set_title(r'Re-identification ratio: M-Scale vs.\ M-Random')
ax.legend(fontsize=8)
ax.set_ylim(0, 10)
plt.tight_layout()

out = OUT_DIR / "fig2_ratio_vs_eps.pdf"
plt.savefig(out, bbox_inches='tight')
print(f"Saved {out}")
