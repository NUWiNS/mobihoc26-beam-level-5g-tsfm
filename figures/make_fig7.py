"""Regenerate Figure 7 (covariate effect) with the legend outside the axes.

Same data, style and figsize as generate_figures.py (Figure 7). Changes:
  - legend moved above the axes (replaces the title, which duplicated the caption)
  - improvement labels placed directly above the covariate bars (no leader lines)
  - a ~0% change is printed as "0%" instead of "−-0%"
"""
import sys as _sys
from pathlib import Path as _P
_sys.path.insert(0, str(_P(__file__).resolve().parent.parent))
from paths import ROOT, GOTSF_DIR, IRISH_DIR, RESULTS_DIR, FIG_DIR
import sys
import numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams.update({
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "font.family": "serif",
    "font.size": 9,
    "axes.labelsize": 10,
    "axes.titlesize": 10,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "legend.fontsize": 8,
    "figure.dpi": 300,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.05,
})

results = pd.read_csv(sys.argv[1])
out = sys.argv[2]

cov_pairs = [
    ("Chronos2-Uni", "Chronos2-Cov", "Chronos-2"),
    ("NHITS", "NHITS+Cov", "N-HiTS"),
    ("TiDE", "TiDE+Cov", "TiDE"),
]
w6_data = results[results["test_week"] == "5w-6w"]

fig, ax = plt.subplots(figsize=(3.76, 2.365))
x_pos = np.arange(len(cov_pairs))
width = 0.3

uni_vals = [w6_data[w6_data["model"] == p[0]]["MASE"].mean() for p in cov_pairs]
cov_vals = [w6_data[w6_data["model"] == p[1]]["MASE"].mean() for p in cov_pairs]
labels_x = [p[2] for p in cov_pairs]
print("uni", np.round(uni_vals, 3), "cov", np.round(cov_vals, 3))

ax.bar(x_pos - width/2, uni_vals, width, label="Univariate", color="#fc8d62", alpha=0.85)
ax.bar(x_pos + width/2, cov_vals, width, label="+ Covariates", color="#1b9e77", alpha=0.85)

for i in range(len(cov_pairs)):
    improv = (uni_vals[i] - cov_vals[i]) / uni_vals[i] * 100
    label = "0%" if abs(improv) < 0.5 else f"−{improv:.0f}%"
    ax.text(i + width/2, cov_vals[i] + 0.012, label, ha="center", va="bottom",
            fontsize=7, color="#1b9e77")

ax.set_xticks(x_pos)
ax.set_xticklabels(labels_x)
ax.set_ylabel("MASE (Week 6)")
ax.set_ylim(0, 0.85)
ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=2, frameon=False,
          handlelength=1.2, columnspacing=1.5, borderaxespad=0.2)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

fig.savefig(out + ".pdf")
fig.savefig(out + ".png", dpi=300)
