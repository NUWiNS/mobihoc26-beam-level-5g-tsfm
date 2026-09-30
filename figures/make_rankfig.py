"""Regenerate the Friedman mean-rank figure (paper Fig. 7) with the paper's model names.
Identical to generate_figures.py 'FIGURE 6' except y-tick labels (NHITS->N-HiTS, NBEATSx->N-BEATSx)."""
import sys as _sys
from pathlib import Path as _P
_sys.path.insert(0, str(_P(__file__).resolve().parent.parent))
from paths import ROOT, GOTSF_DIR, IRISH_DIR, RESULTS_DIR, FIG_DIR
import sys
from pathlib import Path
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
plt.rcParams.update({"pdf.fonttype": 42, "ps.fonttype": 42, "font.family": "serif", "font.size": 9,
    "axes.labelsize": 10, "axes.titlesize": 10, "xtick.labelsize": 8, "ytick.labelsize": 8,
    "legend.fontsize": 8, "figure.dpi": 300, "savefig.bbox": "tight", "savefig.pad_inches": 0.05})
COLORS = {"Chronos2-Cov": "#1b9e77", "Chronos2-Uni": "#66c2a5", "Chronos2-XL": "#b2e2ce", "NHITS+Cov": "#d95f02",
          "NHITS": "#fc8d62", "TiDE+Cov": "#e7298a", "TiDE": "#e78ac3", "NBEATSx+Cov": "#7570b3", "DLinear": "#a6761d",
          "PatchTST": "#666666", "Chronos-Bolt": "#1f78b4", "Moirai": "#cab2d6", "TimesFM": "#ff7f00", "ARIMA": "#999999"}
NAME = lambda m: m.replace("NHITS", "N-HiTS").replace("NBEATSx", "N-BEATSx")
results = pd.read_csv(Path(sys.argv[1]) / "per_beam_results.csv"); out = Path(sys.argv[2])
def panel(tw, ax):
    piv = results[results.test_week == tw].pivot_table(index="beam_id", columns="model", values="MASE").dropna(axis=1)
    mr = piv.rank(axis=1, method="average").mean().sort_values(); k, n = len(mr), len(piv)
    q = {14: 3.354}.get(k, 3.5); cd = q * np.sqrt(k * (k + 1) / (6 * n))
    y = np.arange(k); ax.barh(y, mr.values, height=0.6, color=[COLORS.get(m, "#333") for m in mr.index], alpha=0.85, edgecolor="white", linewidth=0.3)
    ax.set_yticks(y); ax.set_yticklabels([NAME(m) for m in mr.index], fontsize=7.5)
    ax.set_xlabel(f"Mean rank (lower = better, CD = {cd:.2f})")
    for i, r in enumerate(mr.values):
        if r - mr.values[0] < cd: ax.text(r + 0.05, i, "*", va="center", fontsize=10, fontweight="bold", color="#1b9e77")
    ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False); ax.invert_yaxis()
    print(tw, k, round(cd, 3), mr.round(2).to_dict())
fig, axes = plt.subplots(1, 2, figsize=(7.5, 3.5))
panel("5w-6w", axes[0]); axes[0].set_title("(a) Week 6 (in-distribution)")
panel("10w-11w", axes[1]); axes[1].set_title("(b) Week 11 (distribution shift)")
plt.tight_layout(); fig.savefig(out / "fig6_nemenyi_cd.pdf"); fig.savefig(out / "fig6_nemenyi_cd.png", dpi=300)
