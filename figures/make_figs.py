"""Generate paper Figures 3, 4 and 8.

Same data, statistics, style and figsize as generate_figures.py (its Figures 2, 3, 4).
Changes (labels only, no data changes):
  fig2_covariate_correlation: drop the stray "(b)" from the title; last bin labelled
      50-70% (identical beams: max zero fraction is 0.55); shorter y label so it is not clipped.
  fig3_temporal_patterns: legends follow the paper's taxonomy (">30% zeros" group is
      transitional + sparse; the example beam has 35% zeros).
  fig4_mase_by_sparsity: drop the empty 70-100% bin.
Usage: make_figs.py <data_train_dir> <results_dir> <out_dir>
"""
import sys as _sys
from pathlib import Path as _P
_sys.path.insert(0, str(_P(__file__).resolve().parent.parent))
from paths import ROOT, GOTSF_DIR, IRISH_DIR, RESULTS_DIR, FIG_DIR
import sys
from pathlib import Path
import numpy as np, pandas as pd
from scipy import stats as sp_stats
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams.update({
    "pdf.fonttype": 42, "ps.fonttype": 42,
    "font.family": "serif", "font.size": 9, "axes.labelsize": 10, "axes.titlesize": 10,
    "xtick.labelsize": 8, "ytick.labelsize": 8, "legend.fontsize": 8,
    "figure.dpi": 300, "savefig.bbox": "tight", "savefig.pad_inches": 0.05,
})
COLORS = {"Chronos2-Cov": "#1b9e77", "Chronos2-Uni": "#66c2a5", "NHITS+Cov": "#d95f02",
          "NHITS": "#fc8d62", "ARIMA": "#999999"}

DATA_TRAIN, RESULTS, OUT = map(Path, sys.argv[1:4])
vol_train = pd.read_csv(DATA_TRAIN / "DLThpVol_train_0w-5w.csv", index_col=0)
time_train = pd.read_csv(DATA_TRAIN / "DLThpTime_train_0w-5w.csv", index_col=0)
prb_train = pd.read_csv(DATA_TRAIN / "DLPRB_train_0w-5w.csv", index_col=0)
mr_train = pd.read_csv(DATA_TRAIN / "MR_number_train_0w-5w.csv", index_col=0)
meta = pd.read_csv(RESULTS / "beam_metadata.csv")
results = pd.read_csv(RESULTS / "per_beam_results.csv").merge(meta[["beam_id", "zero_frac"]], on="beam_id")

# ---- Covariate correlation by sparsity (paper Figure 3)
cov_trains = {"DLThpTime": time_train, "DLPRB": prb_train, "MR_number": mr_train}
rows = []
zmap = meta.set_index("beam_id").zero_frac
for beam in meta["beam_id"]:
    y = vol_train[beam].values
    for name, df in cov_trains.items():
        x = df[beam].values; m = ~(np.isnan(y) | np.isnan(x))
        r = sp_stats.pearsonr(y[m], x[m])[0] if m.sum() > 10 else np.nan
        rows.append({"covariate": name, "correlation": r, "zero_frac": zmap[beam]})
corr = pd.DataFrame(rows)
labels_sp = ["0-10%", "10-30%", "30-50%", "50-70%"]
corr["bin"] = pd.cut(corr.zero_frac, [0, .1, .3, .5, 1.0], labels=labels_sp, include_lowest=True)
fig, ax = plt.subplots(figsize=(3.67, 2.4))
width = 0.22; x_pos = np.arange(len(labels_sp))
for i, (name, color) in enumerate({"DLThpTime": "#1b9e77", "DLPRB": "#d95f02", "MR_number": "#7570b3"}.items()):
    g = corr[corr.covariate == name].groupby("bin", observed=True)["correlation"]
    ax.bar(x_pos + i * width, [g.mean().get(l, 0) for l in labels_sp], width,
           yerr=[g.std().get(l, 0) for l in labels_sp], label=name, color=color, alpha=0.85,
           capsize=2, error_kw={"linewidth": 0.8})
ax.set_xticks(x_pos + width); ax.set_xticklabels(labels_sp)
ax.set_xlabel("Sparsity bin (zero fraction)"); ax.set_ylabel("Pearson $r$ with DLThpVol")
ax.set_title("Covariate–target correlation by sparsity")
ax.legend(loc="upper right", frameon=False)
ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
fig.savefig(OUT / "fig2_covariate_correlation.pdf"); fig.savefig(OUT / "fig2_covariate_correlation.png", dpi=300); plt.close(fig)
print("cov means:", corr.groupby(["covariate", "bin"], observed=True).correlation.mean().round(2).to_dict())

# ---- Temporal patterns (paper Figure 4)
fig, axes = plt.subplots(1, 2, figsize=(7, 2.5))
active = meta[meta.zero_frac < 0.1].beam_id.tolist(); sparser = meta[meta.zero_frac > 0.3].beam_id.tolist()
hours = np.arange(vol_train.shape[0]) % 24
for beams, color, marker, label in [(active, "#1b9e77", "o-", f"<10% zeros (n={len(active)})"),
                                    (sparser, "#d95f02", "s-", f">30% zeros (n={len(sparser)})")]:
    g = pd.DataFrame({"hour": hours, "val": vol_train[beams].values.mean(axis=1)}).groupby("hour")["val"]
    axes[0].plot(g.mean().index, g.mean().values, marker, color=color, markersize=3, linewidth=1.5, label=label)
    axes[0].fill_between(g.mean().index, g.mean() - g.std(), g.mean() + g.std(), alpha=0.15, color=color)
axes[0].set_xlabel("Hour of day"); axes[0].set_ylabel("Mean DLThpVol")
axes[0].set_title("(a) Diurnal profile by beam activity"); axes[0].legend(loc="upper left", frameon=False, fontsize=7)
axes[0].set_xticks([0, 6, 12, 18, 23])
active_ex = meta.sort_values("zero_frac").iloc[len(active) // 2]["beam_id"]
sparse_ex = meta[meta.zero_frac > 0.35].sort_values("zero_frac").iloc[0]["beam_id"]
zf_ex = zmap[sparse_ex]
t = np.arange(168)
axes[1].plot(t, vol_train[active_ex].values[:168], color="#1b9e77", linewidth=0.8, label=f"Active ({active_ex})", alpha=0.9)
axes[1].plot(t, vol_train[sparse_ex].values[:168], color="#d95f02", linewidth=0.8,
             label=f"{zf_ex:.0%} zeros ({sparse_ex})", alpha=0.9)
axes[1].set_xlabel("Hour (first week)"); axes[1].set_ylabel("DLThpVol")
axes[1].set_title("(b) Example beam time series"); axes[1].legend(loc="upper left", frameon=False, fontsize=7)
for a in axes:
    a.spines["top"].set_visible(False); a.spines["right"].set_visible(False)
fig.subplots_adjust(left=0.08, right=0.96, top=0.92, bottom=0.15, wspace=0.25)
fig.savefig(OUT / "fig3_temporal_patterns.pdf"); fig.savefig(OUT / "fig3_temporal_patterns.png", dpi=300); plt.close(fig)
print("temporal:", len(active), len(sparser), active_ex, sparse_ex, round(zf_ex, 3))

# ---- MASE by sparsity (paper Figure 8)
w6 = results[results.test_week == "5w-6w"].copy()
labels_m = ["0–10%", "10–30%", "30–50%", "50–70%"]
w6["bin"] = pd.cut(w6.zero_frac, [0, .1, .3, .5, .7], labels=labels_m, include_lowest=True)
key = ["Chronos2-Cov", "Chronos2-Uni", "NHITS+Cov", "NHITS", "ARIMA"]
fig, ax = plt.subplots(figsize=(4.5, 2.8)); width = 0.15; x_pos = np.arange(len(labels_m))
for i, m in enumerate(key):
    mm = w6[w6.model == m].groupby("bin", observed=True)["MASE"].mean()
    ax.bar(x_pos + i * width, [mm.get(l, np.nan) for l in labels_m], width, label=m.replace("NHITS", "N-HiTS"), color=COLORS[m],
           alpha=0.85, edgecolor="white", linewidth=0.3)
ax.set_xticks(x_pos + width * (len(key) - 1) / 2); ax.set_xticklabels(labels_m)
ax.set_xlabel("Sparsity bin (training zero fraction)"); ax.set_ylabel("MASE")
ax.set_title("MASE by beam sparsity (GOTSF Week 6)")
ax.axhline(y=1.0, color="gray", linestyle=":", linewidth=0.8, alpha=0.6)
ax.text(len(labels_m) - 0.3, 1.02, "naive baseline", fontsize=6.5, color="gray", ha="right")
ax.legend(loc="center left", bbox_to_anchor=(1.02, 0.5), frameon=False, fontsize=7)
ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False); ax.set_ylim(0, 1.5)
fig.savefig(OUT / "fig4_mase_by_sparsity.pdf"); fig.savefig(OUT / "fig4_mase_by_sparsity.png", dpi=300); plt.close(fig)
print("mase:", w6[w6.model.isin(key)].groupby(["model", "bin"], observed=True).MASE.mean().round(2).to_dict())
