"""
Generate paper figures (Figure 2 sparsity distribution; legacy versions of others).
Figures:
  1. Characterisation: sparsity distribution histogram
  2. Characterisation: covariate correlation heatmap (per sparsity bin)
  3. Characterisation: temporal patterns (diurnal, example beams)
  4. Results: MASE by sparsity bin (grouped bar)
  5. Results: distribution shift degradation (bar chart)
  6. Results: Nemenyi CD diagram
"""
import sys as _sys
from pathlib import Path as _P
_sys.path.insert(0, str(_P(__file__).resolve().parent.parent))
from paths import ROOT, GOTSF_DIR, IRISH_DIR, RESULTS_DIR, FIG_DIR
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from pathlib import Path
from scipy import stats as sp_stats

# ── Paths ────────────────────────────────────────────────────────────────────
DATA_TRAIN = GOTSF_DIR / "train"
DATA_TEST  = GOTSF_DIR / "test"
RESULTS    = RESULTS_DIR
FIG_DIR.mkdir(parents=True, exist_ok=True)

# ── Style ────────────────────────────────────────────────────────────────────
plt.rcParams.update({
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

COLORS = {
    "Chronos2-Cov": "#1b9e77",
    "Chronos2-Uni": "#66c2a5",
    "Chronos2-XL":  "#b2e2ce",
    "NHITS+Cov":    "#d95f02",
    "NHITS":        "#fc8d62",
    "TiDE+Cov":     "#e7298a",
    "TiDE":         "#e78ac3",
    "NBEATSx+Cov":  "#7570b3",
    "DLinear":      "#a6761d",
    "PatchTST":     "#666666",
    "Chronos-Bolt": "#1f78b4",
    "Moirai":       "#cab2d6",
    "TimesFM":      "#ff7f00",
    "ARIMA":        "#999999",
}

def get_color(model):
    return COLORS.get(model, "#333333")

# ── Load data ────────────────────────────────────────────────────────────────
print("Loading data...")
vol_train = pd.read_csv(DATA_TRAIN / "DLThpVol_train_0w-5w.csv", index_col=0)
time_train = pd.read_csv(DATA_TRAIN / "DLThpTime_train_0w-5w.csv", index_col=0)
prb_train = pd.read_csv(DATA_TRAIN / "DLPRB_train_0w-5w.csv", index_col=0)
mr_train = pd.read_csv(DATA_TRAIN / "MR_number_train_0w-5w.csv", index_col=0)

meta = pd.read_csv(RESULTS / "beam_metadata.csv")
results = pd.read_csv(RESULTS / "per_beam_results.csv")
results = results.merge(meta[["beam_id", "zero_frac", "mean_traffic"]], on="beam_id", how="left")

print(f"  Train: {vol_train.shape[0]} hours × {vol_train.shape[1]} beams")
print(f"  Results: {len(results)} rows, {results['model'].nunique()} models")

# ══════════════════════════════════════════════════════════════════════════════
# FIGURE 1: Sparsity distribution histogram
# ══════════════════════════════════════════════════════════════════════════════
print("\nFigure 1: Sparsity distribution...")
fig, ax = plt.subplots(figsize=(3.4, 2.2))
zf = meta["zero_frac"]
# Use analysis-consistent bins: [0,0.1], (0.1,0.3], (0.3,0.5], (0.5,0.7]
bins_edges = [0, 0.1, 0.3, 0.5, 0.7]
counts, _, patches = ax.hist(zf, bins=bins_edges, edgecolor="white", linewidth=0.5,
                              color="#4292c6", alpha=0.85)
# Annotate group counts and labels
group_counts = pd.cut(zf, bins=[0, 0.3, 0.5, 1.0], labels=["active","transitional","sparse"],
                      include_lowest=True).value_counts()
group_info = [
    (0.0, 0.3, "active", "#2166ac"),
    (0.3, 0.5, "transitional", "#d6604d"),
    (0.5, 0.7, "sparse", "#b2182b"),
]
for lo, hi, label, color in group_info:
    n = int(group_counts[label])
    cx = (lo + hi) / 2
    ax.text(cx, ax.get_ylim()[1] * 0.92, f"{label}\n(n={n})",
            ha="center", va="top", fontsize=6.5, color=color, fontweight="bold")

ax.set_xlabel("Zero fraction (training set)")
ax.set_ylabel("Number of beams")
ax.set_title("Sparsity distribution across 2,880 beams")
ax.axvline(x=0.3, color="red", linestyle="--", linewidth=0.8, alpha=0.7)
ax.axvline(x=0.5, color="red", linestyle="--", linewidth=0.8, alpha=0.7)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
fig.savefig(FIG_DIR / "fig1_sparsity_distribution.pdf")
fig.savefig(FIG_DIR / "fig1_sparsity_distribution.png", dpi=300)
plt.close(fig)
print(f"  Saved: fig1_sparsity_distribution.pdf")

# ══════════════════════════════════════════════════════════════════════════════
# FIGURE 2: Covariate correlation vs sparsity
# ══════════════════════════════════════════════════════════════════════════════
print("\nFigure 2: Covariate correlations by sparsity...")
# Compute per-beam Pearson correlation between DLThpVol and each covariate
cov_names = {"DLThpTime": "DLThpTime", "DLPRB": "DLPRB", "MR_number": "MR_number"}
cov_trains = {"DLThpTime": time_train, "DLPRB": prb_train, "MR_number": mr_train}

corr_rows = []
for beam in meta["beam_id"]:
    y = vol_train[beam].values
    zf_val = meta[meta["beam_id"] == beam]["zero_frac"].values[0]
    for cov_name, cov_df in cov_trains.items():
        x = cov_df[beam].values
        mask = ~(np.isnan(y) | np.isnan(x))
        if mask.sum() > 10:
            r, _ = sp_stats.pearsonr(y[mask], x[mask])
        else:
            r = np.nan
        corr_rows.append({"beam_id": beam, "covariate": cov_name,
                          "correlation": r, "zero_frac": zf_val})
corr_df = pd.DataFrame(corr_rows)

# Bin by sparsity
bins_sp = [0, 0.1, 0.3, 0.5, 1.0]
labels_sp = ["0-10%", "10-30%", "30-50%", "50-100%"]
corr_df["sparsity_bin"] = pd.cut(corr_df["zero_frac"], bins=bins_sp, labels=labels_sp,
                                  include_lowest=True)

fig, ax = plt.subplots(figsize=(3.4, 2.4))
cov_colors = {"DLThpTime": "#1b9e77", "DLPRB": "#d95f02", "MR_number": "#7570b3"}
width = 0.22
x_pos = np.arange(len(labels_sp))
for i, (cov_name, color) in enumerate(cov_colors.items()):
    means = corr_df[corr_df["covariate"] == cov_name].groupby("sparsity_bin", observed=True)["correlation"].mean()
    stds = corr_df[corr_df["covariate"] == cov_name].groupby("sparsity_bin", observed=True)["correlation"].std()
    vals = [means.get(l, 0) for l in labels_sp]
    errs = [stds.get(l, 0) for l in labels_sp]
    ax.bar(x_pos + i*width, vals, width, yerr=errs, label=cov_name,
           color=color, alpha=0.85, capsize=2, error_kw={"linewidth": 0.8})

ax.set_xticks(x_pos + width)
ax.set_xticklabels(labels_sp)
ax.set_xlabel("Sparsity bin (zero fraction)")
ax.set_ylabel("Pearson correlation with DLThpVol")
ax.set_title("(b) Covariate–target correlation by sparsity")
ax.legend(loc="upper right", frameon=False, ncol=1)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
fig.savefig(FIG_DIR / "fig2_covariate_correlation.pdf")
fig.savefig(FIG_DIR / "fig2_covariate_correlation.png", dpi=300)
plt.close(fig)
print(f"  Saved: fig2_covariate_correlation.pdf")

# ══════════════════════════════════════════════════════════════════════════════
# FIGURE 3: Temporal patterns — diurnal profile + example beams
# ══════════════════════════════════════════════════════════════════════════════
print("\nFigure 3: Temporal patterns...")
fig, axes = plt.subplots(1, 2, figsize=(7, 2.5))

# (a) Average diurnal profile: active vs sparse beams
active_beams = meta[meta["zero_frac"] < 0.1]["beam_id"].tolist()
sparse_beams = meta[meta["zero_frac"] > 0.3]["beam_id"].tolist()

# Compute hourly average (hour of day) across the 840-hour training period
vol_active = vol_train[active_beams]
vol_sparse = vol_train[sparse_beams]

# Hour of day (repeating 0-23 over 840 hours = 35 days)
hours = np.arange(vol_train.shape[0]) % 24
active_hourly = vol_active.values
sparse_hourly = vol_sparse.values

active_by_hour = pd.DataFrame({"hour": hours, "val": active_hourly.mean(axis=1)}).groupby("hour")["val"]
sparse_by_hour = pd.DataFrame({"hour": hours, "val": sparse_hourly.mean(axis=1)}).groupby("hour")["val"]

ax = axes[0]
ax.plot(active_by_hour.mean().index, active_by_hour.mean().values, "o-",
        color="#1b9e77", markersize=3, linewidth=1.5, label=f"Active (<10% zeros, n={len(active_beams)})")
ax.fill_between(active_by_hour.mean().index,
                active_by_hour.mean().values - active_by_hour.std().values,
                active_by_hour.mean().values + active_by_hour.std().values,
                alpha=0.15, color="#1b9e77")
ax.plot(sparse_by_hour.mean().index, sparse_by_hour.mean().values, "s-",
        color="#d95f02", markersize=3, linewidth=1.5, label=f"Sparse (>30% zeros, n={len(sparse_beams)})")
ax.fill_between(sparse_by_hour.mean().index,
                sparse_by_hour.mean().values - sparse_by_hour.std().values,
                sparse_by_hour.mean().values + sparse_by_hour.std().values,
                alpha=0.15, color="#d95f02")
ax.set_xlabel("Hour of day")
ax.set_ylabel("Mean DLThpVol")
ax.set_title("(a) Diurnal profile by beam activity")
ax.legend(loc="upper left", frameon=False, fontsize=7)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.set_xticks([0, 6, 12, 18, 23])

# (b) Example time series: one active, one sparse
ax = axes[1]
# Pick a representative active beam (low zero_frac, medium traffic)
active_example = meta.sort_values("zero_frac").iloc[len(active_beams)//2]["beam_id"]
sparse_example = meta[meta["zero_frac"] > 0.35].sort_values("zero_frac").iloc[0]["beam_id"]

t = np.arange(168)  # Show first week (168 hours)
ax.plot(t, vol_train[active_example].values[:168], color="#1b9e77", linewidth=0.8,
        label=f"Active ({active_example})", alpha=0.9)
ax.plot(t, vol_train[sparse_example].values[:168], color="#d95f02", linewidth=0.8,
        label=f"Sparse ({sparse_example})", alpha=0.9)
ax.set_xlabel("Hour (first week)")
ax.set_ylabel("DLThpVol")
ax.set_title("(b) Example beam time series")
ax.legend(loc="upper right", frameon=False, fontsize=7)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

# Adjust layout manually instead of tight_layout to avoid artifacts
fig.subplots_adjust(left=0.08, right=0.96, top=0.92, bottom=0.15, wspace=0.25)
fig.savefig(FIG_DIR / "fig3_temporal_patterns.pdf")
fig.savefig(FIG_DIR / "fig3_temporal_patterns.png", dpi=300)
plt.close(fig)
print(f"  Saved: fig3_temporal_patterns.pdf")

# ══════════════════════════════════════════════════════════════════════════════
# FIGURE 4: MASE by sparsity bin (grouped bar — key models)
# ══════════════════════════════════════════════════════════════════════════════
print("\nFigure 4: MASE by sparsity...")
w6 = results[results["test_week"] == "5w-6w"].copy()
bins_mase = [0, 0.1, 0.3, 0.5, 0.7, 1.0]
labels_mase = ["0–10%", "10–30%", "30–50%", "50–70%", "70–100%"]
w6["sparsity_bin"] = pd.cut(w6["zero_frac"], bins=bins_mase, labels=labels_mase, include_lowest=True)

# Show key models only
key_models = ["Chronos2-Cov", "Chronos2-Uni", "NHITS+Cov", "NHITS", "ARIMA"]
fig, ax = plt.subplots(figsize=(4.5, 2.8))
n_models = len(key_models)
width = 0.15
x_pos = np.arange(len(labels_mase))

for i, model in enumerate(key_models):
    sub = w6[w6["model"] == model]
    means = sub.groupby("sparsity_bin", observed=True)["MASE"].mean()
    vals = [means.get(l, np.nan) for l in labels_mase]
    ax.bar(x_pos + i*width, vals, width, label=model, color=get_color(model),
           alpha=0.85, edgecolor="white", linewidth=0.3)

ax.set_xticks(x_pos + width * (n_models-1) / 2)
ax.set_xticklabels(labels_mase)
ax.set_xlabel("Sparsity bin (training zero fraction)")
ax.set_ylabel("MASE")
ax.set_title("MASE by beam sparsity (GOTSF Week 6)")
ax.axhline(y=1.0, color="gray", linestyle=":", linewidth=0.8, alpha=0.6)
ax.text(4.3, 1.02, "naive baseline", fontsize=6.5, color="gray", ha="right")
ax.legend(loc="center left", bbox_to_anchor=(1.02, 0.5), frameon=False, fontsize=7)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.set_ylim(0, 1.5)

fig.savefig(FIG_DIR / "fig4_mase_by_sparsity.pdf")
fig.savefig(FIG_DIR / "fig4_mase_by_sparsity.png", dpi=300)
plt.close(fig)
print(f"  Saved: fig4_mase_by_sparsity.pdf")

# ══════════════════════════════════════════════════════════════════════════════
# FIGURE 5: Distribution shift degradation
# ══════════════════════════════════════════════════════════════════════════════
print("\nFigure 5: Distribution shift...")
shift_models = ["Chronos2-Cov", "Chronos2-Uni", "Chronos2-XL", "NHITS+Cov", "NHITS",
                "TiDE+Cov", "TiDE", "NBEATSx+Cov", "DLinear", "PatchTST",
                "Chronos-Bolt", "Moirai", "TimesFM", "ARIMA"]

degrad = []
for model in shift_models:
    w6_m = results[(results["model"] == model) & (results["test_week"] == "5w-6w")]["MASE"].mean()
    w11_m = results[(results["model"] == model) & (results["test_week"] == "10w-11w")]["MASE"].mean()
    pct = (w11_m - w6_m) / w6_m * 100 if w6_m > 0 else np.nan
    degrad.append({"model": model, "w6": w6_m, "w11": w11_m, "degradation": pct})
degrad_df = pd.DataFrame(degrad).sort_values("degradation")

fig, ax = plt.subplots(figsize=(4.5, 3.2))
bars = ax.barh(range(len(degrad_df)), degrad_df["degradation"].values,
               color=[get_color(m) for m in degrad_df["model"]], alpha=0.85,
               edgecolor="white", linewidth=0.3)
ax.set_yticks(range(len(degrad_df)))
ax.set_yticklabels(degrad_df["model"].values, fontsize=7.5)
ax.set_xlabel("MASE degradation W6 → W11 (%)")
ax.set_title("Distribution shift resilience")
ax.axvline(x=0, color="gray", linewidth=0.5)
# Annotate values
for i, (_, row) in enumerate(degrad_df.iterrows()):
    ax.text(row["degradation"] + 0.3, i, f"+{row['degradation']:.1f}%", va="center", fontsize=6.5)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.invert_yaxis()

fig.savefig(FIG_DIR / "fig5_distribution_shift.pdf")
fig.savefig(FIG_DIR / "fig5_distribution_shift.png", dpi=300)
plt.close(fig)
print(f"  Saved: fig5_distribution_shift.pdf")

# ══════════════════════════════════════════════════════════════════════════════
# FIGURE 6: Nemenyi Critical Difference diagram
# ══════════════════════════════════════════════════════════════════════════════
print("\nFigure 6: Nemenyi CD diagram...")

def nemenyi_cd_diagram(results_df, test_week, metric="MASE", ax=None):
    """Draw a Nemenyi critical difference diagram."""
    sub = results_df[results_df["test_week"] == test_week]
    pivot = sub.pivot_table(index="beam_id", columns="model", values=metric)
    pivot = pivot.dropna(axis=1)
    ranks = pivot.rank(axis=1, method="average")
    mean_ranks = ranks.mean().sort_values()
    k = len(mean_ranks)
    n = len(pivot)
    # Nemenyi CD
    q_alpha = {3: 2.343, 4: 2.569, 5: 2.728, 6: 2.850, 7: 2.949, 8: 3.031,
               9: 3.102, 10: 3.164, 11: 3.219, 12: 3.268, 13: 3.313, 14: 3.354}
    q = q_alpha.get(k, 3.5)
    cd = q * np.sqrt(k * (k + 1) / (6 * n))

    if ax is None:
        fig_cd, ax = plt.subplots(figsize=(5, 3))
    models = mean_ranks.index.tolist()
    ranks_vals = mean_ranks.values

    # Plot
    y_positions = np.arange(k)
    ax.barh(y_positions, ranks_vals, height=0.6,
            color=[get_color(m) for m in models], alpha=0.85,
            edgecolor="white", linewidth=0.3)
    ax.set_yticks(y_positions)
    ax.set_yticklabels(models, fontsize=7.5)
    ax.set_xlabel(f"Mean rank (lower = better, CD = {cd:.2f})")

    # CD value is shown in the x-axis label
    best_rank = ranks_vals[0]

    # Mark models NOT significantly different from best
    for i, r in enumerate(ranks_vals):
        if r - best_rank < cd:
            ax.text(r + 0.05, i, "*", va="center", fontsize=10,
                    fontweight="bold", color="#1b9e77")

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.invert_yaxis()
    return cd

fig, axes = plt.subplots(1, 2, figsize=(7.5, 3.5))
nemenyi_cd_diagram(results, "5w-6w", ax=axes[0])
axes[0].set_title("(a) Week 6 (in-distribution)")
nemenyi_cd_diagram(results, "10w-11w", ax=axes[1])
axes[1].set_title("(b) Week 11 (distribution shift)")
plt.tight_layout()
fig.savefig(FIG_DIR / "fig6_nemenyi_cd.pdf")
fig.savefig(FIG_DIR / "fig6_nemenyi_cd.png", dpi=300)
plt.close(fig)
print(f"  Saved: fig6_nemenyi_cd.pdf")

# ══════════════════════════════════════════════════════════════════════════════
# FIGURE 7: Covariate effect — univariate vs covariate for each model family
# ══════════════════════════════════════════════════════════════════════════════
print("\nFigure 7: Covariate effect...")
cov_pairs = [
    ("Chronos2-Uni", "Chronos2-Cov", "Chronos-2"),
    ("NHITS", "NHITS+Cov", "N-HiTS"),
    ("TiDE", "TiDE+Cov", "TiDE"),
]
w6_data = results[results["test_week"] == "5w-6w"]

fig, ax = plt.subplots(figsize=(3.4, 2.4))
x_pos = np.arange(len(cov_pairs))
width = 0.3

uni_vals = [w6_data[w6_data["model"] == p[0]]["MASE"].mean() for p in cov_pairs]
cov_vals = [w6_data[w6_data["model"] == p[1]]["MASE"].mean() for p in cov_pairs]
labels_x = [p[2] for p in cov_pairs]

ax.bar(x_pos - width/2, uni_vals, width, label="Univariate", color="#fc8d62", alpha=0.85)
ax.bar(x_pos + width/2, cov_vals, width, label="+ Covariates", color="#1b9e77", alpha=0.85)

# Annotate improvement
for i in range(len(cov_pairs)):
    improv = (uni_vals[i] - cov_vals[i]) / uni_vals[i] * 100
    mid_y = (uni_vals[i] + cov_vals[i]) / 2
    ax.annotate(f"−{improv:.0f}%", xy=(i + width/2, cov_vals[i]),
                xytext=(i + 0.45, mid_y), fontsize=7, color="#1b9e77",
                arrowprops=dict(arrowstyle="-", color="#1b9e77", lw=0.5))

ax.set_xticks(x_pos)
ax.set_xticklabels(labels_x)
ax.set_ylabel("MASE (Week 6)")
ax.set_title("Effect of adding covariates")
ax.legend(loc="upper right", frameon=False)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

fig.savefig(FIG_DIR / "fig7_covariate_effect.pdf")
fig.savefig(FIG_DIR / "fig7_covariate_effect.png", dpi=300)
plt.close(fig)
print(f"  Saved: fig7_covariate_effect.pdf")

print(f"\n✓ All figures saved to {FIG_DIR}")
print(f"  Files: {[f.name for f in sorted(FIG_DIR.glob('*.pdf'))]}")
