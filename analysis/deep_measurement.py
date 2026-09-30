"""
Beam-level characterization of GOTSF (paper Section 4, Figures 1 and 5).
Generates figures and statistics for:
1. Cell-vs-beam aggregation comparison
2. STL temporal decomposition per sparsity bin
3. Spatial beam correlation analysis (intra-cell, inter-cell, inter-station)
4. Causal analysis of sparsity (what predicts beam sparsity)
"""

import sys as _sys
from pathlib import Path as _P
_sys.path.insert(0, str(_P(__file__).resolve().parent.parent))
from paths import ROOT, GOTSF_DIR, IRISH_DIR, RESULTS_DIR, FIG_DIR
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy import stats
from collections import defaultdict
import warnings
warnings.filterwarnings('ignore')

plt.rcParams.update({
    'font.size': 9,
    'axes.labelsize': 10,
    'axes.titlesize': 10,
    'legend.fontsize': 8,
    'figure.dpi': 150,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
})

DATA_DIR = str(GOTSF_DIR / "train")
RESULTS_DIR = str(RESULTS_DIR)
FIG_DIR = str(FIG_DIR)

# ── Load data ────────────────────────────────────────────────────────────
print("Loading data...")
vol = pd.read_csv(f"{DATA_DIR}/DLThpVol_train_0w-5w.csv", index_col=0)
thp = pd.read_csv(f"{DATA_DIR}/DLThpTime_train_0w-5w.csv", index_col=0)
prb = pd.read_csv(f"{DATA_DIR}/DLPRB_train_0w-5w.csv", index_col=0)
mr  = pd.read_csv(f"{DATA_DIR}/MR_number_train_0w-5w.csv", index_col=0)
meta = pd.read_csv(f"{RESULTS_DIR}/beam_metadata.csv")

print(f"  vol: {vol.shape}, thp: {thp.shape}, prb: {prb.shape}, mr: {mr.shape}")
print(f"  meta: {meta.shape}")

# Parse beam structure
beam_info = []
for col in vol.columns:
    parts = col.split('_')
    beam_info.append({'beam_id': col, 'bs': int(parts[0]), 'cell': int(parts[1]), 'beam': int(parts[2])})
beam_df = pd.DataFrame(beam_info)

# ═══════════════════════════════════════════════════════════════════════════
# 1. CELL-VS-BEAM AGGREGATION COMPARISON
# ═══════════════════════════════════════════════════════════════════════════
print("\n" + "="*70)
print("1. CELL-VS-BEAM AGGREGATION COMPARISON")
print("="*70)

# Aggregate beams → cell level
cell_vol = {}
for (bs, cell), group in beam_df.groupby(['bs', 'cell']):
    beam_cols = group['beam_id'].tolist()
    cell_vol[f"{bs}_{cell}"] = vol[beam_cols].sum(axis=1)
cell_vol_df = pd.DataFrame(cell_vol)

# Aggregate cells → BS level
bs_vol = {}
for bs, group in beam_df.groupby('bs'):
    beam_cols = group['beam_id'].tolist()
    bs_vol[f"BS_{bs}"] = vol[beam_cols].sum(axis=1)
bs_vol_df = pd.DataFrame(bs_vol)

# Compare properties at each level
def compute_series_properties(series):
    """Compute key properties for a time series."""
    vals = series.values
    zero_frac = (vals == 0).mean()
    cv = vals.std() / (vals.mean() + 1e-10)  # coefficient of variation
    # Autocorrelation at lag 24 (daily) and 168 (weekly)
    if len(vals) > 168:
        acf_24 = np.corrcoef(vals[24:], vals[:-24])[0, 1] if vals.std() > 0 else 0
        acf_168 = np.corrcoef(vals[168:], vals[:-168])[0, 1] if vals.std() > 0 else 0
    else:
        acf_24 = acf_168 = 0
    return {'zero_frac': zero_frac, 'cv': cv, 'acf_24': acf_24, 'acf_168': acf_168, 'mean': vals.mean(), 'std': vals.std()}

print("Computing beam-level properties...")
beam_props = []
for col in vol.columns:
    p = compute_series_properties(vol[col])
    p['level'] = 'beam'
    p['id'] = col
    beam_props.append(p)

print("Computing cell-level properties...")
cell_props = []
for col in cell_vol_df.columns:
    p = compute_series_properties(cell_vol_df[col])
    p['level'] = 'cell'
    p['id'] = col
    cell_props.append(p)

print("Computing BS-level properties...")
bs_props = []
for col in bs_vol_df.columns:
    p = compute_series_properties(bs_vol_df[col])
    p['level'] = 'bs'
    p['id'] = col
    bs_props.append(p)

all_props = pd.DataFrame(beam_props + cell_props + bs_props)

# Print summary table
print("\n  Property comparison (median values):")
print(f"  {'Level':<8} {'N':>5} {'Zero%':>8} {'CV':>8} {'ACF-24':>8} {'ACF-168':>8}")
print(f"  {'-'*46}")
for level in ['beam', 'cell', 'bs']:
    sub = all_props[all_props['level'] == level]
    print(f"  {level:<8} {len(sub):>5} {sub['zero_frac'].median()*100:>7.1f}% {sub['cv'].median():>8.2f} {sub['acf_24'].median():>8.3f} {sub['acf_168'].median():>8.3f}")

# Figure: cell-vs-beam comparison (4 panels)
fig, axes = plt.subplots(2, 2, figsize=(7, 4.5))

# Panel A: Zero fraction distribution at each level (box plots)
box_data = []
box_labels = []
box_colors = []
for level, color, label in [('bs', '#2ecc71', 'BS (n=30)'),
                              ('cell', '#3498db', 'Cell (n=90)'),
                              ('beam', '#e74c3c', 'Beam (n=2880)')]:
    sub = all_props[all_props['level'] == level]
    box_data.append(sub['zero_frac'].values * 100)
    box_labels.append(label)
    box_colors.append(color)
bp = axes[0,0].boxplot(box_data, labels=box_labels, patch_artist=True, widths=0.6,
                        medianprops=dict(color='black', linewidth=1.5))
for patch, color in zip(bp['boxes'], box_colors):
    patch.set_facecolor(color)
    patch.set_alpha(0.6)
axes[0,0].set_ylabel('Zero fraction (%)')
axes[0,0].set_title('(a) Sparsity by aggregation level')

# Panel B: ACF at lag 24 (daily periodicity)
for level, color, label in [('beam', '#e74c3c', 'Beam'),
                              ('cell', '#3498db', 'Cell'),
                              ('bs', '#2ecc71', 'BS')]:
    sub = all_props[all_props['level'] == level]
    axes[0,1].hist(sub['acf_24'], bins=30, alpha=0.5, color=color, label=label, density=True)
axes[0,1].set_xlabel('Autocorrelation at lag 24h')
axes[0,1].set_ylabel('Density')
axes[0,1].set_title('(b) Daily periodicity strength')
axes[0,1].legend(fontsize=7)

# Panel C: Example time series - one BS, its cells, its beams
example_bs = 5
example_cell_cols = [f"{example_bs}_{c}" for c in range(3)]
example_beam_cols = [b['beam_id'] for b in beam_info if b['bs'] == example_bs]

# Plot 1 week (168 hours) for visibility
t = np.arange(168)
axes[1,0].plot(t, bs_vol_df[f"BS_{example_bs}"].values[:168], 'g-', lw=1.5, label=f'BS {example_bs} (sum)', alpha=0.9)
for c in range(3):
    cell_id = f"{example_bs}_{c}"
    axes[1,0].plot(t, cell_vol_df[cell_id].values[:168], '--', lw=1.0, alpha=0.7, label=f'Cell {c}')
axes[1,0].set_xlabel('Hour')
axes[1,0].set_ylabel('DLThpVol')
axes[1,0].set_title(f'(c) BS {example_bs}: station and cell traffic (1 week)')
axes[1,0].legend(fontsize=6, ncol=2)

# Panel D: Same BS, show a few beams vs cell aggregate
axes[1,1].plot(t, cell_vol_df[f"{example_bs}_0"].values[:168], 'b-', lw=1.5, label='Cell 0 (sum of 32 beams)', alpha=0.9)
# Pick 3 beams with different sparsity levels
beam_zeros = [(col, (vol[col].values == 0).mean()) for col in example_beam_cols if col.startswith(f"{example_bs}_0_")]
beam_zeros.sort(key=lambda x: x[1])
for i, idx in enumerate([0, len(beam_zeros)//2, -1]):
    bid, zf = beam_zeros[idx]
    axes[1,1].plot(t, vol[bid].values[:168], '--', lw=0.8, alpha=0.7, label=f'Beam {bid.split("_")[2]} ({zf*100:.0f}% zeros)')
axes[1,1].set_xlabel('Hour')
axes[1,1].set_ylabel('DLThpVol')
axes[1,1].set_title(f'(d) Cell {example_bs}_0: cell vs. constituent beams')
axes[1,1].legend(fontsize=6)

plt.tight_layout()
plt.savefig(f"{FIG_DIR}/fig_cell_vs_beam.pdf")
plt.close()
print("  Saved: fig_cell_vs_beam.pdf")


# ═══════════════════════════════════════════════════════════════════════════
# 2. TEMPORAL DECOMPOSITION (STL-like) PER SPARSITY BIN
# ═══════════════════════════════════════════════════════════════════════════
print("\n" + "="*70)
print("2. TEMPORAL DECOMPOSITION PER SPARSITY BIN")
print("="*70)

def simple_stl(series, period=24):
    """Simple additive decomposition: trend (moving avg) + seasonal + residual."""
    n = len(series)
    # Trend: centered moving average
    trend = series.rolling(window=period, center=True, min_periods=period//2).mean()
    # Detrended
    detrended = series - trend
    # Seasonal: average of each hour-of-day
    seasonal = np.zeros(n)
    for h in range(period):
        mask = np.arange(n) % period == h
        seasonal[mask] = detrended.iloc[mask].mean()
    seasonal = pd.Series(seasonal, index=series.index)
    # Residual
    residual = series - trend - seasonal
    return trend, seasonal, residual

# Assign sparsity bins
meta['sparsity_bin'] = pd.cut(meta['zero_frac'], bins=[0, 0.1, 0.3, 0.5, 0.7, 1.0],
                               labels=['0-10%', '10-30%', '30-50%', '50-70%', '70-100%'],
                               include_lowest=True)

# For each sparsity bin, decompose a representative beam and compute variance ratios
bins_to_plot = ['0-10%', '10-30%', '30-50%', '50-70%']
n_sample_per_bin = 50  # sample beams per bin for variance stats

fig, axes = plt.subplots(len(bins_to_plot), 3, figsize=(10, 8), sharex='col')

variance_ratios = {}

for i, sbin in enumerate(bins_to_plot):
    bin_beams = meta[meta['sparsity_bin'] == sbin]['beam_id'].tolist()
    if len(bin_beams) == 0:
        continue

    # Sample beams for variance statistics
    sample_beams = np.random.choice(bin_beams, min(n_sample_per_bin, len(bin_beams)), replace=False)

    trend_vars, season_vars, resid_vars = [], [], []
    for bid in sample_beams:
        s = vol[bid].dropna()
        if len(s) < 48:
            continue
        tr, se, re = simple_stl(s, period=24)
        tr_v = tr.dropna().var()
        se_v = se.dropna().var()
        re_v = re.dropna().var()
        total = tr_v + se_v + re_v + 1e-15
        trend_vars.append(tr_v / total)
        season_vars.append(se_v / total)
        resid_vars.append(re_v / total)

    variance_ratios[sbin] = {
        'trend': np.mean(trend_vars),
        'seasonal': np.mean(season_vars),
        'residual': np.mean(resid_vars),
        'n': len(trend_vars)
    }

    # Plot one representative beam
    rep_beam = bin_beams[len(bin_beams)//2]
    s = vol[rep_beam]
    tr, se, re = simple_stl(s, period=24)

    # Plot 1 week
    t = np.arange(168)
    axes[i, 0].plot(t, tr.values[:168], 'b-', lw=1)
    axes[i, 0].set_ylabel(sbin, fontweight='bold')
    if i == 0:
        axes[i, 0].set_title('Trend')

    axes[i, 1].plot(t, se.values[:168], 'r-', lw=0.8)
    if i == 0:
        axes[i, 1].set_title('Seasonal (24h)')

    axes[i, 2].plot(t, re.values[:168], 'gray', lw=0.5, alpha=0.7)
    if i == 0:
        axes[i, 2].set_title('Residual')

for j in range(3):
    axes[-1, j].set_xlabel('Hour')

plt.suptitle('Temporal Decomposition by Sparsity Bin (1-week window)', fontsize=11, y=1.01)
plt.tight_layout()
plt.savefig(f"{FIG_DIR}/fig_temporal_decomposition.pdf")
plt.close()
print("  Saved: fig_temporal_decomposition.pdf")

# Print variance ratio table
print("\n  Variance decomposition (fraction of total variance):")
print(f"  {'Bin':<10} {'N':>4} {'Trend':>8} {'Season':>8} {'Residual':>8}")
print(f"  {'-'*42}")
for sbin in bins_to_plot:
    if sbin in variance_ratios:
        v = variance_ratios[sbin]
        print(f"  {sbin:<10} {v['n']:>4} {v['trend']:>8.1%} {v['seasonal']:>8.1%} {v['residual']:>8.1%}")


# ═══════════════════════════════════════════════════════════════════════════
# 3. SPATIAL BEAM CORRELATION ANALYSIS
# ═══════════════════════════════════════════════════════════════════════════
print("\n" + "="*70)
print("3. SPATIAL BEAM CORRELATION ANALYSIS")
print("="*70)

# Compute pairwise correlations at different spatial granularities
intra_cell_corrs = []
inter_cell_corrs = []  # same BS, different cell
inter_bs_corrs = []

# Sample for computational tractability
np.random.seed(42)

print("  Computing intra-cell correlations...")
for (bs, cell), group in beam_df.groupby(['bs', 'cell']):
    beams = group['beam_id'].tolist()
    # Sample pairs
    if len(beams) > 10:
        sample_idx = np.random.choice(len(beams), 10, replace=False)
        beams_sample = [beams[i] for i in sample_idx]
    else:
        beams_sample = beams
    for i in range(len(beams_sample)):
        for j in range(i+1, len(beams_sample)):
            r = vol[beams_sample[i]].corr(vol[beams_sample[j]])
            if not np.isnan(r):
                intra_cell_corrs.append(r)

print("  Computing inter-cell (same BS) correlations...")
for bs, group in beam_df.groupby('bs'):
    cells = group['cell'].unique()
    for c1_idx in range(len(cells)):
        for c2_idx in range(c1_idx+1, len(cells)):
            c1_beams = group[group['cell'] == cells[c1_idx]]['beam_id'].tolist()
            c2_beams = group[group['cell'] == cells[c2_idx]]['beam_id'].tolist()
            # Sample 5 pairs
            for _ in range(5):
                b1 = np.random.choice(c1_beams)
                b2 = np.random.choice(c2_beams)
                r = vol[b1].corr(vol[b2])
                if not np.isnan(r):
                    inter_cell_corrs.append(r)

print("  Computing inter-BS correlations...")
all_bs = beam_df['bs'].unique()
for _ in range(500):  # sample 500 inter-BS pairs
    bs1, bs2 = np.random.choice(all_bs, 2, replace=False)
    b1 = np.random.choice(beam_df[beam_df['bs'] == bs1]['beam_id'].tolist())
    b2 = np.random.choice(beam_df[beam_df['bs'] == bs2]['beam_id'].tolist())
    r = vol[b1].corr(vol[b2])
    if not np.isnan(r):
        inter_bs_corrs.append(r)

print(f"  Intra-cell: n={len(intra_cell_corrs)}, median={np.median(intra_cell_corrs):.3f}")
print(f"  Inter-cell: n={len(inter_cell_corrs)}, median={np.median(inter_cell_corrs):.3f}")
print(f"  Inter-BS:   n={len(inter_bs_corrs)}, median={np.median(inter_bs_corrs):.3f}")

# Figure: spatial correlations
fig, axes = plt.subplots(1, 3, figsize=(11, 3.2))

# Panel A: Correlation distributions
for corrs, color, label in [(intra_cell_corrs, '#e74c3c', f'Intra-cell (n={len(intra_cell_corrs)})'),
                              (inter_cell_corrs, '#3498db', f'Inter-cell (n={len(inter_cell_corrs)})'),
                              (inter_bs_corrs, '#2ecc71', f'Inter-BS (n={len(inter_bs_corrs)})')]:
    axes[0].hist(corrs, bins=40, alpha=0.5, color=color, label=label, density=True)
axes[0].axvline(x=0, color='gray', ls=':', lw=0.5)
axes[0].set_xlabel('Pearson correlation')
axes[0].set_ylabel('Density')
axes[0].set_title('(a) Beam correlation by spatial relationship')
axes[0].legend(fontsize=7)

# Panel B: Correlation heatmap for one BS (cells × beams summary)
example_bs = 5
bs_beams = beam_df[beam_df['bs'] == example_bs].sort_values(['cell', 'beam'])
bs_beam_ids = bs_beams['beam_id'].tolist()
# Subsample to 24 beams (8 per cell) for readability
sample_per_cell = 8
selected = []
for c in range(3):
    cell_beams = [b for b in bs_beam_ids if b.startswith(f"{example_bs}_{c}_")]
    selected.extend(cell_beams[:sample_per_cell])

corr_matrix = vol[selected].corr()
im = axes[1].imshow(corr_matrix.values, cmap='RdBu_r', vmin=-0.3, vmax=1.0, aspect='auto')
# Add cell boundaries
for boundary in [sample_per_cell, 2*sample_per_cell]:
    axes[1].axhline(y=boundary-0.5, color='black', lw=1.5)
    axes[1].axvline(x=boundary-0.5, color='black', lw=1.5)
axes[1].set_title(f'(b) Beam correlations within BS {example_bs}')
axes[1].set_ylabel('Beam index')
plt.colorbar(im, ax=axes[1], shrink=0.8)

# Panel C: Median correlation vs sparsity of beam pair
# For each intra-cell pair, compute avg sparsity and correlation
print("  Computing correlation vs sparsity...")
sparsity_corr_data = []
for (bs, cell), group in beam_df.groupby(['bs', 'cell']):
    beams = group['beam_id'].tolist()
    if len(beams) > 8:
        sample_idx = np.random.choice(len(beams), 8, replace=False)
        beams = [beams[i] for i in sample_idx]
    for i in range(len(beams)):
        for j in range(i+1, len(beams)):
            r = vol[beams[i]].corr(vol[beams[j]])
            zf_i = meta[meta['beam_id'] == beams[i]]['zero_frac'].values[0]
            zf_j = meta[meta['beam_id'] == beams[j]]['zero_frac'].values[0]
            avg_zf = (zf_i + zf_j) / 2
            if not np.isnan(r):
                sparsity_corr_data.append({'avg_zero_frac': avg_zf, 'corr': r})

sc_df = pd.DataFrame(sparsity_corr_data)
sc_df['zf_bin'] = pd.cut(sc_df['avg_zero_frac'], bins=10)
binned = sc_df.groupby('zf_bin')['corr'].agg(['median', 'mean', 'std', 'count'])
bin_centers = [interval.mid for interval in binned.index]
axes[2].errorbar(bin_centers, binned['median'], yerr=binned['std']/np.sqrt(binned['count']),
                  fmt='o-', color='#e74c3c', markersize=4, capsize=3)
axes[2].set_xlabel('Avg. zero fraction of beam pair')
axes[2].set_ylabel('Median Pearson r')
axes[2].set_title('(c) Spatial correlation vs. sparsity')
axes[2].axhline(y=0, color='gray', ls=':', lw=0.5)

# Set cell labels on panel B x-axis AFTER colorbar to prevent reset
cell_mids = [c * sample_per_cell + sample_per_cell/2 - 0.5 for c in range(3)]
axes[1].set_xticks(cell_mids)
axes[1].set_xticklabels(['Cell 0', 'Cell 1', 'Cell 2'], fontsize=7)
axes[1].set_xlabel('')

plt.tight_layout()
plt.savefig(f"{FIG_DIR}/fig_spatial_correlations.pdf")
plt.close()
print("  Saved: fig_spatial_correlations.pdf")


# ═══════════════════════════════════════════════════════════════════════════
# 4. WHAT PREDICTS BEAM SPARSITY? (Causal/structural analysis)
# ═══════════════════════════════════════════════════════════════════════════
print("\n" + "="*70)
print("4. WHAT PREDICTS BEAM SPARSITY?")
print("="*70)

# For each beam, compute: zero_frac of DLThpVol, and also zero_frac and mean of covariates
print("  Computing per-beam covariate statistics...")
beam_stats = []
for _, row in meta.iterrows():
    bid = row['beam_id']
    bs_id = row['bs_id']
    cell_id = row['cell_id']
    beam_idx = row['beam_idx']

    vol_series = vol[bid]
    thp_series = thp[bid]
    prb_series = prb[bid]
    mr_series = mr[bid]

    # Zero fractions
    vol_zf = (vol_series == 0).mean()
    thp_zf = (thp_series == 0).mean()
    prb_zf = (prb_series == 0).mean()
    mr_zf = (mr_series == 0).mean()

    # Means
    vol_mean = vol_series.mean()
    thp_mean = thp_series.mean()
    prb_mean = prb_series.mean()
    mr_mean = mr_series.mean()

    # Correlations with vol (on non-zero hours only)
    nonzero_mask = vol_series > 0
    if nonzero_mask.sum() > 10:
        corr_thp = vol_series[nonzero_mask].corr(thp_series[nonzero_mask])
        corr_prb = vol_series[nonzero_mask].corr(prb_series[nonzero_mask])
        corr_mr = vol_series[nonzero_mask].corr(mr_series[nonzero_mask])
    else:
        corr_thp = corr_prb = corr_mr = np.nan

    # Periodicity strength (ACF at lag 24)
    if vol_series.std() > 0 and len(vol_series) > 24:
        acf_24 = np.corrcoef(vol_series.values[24:], vol_series.values[:-24])[0, 1]
    else:
        acf_24 = 0

    beam_stats.append({
        'beam_id': bid, 'bs_id': bs_id, 'cell_id': cell_id, 'beam_idx': beam_idx,
        'vol_zf': vol_zf, 'thp_zf': thp_zf, 'prb_zf': prb_zf, 'mr_zf': mr_zf,
        'vol_mean': vol_mean, 'thp_mean': thp_mean, 'prb_mean': prb_mean, 'mr_mean': mr_mean,
        'corr_thp': corr_thp, 'corr_prb': corr_prb, 'corr_mr': corr_mr,
        'acf_24': acf_24,
    })

beam_stats_df = pd.DataFrame(beam_stats)

# Sparsity distribution across BS
bs_sparsity = beam_stats_df.groupby('bs_id')['vol_zf'].agg(['mean', 'std', 'min', 'max'])
print("\n  Sparsity distribution across base stations:")
print(f"  Mean zero-frac per BS: min={bs_sparsity['mean'].min():.3f}, max={bs_sparsity['mean'].max():.3f}, "
      f"median={bs_sparsity['mean'].median():.3f}")
print(f"  Std within BS: median={bs_sparsity['std'].median():.3f}")

# Sparsity distribution across beam index (position)
beam_idx_sparsity = beam_stats_df.groupby('beam_idx')['vol_zf'].mean()
print(f"\n  Sparsity by beam index (0-31):")
print(f"  Correlation between beam_idx and zero_frac: r={beam_stats_df['beam_idx'].corr(beam_stats_df['vol_zf']):.3f}")

# Covariate correlation by sparsity
beam_stats_df['sparsity_bin'] = pd.cut(beam_stats_df['vol_zf'], bins=[0, 0.1, 0.3, 0.5, 0.7, 1.0],
                                         labels=['0-10%', '10-30%', '30-50%', '50-70%', '70-100%'],
                                         include_lowest=True)

print("\n  Covariate correlations by sparsity bin:")
print(f"  {'Bin':<10} {'N':>5} {'r(ThpTime)':>12} {'r(PRB)':>10} {'r(MR)':>10} {'ACF-24':>8}")
print(f"  {'-'*55}")
for sbin in ['0-10%', '10-30%', '30-50%', '50-70%']:
    sub = beam_stats_df[beam_stats_df['sparsity_bin'] == sbin]
    print(f"  {sbin:<10} {len(sub):>5} {sub['corr_thp'].median():>12.3f} {sub['corr_prb'].median():>10.3f} "
          f"{sub['corr_mr'].median():>10.3f} {sub['acf_24'].median():>8.3f}")

# Figure: what predicts sparsity
fig, axes = plt.subplots(2, 2, figsize=(7, 5.5))

# Panel A: Sparsity distribution per BS (box plot)
bs_data = [beam_stats_df[beam_stats_df['bs_id'] == bs]['vol_zf'].values * 100 for bs in range(30)]
bp = axes[0,0].boxplot(bs_data, positions=range(30), widths=0.6, patch_artist=True,
                        boxprops=dict(facecolor='#3498db', alpha=0.5),
                        medianprops=dict(color='red', lw=1.5),
                        flierprops=dict(markersize=2))
axes[0,0].set_xlabel('Base station ID')
axes[0,0].set_ylabel('Zero fraction (%)')
axes[0,0].set_title('(a) Beam sparsity distribution per BS')
axes[0,0].set_xticks(range(0, 30, 5))

# Panel B: Sparsity by beam index
beam_idx_data = beam_stats_df.groupby('beam_idx')['vol_zf'].agg(['mean', 'std'])
axes[0,1].errorbar(beam_idx_data.index, beam_idx_data['mean']*100,
                     yerr=beam_idx_data['std']*100/np.sqrt(90),  # 90 = 30 BS × 3 cells
                     fmt='o-', color='#e74c3c', markersize=3, capsize=2)
axes[0,1].set_xlabel('Beam index (0-31)')
axes[0,1].set_ylabel('Mean zero fraction (%)')
axes[0,1].set_title('(b) Sparsity by beam position')

# Panel C: Covariate correlation vs sparsity
for cov, color, label in [('corr_thp', '#e74c3c', 'DLThpTime'),
                            ('corr_prb', '#3498db', 'DLPRB'),
                            ('corr_mr', '#2ecc71', 'MR_number')]:
    binned = beam_stats_df.groupby('sparsity_bin')[cov].median()
    x_pos = range(len(binned))
    axes[1,0].plot(x_pos, binned.values, 'o-', color=color, label=label, markersize=5)
axes[1,0].set_xticks(range(4))
axes[1,0].set_xticklabels(['0-10%', '10-30%', '30-50%', '50-70%'])
axes[1,0].set_xlabel('Sparsity bin')
axes[1,0].set_ylabel('Median Pearson r')
axes[1,0].set_title('(c) Covariate correlation vs. sparsity')
axes[1,0].legend(fontsize=7)

# Panel D: ACF-24 vs sparsity (periodicity degrades with sparsity)
binned_acf = beam_stats_df.groupby('sparsity_bin')['acf_24'].agg(['median', 'std', 'count'])
valid_bins = binned_acf[binned_acf['count'] > 0]
x_pos = range(len(valid_bins))
axes[1,1].bar(x_pos, valid_bins['median'].values, yerr=valid_bins['std'].values/np.sqrt(valid_bins['count'].values),
               color='#9b59b6', alpha=0.7, capsize=3)
axes[1,1].set_xticks(x_pos)
axes[1,1].set_xticklabels(valid_bins.index.tolist(), rotation=0)
axes[1,1].set_xlabel('Sparsity bin')
axes[1,1].set_ylabel('Median ACF at lag 24h')
axes[1,1].set_title('(d) Periodicity strength vs. sparsity')

plt.tight_layout()
plt.savefig(f"{FIG_DIR}/fig_sparsity_drivers.pdf")
plt.close()
print("  Saved: fig_sparsity_drivers.pdf")

# ═══════════════════════════════════════════════════════════════════════════
# 5. COVARIATE ZERO-FRACTION ALIGNMENT
# ═══════════════════════════════════════════════════════════════════════════
print("\n" + "="*70)
print("5. COVARIATE CO-SPARSITY ANALYSIS")
print("="*70)

# Do covariates share the same zero pattern?
print("  Fraction of hours where DLThpVol=0 AND covariate=0:")
vol_zero = (vol == 0)
thp_zero = (thp == 0)
prb_zero = (prb == 0)
mr_zero = (mr == 0)

co_zero_thp = (vol_zero & thp_zero).mean().mean()
co_zero_prb = (vol_zero & prb_zero).mean().mean()
co_zero_mr  = (vol_zero & mr_zero).mean().mean()

vol_zero_frac = vol_zero.mean().mean()
print(f"  DLThpVol zero-hours: {vol_zero_frac:.1%}")
print(f"  Co-zero with DLThpTime: {co_zero_thp:.1%} ({co_zero_thp/vol_zero_frac:.1%} of vol zeros)")
print(f"  Co-zero with DLPRB:     {co_zero_prb:.1%} ({co_zero_prb/vol_zero_frac:.1%} of vol zeros)")
print(f"  Co-zero with MR_number: {co_zero_mr:.1%} ({co_zero_mr/vol_zero_frac:.1%} of vol zeros)")

# Hours where vol=0 but covariates≠0
vol_zero_cov_nonzero = (vol_zero & ~thp_zero).mean().mean()
print(f"  Vol=0 but ThpTime≠0: {vol_zero_cov_nonzero:.1%}")

print("\n" + "="*70)
print("ALL DONE. Figures saved to paper/figures/")
print("="*70)
