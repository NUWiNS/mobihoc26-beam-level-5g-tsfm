"""
Deep measurement analysis v2 — runs on the server with full data access.
Covers:
  A. Cross-KPI analysis: do all 4 KPIs show same beam-level properties?
  B. Distribution shift: how do beam properties change train→week6→week11?
  C. Irish dataset: temporal decomposition + covariate analysis
  D. Formal statistical tests for key findings
  E. Aggregation information loss quantification
"""

import sys as _sys
from pathlib import Path as _P
_sys.path.insert(0, str(_P(__file__).resolve().parent.parent))
from paths import ROOT, GOTSF_DIR, IRISH_DIR, RESULTS_DIR, FIG_DIR
import pandas as pd
import numpy as np
from scipy import stats
from pathlib import Path
import glob
import warnings
warnings.filterwarnings('ignore')

DATA_DIR = GOTSF_DIR
IRISH_DIR = IRISH_DIR
RESULTS_DIR = RESULTS_DIR
OUT_DIR = RESULTS_DIR / "measurement_analysis"
OUT_DIR.mkdir(parents=True, exist_ok=True)

KPIS = ["DLThpVol", "DLThpTime", "DLPRB", "MR_number"]

# ── Helpers ──────────────────────────────────────────────────────────────
def load_wide(kpi, split, test_week="5w-6w"):
    if split == "train":
        return pd.read_csv(DATA_DIR / "train" / f"{kpi}_train_0w-5w.csv", index_col=0)
    else:
        return pd.read_csv(DATA_DIR / "test" / f"{kpi}_test_{test_week}.csv", index_col=0)

def compute_props(series):
    vals = series.values
    n = len(vals)
    zero_frac = (vals == 0).mean()
    mean_val = vals.mean()
    std_val = vals.std()
    cv = std_val / (mean_val + 1e-15)
    if n > 168 and std_val > 0:
        acf_24 = np.corrcoef(vals[24:], vals[:-24])[0, 1]
        acf_168 = np.corrcoef(vals[168:], vals[:-168])[0, 1]
    elif n > 24 and std_val > 0:
        acf_24 = np.corrcoef(vals[24:], vals[:-24])[0, 1]
        acf_168 = np.nan
    else:
        acf_24 = acf_168 = np.nan
    return {'zero_frac': zero_frac, 'cv': cv, 'acf_24': acf_24,
            'acf_168': acf_168, 'mean': mean_val, 'std': std_val}

def parse_beam_id(col):
    parts = col.split('_')
    return int(parts[0]), int(parts[1]), int(parts[2])

# ═══════════════════════════════════════════════════════════════════════════
# A. CROSS-KPI ANALYSIS
# ═══════════════════════════════════════════════════════════════════════════
print("=" * 70)
print("A. CROSS-KPI ANALYSIS: Do all 4 KPIs show the same beam-level properties?")
print("=" * 70)

kpi_results = {}
for kpi in KPIS:
    print(f"\n  Loading {kpi}...")
    df = load_wide(kpi, "train")
    cols = df.columns.tolist()

    # Beam-level stats
    beam_zf = [(df[c] == 0).mean() for c in cols]
    beam_cv = [df[c].std() / (df[c].mean() + 1e-15) for c in cols]
    beam_acf24 = []
    for c in cols:
        v = df[c].values
        if len(v) > 24 and v.std() > 0:
            beam_acf24.append(np.corrcoef(v[24:], v[:-24])[0, 1])
        else:
            beam_acf24.append(np.nan)

    # Cell-level aggregation
    cell_data = {}
    for c in cols:
        bs, cell, beam = parse_beam_id(c)
        key = f"{bs}_{cell}"
        if key not in cell_data:
            cell_data[key] = df[c].copy()
        else:
            cell_data[key] = cell_data[key] + df[c]
    cell_zf = [(cell_data[k] == 0).mean() for k in cell_data]
    cell_acf24 = []
    for k in cell_data:
        v = cell_data[k].values
        if len(v) > 24 and v.std() > 0:
            cell_acf24.append(np.corrcoef(v[24:], v[:-24])[0, 1])
        else:
            cell_acf24.append(np.nan)

    kpi_results[kpi] = {
        'beam_zf_median': np.median(beam_zf),
        'beam_zf_mean': np.mean(beam_zf),
        'beam_cv_median': np.median(beam_cv),
        'beam_acf24_median': np.nanmedian(beam_acf24),
        'cell_zf_median': np.median(cell_zf),
        'cell_acf24_median': np.nanmedian(cell_acf24),
        'n_beams': len(cols),
        'n_cells': len(cell_data),
    }

print("\n  Cross-KPI comparison (beam level):")
print(f"  {'KPI':<14} {'N':>5} {'Med.Zero%':>10} {'Med.CV':>8} {'Med.ACF24':>10}")
print(f"  {'-'*50}")
for kpi in KPIS:
    r = kpi_results[kpi]
    print(f"  {kpi:<14} {r['n_beams']:>5} {r['beam_zf_median']*100:>9.1f}% {r['beam_cv_median']:>8.2f} {r['beam_acf24_median']:>10.3f}")

print("\n  Cross-KPI comparison (cell level):")
print(f"  {'KPI':<14} {'N':>5} {'Med.Zero%':>10} {'Med.ACF24':>10}")
print(f"  {'-'*40}")
for kpi in KPIS:
    r = kpi_results[kpi]
    print(f"  {kpi:<14} {r['n_cells']:>5} {r['cell_zf_median']*100:>9.1f}% {r['cell_acf24_median']:>10.3f}")

# Cross-KPI sparsity correlation: are beams that are sparse in one KPI also sparse in others?
print("\n  Cross-KPI sparsity correlation (Spearman r between beam zero-fracs):")
zf_per_kpi = {}
for kpi in KPIS:
    df = load_wide(kpi, "train")
    zf_per_kpi[kpi] = pd.Series({c: (df[c] == 0).mean() for c in df.columns})

zf_df = pd.DataFrame(zf_per_kpi)
print(zf_df.corr(method='spearman').round(3).to_string())


# ═══════════════════════════════════════════════════════════════════════════
# B. DISTRIBUTION SHIFT ANALYSIS
# ═══════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("B. DISTRIBUTION SHIFT: Properties across train/week6/week11")
print("=" * 70)

shift_results = {}
for period_name, loader_args in [("Train (0-5w)", ("DLThpVol", "train", None)),
                                   ("Test 5-6w", ("DLThpVol", "test", "5w-6w")),
                                   ("Test 10-11w", ("DLThpVol", "test", "10w-11w"))]:
    kpi, split, tw = loader_args
    if tw:
        df = load_wide(kpi, split, tw)
    else:
        df = load_wide(kpi, split)
    cols = df.columns.tolist()

    zf_list = [(df[c] == 0).mean() for c in cols]
    mean_list = [df[c].mean() for c in cols]
    cv_list = [df[c].std() / (df[c].mean() + 1e-15) for c in cols]

    shift_results[period_name] = {
        'zf_median': np.median(zf_list),
        'zf_mean': np.mean(zf_list),
        'mean_median': np.median(mean_list),
        'cv_median': np.median(cv_list),
        'zf_array': np.array(zf_list),
        'mean_array': np.array(mean_list),
    }

print(f"\n  {'Period':<16} {'Med.Zero%':>10} {'Mean.Zero%':>11} {'Med.Traffic':>12} {'Med.CV':>8}")
print(f"  {'-'*60}")
for p, r in shift_results.items():
    print(f"  {p:<16} {r['zf_median']*100:>9.1f}% {r['zf_mean']*100:>10.1f}% {r['mean_median']:>12.4f} {r['cv_median']:>8.2f}")

# KS test: distribution of zero-fracs between periods
print("\n  KS tests for zero-fraction distributions:")
train_zf = shift_results["Train (0-5w)"]['zf_array']
for period in ["Test 5-6w", "Test 10-11w"]:
    test_zf = shift_results[period]['zf_array']
    ks_stat, ks_p = stats.ks_2samp(train_zf, test_zf)
    print(f"  Train vs {period}: KS={ks_stat:.4f}, p={ks_p:.4e}")

# KS test: distribution of mean traffic between periods
print("\n  KS tests for mean-traffic distributions:")
train_mean = shift_results["Train (0-5w)"]['mean_array']
for period in ["Test 5-6w", "Test 10-11w"]:
    test_mean = shift_results[period]['mean_array']
    ks_stat, ks_p = stats.ks_2samp(train_mean, test_mean)
    print(f"  Train vs {period}: KS={ks_stat:.4f}, p={ks_p:.4e}")

# Per-beam shift: how much does each beam's zero-frac change?
print("\n  Per-beam zero-frac change (train → each test):")
train_df = load_wide("DLThpVol", "train")
for tw_name, tw in [("5-6w", "5w-6w"), ("10-11w", "10w-11w")]:
    test_df = load_wide("DLThpVol", "test", tw)
    changes = []
    for c in train_df.columns:
        if c in test_df.columns:
            train_zf_c = (train_df[c] == 0).mean()
            test_zf_c = (test_df[c] == 0).mean()
            changes.append(test_zf_c - train_zf_c)
    changes = np.array(changes)
    print(f"  Train → {tw_name}: mean Δzf={changes.mean():.4f}, std={changes.std():.4f}, "
          f"|Δzf|>0.1: {(np.abs(changes) > 0.1).mean()*100:.1f}% of beams")

# Cross-KPI shift: do covariates shift too?
print("\n  Cross-KPI distribution shift (KS test on mean-traffic, Train vs 10-11w):")
for kpi in KPIS:
    train_k = load_wide(kpi, "train")
    test_k = load_wide(kpi, "test", "10w-11w")
    train_means = np.array([train_k[c].mean() for c in train_k.columns])
    test_means = np.array([test_k[c].mean() for c in test_k.columns if c in train_k.columns])
    ks_stat, ks_p = stats.ks_2samp(train_means, test_means)
    ratio = np.median(test_means) / (np.median(train_means) + 1e-15)
    print(f"  {kpi:<14}: KS={ks_stat:.4f}, p={ks_p:.4e}, median ratio test/train={ratio:.3f}")


# ═══════════════════════════════════════════════════════════════════════════
# C. IRISH DATASET ANALYSIS
# ═══════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("C. IRISH DATASET: Temporal + Covariate Analysis")
print("=" * 70)

irish_files = sorted(glob.glob(str(IRISH_DIR / "**/*.csv"), recursive=True))
print(f"  Found {len(irish_files)} Irish sessions")

IRISH_TARGET = "DL_bitrate"
IRISH_COVS = ["RSRP", "RSRQ", "SNR", "CQI"]
all_irish_cols = [IRISH_TARGET] + IRISH_COVS

# Load all sessions
irish_sessions = []
for f in irish_files:
    try:
        df = pd.read_csv(f)
        for c in all_irish_cols:
            df[c] = pd.to_numeric(df[c], errors="coerce")
        df = df.dropna(subset=all_irish_cols)
        if len(df) >= 180:  # at least 3 minutes
            irish_sessions.append((Path(f).stem, df[all_irish_cols + ['NetworkMode', 'Speed']].reset_index(drop=True)))
    except Exception:
        continue

print(f"  Loaded {len(irish_sessions)} valid sessions")

# Properties per session
irish_props = []
for sid, df in irish_sessions:
    target = df[IRISH_TARGET].values
    zf = (target == 0).mean()
    mean_t = target.mean()
    std_t = target.std()
    cv = std_t / (mean_t + 1e-15)

    # Covariate correlations
    corrs = {}
    nonzero = target > 0
    for cov in IRISH_COVS:
        if nonzero.sum() > 10:
            corrs[f'r_{cov}'] = df[IRISH_TARGET][nonzero].corr(df[cov][nonzero])
        else:
            corrs[f'r_{cov}'] = np.nan

    # Network mode mix
    if 'NetworkMode' in df.columns:
        nm = df['NetworkMode'].value_counts(normalize=True)
        pct_5g = nm.get('5G', 0) + nm.get('NR', 0) + nm.get('ENDC', 0)
    else:
        pct_5g = np.nan

    # Mobility
    if 'Speed' in df.columns:
        speed = pd.to_numeric(df['Speed'], errors='coerce')
        avg_speed = speed.mean()
    else:
        avg_speed = np.nan

    props = {'session_id': sid, 'length': len(df), 'zero_frac': zf,
             'mean_traffic': mean_t, 'cv': cv, 'pct_5g': pct_5g,
             'avg_speed': avg_speed, **corrs}
    irish_props.append(props)

irish_df = pd.DataFrame(irish_props)

print(f"\n  Irish session properties:")
print(f"  {'Metric':<20} {'Median':>10} {'Mean':>10} {'Std':>10}")
print(f"  {'-'*55}")
for col in ['length', 'zero_frac', 'mean_traffic', 'cv', 'pct_5g', 'avg_speed']:
    if col in irish_df.columns:
        print(f"  {col:<20} {irish_df[col].median():>10.3f} {irish_df[col].mean():>10.3f} {irish_df[col].std():>10.3f}")

print(f"\n  Irish covariate correlations with DL_bitrate:")
for cov in IRISH_COVS:
    col = f'r_{cov}'
    print(f"  {cov:<8}: median r={irish_df[col].median():.3f}, mean={irish_df[col].mean():.3f}")

# Irish temporal properties: is there periodicity? (unlikely at per-second granularity)
print("\n  Irish temporal autocorrelation:")
acf_lags = [1, 5, 10, 30, 60]
for lag in acf_lags:
    acfs = []
    for sid, df in irish_sessions:
        v = df[IRISH_TARGET].values
        if len(v) > lag and v.std() > 0:
            acfs.append(np.corrcoef(v[lag:], v[:-lag])[0, 1])
    print(f"  Lag {lag:>3}s: median ACF={np.median(acfs):.3f}, mean={np.mean(acfs):.3f}")

# Irish: how does sparsity relate to network mode and speed?
print("\n  Irish: zero-frac by network mode dominance:")
irish_df['mode_group'] = irish_df['pct_5g'].apply(
    lambda x: '5G-dominant (>50%)' if x > 0.5 else 'LTE-dominant (<=50%)')
for grp, sub in irish_df.groupby('mode_group'):
    print(f"  {grp}: n={len(sub)}, median zero_frac={sub['zero_frac'].median():.3f}, "
          f"median traffic={sub['mean_traffic'].median():.1f}")


# ═══════════════════════════════════════════════════════════════════════════
# D. FORMAL STATISTICAL TESTS FOR KEY FINDINGS
# ═══════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("D. FORMAL STATISTICAL TESTS")
print("=" * 70)

# Test 1: Beam vs Cell sparsity — Mann-Whitney U
print("\n  Test 1: Beam-level vs Cell-level zero fraction (DLThpVol)")
vol_train = load_wide("DLThpVol", "train")
beam_zfs = np.array([(vol_train[c] == 0).mean() for c in vol_train.columns])

# Aggregate to cell level
cell_agg = {}
for c in vol_train.columns:
    bs, cell, beam = parse_beam_id(c)
    key = f"{bs}_{cell}"
    if key not in cell_agg:
        cell_agg[key] = vol_train[c].copy()
    else:
        cell_agg[key] = cell_agg[key] + vol_train[c]
cell_zfs = np.array([(cell_agg[k] == 0).mean() for k in cell_agg])

u_stat, u_p = stats.mannwhitneyu(beam_zfs, cell_zfs, alternative='greater')
print(f"  Beam zero-fracs: median={np.median(beam_zfs):.3f}, mean={np.mean(beam_zfs):.3f}")
print(f"  Cell zero-fracs: median={np.median(cell_zfs):.3f}, mean={np.mean(cell_zfs):.3f}")
print(f"  Mann-Whitney U (beam > cell): U={u_stat:.0f}, p={u_p:.4e}")

# Test 2: ACF-24 beam vs cell — Mann-Whitney U
print("\n  Test 2: Daily periodicity strength (ACF-24) beam vs cell")
beam_acf24 = []
for c in vol_train.columns:
    v = vol_train[c].values
    if v.std() > 0:
        beam_acf24.append(np.corrcoef(v[24:], v[:-24])[0, 1])
    else:
        beam_acf24.append(0)
beam_acf24 = np.array(beam_acf24)

cell_acf24 = []
for k in cell_agg:
    v = cell_agg[k].values
    if v.std() > 0:
        cell_acf24.append(np.corrcoef(v[24:], v[:-24])[0, 1])
    else:
        cell_acf24.append(0)
cell_acf24 = np.array(cell_acf24)

u_stat, u_p = stats.mannwhitneyu(cell_acf24, beam_acf24, alternative='greater')
print(f"  Beam ACF-24: median={np.median(beam_acf24):.3f}")
print(f"  Cell ACF-24: median={np.median(cell_acf24):.3f}")
print(f"  Mann-Whitney U (cell > beam): U={u_stat:.0f}, p={u_p:.4e}")

# Test 3: Sparsity vs periodicity correlation (Spearman)
print("\n  Test 3: Sparsity-periodicity correlation (Spearman)")
rho, p = stats.spearmanr(beam_zfs, beam_acf24)
print(f"  Spearman r(zero_frac, ACF-24) = {rho:.3f}, p={p:.4e}")

# Test 4: Intra-cell vs Inter-BS correlation difference
print("\n  Test 4: Spatial correlation decay (intra-cell vs inter-BS)")
np.random.seed(42)
intra_corrs = []
inter_bs_corrs = []
beam_cols = vol_train.columns.tolist()

# Sample intra-cell
for bs in range(30):
    for cell in range(3):
        beams = [c for c in beam_cols if c.startswith(f"{bs}_{cell}_")]
        if len(beams) > 4:
            sample = np.random.choice(beams, 4, replace=False)
            for i in range(len(sample)):
                for j in range(i+1, len(sample)):
                    r = vol_train[sample[i]].corr(vol_train[sample[j]])
                    if not np.isnan(r):
                        intra_corrs.append(r)

# Sample inter-BS
for _ in range(300):
    bs1, bs2 = np.random.choice(30, 2, replace=False)
    b1 = np.random.choice([c for c in beam_cols if c.startswith(f"{bs1}_")])
    b2 = np.random.choice([c for c in beam_cols if c.startswith(f"{bs2}_")])
    r = vol_train[b1].corr(vol_train[b2])
    if not np.isnan(r):
        inter_bs_corrs.append(r)

u_stat, u_p = stats.mannwhitneyu(intra_corrs, inter_bs_corrs, alternative='greater')
print(f"  Intra-cell: n={len(intra_corrs)}, median={np.median(intra_corrs):.4f}")
print(f"  Inter-BS:   n={len(inter_bs_corrs)}, median={np.median(inter_bs_corrs):.4f}")
print(f"  Mann-Whitney U (intra > inter): U={u_stat:.0f}, p={u_p:.4e}")

# Test 5: Covariate zero-alignment — are zeros synchronized across KPIs?
print("\n  Test 5: Cross-KPI zero synchronization")
vol_z = (vol_train == 0)
for cov_kpi in ["DLThpTime", "DLPRB", "MR_number"]:
    cov_train = load_wide(cov_kpi, "train")
    cov_z = (cov_train == 0)
    # Jaccard similarity of zero patterns per beam
    jaccards = []
    for c in vol_train.columns[:500]:  # sample 500 beams
        v_z = vol_z[c].values
        c_z = cov_z[c].values
        intersection = (v_z & c_z).sum()
        union = (v_z | c_z).sum()
        if union > 0:
            jaccards.append(intersection / union)
    print(f"  DLThpVol vs {cov_kpi}: Jaccard zero overlap = {np.mean(jaccards):.3f} ± {np.std(jaccards):.3f}")


# ═══════════════════════════════════════════════════════════════════════════
# E. AGGREGATION INFORMATION LOSS QUANTIFICATION
# ═══════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("E. AGGREGATION INFORMATION LOSS")
print("=" * 70)

# Information loss: what fraction of cell-level variance is explained
# by evenly distributing traffic across beams vs actual beam distribution?
print("\n  Within-cell beam traffic inequality (Gini coefficient):")
gini_per_cell = []
for bs in range(30):
    for cell in range(3):
        beams = [c for c in vol_train.columns if c.startswith(f"{bs}_{cell}_")]
        beam_means = np.array([vol_train[c].mean() for c in beams])
        # Gini coefficient
        n = len(beam_means)
        if n > 1 and beam_means.sum() > 0:
            sorted_means = np.sort(beam_means)
            index = np.arange(1, n + 1)
            gini = (2 * np.sum(index * sorted_means)) / (n * np.sum(sorted_means)) - (n + 1) / n
            gini_per_cell.append({'bs': bs, 'cell': cell, 'gini': gini,
                                   'top1_share': sorted_means[-1] / sorted_means.sum(),
                                   'top5_share': sorted_means[-5:].sum() / sorted_means.sum()})

gini_df = pd.DataFrame(gini_per_cell)
print(f"  Gini coefficient: median={gini_df['gini'].median():.3f}, mean={gini_df['gini'].mean():.3f}")
print(f"  Top-1 beam share: median={gini_df['top1_share'].median():.1%}, mean={gini_df['top1_share'].mean():.1%}")
print(f"  Top-5 beam share: median={gini_df['top5_share'].median():.1%}, mean={gini_df['top5_share'].mean():.1%}")

# Entropy loss: how much temporal entropy is lost when aggregating?
print("\n  Shannon entropy (temporal variation) comparison:")
def series_entropy(series, bins=50):
    """Discrete entropy of binned values."""
    counts, _ = np.histogram(series.dropna(), bins=bins)
    probs = counts / counts.sum()
    probs = probs[probs > 0]
    return -np.sum(probs * np.log2(probs))

# Beam entropy
beam_entropies = [series_entropy(vol_train[c]) for c in vol_train.columns[:500]]
# Cell entropy
cell_entropies = [series_entropy(pd.Series(cell_agg[k])) for k in list(cell_agg.keys())[:90]]

print(f"  Beam temporal entropy:  median={np.median(beam_entropies):.2f} bits")
print(f"  Cell temporal entropy:  median={np.median(cell_entropies):.2f} bits")
u_stat, u_p = stats.mannwhitneyu(cell_entropies, beam_entropies)
print(f"  Mann-Whitney (cell vs beam entropy): U={u_stat:.0f}, p={u_p:.4e}")

# Final: how many beams per cell carry >50% of traffic?
print("\n  Beam traffic concentration:")
carriers = []
for bs in range(30):
    for cell in range(3):
        beams = [c for c in vol_train.columns if c.startswith(f"{bs}_{cell}_")]
        beam_means = np.array([(vol_train[c].mean(), c) for c in beams],
                               dtype=[('mean', float), ('beam', 'U20')])
        beam_means = np.sort(beam_means, order='mean')[::-1]
        cumsum = np.cumsum([x[0] for x in beam_means])
        total = cumsum[-1]
        if total > 0:
            n_50 = np.searchsorted(cumsum, total * 0.5) + 1
            n_80 = np.searchsorted(cumsum, total * 0.8) + 1
            carriers.append({'bs': bs, 'cell': cell, 'n_50pct': n_50, 'n_80pct': n_80})

carriers_df = pd.DataFrame(carriers)
print(f"  Beams carrying 50% of cell traffic: median={carriers_df['n_50pct'].median():.0f} / 32")
print(f"  Beams carrying 80% of cell traffic: median={carriers_df['n_80pct'].median():.0f} / 32")

# Save key results
print("\n" + "=" * 70)
print("SAVING RESULTS...")
print("=" * 70)
zf_df.to_csv(OUT_DIR / "cross_kpi_sparsity.csv")
gini_df.to_csv(OUT_DIR / "beam_gini.csv", index=False)
carriers_df.to_csv(OUT_DIR / "beam_concentration.csv", index=False)
irish_df.to_csv(OUT_DIR / "irish_properties.csv", index=False)
beam_stats_extra = pd.DataFrame({
    'beam_id': vol_train.columns,
    'zero_frac': beam_zfs,
    'acf_24': beam_acf24
})
beam_stats_extra.to_csv(OUT_DIR / "beam_temporal_stats.csv", index=False)

print(f"  Results saved to {OUT_DIR}")
print("\nDONE.")
