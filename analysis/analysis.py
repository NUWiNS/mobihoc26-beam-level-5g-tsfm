"""
Post-hoc statistical analysis for 5G TSFM benchmark.
Produces: bootstrap CIs, Friedman/Nemenyi, sparsity stratification,
distribution-shift analysis, and LaTeX tables.
"""
import sys as _sys
from pathlib import Path as _P
_sys.path.insert(0, str(_P(__file__).resolve().parent.parent))
from paths import ROOT, GOTSF_DIR, IRISH_DIR, RESULTS_DIR, FIG_DIR
import os

import numpy as np
import pandas as pd
from pathlib import Path
from scipy import stats
from itertools import combinations

RESULTS_DIR = RESULTS_DIR
METRIC_COLS = ["MAE", "RMSE", "MASE", "NMAE"]

# ── helpers ──────────────────────────────────────────────────────────────────
def load(dataset="gotsf"):
    suffix = "_irish" if dataset == "irish" else ""
    meta_name = "beam_metadata_irish.csv" if dataset == "irish" else "beam_metadata.csv"
    df = pd.read_csv(RESULTS_DIR / f"per_beam_results{suffix}.csv")
    meta_path = RESULTS_DIR / meta_name
    if meta_path.exists():
        meta = pd.read_csv(meta_path)
        df = df.merge(meta[["beam_id", "zero_frac", "mean_traffic"]], on="beam_id", how="left")
    return df

# ── 1. Bootstrap confidence intervals ───────────────────────────────────────
def bootstrap_ci(vals, n_boot=10000, ci=0.95, seed=42):
    rng = np.random.RandomState(seed)
    vals = np.asarray(vals, dtype=np.float64)
    vals = vals[~np.isnan(vals)]
    if len(vals) == 0:
        return np.nan, np.nan, np.nan
    means = np.array([rng.choice(vals, size=len(vals), replace=True).mean()
                       for _ in range(n_boot)])
    lo = np.percentile(means, (1 - ci) / 2 * 100)
    hi = np.percentile(means, (1 + ci) / 2 * 100)
    return vals.mean(), lo, hi

def table_with_cis(df, metric="MAE"):
    rows = []
    for tw in sorted(df["test_week"].unique()):
        sub = df[df["test_week"] == tw]
        for mn in sorted(sub["model"].unique()):
            ms = sub[sub["model"] == mn]
            mean, lo, hi = bootstrap_ci(ms[metric].values)
            rows.append({"test_week": tw, "model": mn,
                         f"{metric}_mean": mean, f"{metric}_lo": lo, f"{metric}_hi": hi})
    return pd.DataFrame(rows)

# ── 2. Friedman + Nemenyi ───────────────────────────────────────────────────
def friedman_nemenyi(df, metric="MAE", test_week=None):
    """Per-beam ranks → Friedman χ² → pairwise Nemenyi if significant."""
    if test_week:
        df = df[df["test_week"] == test_week]
    pivot = df.pivot_table(index="beam_id", columns="model", values=metric)
    pivot = pivot.dropna(axis=1)  # drop models with NaN
    models = pivot.columns.tolist()
    k = len(models)
    n = len(pivot)
    if k < 3 or n < 10:
        return None
    # Rank per beam (row), lower metric = rank 1
    ranks = pivot.rank(axis=1, method="average")
    mean_ranks = ranks.mean()
    # Friedman statistic
    chi2 = (12 * n / (k * (k + 1))) * ((mean_ranks ** 2).sum() - k * (k + 1) ** 2 / 4)
    p_friedman = 1 - stats.chi2.cdf(chi2, df=k - 1)
    # Nemenyi CD
    q_alpha = {3: 2.343, 4: 2.569, 5: 2.728, 6: 2.850, 7: 2.949, 8: 3.031,
               9: 3.102, 10: 3.164, 11: 3.219, 12: 3.268, 13: 3.313, 14: 3.354}
    q = q_alpha.get(k, 3.5)  # fallback
    cd = q * np.sqrt(k * (k + 1) / (6 * n))
    # Pairwise differences
    pairs = []
    for a, b in combinations(models, 2):
        diff = abs(mean_ranks[a] - mean_ranks[b])
        sig = diff > cd
        pairs.append({"model_a": a, "model_b": b, "rank_diff": diff,
                       "significant": sig})
    return {"chi2": chi2, "p_friedman": p_friedman, "cd": cd,
            "mean_ranks": mean_ranks.to_dict(), "n_beams": n, "k_models": k,
            "pairwise": pd.DataFrame(pairs)}

# ── 3. Sparsity stratification ──────────────────────────────────────────────
def sparsity_analysis(df, metric="MAE", bins=None):
    if bins is None:
        bins = [0, 0.1, 0.3, 0.5, 0.7, 1.0]
    labels = [f"{bins[i]:.0%}-{bins[i+1]:.0%}" for i in range(len(bins)-1)]
    df = df.copy()
    df["sparsity_bin"] = pd.cut(df["zero_frac"], bins=bins, labels=labels,
                                 include_lowest=True)
    out = df.groupby(["test_week", "model", "sparsity_bin"])[metric].agg(
        ["mean", "std", "count"]).reset_index()
    return out

# ── 4. Distribution-shift analysis ──────────────────────────────────────────
def shift_analysis(df, metric="MAE"):
    """Compare W6 vs W11 degradation per model."""
    rows = []
    for mn in df["model"].unique():
        ms = df[df["model"] == mn]
        w6 = ms[ms["test_week"] == "5w-6w"][metric]
        w11 = ms[ms["test_week"] == "10w-11w"][metric]
        if len(w6) == 0 or len(w11) == 0:
            continue
        m6, m11 = w6.mean(), w11.mean()
        degradation = (m11 - m6) / m6 if m6 > 0 else float("nan")
        # Wilcoxon signed-rank test on paired beams
        common = set(ms[ms["test_week"] == "5w-6w"]["beam_id"]) & \
                 set(ms[ms["test_week"] == "10w-11w"]["beam_id"])
        if len(common) >= 10:
            v6 = ms[(ms["test_week"] == "5w-6w") & (ms["beam_id"].isin(common))].set_index("beam_id")[metric]
            v11 = ms[(ms["test_week"] == "10w-11w") & (ms["beam_id"].isin(common))].set_index("beam_id")[metric]
            common_sorted = sorted(common)
            try:
                stat, p = stats.wilcoxon(v6.loc[common_sorted], v11.loc[common_sorted])
            except Exception:
                stat, p = np.nan, np.nan
        else:
            stat, p = np.nan, np.nan
        rows.append({"model": mn, f"{metric}_w6": m6, f"{metric}_w11": m11,
                     "degradation_pct": degradation * 100, "wilcoxon_p": p})
    return pd.DataFrame(rows).sort_values("degradation_pct")

# ── 5. LaTeX table generation ───────────────────────────────────────────────
def latex_main_table(df):
    """Main results table with CIs."""
    lines = []
    lines.append(r"\begin{table}[t]")
    lines.append(r"\centering\small")
    lines.append(r"\caption{Benchmark results (mean $\pm$ 95\% CI) on 5G beam-level forecasting.}")
    lines.append(r"\label{tab:main}")
    lines.append(r"\begin{tabular}{l cc cc}")
    lines.append(r"\toprule")
    lines.append(r" & \multicolumn{2}{c}{\textbf{Week 6 (IID)}} & \multicolumn{2}{c}{\textbf{Week 11 (Shift)}} \\")
    lines.append(r"\cmidrule(lr){2-3}\cmidrule(lr){4-5}")
    lines.append(r"\textbf{Model} & MAE & MASE & MAE & MASE \\")
    lines.append(r"\midrule")
    for tw_pair in [("5w-6w", "10w-11w")]:
        for mn in sorted(df["model"].unique()):
            parts = []
            parts.append(mn.replace("+", r"\!+\!").replace("_", r"\_"))
            for tw in tw_pair:
                for metric in ["MAE", "MASE"]:
                    sub = df[(df["model"] == mn) & (df["test_week"] == tw)]
                    if len(sub) == 0:
                        parts.append("--")
                        continue
                    mean, lo, hi = bootstrap_ci(sub[metric].values)
                    half = (hi - lo) / 2
                    parts.append(f"{mean:.3f}$\\pm${half:.3f}")
            lines.append(" & ".join(parts) + r" \\")
    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"\end{table}")
    return "\n".join(lines)

# ── 6. Cross-dataset rank comparison ────────────────────────────────────────
def cross_dataset_ranks(gotsf_df, irish_df, metric="MASE"):
    """Compare model MASE rankings across datasets."""
    rows = []
    for label, df, tw in [("GOTSF-W6", gotsf_df, "5w-6w"),
                           ("GOTSF-W11", gotsf_df, "10w-11w"),
                           ("Irish", irish_df, "irish")]:
        sub = df[df["test_week"] == tw] if tw != "irish" else df
        agg = sub.groupby("model")[metric].mean().sort_values()
        for rank, (mn, val) in enumerate(agg.items(), 1):
            rows.append({"dataset": label, "model": mn, metric: val, "rank": rank})
    return pd.DataFrame(rows)

# ── MAIN ────────────────────────────────────────────────────────────────────
def analyze_dataset(df, dataset_name="GOTSF"):
    print(f"\n{'#'*80}")
    print(f"  DATASET: {dataset_name}")
    print(f"  {len(df)} rows, {df['model'].nunique()} models, "
          f"{df['beam_id'].nunique()} series, weeks: {df['test_week'].unique().tolist()}")
    print(f"{'#'*80}")

    # 1. Bootstrap CIs
    print("\n" + "="*80)
    print("  1. RESULTS WITH 95% BOOTSTRAP CONFIDENCE INTERVALS")
    print("="*80)
    for tw in sorted(df["test_week"].unique()):
        print(f"\n  --- {tw} ---")
        print(f"  {'Model':<20} {'MAE':>18} {'RMSE':>18} {'MASE':>18}")
        print(f"  {'-'*74}")
        for mn in sorted(df[df["test_week"]==tw]["model"].unique()):
            sub = df[(df["model"]==mn) & (df["test_week"]==tw)]
            parts = [f"  {mn:<20}"]
            for metric in ["MAE", "RMSE", "MASE"]:
                m, lo, hi = bootstrap_ci(sub[metric].values)
                parts.append(f"{m:.4f} [{lo:.4f},{hi:.4f}]")
            print(" ".join(parts))

    # 2. Friedman + Nemenyi
    print("\n" + "="*80)
    print("  2. FRIEDMAN + NEMENYI TESTS")
    print("="*80)
    for tw in sorted(df["test_week"].unique()):
        print(f"\n  --- {tw} (metric=MASE) ---")
        res = friedman_nemenyi(df, metric="MASE", test_week=tw)
        if res is None:
            print("    Not enough data"); continue
        print(f"    Friedman χ²={res['chi2']:.2f}, p={res['p_friedman']:.2e}, "
              f"CD={res['cd']:.3f} (k={res['k_models']}, n={res['n_beams']})")
        print(f"    Mean ranks:")
        for mn, r in sorted(res["mean_ranks"].items(), key=lambda x: x[1]):
            print(f"      {mn:<20} {r:.2f}")
        sig = res["pairwise"][res["pairwise"]["significant"]]
        if len(sig) > 0:
            print(f"    Significant pairwise differences ({len(sig)}):")
            for _, row in sig.head(20).iterrows():
                print(f"      {row['model_a']} vs {row['model_b']}: Δrank={row['rank_diff']:.2f}")
            if len(sig) > 20:
                print(f"      ... and {len(sig)-20} more")

    # 3. Sparsity stratification
    if "zero_frac" in df.columns and df["zero_frac"].notna().any():
        print("\n" + "="*80)
        print("  3. SPARSITY STRATIFICATION (MASE)")
        print("="*80)
        sp = sparsity_analysis(df, metric="MASE")
        for tw in sorted(sp["test_week"].unique()):
            print(f"\n  --- {tw} ---")
            sub = sp[sp["test_week"]==tw]
            pivot = sub.pivot_table(index="model", columns="sparsity_bin",
                                     values="mean")
            print(pivot.to_string())
        sp.to_csv(RESULTS_DIR / f"sparsity_{dataset_name.lower()}.csv", index=False)

    # 4. Distribution shift (only for GOTSF with 2 weeks)
    weeks = df["test_week"].unique()
    if len(weeks) > 1:
        print("\n" + "="*80)
        print(f"  4. DISTRIBUTION SHIFT ANALYSIS ({weeks[0]} → {weeks[1]})")
        print("="*80)
        sh = shift_analysis(df, metric="MAE")
        print(f"  {'Model':<20} {'MAE_W6':>8} {'MAE_W11':>8} {'Degrad%':>8} {'Wilcoxon_p':>10}")
        print(f"  {'-'*60}")
        for _, r in sh.iterrows():
            print(f"  {r['model']:<20} {r['MAE_w6']:>8.4f} {r['MAE_w11']:>8.4f} "
                  f"{r['degradation_pct']:>+7.1f}% {r['wilcoxon_p']:>10.2e}")
        sh.to_csv(RESULTS_DIR / f"shift_{dataset_name.lower()}.csv", index=False)

    # Save Friedman results
    for tw in sorted(weeks):
        res = friedman_nemenyi(df, metric="MASE", test_week=tw)
        if res:
            pd.DataFrame([res["mean_ranks"]]).to_csv(
                RESULTS_DIR / f"friedman_ranks_{dataset_name.lower()}_{tw}.csv", index=False)
            res["pairwise"].to_csv(
                RESULTS_DIR / f"nemenyi_{dataset_name.lower()}_{tw}.csv", index=False)
    return df

def main():
    # Analyze GOTSF
    gotsf = load("gotsf")
    analyze_dataset(gotsf, "GOTSF")

    # Analyze Irish
    irish_path = RESULTS_DIR / "per_beam_results_irish.csv"
    irish = None
    if irish_path.exists():
        irish = load("irish")
        analyze_dataset(irish, "Irish")

    # Cross-dataset comparison
    if irish is not None:
        print("\n" + "#"*80)
        print("  5. CROSS-DATASET GENERALIZATION (MASE rank comparison)")
        print("#"*80)
        xd = cross_dataset_ranks(gotsf, irish, metric="MASE")
        pivot = xd.pivot_table(index="model", columns="dataset",
                                values="rank", aggfunc="first")
        cols = [c for c in ["GOTSF-W6", "GOTSF-W11", "Irish"] if c in pivot.columns]
        pivot = pivot[cols]
        pivot["avg_rank"] = pivot.mean(axis=1)
        pivot = pivot.sort_values("avg_rank")
        print(f"\n  {'Model':<20}", end="")
        for c in cols:
            print(f"{c:>12}", end="")
        print(f"{'Avg':>8}")
        print(f"  {'-'*(20+12*len(cols)+8)}")
        for mn, row in pivot.iterrows():
            print(f"  {mn:<20}", end="")
            for c in cols:
                print(f"{row[c]:>12.0f}", end="")
            print(f"{row['avg_rank']:>8.1f}")
        xd.to_csv(RESULTS_DIR / "cross_dataset_ranks.csv", index=False)
        print(f"\n  Cross-dataset ranks → {RESULTS_DIR / 'cross_dataset_ranks.csv'}")

    # LaTeX tables
    latex = latex_main_table(gotsf)
    (RESULTS_DIR / "main_table.tex").write_text(latex)
    print(f"  LaTeX table → {RESULTS_DIR / 'main_table.tex'}")
    print(f"  All analysis CSVs saved to {RESULTS_DIR}")

if __name__ == "__main__":
    main()
