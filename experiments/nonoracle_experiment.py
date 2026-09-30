"""Non-oracle (lagged/seasonal-naive) covariate experiment on a stratified beam subset.

Replicates benchmark.py's data pipeline and metrics exactly, but future covariates
are the LAST WEEK OF TRAINING covariates repeated (seasonal-naive), i.e. fully
available at inference time. Also runs a context-length ablation for Chronos2-Uni.
"""
import sys as _sys
from pathlib import Path as _P
_sys.path.insert(0, str(_P(__file__).resolve().parent.parent))
from paths import ROOT, GOTSF_DIR, IRISH_DIR, RESULTS_DIR, FIG_DIR
import numpy as np, pandas as pd, time, sys
from pathlib import Path

DATA = GOTSF_DIR
TARGET = "DLThpVol"
COVS = ["DLThpTime", "DLPRB", "MR_number"]
H = 168
QL = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]

def load(kpi, split, tw="5w-6w"):
    p = DATA / ("train" if split == "train" else "test") / (
        f"{kpi}_train_0w-5w.csv" if split == "train" else f"{kpi}_test_{tw}.csv")
    return pd.read_csv(p, index_col=0)

def _valid(a):
    a = np.asarray(a, dtype=np.float32)
    m = ~np.isnan(a)
    return a if m.all() else a[: (np.where(m)[0][-1] + 1 if m.any() else 0)]

def mase_denom(tv, season=168):
    y = np.asarray(tv, dtype=np.float64)
    if len(y) <= season:
        return max(np.mean(np.abs(np.diff(y))), 1e-8)
    return max(np.mean(np.abs(y[season:] - y[:-season])), 1e-8)

# --- stratified subset (seed 42, ~10% per sparsity bin) ---
md = pd.read_csv(ROOT / "results" / "beam_metadata.csv")
bins = pd.cut(md.zero_frac, [0, .10, .30, .50, .70], include_lowest=True)
rng = np.random.default_rng(42)
subset = (md.assign(bin=bins).groupby("bin", observed=True, group_keys=False)
            .apply(lambda g: g.sample(max(1, int(round(len(g) * 0.10))), random_state=42)))
beams = subset.beam_id.tolist()
print(f"Subset: {len(beams)} beams", subset.groupby("bin", observed=True).size().to_dict())

train_t = load(TARGET, "train")
test_t = load(TARGET, "test")
train_c = {c: load(c, "train") for c in COVS}

def ctx_df(beams, with_cov, ctx_len=None):
    frames = []
    for b in beams:
        y = _valid(train_t[b].values)
        cvs = {c: _valid(train_c[c][b].values) for c in COVS} if with_cov else {}
        if ctx_len:
            y = y[-ctx_len:]
            cvs = {c: v[-ctx_len:] for c, v in cvs.items()}
        d = {"item_id": b, "timestamp": np.arange(len(y)), "target": y}
        d.update(cvs)
        frames.append(pd.DataFrame(d))
    return pd.concat(frames, ignore_index=True)

def futr_df_lagged(beams, ctx_len=None):
    """Seasonal-naive future covariates: repeat last 168h of TRAINING covariates."""
    frames = []
    for b in beams:
        tl = len(_valid(train_t[b].values))
        if ctx_len:
            tl = min(tl, ctx_len)
        d = {"item_id": b, "timestamp": np.arange(tl, tl + H)}
        for c in COVS:
            cv = _valid(train_c[c][b].values)
            d[c] = cv[-168:][:H].astype(np.float32)
        frames.append(pd.DataFrame(d))
    return pd.concat(frames, ignore_index=True)

from chronos import Chronos2Pipeline
import torch
dev = "mps" if torch.backends.mps.is_available() else "cpu"
pipe = Chronos2Pipeline.from_pretrained("amazon/chronos-2", device_map=dev)
print("device:", dev)

def run(label, with_cov, lagged, ctx_len=None):
    ctx = ctx_df(beams, with_cov, ctx_len)
    fut = futr_df_lagged(beams, ctx_len) if (with_cov and lagged) else None
    t0 = time.time()
    pdf = pipe.predict_df(ctx, future_df=fut, prediction_length=H, quantile_levels=QL,
                          id_column="item_id", timestamp_column="timestamp",
                          target="target", batch_size=32)
    print(f"{label}: predict {time.time()-t0:.0f}s")
    rows = []
    for b in beams:
        bp = pdf[pdf["item_id"] == b].sort_values("timestamp")
        true = _valid(test_t[b].values)[:H]
        med = bp["0.5"].values[:H]
        mae = float(np.mean(np.abs(true - med)))
        rows.append({"beam_id": b, "model": label,
                     "MASE": mae / mase_denom(_valid(train_t[b].values))})
    return pd.DataFrame(rows)

res = []
res.append(run("Chronos2-Uni-repro", False, False))
res.append(run("Chronos2-LagCov", True, True))
res.append(run("Chronos2-Uni-c168", False, False, ctx_len=168))
res.append(run("Chronos2-Uni-c336", False, False, ctx_len=336))
res.append(run("Chronos2-Uni-c672", False, False, ctx_len=672))
out = pd.concat(res, ignore_index=True)
out.to_csv(RESULTS_DIR / "nonoracle_results.csv", index=False)

# --- compare with paper's oracle results on the SAME beams ---
paper = pd.read_csv(ROOT / "results" / "per_beam_results.csv")
p6 = paper[(paper.test_week == "5w-6w") & (paper.model.isin(["Chronos2-Uni", "Chronos2-Cov", "NHITS+Cov"]))]
p6 = p6[p6.beam_id.isin(beams)][["beam_id", "model", "MASE"]]
allr = pd.concat([out, p6], ignore_index=True).merge(md[["beam_id", "zero_frac"]], on="beam_id")
allr["bin"] = pd.cut(allr.zero_frac, [0, .10, .30, .50, .70], include_lowest=True)
print("\n=== Mean MASE on subset (n=%d) ===" % len(beams))
print(allr.groupby("model").MASE.mean().round(3).sort_values())
print("\n=== By sparsity bin ===")
print(allr.groupby(["model", "bin"], observed=True).MASE.mean().unstack().round(3))
# retained fraction of oracle gain
piv = allr.pivot_table(index="beam_id", columns="model", values="MASE")
gain_oracle = (piv["Chronos2-Uni"] - piv["Chronos2-Cov"]).median()
gain_lag = (piv["Chronos2-Uni-repro"] - piv["Chronos2-LagCov"]).median()
print(f"\nMedian per-beam MASE reduction vs Uni: oracle={gain_oracle:.4f}, lagged={gain_lag:.4f}, retained={100*gain_lag/gain_oracle:.0f}%")
wins = (piv["Chronos2-LagCov"] < piv["Chronos2-Uni-repro"]).mean()
print(f"LagCov beats Uni on {100*wins:.0f}% of subset beams")
