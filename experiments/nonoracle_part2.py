"""Part 2: past-only covariates, two-stage forecasted covariates, and beam-sleep policy eval."""
import sys as _sys
from pathlib import Path as _P
_sys.path.insert(0, str(_P(__file__).resolve().parent.parent))
from paths import ROOT, GOTSF_DIR, IRISH_DIR, RESULTS_DIR, FIG_DIR
import numpy as np, pandas as pd, time
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

md = pd.read_csv(ROOT / "results" / "beam_metadata.csv")
bins = pd.cut(md.zero_frac, [0, .10, .30, .50, .70], include_lowest=True)
subset = (md.assign(bin=bins).groupby("bin", observed=True, group_keys=False)
            .apply(lambda g: g.sample(max(1, int(round(len(g) * 0.10))), random_state=42)))
beams = subset.beam_id.tolist()

train_t = load(TARGET, "train")
test_t = load(TARGET, "test")
train_c = {c: load(c, "train") for c in COVS}

from chronos import Chronos2Pipeline
import torch
dev = "mps" if torch.backends.mps.is_available() else "cpu"
pipe = Chronos2Pipeline.from_pretrained("amazon/chronos-2", device_map=dev)

def ctx_df(with_cov):
    frames = []
    for b in beams:
        y = _valid(train_t[b].values)
        d = {"item_id": b, "timestamp": np.arange(len(y)), "target": y}
        if with_cov:
            for c in COVS:
                d[c] = _valid(train_c[c][b].values)
        frames.append(pd.DataFrame(d))
    return pd.concat(frames, ignore_index=True)

# --- Stage 1: forecast each covariate univariately with Chronos-2 ---
print("Stage 1: forecasting covariates univariately...")
cov_fcst = {}
for c in COVS:
    frames = [pd.DataFrame({"item_id": b, "timestamp": np.arange(len(_valid(train_c[c][b].values))),
                            "target": _valid(train_c[c][b].values)}) for b in beams]
    ctx = pd.concat(frames, ignore_index=True)
    pdf = pipe.predict_df(ctx, prediction_length=H, quantile_levels=[0.5],
                          id_column="item_id", timestamp_column="timestamp",
                          target="target", batch_size=32)
    cov_fcst[c] = {b: pdf[pdf.item_id == b].sort_values("timestamp")["0.5"].values[:H] for b in beams}

def futr_forecasted():
    frames = []
    for b in beams:
        tl = len(_valid(train_t[b].values))
        d = {"item_id": b, "timestamp": np.arange(tl, tl + H)}
        for c in COVS:
            d[c] = cov_fcst[c][b].astype(np.float32)
        frames.append(pd.DataFrame(d))
    return pd.concat(frames, ignore_index=True)

def run(label, fut, with_cov=True, save_q=False):
    pdf = pipe.predict_df(ctx_df(with_cov), future_df=fut, prediction_length=H,
                          quantile_levels=QL, id_column="item_id",
                          timestamp_column="timestamp", target="target", batch_size=32)
    rows, qstore = [], {}
    for b in beams:
        bp = pdf[pdf["item_id"] == b].sort_values("timestamp")
        true = _valid(test_t[b].values)[:H]
        med = bp["0.5"].values[:H]
        mae = float(np.mean(np.abs(true - med)))
        rows.append({"beam_id": b, "model": label,
                     "MASE": mae / mase_denom(_valid(train_t[b].values))})
        if save_q:
            qstore[b] = {q: bp[str(q)].values[:H] for q in QL}
    return pd.DataFrame(rows), qstore

print("Running PastCov (covariates as historical context only, no future)...")
r_past, q_past = run("Chronos2-PastCov", None, save_q=True)
print("Running FcstCov (two-stage: Chronos-2 forecasts covariates, then target)...")
r_fcst, q_fcst = run("Chronos2-FcstCov", futr_forecasted(), save_q=True)

out = pd.concat([r_past, r_fcst], ignore_index=True)
prev = pd.read_csv(RESULTS_DIR / "nonoracle_results.csv")
allr = pd.concat([prev, out], ignore_index=True)
allr.to_csv(RESULTS_DIR / "nonoracle_results.csv", index=False)

paper = pd.read_csv(ROOT / "results" / "per_beam_results.csv")
p6 = paper[(paper.test_week == "5w-6w") & paper.model.isin(["Chronos2-Uni", "Chronos2-Cov", "NHITS+Cov"])]
p6 = p6[p6.beam_id.isin(beams)][["beam_id", "model", "MASE"]]
full = pd.concat([allr, p6], ignore_index=True).merge(md[["beam_id", "zero_frac"]], on="beam_id")
full["bin"] = pd.cut(full.zero_frac, [0, .10, .30, .50, .70], include_lowest=True)
print("\n=== Mean MASE on subset (n=%d) ===" % len(beams))
print(full.groupby("model").MASE.mean().round(3).sort_values())
print("\n=== By bin ===")
print(full.groupby(["model", "bin"], observed=True).MASE.mean().unstack().round(3))

piv = full.pivot_table(index="beam_id", columns="model", values="MASE")
for lab in ["Chronos2-PastCov", "Chronos2-FcstCov", "Chronos2-LagCov"]:
    g_or = (piv["Chronos2-Uni"] - piv["Chronos2-Cov"]).median()
    g = (piv["Chronos2-Uni-repro"] - piv[lab]).median()
    wins = (piv[lab] < piv["Chronos2-Uni-repro"]).mean()
    print(f"{lab}: median delta vs Uni={g:+.4f} (oracle {g_or:+.4f}, retained {100*g/g_or:.0f}%), wins {100*wins:.0f}%")

# --- Beam-sleep decision-level evaluation ---
print("\n=== Beam-sleep policy (sleep if q90 forecast < eps) ===")
eps = 1e-6  # traffic considered zero
for label, qs in [("PastCov", q_past), ("FcstCov", q_fcst)]:
    for conf, qlev in [("q80", 0.8), ("q90", 0.9)]:
        stats = {"trans": [], "all": []}
        for b in beams:
            true = _valid(test_t[b].values)[:H]
            up = qs[b][qlev]
            sleep = up <= 0.01  # upper quantile below 1% of normalized traffic
            truly_zero = true <= eps
            n_sleep = sleep.sum()
            missed = (sleep & ~truly_zero).sum() / max(n_sleep, 1)          # false sleep rate
            captured = (sleep & truly_zero).sum() / max(truly_zero.sum(), 1)  # sleep capture
            zf = md.set_index("beam_id").loc[b, "zero_frac"]
            rec = (n_sleep / H, missed, captured)
            stats["all"].append(rec)
            if 0.3 < zf <= 0.5:
                stats["trans"].append(rec)
        for k in ["all", "trans"]:
            a = np.array(stats[k])
            print(f"{label} {conf} [{k}]: sleep-eligible {100*a[:,0].mean():.0f}% of hours, "
                  f"false-sleep rate {100*a[:,1].mean():.1f}%, zero-hours captured {100*a[:,2].mean():.0f}%")
