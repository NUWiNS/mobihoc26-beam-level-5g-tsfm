"""Summarise GraphWaveNet seeds against the paper's per-beam results."""
import sys as _sys
from pathlib import Path as _P
_sys.path.insert(0, str(_P(__file__).resolve().parent.parent))
from paths import ROOT, GOTSF_DIR, IRISH_DIR, RESULTS_DIR, FIG_DIR
import glob
import numpy as np, pandas as pd
from scipy.stats import wilcoxon

R = str(RESULTS_DIR)
g = pd.concat([pd.read_csv(f) for f in sorted(glob.glob(str(RESULTS_DIR / "gwn" / "gwn_seed*.csv")))])
seeds = sorted(g.seed.unique())
print(f"seeds: {seeds}")

print("\n== mean MASE per seed")
per_seed = g.groupby(["test_week", "seed"]).MASE.mean().unstack(0)
print(per_seed.round(4).to_string())
med = per_seed.median()
print("median over seeds:", med.round(3).to_dict())

# per-beam median over seeds -> comparable per-beam series
pb = g.groupby(["beam_id", "test_week"]).MASE.median().reset_index()
pb["model"] = "GraphWaveNet"
base = pd.read_csv(f"{R}/per_beam_results.csv")
meta = pd.read_csv(f"{R}/beam_metadata.csv")[["beam_id", "zero_frac"]]

print("\n== shift degradation W6->W11 (mean MASE, median-of-seeds)")
print(f"GraphWaveNet: {100*(med['10w-11w']/med['5w-6w']-1):+.1f}%")

print("\n== sparsity bins, W6 (per-beam median over seeds)")
bins = [(0, .1), (.1, .3), (.3, .5), (.5, 1.01)]
def binned(df):
    d = df.merge(meta, on="beam_id")
    return [d[(d.zero_frac >= lo) & (d.zero_frac < hi)].MASE.mean() for lo, hi in bins]
w6 = pb[pb.test_week == "5w-6w"]
rows = {"GraphWaveNet": binned(w6)}
for m in ["Chronos2-Uni", "Chronos2-Cov", "Chronos2-XL", "NHITS", "NHITS+Cov", "PatchTST"]:
    rows[m] = binned(base[(base.model == m) & (base.test_week == "5w-6w")])
print(pd.DataFrame(rows, index=["0-10", "10-30", "30-50", "50+"]).T.round(3).to_string())

print("\n== paired per-beam Wilcoxon (GWN vs model): win rate of the OTHER model, p")
for tw in ["5w-6w", "10w-11w"]:
    a = pb[pb.test_week == tw].set_index("beam_id").MASE
    for m in ["Chronos2-Uni", "Chronos2-Cov", "Chronos2-XL", "NHITS", "NHITS+Cov", "PatchTST"]:
        b = base[(base.model == m) & (base.test_week == tw)].set_index("beam_id").MASE
        a2, b2 = a.align(b, join="inner")
        p = wilcoxon(a2, b2).pvalue
        print(f"  {tw} {m:13s} mean {b2.mean():.3f} vs GWN {a2.mean():.3f} | {m} wins {100*(b2<a2).mean():.0f}% p={p:.1e}")

print("\n== timing (s) per seed:", g.groupby("seed").time_s.first().round(0).to_dict(),
      "epochs:", g.groupby("seed").epochs.first().to_dict())
