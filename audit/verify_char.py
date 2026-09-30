import sys as _sys
from pathlib import Path as _P
_sys.path.insert(0, str(_P(__file__).resolve().parent.parent))
from paths import ROOT, GOTSF_DIR, IRISH_DIR, RESULTS_DIR, FIG_DIR
import sys, numpy as np, pandas as pd
from benchmark import load_wide, load_irish, IRISH_TARGET
V = load_wide("DLThpVol", "train"); T = load_wide("DLThpTime", "train")
P = load_wide("DLPRB", "train"); M = load_wide("MR_number", "train")
cols = V.columns; bs = np.array([int(c.split("_")[0]) for c in cols]); cell = np.array([int(c.split("_")[1]) for c in cols])
C = np.corrcoef(V.values.T)  # all pairs
iu = np.triu_indices(len(cols), 1)
same_bs = bs[iu[0]] == bs[iu[1]]; same_cell = same_bs & (cell[iu[0]] == cell[iu[1]])
r = C[iu]
print("SPATIAL all-pairs median r: intra-cell %.3f  inter-cell %.3f  inter-BS %.3f" % (
    np.median(r[same_cell]), np.median(r[same_bs & ~same_cell]), np.median(r[~same_bs])))
# ACF24
def acf(x, k):
    x = np.asarray(x, float); return np.corrcoef(x[k:], x[:-k])[0, 1] if x.std() > 0 else 0
bsV = V.T.groupby(bs).sum().T
a_bs = np.array([acf(bsV[c], 24) for c in bsV.columns])
print("BS ACF24: min %.2f median %.2f max %.2f frac>0.8 %.2f" % (a_bs.min(), np.median(a_bs), a_bs.max(), (a_bs > .8).mean()))
zf = (V == 0).mean()
bins = pd.cut(zf, [0, .1, .3, .5, .7], include_lowest=True)
a24 = pd.Series([acf(V[c], 24) for c in cols], index=cols); a168 = pd.Series([acf(V[c], 168) for c in cols], index=cols)
print("beam ACF24 median by bin:", a24.groupby(bins, observed=True).median().round(2).tolist(), " ACF168:", a168.groupby(bins, observed=True).median().round(2).tolist())
for name, X in [("DLThpTime", T), ("DLPRB", P), ("MR_number", M)]:
    rr = pd.Series([np.corrcoef(V[c], X[c])[0, 1] if V[c].std() > 0 and X[c].std() > 0 else np.nan for c in cols], index=cols)
    print(f"cov r {name} median by bin:", rr.groupby(bins, observed=True).median().round(2).tolist(), " mean:", rr.groupby(bins, observed=True).mean().round(2).tolist())
z = V.values == 0
print("co-sparsity: P(vol=0) %.3f  P(time=0|vol=0) %.3f  P(time!=0|vol=0) %.3f  P(vol=0 & time!=0) %.3f" % (
    z.mean(), (T.values[z] == 0).mean(), (T.values[z] != 0).mean(), (z & (T.values != 0)).mean()))
w11 = load_wide("DLThpVol", "test", "10w-11w"); w6 = load_wide("DLThpVol", "test", "5w-6w")
m5 = V.iloc[672:840].mean(); ch11 = ((w11.mean() - m5).abs() / m5.replace(0, np.nan)); ch6 = ((w6.mean() - m5).abs() / m5.replace(0, np.nan))
print("weekly mean change W5->W11 median %.3f (>40%%: %.3f) ; W5->W6 median %.3f" % (ch11.median(), (ch11 > .4).mean(), ch6.median()))
data, sids, meta = load_irish()
y = data[f"{IRISH_TARGET}_train"]
lens = np.array([int(y[s].notna().sum()) + 60 for s in sids])
print("IRISH n=%d zero frac mean %.3f ; session length s min %d median %d max %d ; >600: %d" % (len(sids), meta.zero_frac.mean(), lens.min(), np.median(lens), lens.max(), (lens > 600).sum()))
