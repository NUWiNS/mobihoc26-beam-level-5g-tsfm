import sys as _sys
from pathlib import Path as _P
_sys.path.insert(0, str(_P(__file__).resolve().parent.parent))
from paths import ROOT, GOTSF_DIR, IRISH_DIR, RESULTS_DIR, FIG_DIR
import sys, numpy as np, pandas as pd
from benchmark import load_wide
V = load_wide("DLThpVol", "train"); zf = (V == 0).mean(); bins = pd.cut(zf, [0, .1, .3, .5, .7], include_lowest=True)
def acf(x, k=24):
    x = np.asarray(x, float); return np.corrcoef(x[k:], x[:-k])[0, 1] if x.std() > 0 else 0
for k in ["DLThpTime", "DLPRB", "MR_number"]:
    X = load_wide(k, "train"); a = pd.Series([acf(X[c]) for c in X.columns], index=X.columns)
    print(k, "ACF24 median by bin:", a.groupby(bins, observed=True).median().round(2).tolist())
cols = V.columns; cell = V.T.groupby([c.rsplit("_", 1)[0] for c in cols]).sum().T
print("cell ACF24 median %.2f ; cell zero-frac max %.3f ; BS zero-frac max %.3f" % (np.median([acf(cell[c]) for c in cell.columns]), (cell == 0).mean().max(), (V.T.groupby([c.split('_')[0] for c in cols]).sum().T == 0).mean().max()))
C = np.corrcoef(V.values.T); iu = np.triu_indices(len(cols), 1)
key = np.array([c.rsplit("_", 1)[0] for c in cols]); same = key[iu[0]] == key[iu[1]]
avgz = (zf.values[iu[0]] + zf.values[iu[1]]) / 2; r = C[iu]
for lo, hi in [(0, .1), (.1, .2), (.2, .3), (.3, .4), (.4, .6)]:
    s = same & (avgz >= lo) & (avgz < hi); print(f"intra-cell pairs avg zf {lo}-{hi}: n={s.sum()} median r={np.median(r[s]):.3f}")
share = V.mean(); print("traffic share of 0-10%% beams: %.3f" % (share[bins == bins.cat.categories[0]].sum() / share.sum()))
