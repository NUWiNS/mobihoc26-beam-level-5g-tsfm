"""STL (period=168, robust) variance shares per sparsity bin, all 2,880 beams (paper §4.4 wording)."""
import sys as _sys
from pathlib import Path as _P
_sys.path.insert(0, str(_P(__file__).resolve().parent.parent))
from paths import ROOT, GOTSF_DIR, IRISH_DIR, RESULTS_DIR, FIG_DIR
import sys, numpy as np, pandas as pd
from statsmodels.tsa.seasonal import STL
from joblib import Parallel, delayed
from benchmark import load_wide
V = load_wide("DLThpVol", "train"); zf = (V == 0).mean()
def shares(x):
    r = STL(np.asarray(x, float), period=168, robust=True).fit()
    v = np.array([np.var(r.seasonal), np.var(r.trend), np.var(r.resid)]); return v / v.sum()
S = np.array(Parallel(n_jobs=32)(delayed(shares)(V[c]) for c in V.columns))
df = pd.DataFrame(S, columns=["seasonal", "trend", "resid"], index=V.columns)
df["bin"] = pd.cut(zf, [0, .1, .3, .5, .7], include_lowest=True)
print((df.groupby("bin", observed=True)[["seasonal", "trend", "resid"]].mean() * 100).round(1).to_string())
fine = pd.cut(zf, np.arange(0, .6, .05))
g = df.groupby(fine, observed=True)[["seasonal", "resid"]].mean()
print("crossover (first 5%-bin where resid > seasonal):", g.index[(g.resid > g.seasonal).values][0])
df.to_csv(str(RESULTS_DIR / "audit" / "stl_shares.csv"))
