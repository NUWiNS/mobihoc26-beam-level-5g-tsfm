"""Beam-sleep decision-level eval with threshold sweep, using deployable (univariate) Chronos-2 quantiles."""
import sys as _sys
from pathlib import Path as _P
_sys.path.insert(0, str(_P(__file__).resolve().parent.parent))
from paths import ROOT, GOTSF_DIR, IRISH_DIR, RESULTS_DIR, FIG_DIR
import numpy as np, pandas as pd
from pathlib import Path

DATA = GOTSF_DIR
H = 168
QL = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]

def _valid(a):
    a = np.asarray(a, dtype=np.float32)
    m = ~np.isnan(a)
    return a if m.all() else a[: (np.where(m)[0][-1] + 1 if m.any() else 0)]

md = pd.read_csv(ROOT / "results" / "beam_metadata.csv")
bins = pd.cut(md.zero_frac, [0, .10, .30, .50, .70], include_lowest=True)
subset = (md.assign(bin=bins).groupby("bin", observed=True, group_keys=False)
            .apply(lambda g: g.sample(max(1, int(round(len(g) * 0.10))), random_state=42)))
beams = subset.beam_id.tolist()
train_t = pd.read_csv(DATA / "train" / "DLThpVol_train_0w-5w.csv", index_col=0)
test_t = pd.read_csv(DATA / "test" / "DLThpVol_test_5w-6w.csv", index_col=0)

from chronos import Chronos2Pipeline
import torch
pipe = Chronos2Pipeline.from_pretrained("amazon/chronos-2",
    device_map="mps" if torch.backends.mps.is_available() else "cpu")
frames = [pd.DataFrame({"item_id": b, "timestamp": np.arange(len(_valid(train_t[b].values))),
                        "target": _valid(train_t[b].values)}) for b in beams]
pdf = pipe.predict_df(pd.concat(frames, ignore_index=True), prediction_length=H,
                      quantile_levels=QL, id_column="item_id", timestamp_column="timestamp",
                      target="target", batch_size=32)

# diagnose: distribution of q80 upper bound during truly-zero hours
q80_zero, q80_nonzero = [], []
for b in beams:
    bp = pdf[pdf.item_id == b].sort_values("timestamp")
    true = _valid(test_t[b].values)[:H]
    q80 = bp["0.8"].values[:H]
    q80_zero.extend(q80[true == 0]); q80_nonzero.extend(q80[true > 0])
q80_zero, q80_nonzero = np.array(q80_zero), np.array(q80_nonzero)
print("q80 during zero hours: median %.4f p25 %.4f | during nonzero: median %.4f" % (
    np.median(q80_zero), np.percentile(q80_zero, 25), np.median(q80_nonzero)))

print("\nthreshold sweep (sleep if q80 < eps), transitional beams (30-50%):")
zmap = md.set_index("beam_id").zero_frac
for eps in [0.01, 0.02, 0.05, 0.10]:
    rows = []
    for b in beams:
        if not (0.3 < zmap[b] <= 0.5):
            continue
        bp = pdf[pdf.item_id == b].sort_values("timestamp")
        true = _valid(test_t[b].values)[:H]
        sleep = bp["0.8"].values[:H] < eps
        tz = true == 0
        rows.append((sleep.mean(), (sleep & ~tz).sum() / max(sleep.sum(), 1),
                     (sleep & tz).sum() / max(tz.sum(), 1),
                     true[sleep & ~tz].sum() / max(true.sum(), 1e-9)))
    a = np.array(rows)
    print(f"eps={eps}: sleep {100*a[:,0].mean():.0f}% of hours | false-sleep {100*a[:,1].mean():.1f}% "
          f"| zero-hours captured {100*a[:,2].mean():.0f}% | traffic missed {100*a[:,3].mean():.2f}%")
