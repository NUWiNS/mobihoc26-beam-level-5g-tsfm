"""Decision-level beam-sleep evaluation (paper Sec. 7.2), threshold sweep.

Same stratified 10% subset as experiments/sleep_policy.py (289 beams,
random_state=42). Policy: sleep hour t if the forecast 80th percentile q80_t < eps,
i.e. the model is >=80% confident traffic is below eps. Evaluated on GOTSF Week 6
for (i) deployable univariate Chronos-2 and (ii) oracle-covariate Chronos-2.

Metrics per eps, over transitional beams (30-50% zeros) and over all subset beams:
  sleep_elig   fraction of hours put to sleep
  false_sleep  fraction of slept hours that actually carried traffic
  missed_act   fraction of active hours (traffic > 0) that were slept (missed activations)
  zero_capt    fraction of truly-zero hours that were slept (captured sleep opportunity)
  traffic_miss fraction of the beam's traffic volume falling in slept hours
"""
import sys as _sys
from pathlib import Path as _P
_sys.path.insert(0, str(_P(__file__).resolve().parent.parent))
from paths import ROOT, GOTSF_DIR, IRISH_DIR, RESULTS_DIR, FIG_DIR
import sys
import numpy as np, pandas as pd

from benchmark import (load_all, to_chronos_df, to_chronos_future, _valid, eval_beam,
                       TARGET_KPI, QUANTILE_LEVELS, RESULTS_DIR)

H = 168
md = pd.read_csv(RESULTS_DIR / "beam_metadata.csv")
bins = pd.cut(md.zero_frac, [0, .10, .30, .50, .70], include_lowest=True)
subset = (md.assign(bin=bins).groupby("bin", observed=True, group_keys=False)
            .apply(lambda g: g.sample(max(1, int(round(len(g) * 0.10))), random_state=42)))
beams = subset.beam_id.tolist()
print(f"subset beams: {len(beams)}")
zmap = md.set_index("beam_id").zero_frac

data = load_all("5w-6w")
from chronos import Chronos2Pipeline
pipe = Chronos2Pipeline.from_pretrained("amazon/chronos-2", device_map="cuda")
# transformers/chronos version skew in this venv: the attention layers read
# config._attn_implementation, which newer transformers no longer sets.
for m in pipe.model.modules():
    cfg = getattr(m, "config", None)
    if cfg is not None and not hasattr(cfg, "_attn_implementation"):
        object.__setattr__(cfg, "_attn_implementation", getattr(cfg, "attn_implementation", None) or "eager")

preds = {}
for label, mode in [("Chronos2-Uni", "univariate"), ("Chronos2-Cov (oracle)", "covariates")]:
    pdf = pipe.predict_df(to_chronos_df(data, beams, mode), future_df=to_chronos_future(data, beams, mode),
                          prediction_length=H, quantile_levels=QUANTILE_LEVELS, id_column="item_id",
                          timestamp_column="timestamp", target="target", batch_size=64)
    preds[label] = pdf

test = data[f"{TARGET_KPI}_test"]
train = data[f"{TARGET_KPI}_train"]
# sanity check: must reproduce the paper's subset MASE (Uni 0.722, Cov 0.612)
for label, pdf in preds.items():
    ms = [eval_beam(_valid(test[b].values)[:H],
                    pdf[pdf.item_id == b].sort_values("timestamp")["0.5"].values[:H],
                    _valid(train[b].values))["MASE"] for b in beams]
    print(f"REPRO {label}: subset MASE {np.mean(ms):.3f}")
rows = []
for label, pdf in preds.items():
    for b in beams:
        bp = pdf[pdf.item_id == b].sort_values("timestamp")
        true = _valid(test[b].values)[:H]
        q80 = bp["0.8"].values[:H]
        tz = true == 0
        for eps in [0.01, 0.02, 0.05, 0.10, 0.20]:
            s = q80 < eps
            rows.append(dict(model=label, beam=b, zf=zmap[b], eps=eps,
                             sleep_elig=s.mean(),
                             false_sleep=(s & ~tz).sum() / max(s.sum(), 1),
                             missed_act=(s & ~tz).sum() / max((~tz).sum(), 1),
                             zero_capt=(s & tz).sum() / max(tz.sum(), 1),
                             traffic_miss=true[s & ~tz].sum() / max(true.sum(), 1e-9),
                             q80_zero_med=np.median(q80[tz]) if tz.any() else np.nan))
df = pd.DataFrame(rows)
df.to_csv(str(RESULTS_DIR / "sleep_sweep.csv"), index=False)
cols = ["sleep_elig", "false_sleep", "missed_act", "zero_capt", "traffic_miss"]
for name, sel in [("TRANSITIONAL (30-50%)", (df.zf > .3) & (df.zf <= .5)), ("ALL SUBSET", df.zf >= 0)]:
    print(f"\n== {name}: n_beams={df[sel].beam.nunique()} (means over beams, %)")
    print((df[sel].groupby(["model", "eps"])[cols].mean() * 100).round(1).to_string())
print("\nmedian q80 during truly-zero hours:", df.groupby("model").q80_zero_med.median().round(3).to_dict())
print("median transitional-beam zero fraction in test week:",
      round(float(np.median([(_valid(test[b].values)[:H] == 0).mean() for b in beams if .3 < zmap[b] <= .5])), 3))
