"""Chronos-2 cross-learning WITHOUT covariates (true univariate XL), per BS, W6 and W11.
Also re-runs cross-learning WITH covariates as a sanity check against the paper's XL (0.650/0.707)."""
import sys as _sys
from pathlib import Path as _P
_sys.path.insert(0, str(_P(__file__).resolve().parent.parent))
from paths import ROOT, GOTSF_DIR, IRISH_DIR, RESULTS_DIR, FIG_DIR
import sys, time
import numpy as np, pandas as pd
from benchmark import (load_all, get_beam_cols, beams_for_bs, to_chronos_df, to_chronos_future,
                       eval_beam, _valid, TARGET_KPI, QUANTILE_LEVELS)
from chronos import Chronos2Pipeline
pipe = Chronos2Pipeline.from_pretrained("amazon/chronos-2", device_map="cuda")
for m in pipe.model.modules():  # transformers/chronos version skew in this venv
    cfg = getattr(m, "config", None)
    if cfg is not None and not hasattr(cfg, "_attn_implementation"):
        object.__setattr__(cfg, "_attn_implementation", getattr(cfg, "attn_implementation", None) or "eager")
H = 168
rows = []
for tw in ["5w-6w", "10w-11w"]:
    data = load_all(tw); bc = get_beam_cols(data, 30)
    for label, ctx_mode in [("Chronos2-XL-Uni", "univariate"), ("Chronos2-XL-Cov(repro)", "cross_learning")]:
        t0 = time.time()
        for bs in range(30):
            b = beams_for_bs(bc, bs)
            pdf = pipe.predict_df(to_chronos_df(data, b, ctx_mode), future_df=to_chronos_future(data, b, ctx_mode),
                                  prediction_length=H, quantile_levels=QUANTILE_LEVELS, id_column="item_id",
                                  timestamp_column="timestamp", target="target", batch_size=64, cross_learning=True)
            for x in b:
                bp = pdf[pdf.item_id == x].sort_values("timestamp")
                qd = {q: bp[str(q)].values[:H] for q in QUANTILE_LEVELS}
                r = eval_beam(_valid(data[f"{TARGET_KPI}_test"][x].values)[:H], bp["0.5"].values[:H],
                              _valid(data[f"{TARGET_KPI}_train"][x].values), qd)
                r.update(beam_id=x, model=label, test_week=tw); rows.append(r)
        d = pd.DataFrame([r for r in rows if r["model"] == label and r["test_week"] == tw])
        print(f"{tw} {label}: MASE {d.MASE.mean():.3f} PICP {d.PICP_80.mean():.3f} CRPS {d.CRPS.mean():.3f} ({time.time()-t0:.0f}s)", flush=True)
pd.DataFrame(rows).to_csv(str(RESULTS_DIR / "audit" / "xl_uni.csv"), index=False)
print("DONE")
