"""
Are Time Series Foundation Models Ready for 5G?
Systematic benchmark on beam-level 5G traffic forecasting.
"""
import sys as _sys
from pathlib import Path as _P
_sys.path.insert(0, str(_P(__file__).resolve().parent))
from paths import ROOT, GOTSF_DIR, IRISH_DIR, RESULTS_DIR, FIG_DIR
import os, sys, time, json, warnings, argparse

os.environ["TOKENIZERS_PARALLELISM"] = "false"

import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.metrics import mean_absolute_error, mean_squared_error
warnings.filterwarnings("ignore")

DATA_DIR = GOTSF_DIR
IRISH_DIR = IRISH_DIR
RESULTS_DIR = RESULTS_DIR
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
KPIS = ["DLThpVol", "DLThpTime", "DLPRB", "MR_number"]
TARGET_KPI = "DLThpVol"
COVARIATE_KPIS = ["DLThpTime", "DLPRB", "MR_number"]
PREDICTION_LENGTH = 168
QUANTILE_LEVELS = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]

# Irish 5G dataset config
IRISH_TARGET = "DL_bitrate"
IRISH_COVARIATES = ["RSRP", "RSRQ", "SNR", "CQI"]
IRISH_PRED_LEN = 60  # 60 seconds

# === DATA: GOTSF (beam-level 5G traffic) ===
def load_wide(kpi, split, test_week="5w-6w"):
    if split == "train":
        p = DATA_DIR / "train" / f"{kpi}_train_0w-5w.csv"
    else:
        p = DATA_DIR / "test" / f"{kpi}_test_{test_week}.csv"
    return pd.read_csv(p, index_col=0)

def load_all(test_week="5w-6w"):
    d = {}
    for kpi in KPIS:
        d[f"{kpi}_train"] = load_wide(kpi, "train")
        d[f"{kpi}_test"] = load_wide(kpi, "test", test_week)
    return d

def get_beam_cols(data, num_bs=30):
    cols = data[f"{TARGET_KPI}_train"].columns.tolist()
    return [c for c in cols if int(c.split("_")[0]) < num_bs]

def beams_for_bs(beam_cols, bs_id):
    return [c for c in beam_cols if c.startswith(f"{bs_id}_")]

def beam_metadata(beam_cols, data):
    train = data[f"{TARGET_KPI}_train"]
    rows = []
    for b in beam_cols:
        parts = b.split("_")
        s = train[b]
        rows.append({"beam_id": b, "bs_id": int(parts[0]),
                      "cell_id": int(parts[1]), "beam_idx": int(parts[2]),
                      "zero_frac": float((s==0).mean()),
                      "mean_traffic": float(s.mean())})
    return pd.DataFrame(rows)

# === DATA: Irish 5G (client-side throughput traces) ===
import glob

def load_irish():
    """Load Irish 5G dataset: 83 per-second traces → wide-format DataFrames.
    Returns (data_dict, series_ids, metadata_df) with same interface as GOTSF."""
    files = sorted(glob.glob(str(IRISH_DIR / "**/*.csv"), recursive=True))
    h = IRISH_PRED_LEN
    target_col = IRISH_TARGET
    cov_cols = IRISH_COVARIATES
    all_kpis = [target_col] + cov_cols
    # Read and filter sessions with enough data
    sessions = []
    for f in files:
        try:
            df = pd.read_csv(f)
            # Ensure numeric columns exist
            for c in all_kpis:
                df[c] = pd.to_numeric(df[c], errors="coerce")
            df = df.dropna(subset=all_kpis)
            if len(df) < h * 3:  # need at least 3x prediction len
                continue
            sessions.append((Path(f).stem, df[all_kpis].reset_index(drop=True)))
        except Exception:
            continue
    print(f"  Irish 5G: {len(sessions)} sessions loaded (of {len(files)} total)")
    # Build wide DataFrames: columns = session IDs, rows = time steps
    # Align all sessions to same length by padding shorter with NaN
    # Train = all but last h steps, Test = last h steps
    series_ids = [s[0] for s in sessions]
    data = {}
    for kpi in all_kpis:
        train_dict, test_dict = {}, {}
        for sid, df in sessions:
            vals = df[kpi].values.astype(np.float32)
            train_dict[sid] = pd.Series(vals[:-h])
            test_dict[sid] = pd.Series(vals[-h:])
        data[f"{kpi}_train"] = pd.DataFrame(train_dict)
        data[f"{kpi}_test"] = pd.DataFrame(test_dict)
    # Metadata
    meta_rows = []
    for sid, df in sessions:
        s = df[target_col].values[:-h]
        meta_rows.append({"beam_id": sid, "bs_id": 0, "cell_id": 0, "beam_idx": 0,
                          "zero_frac": float((s == 0).mean()),
                          "mean_traffic": float(np.nanmean(s))})
    meta = pd.DataFrame(meta_rows)
    return data, series_ids, meta

def to_nf_long(data, beam_cols, with_cov=False):
    train = data[f"{TARGET_KPI}_train"]
    frames = []
    for b in beam_cols:
        y = train[b].values.astype(np.float32)
        valid = ~np.isnan(y)
        d = {"unique_id": b, "ds": np.arange(int(valid.sum())),
             "y": y[valid]}
        if with_cov:
            for cov in COVARIATE_KPIS:
                cv = data[f"{cov}_train"][b].values.astype(np.float32)
                d[cov] = cv[valid]
        frames.append(pd.DataFrame(d))
    return pd.concat(frames, ignore_index=True)

def nf_futr_df(data, beam_cols):
    """Future covariates DataFrame for NeuralForecast predict."""
    frames = []
    for b in beam_cols:
        y_train = data[f"{TARGET_KPI}_train"][b].values
        tl = int((~np.isnan(y_train)).sum())
        d = {"unique_id": b, "ds": np.arange(tl, tl + PREDICTION_LENGTH)}
        for cov in COVARIATE_KPIS:
            d[cov] = data[f"{cov}_test"][b].values[:PREDICTION_LENGTH].astype(np.float32)
        frames.append(pd.DataFrame(d))
    return pd.concat(frames, ignore_index=True)

def to_chronos_df(data, beam_cols, mode="univariate"):
    train = data[f"{TARGET_KPI}_train"]
    frames = []
    for b in beam_cols:
        y = train[b].values.astype(np.float32)
        valid = ~np.isnan(y)
        d = {"item_id": b, "timestamp": np.arange(int(valid.sum())),
             "target": y[valid]}
        if mode in ("covariates", "cross_learning"):
            for cov in COVARIATE_KPIS:
                cv = data[f"{cov}_train"][b].values.astype(np.float32)
                d[cov] = cv[valid]
        frames.append(pd.DataFrame(d))
    return pd.concat(frames, ignore_index=True)

def to_chronos_future(data, beam_cols, mode="univariate"):
    if mode == "univariate":
        return None
    frames = []
    for b in beam_cols:
        y_train = data[f"{TARGET_KPI}_train"][b].values
        tl = int((~np.isnan(y_train)).sum())
        d = {"item_id": b, "timestamp": np.arange(tl, tl + PREDICTION_LENGTH)}
        for cov in COVARIATE_KPIS:
            d[cov] = data[f"{cov}_test"][b].values[:PREDICTION_LENGTH].astype(np.float32)
        frames.append(pd.DataFrame(d))
    return pd.concat(frames, ignore_index=True)

def _valid(arr):
    """Strip trailing NaN from variable-length padded series."""
    a = np.asarray(arr, dtype=np.float32)
    mask = ~np.isnan(a)
    if mask.all():
        return a
    last_valid = np.where(mask)[0][-1] + 1 if mask.any() else 0
    return a[:last_valid]

# === METRICS ===
def mase_denom(tv, season=168):
    y = np.asarray(tv, dtype=np.float64)
    if len(y) <= season:
        d = np.mean(np.abs(np.diff(y))); return max(d, 1e-8)
    return max(np.mean(np.abs(y[season:] - y[:-season])), 1e-8)

def quantile_crps(yt, qd, qls):
    c = 0.0
    for ql in qls:
        e = yt - qd[ql]
        c += np.mean(np.where(e >= 0, ql*e, (ql-1.0)*e))
    return 2.0 * c / len(qls)

def eval_beam(yt, yp, tv, qd=None):
    yt, yp = np.asarray(yt, dtype=np.float64), np.asarray(yp, dtype=np.float64)
    mae = float(mean_absolute_error(yt, yp))
    rmse = float(np.sqrt(mean_squared_error(yt, yp)))
    r = {"MAE": mae, "RMSE": rmse, "MASE": mae/mase_denom(tv),
         "NMAE": mae/(np.mean(np.abs(yt))+1e-8)}
    if qd and 0.1 in qd and 0.9 in qd:
        q10, q90 = np.asarray(qd[0.1]), np.asarray(qd[0.9])
        r["PICP_80"] = float(np.mean((yt >= q10) & (yt <= q90)))
        r["CRPS"] = float(quantile_crps(yt, qd, sorted(qd.keys())))
    return r

# === MODEL: ARIMA ===
def run_arima(data, beam_cols, test_week):
    from statsmodels.tsa.arima.model import ARIMA as _ARIMA
    print("    Running ARIMA per-beam...")
    y_test, train = data[f"{TARGET_KPI}_test"], data[f"{TARGET_KPI}_train"]
    results, t0 = [], time.time()
    for i, b in enumerate(beam_cols):
        if (i+1) % 100 == 0: print(f"      {i+1}/{len(beam_cols)}")
        tv = _valid(train[b].values)
        try:
            pred = _ARIMA(tv, order=(5,1,0)).fit().forecast(steps=PREDICTION_LENGTH)
        except Exception:
            pred = np.full(PREDICTION_LENGTH, tv[-1])
        true = _valid(y_test[b].values)[:PREDICTION_LENGTH]
        m = eval_beam(true, pred, tv)
        m.update({"beam_id": b, "model": "ARIMA", "test_week": test_week})
        results.append(m)
    elapsed = time.time() - t0
    for r in results: r["time_s"] = elapsed
    return results

# === MODEL: NeuralForecast (Trained DL) ===
def run_nf_models(data, beam_cols, test_week, max_steps=5000, input_size=336,
                  with_cov=False):
    from neuralforecast import NeuralForecast
    from neuralforecast.models import PatchTST, TiDE, NHITS, DLinear, NBEATSx
    h = PREDICTION_LENGTH
    suffix = "+Cov" if with_cov else ""
    kw = dict(h=h, input_size=input_size, max_steps=max_steps,
              early_stop_patience_steps=10, val_check_steps=100,
              scaler_type="standard", random_seed=42)
    if with_cov:
        kw["futr_exog_list"] = COVARIATE_KPIS
        models = [TiDE(**kw), NHITS(**kw), NBEATSx(**kw)]
    else:
        models = [PatchTST(patch_len=24, stride=24, **kw),
                  TiDE(**kw), NHITS(**kw), DLinear(**kw)]
    cov_label = " +covariates" if with_cov else ""
    print(f"    Preparing NF long-format data{cov_label}...")
    train_df = to_nf_long(data, beam_cols, with_cov=with_cov)
    nf = NeuralForecast(models=models, freq=1)
    print(f"    Fitting {len(models)} models (max_steps={max_steps})...")
    t0 = time.time(); nf.fit(df=train_df, val_size=h); t_fit = time.time()-t0
    print(f"    Fit: {t_fit:.1f}s")
    print("    Predicting...")
    t0 = time.time()
    if with_cov:
        futr = nf_futr_df(data, beam_cols)
        preds = nf.predict(futr_df=futr)
    else:
        preds = nf.predict(df=train_df)
    t_pred = time.time()-t0
    print(f"    Predict: {t_pred:.1f}s")
    y_test, train_w = data[f"{TARGET_KPI}_test"], data[f"{TARGET_KPI}_train"]
    all_res = []
    model_names = [type(mdl).__name__ for mdl in models]
    for mn in model_names:
        for b in beam_cols:
            bp = preds[preds["unique_id"] == b][mn].values[:PREDICTION_LENGTH]
            true = _valid(y_test[b].values)[:PREDICTION_LENGTH]
            m = eval_beam(true, bp, _valid(train_w[b].values))
            m.update({"beam_id": b, "model": mn + suffix, "test_week": test_week,
                       "time_s": t_fit+t_pred})
            all_res.append(m)
    return all_res

# === MODEL: Chronos-2 ===
def run_chronos2(data, beam_cols, mode, test_week, batch_size=64):
    from chronos import Chronos2Pipeline
    label = {"univariate":"Chronos2-Uni","covariates":"Chronos2-Cov",
             "cross_learning":"Chronos2-XL"}[mode]
    pipeline = Chronos2Pipeline.from_pretrained("amazon/chronos-2", device_map="cuda")
    ctx = to_chronos_df(data, beam_cols, mode)
    fut = to_chronos_future(data, beam_cols, mode)
    t0 = time.time()
    pdf = pipeline.predict_df(ctx, future_df=fut, prediction_length=PREDICTION_LENGTH,
        quantile_levels=QUANTILE_LEVELS, id_column="item_id",
        timestamp_column="timestamp", target="target",
        batch_size=batch_size, cross_learning=(mode=="cross_learning"))
    elapsed = time.time() - t0
    y_test, tw = data[f"{TARGET_KPI}_test"], data[f"{TARGET_KPI}_train"]
    results = []
    for b in beam_cols:
        bp = pdf[pdf["item_id"]==b].sort_values("timestamp")
        true = _valid(y_test[b].values)[:PREDICTION_LENGTH]
        median = bp["0.5"].values[:PREDICTION_LENGTH]
        qd = {ql: bp[str(ql)].values[:PREDICTION_LENGTH]
              for ql in QUANTILE_LEVELS if str(ql) in bp.columns}
        m = eval_beam(true, median, _valid(tw[b].values), qd)
        m.update({"beam_id": b, "model": label, "test_week": test_week,
                   "time_s": elapsed})
        results.append(m)
    return results

# === MODEL: Chronos-Bolt ===
def run_bolt(data, beam_cols, test_week, batch_size=64):
    import torch
    from chronos import BaseChronosPipeline
    pipeline = BaseChronosPipeline.from_pretrained("amazon/chronos-bolt-base",
                                                    device_map="cuda")
    tw, y_test = data[f"{TARGET_KPI}_train"], data[f"{TARGET_KPI}_test"]
    inputs = [torch.tensor(_valid(tw[b].values), dtype=torch.float32) for b in beam_cols]
    mid = QUANTILE_LEVELS.index(0.5)
    results = []
    t0 = time.time()
    # Batch to avoid OOM on large beam counts
    CHUNK = min(batch_size, 256)
    for start in range(0, len(inputs), CHUNK):
        chunk_inputs = inputs[start:start+CHUNK]
        chunk_beams = beam_cols[start:start+CHUNK]
        qp = pipeline.predict_quantiles(chunk_inputs, prediction_length=PREDICTION_LENGTH,
                                         quantile_levels=QUANTILE_LEVELS)
        quantiles_tensor = qp[0]  # (chunk, h, n_quantiles)
        for i, b in enumerate(chunk_beams):
            true = _valid(y_test[b].values)[:PREDICTION_LENGTH]
            qa = quantiles_tensor[i].numpy()
            median = qa[:PREDICTION_LENGTH, mid]
            qd = {ql: qa[:PREDICTION_LENGTH, j] for j, ql in enumerate(QUANTILE_LEVELS)}
            m = eval_beam(true, median, _valid(tw[b].values), qd)
            m.update({"beam_id": b, "model": "Chronos-Bolt", "test_week": test_week})
            results.append(m)
        torch.cuda.empty_cache()
    elapsed = time.time() - t0
    for r in results: r["time_s"] = elapsed
    return results

# === MODEL: Moirai ===
def run_moirai(data, beam_cols, test_week, batch_size=32):
    import torch
    from uni2ts.model.moirai import MoiraiForecast, MoiraiModule
    from gluonts.dataset.common import ListDataset
    tw, y_test = data[f"{TARGET_KPI}_train"], data[f"{TARGET_KPI}_test"]
    module = MoiraiModule.from_pretrained("Salesforce/moirai-1.1-R-base")
    ctx_len = min(len(tw), 512)
    model = MoiraiForecast(module=module, prediction_length=PREDICTION_LENGTH,
        context_length=ctx_len, patch_size="auto", num_samples=100,
        target_dim=1, feat_dynamic_real_dim=0, past_feat_dynamic_real_dim=0)
    predictor = model.create_predictor(batch_size=batch_size,
                                        device=torch.device("cuda"))
    ds = ListDataset([{"start": pd.Period("2024-01-01", freq="h"),
                        "target": _valid(tw[b].values)[-ctx_len:]}
                       for b in beam_cols], freq="h")
    t0 = time.time()
    forecasts = list(predictor.predict(ds))
    elapsed = time.time() - t0
    results = []
    for i, b in enumerate(beam_cols):
        true = _valid(y_test[b].values)[:PREDICTION_LENGTH]
        samples = forecasts[i].samples
        median = np.median(samples, axis=0)[:PREDICTION_LENGTH]
        qd = {ql: np.quantile(samples, ql, axis=0)[:PREDICTION_LENGTH]
              for ql in QUANTILE_LEVELS}
        m = eval_beam(true, median, _valid(tw[b].values), qd)
        m.update({"beam_id": b, "model": "Moirai", "test_week": test_week,
                   "time_s": elapsed})
        results.append(m)
    return results

# === MODEL: TimesFM ===
def run_timesfm_model(data, beam_cols, test_week, batch_size=64):
    import timesfm
    tw, y_test = data[f"{TARGET_KPI}_train"], data[f"{TARGET_KPI}_test"]
    tfm = timesfm.TimesFm(
        hparams=timesfm.TimesFmHparams(backend="gpu",
            per_core_batch_size=batch_size, horizon_len=PREDICTION_LENGTH),
        checkpoint=timesfm.TimesFmCheckpoint(
            huggingface_repo_id="google/timesfm-1.0-200m-pytorch"))
    inputs = [_valid(tw[b].values) for b in beam_cols]
    t0 = time.time()
    point_fc, _ = tfm.forecast(inputs, freq=[0]*len(inputs))
    elapsed = time.time() - t0
    results = []
    for i, b in enumerate(beam_cols):
        true = _valid(y_test[b].values)[:PREDICTION_LENGTH]
        pred = point_fc[i][:PREDICTION_LENGTH]
        m = eval_beam(true, pred, _valid(tw[b].values))
        m.update({"beam_id": b, "model": "TimesFM", "test_week": test_week,
                   "time_s": elapsed})
        results.append(m)
    return results

# === ORCHESTRATOR ===
def _log(results):
    df = pd.DataFrame(results); mn = df["model"].iloc[0]
    s = f"    {mn}: MAE={df['MAE'].mean():.4f} RMSE={df['RMSE'].mean():.4f} MASE={df['MASE'].mean():.4f}"
    if "CRPS" in df and df["CRPS"].notna().any(): s += f" CRPS={df['CRPS'].mean():.4f}"
    if "PICP_80" in df and df["PICP_80"].notna().any(): s += f" PICP={df['PICP_80'].mean():.3f}"
    s += f" ({df['time_s'].iloc[0]:.1f}s)"; print(s)

def run_benchmark(args):
    print("="*85)
    print("  Are Time Series Foundation Models Ready for 5G?")
    print(f"  Dataset: {args.dataset}")
    print("="*85)
    global TARGET_KPI, COVARIATE_KPIS, PREDICTION_LENGTH
    all_res = []
    if args.dataset == "irish":
        TARGET_KPI = IRISH_TARGET
        COVARIATE_KPIS = IRISH_COVARIATES
        PREDICTION_LENGTH = IRISH_PRED_LEN
        data, bc, meta = load_irish()
        meta.to_csv(RESULTS_DIR / "beam_metadata_irish.csv", index=False)
        test_weeks = ["irish"]
        print(f"  Series={len(bc)} Sparsity={meta['zero_frac'].mean():.1%}")
    else:
        test_weeks = args.test_weeks
    for tw in test_weeks:
        print(f"\n{'#'*85}\n  TEST: {tw}\n{'#'*85}")
        if args.dataset == "gotsf":
            data = load_all(test_week=tw)
            bc = get_beam_cols(data, num_bs=args.num_bs)
            meta = beam_metadata(bc, data)
            meta.to_csv(RESULTS_DIR / "beam_metadata.csv", index=False)
            print(f"  Beams={len(bc)} BS={args.num_bs} Sparsity={meta['zero_frac'].mean():.1%}")

        if "chronos2" in args.stages:
            modes = ["univariate", "covariates"]
            if args.dataset == "gotsf":
                modes.append("cross_learning")
            for mode in modes:
                print(f"\n  >> Chronos-2 ({mode})")
                if mode == "cross_learning":
                    r = []
                    for bs in range(args.num_bs):
                        r.extend(run_chronos2(data, beams_for_bs(bc,bs), mode, tw, args.batch_size))
                else:
                    r = run_chronos2(data, bc, mode, tw, args.batch_size)
                all_res.extend(r); _log(r)

        if "bolt" in args.stages:
            print(f"\n  >> Chronos-Bolt")
            try: r = run_bolt(data, bc, tw, args.batch_size); all_res.extend(r); _log(r)
            except Exception as e: print(f"    ERROR: {e}")

        if "moirai" in args.stages:
            print(f"\n  >> Moirai")
            try: r = run_moirai(data, bc, tw); all_res.extend(r); _log(r)
            except Exception as e: print(f"    ERROR: {e}")

        if "timesfm" in args.stages:
            print(f"\n  >> TimesFM")
            try: r = run_timesfm_model(data, bc, tw, args.batch_size); all_res.extend(r); _log(r)
            except Exception as e: print(f"    ERROR: {e}")

        if "trained" in args.stages:
            print(f"\n  >> NeuralForecast (PatchTST, TiDE, NHITS, DLinear) — no covariates")
            try:
                r = run_nf_models(data, bc, tw, args.max_steps, args.input_size,
                                  with_cov=False)
                all_res.extend(r)
                rdf = pd.DataFrame(r)
                for mn in rdf["model"].unique():
                    _log(rdf[rdf["model"]==mn].to_dict("records"))
            except Exception as e: print(f"    ERROR: {e}")

        if "trained_cov" in args.stages:
            print(f"\n  >> NeuralForecast (PatchTST, TiDE, NHITS, DLinear) + covariates")
            try:
                r = run_nf_models(data, bc, tw, args.max_steps, args.input_size,
                                  with_cov=True)
                all_res.extend(r)
                rdf = pd.DataFrame(r)
                for mn in rdf["model"].unique():
                    _log(rdf[rdf["model"]==mn].to_dict("records"))
            except Exception as e: print(f"    ERROR: {e}")

        if "arima" in args.stages:
            print(f"\n  >> ARIMA")
            r = run_arima(data, bc, tw); all_res.extend(r); _log(r)

    rdf = pd.DataFrame(all_res)
    if len(rdf) > 0:
        suffix = "_irish" if args.dataset == "irish" else ""
        out_path = RESULTS_DIR / f"per_beam_results{suffix}.csv"
        # Append to existing results (merge, dedup by beam+model+week)
        if out_path.exists():
            prev = pd.read_csv(out_path)
            rdf = pd.concat([prev, rdf], ignore_index=True)
            rdf = rdf.drop_duplicates(
                subset=["beam_id", "model", "test_week"], keep="last")
        rdf.to_csv(out_path, index=False)
        print_summary(rdf)
    else:
        print("\n  No results collected.")
    return rdf

def print_summary(df):
    print("\n" + "="*100 + "\n  RESULTS\n" + "="*100)
    for tw in df["test_week"].unique():
        sub = df[df["test_week"]==tw]
        print(f"\n  Test: {tw} | Beams: {sub['beam_id'].nunique()}")
        hc = "CRPS" in sub and sub["CRPS"].notna().any()
        hp = "PICP_80" in sub and sub["PICP_80"].notna().any()
        h = f"  {'Model':<20}{'MAE':>8}{'RMSE':>8}{'MASE':>8}{'NMAE':>8}"
        if hc: h += f"{'CRPS':>8}"
        if hp: h += f"{'PICP':>8}"
        h += f"{'Time':>8}"; print(h); print("  "+"-"*len(h))
        for mn in sub["model"].unique():
            ms = sub[sub["model"]==mn]
            l = f"  {mn:<20}{ms['MAE'].mean():>8.4f}{ms['RMSE'].mean():>8.4f}{ms['MASE'].mean():>8.4f}{ms['NMAE'].mean():>8.4f}"
            if hc:
                v = ms['CRPS'].mean() if ms['CRPS'].notna().any() else float('nan')
                l += f"{v:>8.4f}"
            if hp:
                v = ms['PICP_80'].mean() if ms['PICP_80'].notna().any() else float('nan')
                l += f"{v:>8.3f}"
            l += f"{ms['time_s'].iloc[0]:>7.1f}s"; print(l)
    print(f"\n  Saved → {RESULTS_DIR / 'per_beam_results.csv'}")

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", default="gotsf", choices=["gotsf", "irish"])
    p.add_argument("--stages", nargs="+",
        default=["chronos2","bolt","moirai","timesfm","trained","trained_cov","arima"])
    p.add_argument("--test-weeks", nargs="+", default=["5w-6w"])
    p.add_argument("--num-bs", type=int, default=5)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--max-steps", type=int, default=5000)
    p.add_argument("--input-size", type=int, default=336)
    args = p.parse_args()
    run_benchmark(args)
