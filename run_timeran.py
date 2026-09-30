import sys as _sys
from pathlib import Path as _P
_sys.path.insert(0, str(_P(__file__).resolve().parent))
from paths import ROOT, GOTSF_DIR, IRISH_DIR, RESULTS_DIR, FIG_DIR
#!/usr/bin/env python3
"""
TimeRAN baseline evaluation for beam-level 5G traffic forecasting.
Fine-tunes TimeRAN (built on MOMENT) then evaluates per-beam.
Results are saved in the same CSV format as benchmark.py.

Usage:
    python run_timeran.py --dataset gotsf
    python run_timeran.py --dataset irish
    python run_timeran.py --dataset gotsf --mode linear_probing
"""
import os, sys, time, warnings, argparse, random, glob
os.environ["TOKENIZERS_PARALLELISM"] = "false"
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import torch
import torch.cuda.amp as amp
from torch.utils.data import Dataset, DataLoader
from torch.optim.lr_scheduler import OneCycleLR
from tqdm import tqdm
from pathlib import Path
from sklearn.metrics import mean_absolute_error, mean_squared_error

# ──────────────────────────────────────────────────────────
# PATHS & CONSTANTS (same as benchmark.py)
# ──────────────────────────────────────────────────────────
DATA_DIR = GOTSF_DIR
IRISH_DIR = IRISH_DIR
RESULTS_DIR = RESULTS_DIR
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

KPIS = ["DLThpVol", "DLThpTime", "DLPRB", "MR_number"]
TARGET_KPI = "DLThpVol"
COVARIATE_KPIS = ["DLThpTime", "DLPRB", "MR_number"]
PREDICTION_LENGTH = 168

IRISH_TARGET = "DL_bitrate"
IRISH_COVARIATES = ["RSRP", "RSRQ", "SNR", "CQI"]
IRISH_PRED_LEN = 60

TIMERAN_CKPT_DIR = ROOT / "TimeRAN" / "TimeRAN_Foundation_Model" / "data" / "checkpoints"

MOMENT_SEQ_LEN = 512  # MOMENT's native input length

# TimeRAN variant names map to DIFFERENT MOMENT backbones:
# TimeRAN-base (d_model=512, ~40M) → MOMENT-1-small
# TimeRAN-large (d_model=768, ~125M) → MOMENT-1-base
VARIANT_TO_MODEL = {
    "small": "AutonLab/MOMENT-1-small",   # TimeRAN-small (~14M) — may need custom config
    "base":  "AutonLab/MOMENT-1-small",   # TimeRAN-base (d_model=512, ~40M)
    "large": "AutonLab/MOMENT-1-base",    # TimeRAN-large (d_model=768, ~125M)
}

VARIANT_TO_CHECKPOINT = {
    "small": TIMERAN_CKPT_DIR / "small" / "TimeRAN_small.pth",
    "base":  TIMERAN_CKPT_DIR / "base"  / "TimeRAN_base.pth",
    "large": TIMERAN_CKPT_DIR / "large" / "TimeRAN_large.pth",
}


# ──────────────────────────────────────────────────────────
# DATA LOADING (reuse from benchmark.py)
# ──────────────────────────────────────────────────────────
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


def get_beam_cols(data, num_bs=5):
    cols = data[f"{TARGET_KPI}_train"].columns.tolist()
    return [c for c in cols if int(c.split("_")[0]) < num_bs]


def beam_metadata(beam_cols, data):
    train = data[f"{TARGET_KPI}_train"]
    rows = []
    for b in beam_cols:
        parts = b.split("_")
        s = train[b]
        rows.append({"beam_id": b, "bs_id": int(parts[0]),
                      "cell_id": int(parts[1]), "beam_idx": int(parts[2]),
                      "zero_frac": float((s == 0).mean()),
                      "mean_traffic": float(s.mean())})
    return pd.DataFrame(rows)


def load_irish():
    """Load Irish 5G dataset (same as benchmark.py)."""
    files = sorted(glob.glob(str(IRISH_DIR / "**/*.csv"), recursive=True))
    h = IRISH_PRED_LEN
    target_col = IRISH_TARGET
    cov_cols = IRISH_COVARIATES
    all_kpis = [target_col] + cov_cols
    sessions = []
    for f in files:
        try:
            df = pd.read_csv(f)
            for c in all_kpis:
                df[c] = pd.to_numeric(df[c], errors="coerce")
            df = df.dropna(subset=all_kpis)
            if len(df) < h * 3:
                continue
            sessions.append((Path(f).stem, df[all_kpis].reset_index(drop=True)))
        except Exception:
            continue
    print(f"  Irish 5G: {len(sessions)} sessions loaded (of {len(files)} total)")
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
    meta_rows = []
    for sid, df in sessions:
        s = df[target_col].values[:-h]
        meta_rows.append({"beam_id": sid, "bs_id": 0, "cell_id": 0, "beam_idx": 0,
                          "zero_frac": float((s == 0).mean()),
                          "mean_traffic": float(np.nanmean(s))})
    meta = pd.DataFrame(meta_rows)
    return data, series_ids, meta


def _valid(arr):
    """Strip trailing NaN from variable-length padded series."""
    a = np.asarray(arr, dtype=np.float32)
    mask = ~np.isnan(a)
    if mask.all():
        return a
    last_valid = np.where(mask)[0][-1] + 1 if mask.any() else 0
    return a[:last_valid]


# ──────────────────────────────────────────────────────────
# METRICS (same as benchmark.py)
# ──────────────────────────────────────────────────────────
def mase_denom(tv, season=168):
    y = np.asarray(tv, dtype=np.float64)
    if len(y) <= season:
        d = np.mean(np.abs(np.diff(y)))
        return max(d, 1e-8)
    return max(np.mean(np.abs(y[season:] - y[:-season])), 1e-8)


def eval_beam(yt, yp, tv):
    yt = np.asarray(yt, dtype=np.float64)
    yp = np.asarray(yp, dtype=np.float64)
    mae = float(mean_absolute_error(yt, yp))
    rmse = float(np.sqrt(mean_squared_error(yt, yp)))
    return {
        "MAE": mae,
        "RMSE": rmse,
        "MASE": mae / mase_denom(tv),
        "NMAE": mae / (np.mean(np.abs(yt)) + 1e-8),
    }


# ──────────────────────────────────────────────────────────
# CUSTOM DATASET for TimeRAN fine-tuning
# ──────────────────────────────────────────────────────────
class BeamForecastDataset(Dataset):
    """
    Creates sliding-window (context, forecast) pairs from beam training data.
    Each sample: x shape (1, MOMENT_SEQ_LEN), mask (MOMENT_SEQ_LEN,), y (1, horizon).
    Shorter contexts are left-padded with zeros and mask indicates valid positions.
    """
    def __init__(self, data, beam_cols, target_kpi, horizon, context_len=336, stride=None):
        self.samples = []  # (x, mask, y) tuples
        self.beam_ids = []
        if stride is None:
            stride = horizon

        train_df = data[f"{target_kpi}_train"]

        for b in beam_cols:
            vals = _valid(train_df[b].values)
            T = len(vals)
            min_required = context_len + horizon
            if T < min_required:
                # Use whatever context we can
                if T > horizon:
                    actual_ctx = T - horizon
                    x_raw = vals[:actual_ctx]
                    y_raw = vals[actual_ctx:actual_ctx + horizon]
                    x_padded, mask = self._pad_to_moment(x_raw)
                    y_tensor = y_raw.reshape(1, -1)
                    self.samples.append((x_padded, mask, y_tensor))
                    self.beam_ids.append(b)
                continue

            max_start = T - min_required
            for start in range(0, max_start + 1, stride):
                x_raw = vals[start:start + context_len]
                y_raw = vals[start + context_len:start + context_len + horizon]
                x_padded, mask = self._pad_to_moment(x_raw)
                y_tensor = y_raw.reshape(1, -1)
                self.samples.append((x_padded, mask, y_tensor))
                self.beam_ids.append(b)

        print(f"    Created {len(self.samples)} training windows from {len(beam_cols)} series")

    @staticmethod
    def _pad_to_moment(x_raw):
        """Left-pad to MOMENT_SEQ_LEN, return (x, mask)."""
        L = len(x_raw)
        if L >= MOMENT_SEQ_LEN:
            x_out = x_raw[-MOMENT_SEQ_LEN:].reshape(1, MOMENT_SEQ_LEN)
            mask = np.ones(MOMENT_SEQ_LEN, dtype=np.float32)
        else:
            pad_len = MOMENT_SEQ_LEN - L
            x_out = np.concatenate([np.zeros(pad_len, dtype=np.float32), x_raw])
            x_out = x_out.reshape(1, MOMENT_SEQ_LEN)
            mask = np.concatenate([np.zeros(pad_len, dtype=np.float32),
                                   np.ones(L, dtype=np.float32)])
        return x_out.astype(np.float32), mask.astype(np.float32)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        x, mask, y = self.samples[idx]
        return (torch.as_tensor(x, dtype=torch.float32),
                torch.as_tensor(mask, dtype=torch.float32),
                torch.as_tensor(y, dtype=torch.float32))


# ──────────────────────────────────────────────────────────
# MODEL
# ──────────────────────────────────────────────────────────
def load_timeran_model(variant, horizon, mode, device):
    """Load MOMENT backbone + TimeRAN checkpoint for forecasting."""
    from momentfm import MOMENTPipeline

    model_name = VARIANT_TO_MODEL[variant]
    ckpt_path = VARIANT_TO_CHECKPOINT[variant]

    freeze = mode == "linear_probing"

    model = MOMENTPipeline.from_pretrained(
        model_name,
        model_kwargs={
            "task_name": "forecasting",
            "forecast_horizon": horizon,
            "freeze_encoder": freeze,
            "freeze_embedder": freeze,
            "freeze_head": False,
        },
    )
    model.init()

    # Load TimeRAN pre-trained weights
    if ckpt_path.exists():
        print(f"  Loading TimeRAN checkpoint: {ckpt_path}")
        state_dict = torch.load(str(ckpt_path), map_location="cpu", weights_only=True)

        # Remove "module." prefix if from DataParallel
        if any(k.startswith("module.") for k in state_dict.keys()):
            state_dict = {k.replace("module.", "", 1): v for k, v in state_dict.items()}

        # Remove forecasting head weights (they have wrong shape for our horizon)
        for k in ["head.linear.weight", "head.linear.bias"]:
            state_dict.pop(k, None)

        # Load backbone weights (strict=False to skip missing head)
        missing, unexpected = model.load_state_dict(state_dict, strict=False)
        print(f"    Missing keys: {len(missing)}, Unexpected: {len(unexpected)}")
    else:
        print(f"  WARNING: Checkpoint not found at {ckpt_path}, using base MOMENT weights")

    model = model.to(device)

    # Print parameter stats
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Parameters: total={total:,} trainable={trainable:,} frozen={total - trainable:,}")

    return model


# ──────────────────────────────────────────────────────────
# FINE-TUNING
# ──────────────────────────────────────────────────────────
def finetune_timeran(model, train_loader, args, device):
    """Fine-tune TimeRAN for forecasting."""
    criterion = torch.nn.MSELoss().to(device)
    optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=float(args.init_lr),
    )
    total_steps = len(train_loader) * args.epochs
    if total_steps == 0:
        print("  WARNING: No training steps, skipping fine-tuning")
        return model

    scheduler = OneCycleLR(
        optimizer, max_lr=float(args.max_lr),
        total_steps=total_steps, pct_start=0.3,
    )
    scaler = amp.GradScaler()

    for epoch in range(args.epochs):
        model.train()
        train_losses = []
        pbar = tqdm(train_loader, desc=f"  Epoch {epoch + 1}/{args.epochs} [train]")

        for timeseries, input_mask, forecast in pbar:
            timeseries = timeseries.float().to(device)  # (B, 1, 512)
            forecast = forecast.float().to(device)       # (B, 1, H)
            input_mask = input_mask.to(device)           # (B, 512)

            optimizer.zero_grad(set_to_none=True)

            with amp.autocast():
                out = model(x_enc=timeseries, input_mask=input_mask)
                loss = criterion(out.forecast, forecast)

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_norm)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()

            train_losses.append(loss.item())
            pbar.set_postfix({"loss": f"{np.mean(train_losses):.4f}"})

        print(f"    Epoch {epoch + 1}: Train MSE = {np.mean(train_losses):.4f}")

    return model


# ──────────────────────────────────────────────────────────
# EVALUATION
# ──────────────────────────────────────────────────────────
def evaluate_timeran(model, data, beam_cols, target_kpi, prediction_length,
                     context_len, test_week, model_label, device):
    """Evaluate fine-tuned TimeRAN per-beam."""
    model.eval()
    train_df = data[f"{target_kpi}_train"]
    test_df = data[f"{target_kpi}_test"]

    results = []
    t0 = time.time()

    # Batch beams for efficient inference
    batch_x = []
    batch_mask = []
    batch_beams = []

    for b in beam_cols:
        train_vals = _valid(train_df[b].values)
        # Use last context_len steps as context
        if len(train_vals) >= context_len:
            ctx = train_vals[-context_len:]
        else:
            ctx = train_vals

        x_padded, mask = BeamForecastDataset._pad_to_moment(ctx)
        batch_x.append(x_padded)
        batch_mask.append(mask)
        batch_beams.append(b)

    # Run inference in batches
    all_preds = []
    BATCH_SIZE = 64
    with torch.no_grad():
        for start in range(0, len(batch_x), BATCH_SIZE):
            end = min(start + BATCH_SIZE, len(batch_x))
            x = torch.tensor(np.array(batch_x[start:end]), dtype=torch.float32).to(device)
            m = torch.tensor(np.array(batch_mask[start:end]), dtype=torch.float32).to(device)

            with amp.autocast():
                out = model(x_enc=x, input_mask=m)

            preds = out.forecast.detach().cpu().numpy()  # (B, 1, H)
            all_preds.append(preds)

    all_preds = np.concatenate(all_preds, axis=0)  # (N, 1, H)
    elapsed = time.time() - t0

    for i, b in enumerate(batch_beams):
        true = _valid(test_df[b].values)[:prediction_length]
        pred = all_preds[i, 0, :prediction_length]  # univariate, channel 0

        train_vals = _valid(train_df[b].values)
        m = eval_beam(true, pred, train_vals)
        m.update({
            "beam_id": b,
            "model": model_label,
            "test_week": test_week,
            "time_s": elapsed,
        })
        results.append(m)

    return results


# ──────────────────────────────────────────────────────────
# MAIN
# ──────────────────────────────────────────────────────────
def control_randomness(seed=77):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def run_gotsf(args):
    global TARGET_KPI, COVARIATE_KPIS, PREDICTION_LENGTH
    TARGET_KPI = "DLThpVol"
    COVARIATE_KPIS = ["DLThpTime", "DLPRB", "MR_number"]
    PREDICTION_LENGTH = 168

    device = f"cuda:{args.device}" if torch.cuda.is_available() else "cpu"
    all_results = []

    for tw in args.test_weeks:
        print(f"\n{'#' * 80}")
        print(f"  GOTSF — Test: {tw} — Mode: {args.mode} — Variant: {args.variant}")
        print(f"{'#' * 80}")

        data = load_all(test_week=tw)
        bc = get_beam_cols(data, num_bs=args.num_bs)
        meta = beam_metadata(bc, data)
        print(f"  Beams={len(bc)} BS={args.num_bs} Sparsity={meta['zero_frac'].mean():.1%}")

        # Create training dataset
        train_ds = BeamForecastDataset(
            data, bc, TARGET_KPI, PREDICTION_LENGTH,
            context_len=args.context_len, stride=args.stride,
        )
        train_loader = DataLoader(
            train_ds, batch_size=args.batch_size, shuffle=True, drop_last=True,
        )

        # Load model fresh for each test week
        control_randomness(args.seed)
        model = load_timeran_model(args.variant, PREDICTION_LENGTH, args.mode, device)

        # Fine-tune
        print(f"\n  >> Fine-tuning TimeRAN-{args.variant} ({args.mode})")
        t0 = time.time()
        model = finetune_timeran(model, train_loader, args, device)
        ft_time = time.time() - t0
        print(f"  Fine-tuning completed in {ft_time:.1f}s")

        # Evaluate
        mode_label = "FF" if args.mode == "full_finetuning" else "LP"
        model_label = f"TimeRAN-{args.variant}-{mode_label}"
        results = evaluate_timeran(
            model, data, bc, TARGET_KPI, PREDICTION_LENGTH,
            args.context_len, tw, model_label, device,
        )
        all_results.extend(results)

        # Summary
        rdf = pd.DataFrame(results)
        print(f"\n  {model_label}: MAE={rdf['MAE'].mean():.4f} "
              f"RMSE={rdf['RMSE'].mean():.4f} MASE={rdf['MASE'].mean():.4f} "
              f"NMAE={rdf['NMAE'].mean():.4f}")

        # Cleanup
        del model
        torch.cuda.empty_cache()

    return all_results


def run_irish(args):
    global TARGET_KPI, COVARIATE_KPIS, PREDICTION_LENGTH
    TARGET_KPI = IRISH_TARGET
    COVARIATE_KPIS = IRISH_COVARIATES
    PREDICTION_LENGTH = IRISH_PRED_LEN

    device = f"cuda:{args.device}" if torch.cuda.is_available() else "cpu"

    print(f"\n{'#' * 80}")
    print(f"  Irish 5G — Mode: {args.mode} — Variant: {args.variant}")
    print(f"{'#' * 80}")

    data, bc, meta = load_irish()
    print(f"  Series={len(bc)} Sparsity={meta['zero_frac'].mean():.1%}")

    # Create training dataset
    train_ds = BeamForecastDataset(
        data, bc, TARGET_KPI, PREDICTION_LENGTH,
        context_len=args.context_len, stride=args.stride,
    )
    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True, drop_last=True,
    )

    # Load model
    control_randomness(args.seed)
    model = load_timeran_model(args.variant, PREDICTION_LENGTH, args.mode, device)

    # Fine-tune
    print(f"\n  >> Fine-tuning TimeRAN-{args.variant} ({args.mode})")
    t0 = time.time()
    model = finetune_timeran(model, train_loader, args, device)
    ft_time = time.time() - t0
    print(f"  Fine-tuning completed in {ft_time:.1f}s")

    # Evaluate
    mode_label = "FF" if args.mode == "full_finetuning" else "LP"
    model_label = f"TimeRAN-{args.variant}-{mode_label}"
    results = evaluate_timeran(
        model, data, bc, TARGET_KPI, PREDICTION_LENGTH,
        args.context_len, "irish", model_label, device,
    )

    rdf = pd.DataFrame(results)
    print(f"\n  {model_label}: MAE={rdf['MAE'].mean():.4f} "
          f"RMSE={rdf['RMSE'].mean():.4f} MASE={rdf['MASE'].mean():.4f} "
          f"NMAE={rdf['NMAE'].mean():.4f}")

    del model
    torch.cuda.empty_cache()

    return results


def main():
    p = argparse.ArgumentParser(description="TimeRAN baseline evaluation")
    p.add_argument("--dataset", default="gotsf", choices=["gotsf", "irish"])
    p.add_argument("--variant", default="base", choices=["small", "base", "large"])
    p.add_argument("--mode", default="full_finetuning",
                   choices=["full_finetuning", "linear_probing"])
    p.add_argument("--test-weeks", nargs="+", default=["5w-6w"])
    p.add_argument("--num-bs", type=int, default=5)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--epochs", type=int, default=5)
    p.add_argument("--init-lr", type=float, default=1e-5)
    p.add_argument("--max-lr", type=float, default=1e-4)
    p.add_argument("--max-norm", type=float, default=5.0)
    p.add_argument("--context-len", type=int, default=336)
    p.add_argument("--stride", type=int, default=None,
                   help="Sliding window stride (default: prediction_length)")
    p.add_argument("--device", type=int, default=0)
    p.add_argument("--seed", type=int, default=77)
    args = p.parse_args()

    print("=" * 80)
    print("  TimeRAN Baseline Evaluation for 5G Traffic Forecasting")
    print(f"  Dataset: {args.dataset} | Variant: {args.variant} | Mode: {args.mode}")
    print("=" * 80)

    if args.dataset == "gotsf":
        results = run_gotsf(args)
        suffix = ""
    else:
        results = run_irish(args)
        suffix = "_irish"

    if results:
        rdf = pd.DataFrame(results)
        out_path = RESULTS_DIR / f"per_beam_results{suffix}.csv"
        # Append to existing results (merge, dedup by beam+model+week)
        if out_path.exists():
            prev = pd.read_csv(out_path)
            rdf = pd.concat([prev, rdf], ignore_index=True)
            rdf = rdf.drop_duplicates(
                subset=["beam_id", "model", "test_week"], keep="last")
        rdf.to_csv(out_path, index=False)

        # Print summary
        print(f"\n{'=' * 80}")
        print("  RESULTS SUMMARY")
        print(f"{'=' * 80}")
        for tw in rdf["test_week"].unique():
            sub = rdf[rdf["test_week"] == tw]
            for mn in sub["model"].unique():
                if "TimeRAN" not in mn:
                    continue
                ms = sub[sub["model"] == mn]
                print(f"  {tw} | {mn:<30} MAE={ms['MAE'].mean():8.4f} "
                      f"RMSE={ms['RMSE'].mean():8.4f} "
                      f"MASE={ms['MASE'].mean():8.4f} "
                      f"NMAE={ms['NMAE'].mean():8.4f}")

        print(f"\n  Saved → {out_path}")
    else:
        print("\n  No results collected.")


if __name__ == "__main__":
    main()
