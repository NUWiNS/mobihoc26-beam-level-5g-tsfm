"""
Graph WaveNet (Wu et al., IJCAI 2019) baseline on the per-BS beam graph (GOTSF).

Protocol matches benchmark.py's trained baselines:
  - train on Weeks 1-5 (840 h); the last H=168 h are held out for early stopping
  - input window 336 h, forecast H=168 h from the end of the training history
  - evaluate against the Week-6 and Week-11 test files with benchmark.py's MASE
  - univariate target only (no future telemetry), one global model over all 30 BSs

Graph per BS (96 beams): top-k |Pearson r| of the training target (first 672 h,
i.e. excluding the validation block), symmetrised, used as forward/backward
transition supports, plus GWN's self-adaptive adjacency (shared across BSs,
indexed by beam position within the BS).

Long input: kernel-2 dilated gated TCN with dilations 1..256 (receptive field
512); the 336-h input is left-zero-padded to 512, as in the original GWN code.
"""
import sys as _sys
from pathlib import Path as _P
_sys.path.insert(0, str(_P(__file__).resolve().parent.parent))
from paths import ROOT, GOTSF_DIR, IRISH_DIR, RESULTS_DIR, FIG_DIR
import os, sys, time, argparse, json
import numpy as np, pandas as pd, torch, torch.nn as nn, torch.nn.functional as F

from benchmark import load_wide, eval_beam, _valid, TARGET_KPI  # same metric code

H, L, NB, NBEAM = 168, 336, 30, 96
VAL_START = 840 - H  # 672


# ---------------------------------------------------------------- model
class NConv(nn.Module):
    def forward(self, x, A):  # x: (B,C,N,T), A: (B,N,N) or (N,N)
        if A.dim() == 2:
            return torch.einsum("bcvt,vw->bcwt", x, A).contiguous()
        return torch.einsum("bcvt,bvw->bcwt", x, A).contiguous()


class GCN(nn.Module):
    def __init__(self, c_in, c_out, n_supports, order=2, dropout=0.3):
        super().__init__()
        self.nconv, self.order, self.dropout = NConv(), order, dropout
        self.mlp = nn.Conv2d(c_in * (order * n_supports + 1), c_out, 1)

    def forward(self, x, supports):
        out = [x]
        for A in supports:
            x1 = self.nconv(x, A); out.append(x1)
            for _ in range(2, self.order + 1):
                x1 = self.nconv(x1, A); out.append(x1)
        h = self.mlp(torch.cat(out, dim=1))
        return F.dropout(h, self.dropout, training=self.training)


class GraphWaveNet(nn.Module):
    def __init__(self, n_nodes, in_dim=2, out_dim=H, res=32, dil=32, skip=256,
                 end=512, dilations=(1, 2, 4, 8, 16, 32, 64, 128, 256),
                 emb=10, dropout=0.3):
        super().__init__()
        self.dilations = dilations
        self.rf = 1 + sum(dilations)  # kernel 2
        self.start = nn.Conv2d(in_dim, res, 1)
        self.E1 = nn.Parameter(torch.randn(n_nodes, emb))
        self.E2 = nn.Parameter(torch.randn(emb, n_nodes))
        n_sup = 3  # fwd, bwd, adaptive
        self.filt, self.gate, self.skips, self.gcns, self.bns = (nn.ModuleList() for _ in range(5))
        for d in dilations:
            self.filt.append(nn.Conv2d(res, dil, (1, 2), dilation=d))
            self.gate.append(nn.Conv2d(res, dil, (1, 2), dilation=d))
            self.skips.append(nn.Conv2d(dil, skip, 1))
            self.gcns.append(GCN(dil, res, n_sup, dropout=dropout))
            self.bns.append(nn.BatchNorm2d(res))
        self.end1 = nn.Conv2d(skip, end, 1)
        self.end2 = nn.Conv2d(end, out_dim, 1)

    def forward(self, x, P_fwd, P_bwd):  # x: (B,in_dim,N,T)
        if x.size(3) < self.rf:
            x = F.pad(x, (self.rf - x.size(3), 0, 0, 0))
        x = self.start(x)
        adp = F.softmax(F.relu(self.E1 @ self.E2), dim=1)
        supports = [P_fwd, P_bwd, adp]
        skip = 0
        for i in range(len(self.dilations)):
            residual = x
            h = torch.tanh(self.filt[i](x)) * torch.sigmoid(self.gate[i](x))
            s = self.skips[i](h)
            skip = s if isinstance(skip, int) else s + skip[:, :, :, -s.size(3):]
            h = self.gcns[i](h, supports)
            x = self.bns[i](h + residual[:, :, :, -h.size(3):])
        # no final ReLU: the head predicts standardised targets, which can be negative
        out = self.end2(F.relu(self.end1(F.relu(skip))))  # (B,H,N,1)
        return out[..., -1].transpose(1, 2)  # (B,N,H)


# ---------------------------------------------------------------- data
def build(k=10):
    tr = load_wide(TARGET_KPI, "train")  # (840, 2880)
    beams = [[c for c in tr.columns if int(c.split("_")[0]) == bs] for bs in range(NB)]
    assert all(len(b) == NBEAM for b in beams)
    Y = np.stack([tr[b].values.T for b in beams]).astype(np.float32)  # (NB,N,840)
    fit = Y[:, :, :VAL_START]
    mu, sd = fit.mean(-1, keepdims=True), fit.std(-1, keepdims=True) + 1e-6
    Z = (Y - mu) / sd
    Pf, Pb = [], []
    for bs in range(NB):
        r = np.nan_to_num(np.corrcoef(fit[bs]))
        np.fill_diagonal(r, 0); a = np.abs(r)
        A = np.zeros_like(a)
        idx = np.argsort(-a, axis=1)[:, :k]
        for i in range(NBEAM): A[i, idx[i]] = a[i, idx[i]]
        A = np.maximum(A, A.T) + np.eye(NBEAM)
        Pf.append(A / A.sum(1, keepdims=True)); Pb.append(A.T / A.T.sum(1, keepdims=True))
    t = np.arange(840)
    tod = np.stack([(t % 24) / 24.0, ((t // 24) % 7) / 7.0])  # time-of-day, day-of-week
    return beams, Y, Z, mu, sd, np.array(Pf, np.float32), np.array(Pb, np.float32), tod


def window(Z, tod, bs, t, n_time=1):  # input ending at t (exclusive)
    feats = [Z[bs, :, t - L:t]] + [np.broadcast_to(tod[k, t - L:t], (NBEAM, L)) for k in range(n_time)]
    return np.stack(feats).astype(np.float32)  # (1+n_time,N,L)


# ---------------------------------------------------------------- train/eval
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--patience", type=int, default=10)
    ap.add_argument("--bs", type=int, default=32)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--dropout", type=float, default=0.3)
    ap.add_argument("--res", type=int, default=32, help="residual/dilation channels")
    ap.add_argument("--dow", action="store_true", help="add day-of-week input channel")
    ap.add_argument("--std_loss", action="store_true", help="MAE on standardised scale (equal beam weight)")
    ap.add_argument("--tag", default="", help="config name used in output filenames")
    ap.add_argument("--out", default=str(RESULTS_DIR / "gwn"))
    a = ap.parse_args()
    nt = 2 if a.dow else 1
    os.makedirs(a.out, exist_ok=True)
    torch.manual_seed(a.seed); np.random.seed(a.seed)
    dev = torch.device("cuda")
    beams, Y, Z, mu, sd, Pf, Pb, tod = build()
    Pf_t, Pb_t = torch.tensor(Pf, device=dev), torch.tensor(Pb, device=dev)
    mu_t, sd_t = torch.tensor(mu, device=dev), torch.tensor(sd, device=dev)
    Y_t = torch.tensor(Y, device=dev)

    # training windows: targets fully before VAL_START; val window: t = VAL_START
    train_idx = [(bs, t) for bs in range(NB) for t in range(L, VAL_START - H + 1)]
    Xtr = torch.tensor(np.stack([window(Z, tod, bs, t, nt) for bs, t in train_idx]), device=dev)
    Ytr = torch.stack([Y_t[bs, :, t:t + H] for bs, t in train_idx])
    Btr = torch.tensor([bs for bs, _ in train_idx], device=dev)
    Xv = torch.tensor(np.stack([window(Z, tod, bs, VAL_START, nt) for bs in range(NB)]), device=dev)
    Yv = Y_t[:, :, VAL_START:VAL_START + H]

    model = GraphWaveNet(NBEAM, in_dim=1 + nt, res=a.res, dil=a.res, dropout=a.dropout).to(dev)
    opt = torch.optim.Adam(model.parameters(), lr=a.lr, weight_decay=1e-4)

    def predict(X, bsi):  # returns real-scale (B,N,H)
        z = model(X, Pf_t[bsi], Pb_t[bsi])
        return z * sd_t[bsi] + mu_t[bsi]

    best, best_state, bad, t0 = 1e18, None, 0, time.time()
    for ep in range(a.epochs):
        model.train()
        perm = np.random.permutation(len(train_idx)); tl = 0.0
        for i in range(0, len(perm), a.bs):
            j = torch.as_tensor(perm[i:i + a.bs], device=dev)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                pred = predict(Xtr[j], Btr[j])
            err = pred.float() - Ytr[j]
            if a.std_loss:
                err = err / sd_t[Btr[j]]
            loss = err.abs().mean()
            opt.zero_grad(); loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5.0); opt.step()
            tl += loss.item() * len(j)
        model.eval()
        with torch.no_grad():
            vl = (predict(Xv, torch.arange(NB, device=dev)) - Yv).abs().mean().item()
        if vl < best - 1e-6:
            best, bad = vl, 0
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
        print(f"seed {a.seed} ep {ep:3d} train {tl/len(perm):.4f} val {vl:.4f} "
              f"best {best:.4f} ({time.time()-t0:.0f}s)", flush=True)
        if bad >= a.patience: break
    t_fit = time.time() - t0
    model.load_state_dict(best_state); model.eval()

    # forecast H steps from the end of the full 840-h training history
    with torch.no_grad():
        Xf = torch.tensor(np.stack([window(Z, tod, bs, 840, nt) for bs in range(NB)]), device=dev)
        P = predict(Xf, torch.arange(NB, device=dev)).cpu().numpy()  # (NB,N,H)
    tr = load_wide(TARGET_KPI, "train")
    rows = []
    for tw in ["5w-6w", "10w-11w"]:
        te = load_wide(TARGET_KPI, "test", tw)
        for bs in range(NB):
            for n, b in enumerate(beams[bs]):
                true = _valid(te[b].values)[:H]
                m = eval_beam(true, P[bs, n], _valid(tr[b].values))
                m.update({"beam_id": b, "model": "GraphWaveNet", "test_week": tw,
                          "seed": a.seed, "time_s": t_fit, "epochs": ep + 1,
                          "config": a.tag or "default", "best_val_mae": best})
                rows.append(m)
    df = pd.DataFrame(rows)
    tag = f"_{a.tag}" if a.tag else ""
    df.to_csv(f"{a.out}/gwn{tag}_seed{a.seed}.csv", index=False)
    print("DONE", a.tag or "default", a.seed, f"best_val {best:.4f}", df.groupby("test_week").MASE.mean().round(4).to_dict(),
          f"fit {t_fit:.0f}s", flush=True)


if __name__ == "__main__":
    main()
