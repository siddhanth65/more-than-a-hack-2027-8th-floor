"""Model B (BiGRU) inference as an external tool, run in the PathCRF venv (torch) via subprocess like PathCRF.

Standalone: imports numpy and torch only (nothing from mtah). Same network and inference as the training script
scripts/exp_seq_gru.py (prototype) / scripts/exp_seq_lomo_gru.py: standardised step features, 600-step windows
(stride 300, the step from the window where it is most central), team-symmetrised, log(clip(p, 1e-4, 1)).
Deterministic: CPU only, torch threads pinned (default 4), torch.use_deterministic_algorithms(True).

  <pathcrf venv python> seq_b_infer.py --in X.npz --ckpt modelB.pt --out logp.npz [--threads 4]
--in: X (T, n) float32 step features, seg (T,) half ids, names (n,) feature names. --out: logp (T, 3) float32 and
torch_version.
"""
import argparse, os

os.environ["CUDA_VISIBLE_DEVICES"] = ""

import numpy as np
import torch
import torch.nn as nn

STREAMS = ("ball", "tag", "dead", "pathcrf", "teamctrl")
SIGNED = {f"{k}_ha" for k in STREAMS} | {"last_touch_team", "next_touch_team", "pcrf_phome"}
SWAP = [("touch_H", "touch_A"), ("strong_H", "strong_A")]


LOC = ["ball_speed", "ball_absx", "ball_absy", "log_since_touch", "log_until_touch", "pcrf_imputed"]
SIGNED_SCALE = [f"{k}_ha" for k in STREAMS] + ["pcrf_phome"]


def normalise_v1(X, names):
    """Per-half robust normalisation 'v1' (identical to scripts/seq3_common.normalise, mode v1) of ONE half's steps:
    location+scale (median / IQR) for LOC; scale-only (median |x| over x != 0) for the signed H-A margins."""
    X = X.astype(np.float32, copy=True); ix = {n: i for i, n in enumerate(names)}
    for n in LOC:
        v = X[:, ix[n]]; q1, q2, q3 = np.percentile(v, [25, 50, 75])
        X[:, ix[n]] = (v - q2) / max(q3 - q1, 1e-3)
    for n in SIGNED_SCALE:
        v = X[:, ix[n]]; nz = np.abs(v[v != 0])
        X[:, ix[n]] = v / (max(float(np.median(nz)), 1e-3) if len(nz) else 1.0)
    return X


class Net(nn.Module):
    def __init__(self, nf, h=64):
        super().__init__()
        self.inp = nn.Linear(nf, h)
        self.gru = nn.GRU(h, h, num_layers=2, batch_first=True, bidirectional=True, dropout=0.1)
        self.cls = nn.Linear(2 * h, 3); self.bnd = nn.Linear(2 * h, 1)

    def forward(self, x):
        o, _ = self.gru(torch.relu(self.inp(x)))
        return self.cls(o), self.bnd(o).squeeze(-1)


def mirror_np(X, names):
    X = X.copy(); ix = {n: i for i, n in enumerate(names)}
    for n in SIGNED:
        X[..., ix[n]] = -X[..., ix[n]]
    for a, b in SWAP:
        X[..., [ix[a], ix[b]]] = X[..., [ix[b], ix[a]]]
    return X


@torch.no_grad()
def infer(net, X, mu, sd, names, win, stride):
    net.eval()
    T = len(X); out = np.zeros((T, 3), np.float32); best = np.full(T, -np.inf)
    Xn = ((X - mu) / sd).astype(np.float32)
    L = min(win, T)
    starts = list(range(0, T - L + 1, stride))
    if starts[-1] + L < T:
        starts.append(T - L)
    xb = np.stack([Xn[a:a + L] for a in starts])
    p1 = torch.softmax(net(torch.tensor(xb))[0], -1).numpy()
    p2 = torch.softmax(net(torch.tensor(mirror_np(xb, names)))[0], -1).numpy()[..., [1, 0, 2]]
    P = 0.5 * (p1 + p2)
    cen = -np.abs(np.arange(L) - (L - 1) / 2)
    for i, a in enumerate(starts):
        sl = slice(a, a + L); upd = cen > best[sl]
        seg = out[sl]; seg[upd] = np.log(np.clip(P[i][upd], 1e-4, 1)); best[sl] = np.where(upd, cen, best[sl])
    assert np.isfinite(best).all()
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", required=True); ap.add_argument("--ckpt", required=True)
    ap.add_argument("--out", required=True); ap.add_argument("--threads", type=int, default=4)
    a = ap.parse_args()
    torch.set_num_threads(a.threads); torch.use_deterministic_algorithms(True); torch.manual_seed(0)
    z = np.load(a.inp)
    X, seg, names = z["X"].astype(np.float32), z["seg"], [str(s) for s in z["names"]]
    ck = torch.load(a.ckpt, weights_only=False)
    if list(ck["names"]) != names:
        raise SystemExit("feature names differ from the checkpoint's")
    net = Net(ck["nf"], ck["hidden"]); net.load_state_dict(ck["state_dict"])
    mu, sd = ck["mu"].numpy(), ck["sd"].numpy()
    lp = np.zeros((len(X), 3), np.float32)
    for h in np.unique(seg):                  # one sequence per half: windows never cross halves
        ii = np.flatnonzero(seg == h)
        norm = ck.get("norm", "none")
        if norm not in ("none", "v1"):
            raise SystemExit(f"unsupported input normalisation {norm!r}")
        Xh = normalise_v1(X[ii], names) if norm == "v1" else X[ii]
        lp[ii] = infer(net, Xh, mu, sd, names, int(ck["win"]), int(ck["stride"]))
    np.savez(a.out, logp=lp, torch_version=np.array(torch.__version__))


if __name__ == "__main__":
    main()
