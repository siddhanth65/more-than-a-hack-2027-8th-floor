"""exp-seq Model B (PathCRF venv: CPU torch): 2-layer BiGRU over 5 fps step features (stage-2 feature caches).

Loss: cross-entropy on H/A/D (ungraded steps ignored) + 0.5 * BCE auxiliary boundary head (truth changes within
+-0.5 s = +-2 steps). Training windows of 2 min (600 steps), random offsets each epoch, team-swap mirror augmentation.
Inner run: train 9002-9005, epoch count chosen by 9006 loss (writes 9006 log-probs); final run: 9002-9006 with that
epoch count (writes 9001 log-probs). Inference: overlapping 600-step windows (stride 300), centre halves kept,
team-symmetrised. CPU only, 4 threads; checkpoints every epoch under exp_seq/gru/.
"""
import json, os, sys, time
from pathlib import Path

os.environ["CUDA_VISIBLE_DEVICES"] = ""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as Fn

torch.set_num_threads(4)
OUT = (Path(__file__).resolve().parents[1] / "runs" / "select" / "exp_seq")
G = OUT / "gru"; G.mkdir(exist_ok=True)
SEEDS = [100, 101]
WIN, STRIDE, BATCH = 600, 300, 64
MAX_EPOCHS, BUDGET_S = 40, 13 * 60     # per run; two runs <= ~30 min
STREAMS = ("ball", "tag", "dead", "pathcrf", "teamctrl")
SIGNED = {f"{k}_ha" for k in STREAMS} | {"last_touch_team", "next_touch_team", "pcrf_phome"}
SWAP = [("touch_H", "touch_A"), ("strong_H", "strong_A")]


def names_for(ms):
    return [f"game_{m}_{h}_s{s}" for m in ms for h in ("H1", "H2") for s in SEEDS]


def load(n):
    z = np.load(OUT / "feat" / f"{n}.npz")
    return z["X"], z["starts"], z["seg"], z["y"], [str(s) for s in z["names"]]


def mirror_np(X, names):
    X = X.copy(); ix = {n: i for i, n in enumerate(names)}
    for n in SIGNED:
        X[..., ix[n]] = -X[..., ix[n]]
    for a, b in SWAP:
        X[..., [ix[a], ix[b]]] = X[..., [ix[b], ix[a]]]
    return X


def boundary(y):
    b = np.zeros(len(y), np.float32)
    ch = np.flatnonzero((y[1:] != y[:-1]) & (y[1:] < 3) & (y[:-1] < 3)) + 1
    for d in range(-2, 3):
        j = np.clip(ch + d, 0, len(y) - 1); b[j] = 1
    return b


class Net(nn.Module):
    def __init__(self, nf, h=64):
        super().__init__()
        self.inp = nn.Linear(nf, h)
        self.gru = nn.GRU(h, h, num_layers=2, batch_first=True, bidirectional=True, dropout=0.1)
        self.cls = nn.Linear(2 * h, 3); self.bnd = nn.Linear(2 * h, 1)

    def forward(self, x):
        o, _ = self.gru(torch.relu(self.inp(x)))
        return self.cls(o), self.bnd(o).squeeze(-1)


def seqs(ms, mu=None, sd=None):
    S = []
    for n in names_for(ms):
        X, starts, seg, y, names = load(n)
        S.append((n, X, y, starts))
    if mu is None:
        A = np.concatenate([s[1] for s in S]); mu = A.mean(0); sd = A.std(0) + 1e-3
        ix = {k: i for i, k in enumerate(names)}
        for k in SIGNED:                      # keep signed features centred at 0 so the mirror stays a sign flip
            mu[ix[k]] = 0.0
        for a, b in SWAP:                     # swapped pairs share one scaling so the mirror commutes with it
            mu[[ix[a], ix[b]]] = mu[[ix[a], ix[b]]].mean(); sd[[ix[a], ix[b]]] = sd[[ix[a], ix[b]]].mean()
    return S, mu, sd, names


def batches(S, mu, sd, names, rng):
    W = []
    for _, X, y, _ in S:
        Xn = (X - mu) / sd; T = len(y); off = rng.integers(0, STRIDE)
        for a in range(off, max(T - WIN, 0) + 1, STRIDE):
            W.append((Xn[a:a + WIN], y[a:a + WIN]))
    idx = rng.permutation(2 * len(W))
    for k in range(0, len(idx), BATCH):
        xb, yb = [], []
        for i in idx[k:k + BATCH]:
            x, y = W[i % len(W)]
            if i >= len(W):
                x = mirror_np(x, names); y = np.where(y == 0, 1, np.where(y == 1, 0, y))
            xb.append(x); yb.append(y)
        yield torch.tensor(np.stack(xb)), torch.tensor(np.stack(yb).astype(np.int64))


def loss_fn(net, xb, yb):
    lo, lb = net(xb)
    ce = Fn.cross_entropy(lo.reshape(-1, 3), yb.reshape(-1), ignore_index=3)
    bt = torch.tensor(np.stack([boundary(y) for y in yb.numpy()]))
    bce = Fn.binary_cross_entropy_with_logits(lb, bt)
    return ce + 0.5 * bce, ce


@torch.no_grad()
def infer(net, X, mu, sd, names):
    """(T, 3) log-probs: windows of WIN steps, stride STRIDE (batched, team-symmetrised), each step taken from the
    window where it is most central."""
    net.eval()
    T = len(X); out = np.zeros((T, 3), np.float32); best = np.full(T, -np.inf)
    Xn = ((X - mu) / sd).astype(np.float32)
    L = min(WIN, T)
    starts = list(range(0, T - L + 1, STRIDE))
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


def val_loss(net, S, mu, sd, names):
    tot, n = 0.0, 0
    for _, X, y, _ in S:
        lp = infer(net, X, mu, sd, names); m = y < 3
        tot += -lp[m, y[m]].sum(); n += m.sum()
    return tot / n


def run(train_ms, val_ms, tag, n_epochs=None, log=print):
    torch.manual_seed(0); rng = np.random.default_rng(0)
    S, mu, sd, names = seqs(train_ms)
    V = seqs(val_ms, mu, sd)[0] if val_ms else None
    net = Net(S[0][1].shape[1]); opt = torch.optim.Adam(net.parameters(), lr=3e-3, weight_decay=1e-5)
    t0 = time.time(); hist = []; best = (np.inf, 0)
    E = n_epochs or MAX_EPOCHS
    for ep in range(1, E + 1):
        net.train(); tl, nb = 0.0, 0
        for xb, yb in batches(S, mu, sd, names, rng):
            opt.zero_grad(); l, ce = loss_fn(net, xb, yb); l.backward()
            nn.utils.clip_grad_norm_(net.parameters(), 1.0); opt.step(); tl += ce.item(); nb += 1
        rec = {"epoch": ep, "train_ce": tl / nb, "s": round(time.time() - t0, 1)}
        if V is not None:
            rec["val_nll"] = float(val_loss(net, V, mu, sd, names))
            if rec["val_nll"] < best[0]:
                best = (rec["val_nll"], ep); torch.save(net.state_dict(), G / f"{tag}_best.pt")
        torch.save(net.state_dict(), G / f"{tag}_last.pt"); hist.append(rec); log(f"{tag} {rec}")
        if V is not None and ep - best[1] >= 5:
            break
        if time.time() - t0 > BUDGET_S:
            log(f"{tag}: time budget reached"); break
    if V is not None:
        net.load_state_dict(torch.load(G / f"{tag}_best.pt"))
    return net, mu, sd, names, hist, best, time.time() - t0


def save_preds(net, ms, mu, sd, names):
    d = OUT / "pred_B"; d.mkdir(exist_ok=True); t0 = time.time()
    for n in names_for(ms):
        X, starts, seg, y, _ = load(n)
        np.savez_compressed(d / f"{n}.npz", logp=infer(net, X, mu, sd, names), starts=starts, y=y)
    return time.time() - t0


def main():
    logf = open(OUT / "gru.log", "a")

    def log(msg):
        line = f"[{time.strftime('%H:%M:%S')}] {msg}"; print(line, flush=True); logf.write(line + "\n"); logf.flush()

    net, mu, sd, names, hist, best, t_in = run([9002, 9003, 9004, 9005], [9006], "inner", log=log)
    save_preds(net, [9006], mu, sd, names)
    log(f"inner best epoch {best[1]} val_nll {best[0]:.4f} ({t_in:.0f}s)")
    net, mu, sd, names, hist_f, _, t_fin = run([9002, 9003, 9004, 9005, 9006], None, "final", n_epochs=best[1], log=log)
    t_inf = save_preds(net, [9001], mu, sd, names)
    res = {"best_epoch": best[1], "inner_val_nll": best[0], "inner_seconds": round(t_in, 1),
           "final_train_seconds": round(t_fin, 1), "test_inference_seconds_4_halves": round(t_inf, 1),
           "hist_inner": hist, "arch": "Linear(nf,64)+ReLU -> BiGRU(64, 2 layers) -> cls(3) + boundary(1)",
           "n_params": int(sum(p.numel() for p in net.parameters()))}
    (OUT / "modelB.json").write_text(json.dumps(res, indent=1))
    log(f"MODEL B done best_epoch {best[1]} train {t_in + t_fin:.0f}s infer {t_inf:.0f}s")


if __name__ == "__main__":
    main()
