"""exp-seq2 Model Bs = Model B with a SPELL-WEIGHTED loss (each true spell counts ~once, as in spellF1):
per-step CE weight min(1, L0 / len(true run containing the step)), L0 = 25 steps (5 s), renormalised per batch;
same architecture / features / epochs rule; outputs pred_Bs, modelBs_f*.json, gru_sw checkpoints, final/modelBs.pt.

Original: exp-seq T2 Model B (BiGRU, PathCRF venv CPU torch) for the 6-fold LOMO and the final all-match model.

Per fold F (layout of scripts/exp_seq_lomo.py): inner run on train minus V (V = max(train)), epochs chosen by V's
NLL (patience 5, <= 40 epochs, 13 min budget as the prototype) -> predictions on V; fold-final run on all five
train matches with that epoch count -> predictions on F. `--final`: all six matches (fold{m} caches), epochs = median
fold choice, no time budget -> runs/select/exp_seq/final/modelB.pt (state dict + normalisation + feature names).
Per-fold checkpoint: lomo/modelBs_f{F}.json (a finished fold is skipped).
Usage: <venv python> exp_seq_lomo_gru.py [--folds 9002 ...] | --final
"""
import argparse, json, os, sys, time
from pathlib import Path

os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["OMP_NUM_THREADS"] = "4"
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import numpy as np
import torch
import torch.nn as nn

import exp_seq_gru as g   # Net, batches, infer, constants (prototype; same architecture)
import torch.nn.functional as Fn

L0 = 25


def run_weights(y):
    w = np.ones(len(y), np.float32)
    ch = np.flatnonzero(y[1:] != y[:-1]) + 1; st = np.r_[0, ch]; en = np.r_[ch, len(y)]
    for a, b in zip(st, en):
        w[a:b] = min(1.0, L0 / (b - a))
    w[y == 3] = 0.0
    return w


def loss_sw(net, xb, yb):
    lo, lb = net(xb)
    yn = yb.numpy(); w = torch.tensor(np.stack([run_weights(y) for y in yn]))
    ce_t = Fn.cross_entropy(lo.reshape(-1, 3), yb.reshape(-1).clamp(max=2), reduction="none") * w.reshape(-1)
    ce = ce_t.sum() / w.sum().clamp(min=1e-6)
    bt = torch.tensor(np.stack([g.boundary(y) for y in yn]))
    bce = Fn.binary_cross_entropy_with_logits(lb, bt)
    return ce + 0.5 * bce, ce


g.loss_fn = loss_sw

torch.set_num_threads(4)
torch.use_deterministic_algorithms(True)
P = (Path(__file__).resolve().parents[1] / "runs" / "select" / "exp_seq2")
L = P / "lomo"; FINAL = P / "final"
CK = L / "gru_sw"; CK.mkdir(parents=True, exist_ok=True)
MATCHES = [9001, 9002, 9003, 9004, 9005, 9006]
SEEDS = [100, 101]


def halves_of(m):
    return [(f"game_{m}_{h}", s) for h in ("H1", "H2") for s in SEEDS]


def src_of(F, m):
    if m == F:
        return f"fold{F}"
    a, b = sorted((F, m)); return f"pair{a}_{b}"


def fpath(src, key, seed):
    return L / "feat" / src / f"{key}_s{seed}.npz"


def load(p):
    z = np.load(p)
    return z["X"], z["starts"], z["seg"], z["y"], [str(s) for s in z["names"]]


def seqs(paths, mu=None, sd=None):
    S = []
    for p in paths:
        X, starts, seg, y, names = load(p)
        S.append((p.name, X, y, starts))
    if mu is None:
        A = np.concatenate([s[1] for s in S]); mu = A.mean(0); sd = A.std(0) + 1e-3
        ix = {k: i for i, k in enumerate(names)}
        for k in g.SIGNED:
            mu[ix[k]] = 0.0
        for a, b in g.SWAP:
            mu[[ix[a], ix[b]]] = mu[[ix[a], ix[b]]].mean(); sd[[ix[a], ix[b]]] = sd[[ix[a], ix[b]]].mean()
    return S, mu, sd, names


def run(train_paths, val_paths, tag, n_epochs=None, budget=g.BUDGET_S, log=print):
    torch.manual_seed(0); rng = np.random.default_rng(0)
    S, mu, sd, names = seqs(train_paths)
    V = seqs(val_paths, mu, sd)[0] if val_paths else None
    net = g.Net(S[0][1].shape[1]); opt = torch.optim.Adam(net.parameters(), lr=3e-3, weight_decay=1e-5)
    t0 = time.time(); hist = []; best = (np.inf, 0)
    for ep in range(1, (n_epochs or g.MAX_EPOCHS) + 1):
        net.train(); tl, nb = 0.0, 0
        for xb, yb in g.batches(S, mu, sd, names, rng):
            opt.zero_grad(); l, ce = g.loss_fn(net, xb, yb); l.backward()
            nn.utils.clip_grad_norm_(net.parameters(), 1.0); opt.step(); tl += ce.item(); nb += 1
        rec = {"epoch": ep, "train_ce": tl / nb, "s": round(time.time() - t0, 1)}
        if V is not None:
            rec["val_nll"] = float(g.val_loss(net, V, mu, sd, names))
            if rec["val_nll"] < best[0]:
                best = (rec["val_nll"], ep); torch.save(net.state_dict(), CK / f"{tag}_best.pt")
        hist.append(rec); log(f"{tag} {rec}")
        if V is not None and ep - best[1] >= 5:
            break
        if budget is not None and time.time() - t0 > budget:
            log(f"{tag}: time budget reached"); break
    if V is not None:
        net.load_state_dict(torch.load(CK / f"{tag}_best.pt"))
    return net, mu, sd, names, hist, best, time.time() - t0


def save_preds(net, F, m, mu, sd, names):
    for k, s in halves_of(m):
        X, starts, seg, y, _ = load(fpath(src_of(F, m), k, s))
        out = L / "pred_Bs" / f"f{F}" / f"{k}_s{s}.npz"; out.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(out, logp=g.infer(net, X, mu, sd, names), starts=starts, y=y)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--folds", type=int, nargs="*", default=MATCHES); ap.add_argument("--final", action="store_true")
    a = ap.parse_args()
    logf = open(L / "gru_sw_lomo.log", "a")

    def log(msg):
        line = f"[{time.strftime('%H:%M:%S')}] {msg}"; print(line, flush=True); logf.write(line + "\n"); logf.flush()

    if a.final:
        FINAL.mkdir(parents=True, exist_ok=True)
        eps = [json.loads((L / f"modelBs_f{F}.json").read_text())["best_epoch"] for F in MATCHES]
        E = int(round(float(np.median(eps))))
        paths = [fpath(f"fold{m}", k, s) for m in MATCHES for k, s in halves_of(m)]
        net, mu, sd, names, hist, _, t = run(paths, None, "final_all", n_epochs=E, budget=None, log=log)
        torch.save({"state_dict": net.state_dict(), "mu": torch.tensor(mu), "sd": torch.tensor(sd), "names": names,
                    "nf": int(len(names)), "hidden": 64, "win": g.WIN, "stride": g.STRIDE}, FINAL / "modelBs.pt")
        (FINAL / "modelBs.json").write_text(json.dumps({"epochs": E, "fold_best_epochs": eps, "seconds": round(t, 1),
                                                       "hist": hist, "arch": "Linear(nf,64)+ReLU -> BiGRU(64, 2 layers)"
                                                       " -> cls(3) + boundary(1)"}, indent=1))
        log(f"final B: epochs {E} (folds {eps}) {t:.0f}s")
        return
    for F in a.folds:
        js = L / f"modelBs_f{F}.json"
        if js.exists():
            log(f"B fold {F}: done"); continue
        train = [m for m in MATCHES if m != F]; V = max(train); inner = [m for m in train if m != V]
        tr = [fpath(src_of(F, m), k, s) for m in inner for k, s in halves_of(m)]
        va = [fpath(src_of(F, V), k, s) for k, s in halves_of(V)]
        net, mu, sd, names, hist, best, t_in = run(tr, va, f"f{F}_inner", log=log)
        save_preds(net, F, V, mu, sd, names)
        tr = [fpath(src_of(F, m), k, s) for m in train for k, s in halves_of(m)]
        net, mu, sd, names, _, _, t_fin = run(tr, None, f"f{F}_final", n_epochs=best[1], log=log)
        save_preds(net, F, F, mu, sd, names)
        js.write_text(json.dumps({"best_epoch": best[1], "inner_val_nll": best[0], "inner_seconds": round(t_in, 1),
                                  "final_seconds": round(t_fin, 1), "V": V, "hist_inner": hist}, indent=1))
        log(f"B fold {F} done best_epoch {best[1]} ({t_in + t_fin:.0f}s)")


if __name__ == "__main__":
    main()
