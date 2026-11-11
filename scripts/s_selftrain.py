"""Candidate S (Plan 9 amendment 3): cross-half self-training of the BiGRU stream B, simulator acceptance run.

Stages (the torch stages run in the torch venv, the others in the main env; all CPU):
  base   --fold F                  torch  base B of fold F (exp_seq_lomo_gru recipe: five training matches, the fold's
                                          registered epoch count); predictions of F's halves -> <out>/pred/base/f{F}
  pseudo --fold F                  main   pseudo step labels of F's halves (V1/V2: confident ball control; V3: plus the
                                          a17 decode of the base B at merged posterior >= 0.9) -> <out>/pseudo/f{F}
  ft     --fold F --variant V --key K --seed S   torch  fine-tune the base on the simulator data plus the OTHER half's
                                          pseudo labels, predict half K -> <out>/pred/{V}/f{F}
  score  [--folds ...]             main   decode every variant at the a17 cell (dead -3.0, seq_d -0.5, B weight 12),
                                          four truth variants -> <out>/lomo_scores.csv, summary.json
Variants: V1 (conf, w 1.0), V2 (conf, w 0.5), V3 (conf + a17 decode >= 0.9, w 0.5), V0 (diagnostic: V1 labels at weight 0).
Hyper-parameters are fixed here (amendment 3, item 4).
"""
import argparse, json, os, sys, time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "scripts"))
os.environ.setdefault("OMP_NUM_THREADS", "4")

import numpy as np

from mtah import selftrain as st

MATCHES = [9001, 9002, 9003, 9004, 9005, 9006]
SEEDS = [100, 101]
VARIANTS = {"V1": {"labels": "conf", "w": 1.0}, "V2": {"labels": "conf", "w": 0.5}, "V3": {"labels": "conf+a17", "w": 0.5},
            "V0": {"labels": "conf", "w": 0.0}}
FT = {"epochs": 4, "lr": 5e-4, "wd": 1e-5, "batch": 64, "clip": 1.0, "boundary_w": 0.5, "seed": 0}
CELL = {"dead_offset": -3.0, "seq_d_offset": -0.5, "w_B": 12.0}
MIN_POST = 0.9
DEFAULT_LOMO = ROOT / "runs" / "select" / "exp_seq2" / "lomo"
DEFAULT_SIM = ROOT / "runs" / "sim"
RULE = ROOT / "docs" / "arb" / "rule.json"


def args_paths(a):
    return Path(a.out), Path(a.lomo)


def keys_of(F):
    return [f"game_{F}_H{h}" for h in (1, 2)]


def other_key(key):
    return key[:-1] + ("2" if key.endswith("1") else "1")


def half_id(key):
    return int(key[-1])


# ----------------------------------------------------------------------------------------------------------- torch
def _torch_env():
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    import torch
    torch.set_num_threads(int(os.environ.get("OMP_NUM_THREADS", "4"))); torch.use_deterministic_algorithms(True)
    return torch


def finetune(base_ck, sim_seqs, pseudo_seq, w_pseudo, hp=FT, log=print):
    """Fine-tune a copy of the base net. sim_seqs [(Xn, y)] normalised simulator sequences; pseudo_seq (Xn, y_pseudo)
    of the OTHER half or None. Returns the net."""
    torch = _torch_env()
    import torch.nn as nn
    import torch.nn.functional as Fn
    import exp_seq_gru as g
    torch.manual_seed(hp["seed"]); rng = np.random.default_rng(hp["seed"])
    net = g.Net(base_ck["nf"], base_ck.get("hidden", 64)); net.load_state_dict(base_ck["state_dict"])
    opt = torch.optim.Adam(net.parameters(), lr=hp["lr"], weight_decay=hp["wd"])
    names = base_ck["names"]
    mirror = lambda x: g.mirror_np(x, names)
    swap = lambda y: np.where(y == 0, 1, np.where(y == 1, 0, y))
    for ep in range(1, hp["epochs"] + 1):
        net.train(); tl, nb = 0.0, 0
        wins = st.build_epoch(sim_seqs, pseudo_seq, w_pseudo, rng, g.WIN, g.STRIDE, mirror, swap)
        for k in range(0, len(wins), hp["batch"]):
            b = wins[k:k + hp["batch"]]
            xb = torch.tensor(np.stack([e["x"] for e in b])); yb = torch.tensor(np.stack([e["y"] for e in b]).astype(np.int64))
            wb = torch.tensor(np.stack([e["w"] for e in b]).astype(np.float32))
            simm = torch.tensor([0.0 if e["pseudo"] else 1.0 for e in b])
            opt.zero_grad()
            lo, lb = net(xb)
            ce = Fn.cross_entropy(lo.reshape(-1, 3), yb.reshape(-1), ignore_index=st.IGNORE, reduction="none").reshape(yb.shape)
            wce = (ce * wb).sum() / wb.sum().clamp(min=1e-6)
            bt = torch.tensor(np.stack([g.boundary(y) for y in yb.numpy()]))
            bce = Fn.binary_cross_entropy_with_logits(lb, bt, reduction="none").mean(1)
            bnd = (bce * simm).sum() / simm.sum().clamp(min=1.0)
            loss = wce + hp["boundary_w"] * bnd
            loss.backward(); nn.utils.clip_grad_norm_(net.parameters(), hp["clip"]); opt.step()
            tl += float(wce); nb += 1
        log(f"  ft epoch {ep}: weighted ce {tl / max(nb, 1):.4f} windows {len(wins)} (pseudo {sum(e['pseudo'] for e in wins)})")
    return net


def load_base(path):
    torch = _torch_env()
    ck = torch.load(path, weights_only=False)
    ck["mu"] = np.asarray(ck["mu"]); ck["sd"] = np.asarray(ck["sd"])
    return ck


def sim_train_seqs(gl, F, ck):
    """Normalised simulator training sequences of fold F (the five training matches, cross-fitted evidence)."""
    paths = [gl.fpath(gl.src_of(F, m), k, s) for m in MATCHES if m != F for k, s in gl.halves_of(m)]
    S, _, _, names = gl.seqs(paths, ck["mu"], ck["sd"])
    assert list(names) == list(ck["names"])
    return [(((X - ck["mu"]) / ck["sd"]).astype(np.float32), y) for _, X, y, _ in S]


def stage_base(a):
    out, lomo = args_paths(a)
    import exp_seq_lomo_gru as gl, exp_seq_gru as g
    torch = _torch_env()
    F = a.fold; (out / "base").mkdir(parents=True, exist_ok=True)
    ep = json.loads((lomo / f"modelB_f{F}.json").read_text())["best_epoch"]
    paths = [gl.fpath(gl.src_of(F, m), k, s) for m in MATCHES if m != F for k, s in gl.halves_of(m)]
    t0 = time.time()
    net, mu, sd, names, hist, _, _ = gl.run(paths, None, f"s_base_f{F}", n_epochs=ep, budget=None, log=print)
    torch.save({"state_dict": net.state_dict(), "mu": torch.tensor(mu), "sd": torch.tensor(sd), "names": names,
                "nf": len(names), "hidden": 64, "epochs": ep}, out / "base" / f"f{F}.pt")
    d = out / "pred" / "base" / f"f{F}"; d.mkdir(parents=True, exist_ok=True)
    for k, s in gl.halves_of(F):
        X, starts, seg, y, _ = gl.load(gl.fpath(gl.src_of(F, F), k, s))
        np.savez_compressed(d / f"{k}_s{s}.npz", logp=g.infer(net, X, mu, sd, names), starts=starts, y=y)
    print(f"base f{F} epochs {ep} {time.time() - t0:.0f}s")


def stage_ft(a):
    out, lomo = args_paths(a)
    import exp_seq_lomo_gru as gl, exp_seq_gru as g
    F, V, key, seed = a.fold, a.variant, a.key, a.seed
    spec = VARIANTS[V]
    src = other_key(key)
    st.leakage_check(half_id(key), half_id(src), [])
    t0 = time.time()
    ck = load_base(out / "base" / f"f{F}.pt")
    sims = sim_train_seqs(gl, F, ck)
    pz = np.load(out / "pseudo" / f"f{F}" / f"{src}_s{seed}.npz")
    lab = pz["y_conf"] if spec["labels"] == "conf" else pz["y_conf_a17"]
    Xo = np.load(gl.fpath(gl.src_of(F, F), src, seed))["X"]
    assert len(Xo) == len(lab)
    pseudo = (((Xo - ck["mu"]) / ck["sd"]).astype(np.float32), lab.astype(np.int64))
    net = finetune(ck, sims, pseudo, spec["w"])
    X, starts, seg, y, names = gl.load(gl.fpath(gl.src_of(F, F), key, seed))
    d = out / "pred" / V / f"f{F}"; d.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(d / f"{key}_s{seed}.npz", logp=g.infer(net, X, ck["mu"], ck["sd"], names), starts=starts, y=y)
    print(f"ft f{F} {V} {key} s{seed} {time.time() - t0:.0f}s")


# ------------------------------------------------------------------------------------------------------ main env
def decode_cell(h, wts_B=CELL["w_B"], seq=None):
    """(labels, cost matrix) of the registered decoder at the a17 cell for exp_seq_lomo.Half-like `h` with h.seq['B']."""
    import copy
    import exp_seq_lomo as L
    import mtah.select as sel
    from mtah.decode.viterbi import cost_matrix
    from mtah.evidence.dead import dead_evidence
    base = L.base_cfg()
    cfg = sel.apply_cell(base, {**sel.GRID8[0], "dead_offset": CELL["dead_offset"]})
    cfg.decoder.lengths = sel.estimate_lengths([], cfg, None)
    ev = dict(h.ev); ev["dead"] = dead_evidence(h.p_dead, logit_offset=CELL["dead_offset"])
    sb = (seq if seq is not None else h.seq["B"]).copy(); sb[:, 2] += np.float32(CELL["seq_d_offset"])
    hh = copy.copy(h); hh.seq = {"B": sb}
    lab = L.decode_labels(hh, ev, {"B": wts_B}, cfg)
    dcfg = copy.deepcopy(cfg.decoder); evd = dict(ev); evd["seqB"] = sb
    dcfg.weights["seqB"] = float(wts_B); dcfg.lengths["seqB"] = 1.0
    return lab, cost_matrix(evd, dcfg)


class HalfB:
    """Cached evidence of one sim half (exp_seq_lomo layout), Model-B stream from `pred_path` (or None)."""
    def __init__(self, lomo, F, key, seed, sim_dir, pred_path=None):
        import exp_seq_lomo as L
        import exp_seq_feats as fx
        from mtah.sim.dataset import load_half_sim
        self.s = load_half_sim(sim_dir, key, seed)
        z = np.load(L.hpath(L.src_of(F, F), key, seed))
        self.ev = {k: z[f"ev_{k}"] for k in L.STREAMS}
        self.strong = (z["sH"], z["sA"], z["touches"]); self.p_dead = z["p_dead"]
        self.seq = {}
        if pred_path is not None:
            p = np.load(pred_path)
            self.seq["B"] = fx.upsample(p["logp"], p["starts"], self.s.match.n_frames, self.s.match.half)


def stage_pseudo(a):
    out, lomo = args_paths(a)
    import arb_common as ac
    F = a.fold
    rule = ac.RuleParams(**json.loads(RULE.read_text())["rule"])
    d = out / "pseudo" / f"f{F}"; d.mkdir(parents=True, exist_ok=True); stats = []
    for key in keys_of(F):
        for seed in SEEDS:
            h = HalfB(lomo, F, key, seed, a.sim, out / "pred" / "base" / f"f{F}" / f"{key}_s{seed}.npz")
            m = h.s.match
            conf = st.exclude_frames(ac.confident_control(m, rule), ())     # no public window in the simulator
            lab, C = decode_cell(h)
            post = st.merged_posterior_labels(C, lab, MIN_POST)
            conf_a17 = st.combine_frame_labels(conf, post)
            z = np.load(lomo / "feat" / f"fold{F}" / f"{key}_s{seed}.npz"); starts = z["starts"]
            y1 = st.to_step_labels(conf, starts); y3 = st.to_step_labels(conf_a17, starts)
            np.savez_compressed(d / f"{key}_s{seed}.npz", y_conf=y1, y_conf_a17=y3)
            tr = np.array([{"H": 0, "A": 1, "D": 2}.get(c, 3) for c in h.s.truth["raw_Dfull"]])
            ok = lambda f: float((f[f >= 0] == tr[f >= 0]).mean()) if (f >= 0).any() else float("nan")
            stats.append({"fold": F, "key": key, "seed": seed, "n_conf": int((conf >= 0).sum()),
                          "n_conf_a17": int((conf_a17 >= 0).sum()), "n_post_only": int(((post >= 0) & (conf < 0)).sum()),
                          "prec_conf_raw": ok(conf), "prec_post_raw": ok(post), "n_post_D": int((post == 2).sum()),
                          "steps_conf": int((y1 != st.IGNORE).sum()), "steps_conf_a17": int((y3 != st.IGNORE).sum())})
            print(stats[-1], flush=True)
    (d / "stats.json").write_text(json.dumps(stats, indent=1))


def stage_score(a):
    out, lomo = args_paths(a)
    import pandas as pd
    import exp_seq_lomo as L
    rows = []
    folds = a.folds or MATCHES
    srcs = {"ref_stored": lambda F, k, s: lomo / "pred_B" / f"f{F}" / f"{k}_s{s}.npz",
            "ref_base_retrained": lambda F, k, s: out / "pred" / "base" / f"f{F}" / f"{k}_s{s}.npz"}
    for V in ("V0", "V1", "V2", "V3"):
        srcs[V] = (lambda V: lambda F, k, s: out / "pred" / V / f"f{F}" / f"{k}_s{s}.npz")(V)
    for name, fn in srcs.items():
        for F in folds:
            for k in keys_of(F):
                for s in SEEDS:
                    p = fn(F, k, s)
                    if not p.exists():
                        continue
                    h = HalfB(lomo, F, k, s, a.sim, p)
                    lab, _ = decode_cell(h)
                    rows += L.score_rows(h, lab, {"fold": F, "name": name})
        print(name, "done", flush=True)
    df = pd.DataFrame(rows); df.to_csv(out / "lomo_scores.csv", index=False)
    summ = {}
    for name, g in df.groupby("name"):
        summ[name] = {"n_folds": int(g.fold.nunique()), "n_halves": int(len(g) / 4)}
        for v in L.VARS:
            gv = g[g.variant == v]
            summ[name][v] = float(gv.spell_f1.mean())
            summ[name][v + "_per_fold"] = {int(f): float(x) for f, x in gv.groupby("fold").spell_f1.mean().items()}
    (out / "summary.json").write_text(json.dumps(summ, indent=1))
    for name, s in summ.items():
        print(name, s["n_folds"], "ctrl_Dfull %.4f raw_Dfull %.4f" % (s["ctrl_Dfull"], s["raw_Dfull"]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["base", "pseudo", "ft", "score"])
    ap.add_argument("--fold", type=int); ap.add_argument("--folds", type=int, nargs="*")
    ap.add_argument("--variant", choices=list(VARIANTS)); ap.add_argument("--key"); ap.add_argument("--seed", type=int)
    ap.add_argument("--out", default=str(ROOT / "runs" / "s" / "lomo")); ap.add_argument("--lomo", default=str(DEFAULT_LOMO))
    ap.add_argument("--sim", default=str(DEFAULT_SIM))
    a = ap.parse_args()
    {"base": stage_base, "pseudo": stage_pseudo, "ft": stage_ft, "score": stage_score}[a.stage](a)


if __name__ == "__main__":
    main()
