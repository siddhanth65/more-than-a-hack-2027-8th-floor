"""Boundary-placement experiment: baseline vs oracle vs touch-snap vs constant-offset (LOMO folds 9001, 9004)."""
import json, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
import os
sys.path.insert(0, os.environ.get('MTAH_CODE', str(ROOT)))
import numpy as np

from mtah.eval.official import possession_metrics as pm
from mtah.eval.scorer import codes
from mtah.evidence.touches import TouchParams, detect_touches
from mtah.pipeline import PipelineConfig, compute_evidence, compute_strong, predict
from mtah.select import apply_cell, estimate_lengths, train_dead
from mtah.sim.dataset import SEALED_MATCH, list_sim_keys, load_half_sim, lomo_folds, match_id

HELD = (9001, 9004); SEEDS = (100, 101); H, A, D = 0, 1, 2
TYPES = ("HA", "toD", "fromD"); TOLS = (1, 3, 8, 25); WIN = (12, 25, 50)
TP = TouchParams()


def ttype(fr, to):
    if to == D: return "toD"
    if fr == D: return "fromD"
    return "HA"


def f1(g, p, runs):
    return float(pm.evaluate(g, p, bootstrap_n=0)["spell_f1_mean"])


def errors(g, p, runs):
    gt = pm.transitions(g, runs); pt = pm.transitions(p, runs)
    out = {t: [] for t in TYPES}; n = {t: 0 for t in TYPES}
    for f, a, b in pt:
        t = ttype(a, b); n[t] += 1
        c = [tf for tf, ta, tb in gt if tb == b and ttype(ta, tb) == t and abs(tf - f) <= 50]
        if c:
            out[t].append(f - min(c, key=lambda x: (abs(x - f), x)))
    return out, n


def summ(e, n):
    r = {}
    for t in TYPES:
        x = np.asarray(e[t], float)
        d = {"n_pred_trans": n[t], "n_matched50": int(len(x))}
        if len(x):
            d.update(median=float(np.median(x)), mean=float(x.mean()), p10=float(np.percentile(x, 10)),
                     p90=float(np.percentile(x, 90)),
                     **{f"within{k}": float((np.abs(x) <= k).mean()) for k in TOLS})
        r[t] = d
    return r


def rebuild(p, runs, newb):
    """newb(frame, from, to, lo, hi) -> new boundary frame; applied sequentially over segments per run."""
    out = p.copy()
    for s0, e0 in runs:
        segs = [list(x) for x in pm.segments(p, [(s0, e0)])]
        for i in range(len(segs) - 1):
            fr, to = segs[i][2], segs[i + 1][2]
            lo, hi = segs[i][0], segs[i + 1][1]
            nb = newb(segs[i + 1][0], fr, to, lo, hi)
            if nb is None: continue
            nb = int(min(max(nb, lo + 1), hi - 1))
            segs[i][1] = nb; segs[i + 1][0] = nb
        for s, e, c in segs:
            out[s:e] = c
    return out


def snap(p, runs, touches, w, mode):
    fr_ = touches[:, 0].astype(int) if len(touches) else np.zeros(0, int)
    def nb(f, a, b, lo, hi):
        if b == D or a == D: return None
        m = np.abs(fr_ - f) <= w
        if mode in ("team", "strong"): m &= touches[:, 2] == b if len(touches) else m
        if mode == "strong": m &= touches[:, 3] >= TP.theta_hi if len(touches) else m
        if not m.any(): return None
        c = fr_[m]; return int(c[np.lexsort((c, np.abs(c - f)))[0]])
    return rebuild(p, runs, nb)


def offset(p, runs, med):
    def nb(f, a, b, lo, hi):
        m = med.get(ttype(a, b))
        return None if m is None else f - int(round(m))
    return rebuild(p, runs, nb)


def oracle(g, p, runs):
    gs = pm.possession_spells(g, runs); ps = pm.possession_spells(p, runs)
    ov = pm.spell_overlaps(gs, ps); used_g, used_p, pairs = set(), set(), []
    for iou, i, j in ov:
        if iou < 0.3: break
        if i in used_g or j in used_p: continue
        used_g.add(i); used_p.add(j); pairs.append((i, j))
    out = p.copy()
    for i, j in pairs: out[ps[j][0]:ps[j][1]] = D
    for i, j in pairs: out[gs[i][0]:gs[i][1]] = gs[i][2]
    return out


def main():
    best = json.loads((ROOT / "runs/select/a1/best.json").read_text()); cell = best["cell"]
    sim = str(ROOT / "runs" / "sim")
    keys = [k for k in list_sim_keys(sim) if match_id(k) != SEALED_MATCH]
    base = PipelineConfig(); data = {}
    for held, trk, tek in lomo_folds(keys):
        if held not in HELD: continue
        train = [load_half_sim(sim, k, s) for k in trk for s in SEEDS]
        test = [load_half_sim(sim, k, s) for k in tek for s in SEEDS]
        lengths = estimate_lengths(train, base, train_dead); dead = train_dead(train)
        cfg = apply_cell(base, cell); cfg.decoder.lengths = lengths
        L = []
        for s in test:
            ev = compute_evidence(s.match, s.tags, cfg, dead)
            lab = predict(s.match, s.tags, cfg, dead, evidence=ev, strong=compute_strong(s.match, cfg))
            g = codes(s.truth["ctrl_Dfull"]); runs = pm.valid_runs(g)
            p = pm.fill_abstentions(codes(lab), runs)
            L.append((g, runs, p, detect_touches(s.match, TP)))
        data[held] = L; print("fold", held, "predicted", flush=True)

    variants = {"baseline": lambda g, r, p, t, ctx: p, "oracle": lambda g, r, p, t, ctx: oracle(g, r, p)}
    variants["oracle"] = lambda g, r, p, t, ctx: oracle(g, p, r)
    for w in WIN:
        for mode in ("any", "team", "strong"):
            variants[f"snap_{mode}_w{w}"] = (lambda mode, w: lambda g, r, p, t, ctx: snap(p, r, t, w, mode))(mode, w)
    variants["offset_other_fold"] = lambda g, r, p, t, ctx: offset(p, r, ctx)

    # per-fold baseline median errors (for offset)
    meds = {}
    for held, L in data.items():
        e = {t: [] for t in TYPES}
        for g, r, p, _ in L:
            ee, _n = errors(g, r and p, r) if False else errors(g, p, r)
            for t in TYPES: e[t] += ee[t]
        meds[held] = {t: float(np.median(e[t])) for t in TYPES if e[t]}
    res = {"cell": cell, "seeds": list(SEEDS), "folds": list(HELD), "truth": "ctrl_Dfull",
           "offset_medians_by_estimating_fold": {str(k): v for k, v in meds.items()}, "variants": {}}
    for name, fn in variants.items():
        per_fold = {}; pool = {t: [] for t in TYPES}; npool = {t: 0 for t in TYPES}
        for held, L in data.items():
            ctx = meds[[h for h in HELD if h != held][0]]
            sc = []
            for g, r, p, t in L:
                q = fn(g, r, p, t, ctx); sc.append(f1(g, q, r))
                ee, nn = errors(g, q, r)
                for k in TYPES: pool[k] += ee[k]; npool[k] += nn[k]
            per_fold[str(held)] = float(np.mean(sc))
        res["variants"][name] = {"spell_f1_per_fold": per_fold, "spell_f1": float(np.mean(list(per_fold.values()))),
                                 "errors": summ(pool, npool)}
        print(name, res["variants"][name]["spell_f1"], flush=True)
    b = res["variants"]["baseline"]["spell_f1"]
    for v in res["variants"].values(): v["delta"] = v["spell_f1"] - b
    (ROOT / "runs/select/exp_boundaries.json").write_text(json.dumps(res, indent=1))
    print(f"{'variant':22s}{'f1':>7s}{'delta':>8s}  med_err HA/toD/fromD   within3 HA/toD/fromD")
    for k, v in res["variants"].items():
        m = [v["errors"][t].get("median", float("nan")) for t in TYPES]
        w3 = [v["errors"][t].get("within3", float("nan")) for t in TYPES]
        print(f"{k:22s}{v['spell_f1']:7.4f}{v['delta']:+8.4f}  " + "/".join(f"{x:+.1f}" for x in m) + "   " + "/".join(f"{x:.2f}" for x in w3))
    print("baseline errors", json.dumps(res["variants"]["baseline"]["errors"], indent=0))


if __name__ == "__main__":
    main()
