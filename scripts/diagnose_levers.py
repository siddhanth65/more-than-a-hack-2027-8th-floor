"""Plan 2d T0 lever diagnostic (sim only, seed 100, LOMO folds 9001-9006; the sealed 9007 is never loaded).

For the selected a3b cell (runs/select/a3b/best.json), per fold: dead model retrained on the fold's train halves,
PathCRF posteriors from the cache, labels decoded exactly as select_lomo does. Then:
 (a) error anatomy on visible alive frames (ball present, ctrl_Dfull truth H/A): team error rate and error-run
     lengths of the ball-stream argmax and of the final labels; per-stream own-team evidence margins inside false
     (unmatched at IoU 0.3) vs matched predicted spells; false splits / merges / boundary-only / unmatched counts
     by whether the spell starts on a visible frame.
 (b) ceiling A: the ball stream on visible frames replaced by a synthetic team-control stream (truth team with
     block noise whose error runs are resampled from the measured ball-argmax error runs) at accuracies a; and a
     one-fold learned HGB team-control model (train 9002-9006, test 9001) fed in the same way.
 (c) ceiling C: greedy oracle merge-only pass over the a3b labels (relabel a predicted H/A spell to the other team
     when a neighbour holds that team, iff spellF1 vs ctrl_Dfull rises), and a length-only merge rule.
Writes aggregates only to <dir>/levers.json (no per-frame data)."""
import argparse, copy, json, sys, time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
for p in (ROOT, ROOT / "scripts"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier

from mtah.eval.scorer import score
from mtah.evidence.dead import direction_of_play
from mtah.evidence.touches import TouchParams, detect_touches
from mtah.io.match import distance_to_line, half_runs
from mtah.pipeline import compute_evidence, compute_strong, predict
from mtah.select import apply_cell, estimate_lengths, train_dead
from mtah.sim.dataset import SEALED_MATCH, list_sim_keys, lomo_folds, match_id
from select_lomo import base_for_grid, load_sims

SEED = 100
IOUS = (0.3, 0.4, 0.5, 0.6, 0.7)
ACC_LEVELS = (0.85, 0.90, 0.95, 1.0)
MERGE_S = (0.5, 1.0, 2.0)
N_REPS = 3
EPS = 1e-3
TEAM = {"H": 0, "A": 1}


# ---------------------------------------------------------------- fast spell F1 (equal to the official scorer)
def to_codes(lab):
    lab = np.asarray(lab)
    c = np.full(len(lab), 3, np.int8)
    c[lab == "H"] = 0; c[lab == "A"] = 1; c[lab == "D"] = 2
    return c


def valid_runs(g):
    graded = (g != 3).astype(np.int8)
    e = np.diff(np.concatenate(([0], graded, [0])))
    return list(zip(np.flatnonzero(e == 1).tolist(), np.flatnonzero(e == -1).tolist()))


def spells(c, runs):
    """(n,3) int64 [start, stop, team] maximal H/A runs inside each valid run, in time order."""
    out = []
    for a, b in runs:
        w = c[a:b]
        if not len(w):
            continue
        br = np.flatnonzero(w[1:] != w[:-1]) + 1
        st = np.r_[0, br]; en = np.r_[br, len(w)]; k = w[st]
        keep = k < 2
        out.append(np.c_[st[keep] + a, en[keep] + a, k[keep]].astype(np.int64))
    return np.concatenate(out) if out else np.zeros((0, 3), np.int64)


def overlaps(gs, ps):
    """(iou, gi, pj) arrays sorted like pm.spell_overlaps: (-iou, gi, pj)."""
    G, P, I = [], [], []
    for t in (0, 1):
        gi = np.flatnonzero(gs[:, 2] == t); pj = np.flatnonzero(ps[:, 2] == t)
        if not len(gi) or not len(pj):
            continue
        g = gs[gi]; p = ps[pj]
        lo = np.searchsorted(p[:, 1], g[:, 0], side="right")
        hi = np.searchsorted(p[:, 0], g[:, 1], side="left")
        cnt = np.maximum(hi - lo, 0)
        if not cnt.sum():
            continue
        rg = np.repeat(np.arange(len(g)), cnt)
        rp = np.concatenate([np.arange(l, h) for l, h in zip(lo, hi) if h > l])
        a0, a1 = g[rg, 0], g[rg, 1]; b0, b1 = p[rp, 0], p[rp, 1]
        inter = np.minimum(a1, b1) - np.maximum(a0, b0); union = np.maximum(a1, b1) - np.minimum(a0, b0)
        G.append(gi[rg]); P.append(pj[rp]); I.append(inter / union.astype(np.float64))
    if not G:
        return np.zeros(0), np.zeros(0, int), np.zeros(0, int)
    G = np.concatenate(G); P = np.concatenate(P); I = np.concatenate(I)
    o = np.lexsort((P, G, -I))
    return I[o], G[o], P[o]


def greedy_flags(ov, n_gt, n_pred, thr):
    iou, gi, pj = ov
    mg = np.zeros(n_gt, bool); mp = np.zeros(n_pred, bool)
    for v, i, j in zip(iou.tolist(), gi.tolist(), pj.tolist()):
        if v < thr:
            break
        if mg[i] or mp[j]:
            continue
        mg[i] = True; mp[j] = True
    return mg, mp


def spell_f1(gt, pc):
    """gt = (codes, runs, gt spells); pc = predicted codes. Mean F1 over IOUS, identical to pm.spell_scores."""
    _, runs, gs = gt
    ps = spells(pc, runs)
    iou, gi, pj = overlaps(gs, ps)
    nt, npred = len(gs), len(ps)
    if nt + npred == 0:
        return 0.0
    mg = np.zeros(nt, bool); mp = np.zeros(npred, bool); m = 0
    # one greedy pass in IoU-descending order; the match count at threshold t is the state on leaving iou >= t
    order_thr = sorted(IOUS, reverse=True)
    counts = {}; k = 0
    for v, i, j in zip(iou.tolist(), gi.tolist(), pj.tolist()):
        while k < len(order_thr) and v < order_thr[k]:
            counts[order_thr[k]] = m; k += 1
        if k == len(order_thr):
            break
        if mg[i] or mp[j]:
            continue
        mg[i] = True; mp[j] = True; m += 1
    while k < len(order_thr):
        counts[order_thr[k]] = m; k += 1
    return float(np.mean([2.0 * counts[t] / (nt + npred) for t in IOUS]))


def gt_struct(truth):
    c = to_codes(truth); runs = valid_runs(c)
    return (c, runs, spells(c, runs))


# ---------------------------------------------------------------- helpers
def run_lengths(mask):
    m = np.asarray(mask, np.int8)
    e = np.diff(np.concatenate(([0], m, [0])))
    return np.flatnonzero(e == -1) - np.flatnonzero(e == 1)


def len_hist(lengths, fps):
    s = np.asarray(lengths, float) / fps
    n = len(s); fr = float(np.sum(lengths)) if n else 0.0
    out = {"n_runs": int(n), "frames": int(fr)}
    for name, lo, hi in (("lt1s", 0, 1), ("1to3s", 1, 3), ("gt3s", 3, np.inf)):
        sel = (s >= lo) & (s < hi)
        out[name] = {"n": int(sel.sum()), "share_runs": float(sel.mean()) if n else None,
                     "share_frames": float(np.asarray(lengths)[sel].sum() / fr) if fr else None}
    if n:
        out["quantiles_s"] = {f"p{q}": float(np.percentile(s, q)) for q in (10, 25, 50, 75, 90, 99)}
        out["mean_s"] = float(s.mean())
    return out


def argmax_team(ev):
    """0 = H, 1 = A, -1 = no opinion."""
    out = np.full(len(ev), -1, np.int8)
    ok = np.isfinite(ev[:, 0]) & np.isfinite(ev[:, 1])
    out[ok] = np.where(ev[ok, 0] >= ev[ok, 1], 0, 1)
    return out


def team_codes(lab):
    c = to_codes(lab).astype(np.int8)
    c[c > 1] = -1
    return c


def ffill_team(tc, half):
    """Truth team with D/X frames forward-filled (back-filled at the head) inside each half."""
    out = tc.copy()
    for _, a, b in half_runs(half):
        w = out[a:b]; known = np.flatnonzero(w >= 0)
        if not len(known):
            continue
        carry = np.maximum.accumulate(np.where(w >= 0, np.arange(len(w)), -1)); carry[carry < 0] = known[0]
        out[a:b] = w[carry]
    return out


def team_stream(p_home, valid):
    ev = np.full((len(p_home), 3), np.nan, np.float32)
    p = np.clip(np.asarray(p_home, np.float64), 0.0, 1.0)
    ev[valid, 0] = np.log(p[valid] + EPS); ev[valid, 1] = np.log(1.0 - p[valid] + EPS)
    return ev


def block_noise(n, err_runs, acc, rng):
    """(n,) bool flip mask: alternating correct (geometric, mean m_e*a/(1-a)) and error runs (resampled)."""
    flip = np.zeros(n, bool)
    if acc >= 1.0:
        return flip
    m_e = float(np.mean(err_runs)); q = 1.0 / max(m_e * acc / (1.0 - acc), 1.0)
    t = int(rng.geometric(q)) - 1                     # start inside a correct run of random length
    while t < n:
        e = int(err_runs[rng.integers(len(err_runs))])
        flip[t:t + e] = True; t += e + int(rng.geometric(q))
    return flip


# ---------------------------------------------------------------- learned team-control features (T1 sketch)
def _vel(xy, fps):
    v = np.full(xy.shape, np.nan, np.float64)
    v[1:-1] = (xy[2:] - xy[:-2]) * (fps / 2.0)
    return v


def _centred_mean(x, w):
    f = np.isfinite(x); v = np.where(f, x, 0.0)
    cv = np.concatenate([[0.0], np.cumsum(v)]); cf = np.concatenate([[0.0], np.cumsum(f)])
    n = len(x); i = np.arange(n); lo = np.clip(i - w, 0, n); hi = np.clip(i + w + 1, 0, n)
    with np.errstate(invalid="ignore", divide="ignore"):
        return (cv[hi] - cv[lo]) / (cf[hi] - cf[lo])


def _touch_times(touches, F, fps, team):
    """seconds since the last / until the next detected touch by `team` (capped 10 s)."""
    since = np.full(F, 10.0); until = np.full(F, 10.0)
    if len(touches):
        fr = np.sort(touches[touches[:, 2] == team, 0].astype(int))
        if len(fr):
            idx = np.arange(F)
            k = np.searchsorted(fr, idx, side="right") - 1
            ok = k >= 0; since[ok] = np.minimum((idx[ok] - fr[k[ok]]) / fps, 10.0)
            k2 = np.searchsorted(fr, idx, side="left"); ok2 = k2 < len(fr)
            until[ok2] = np.minimum((fr[k2[ok2]] - idx[ok2]) / fps, 10.0)
    return since, until


def team_ctrl_features(match, touches):
    """(F, K) float32 features and per-column swap kind ('H'/'A' paired, 'S' symmetric, 'N' negate on swap).
    Rows are NaN where the ball is absent. x is attack-signed per team (direction of play)."""
    fps = match.fps; F = match.n_frames
    ball = match.ball.astype(np.float64); pos = match.pos.astype(np.float64)
    bv = _vel(ball, fps); pv = _vel(pos, fps)
    bspeed = np.hypot(bv[:, 0], bv[:, 1])
    bacc = np.full(F, np.nan); bacc[1:-1] = (bspeed[2:] - bspeed[:-2]) * (fps / 2.0)
    dop = direction_of_play(match).astype(np.float64)
    cols, kinds, names = [], [], []
    d1 = {}
    per_team = {}
    for k, tag in ((0, "H"), (1, "A")):
        sl = np.flatnonzero(match.team == k)
        rel = pos[:, sl] - ball[:, None]
        d = np.hypot(rel[..., 0], rel[..., 1]); d = np.where(np.isfinite(d), d, np.inf)
        order = np.argsort(d, axis=1)[:, :3]
        rows = np.arange(F)[:, None]
        dn = d[rows, order]
        r = rel[rows, order]; v = pv[:, sl][rows, order]
        relv = v - bv[:, None]
        with np.errstate(invalid="ignore", divide="ignore"):
            closing = -(r[..., 0] * relv[..., 0] + r[..., 1] * relv[..., 1]) / dn
            ps = np.hypot(v[..., 0], v[..., 1])
            cosang = (v[..., 0] * bv[:, None, 0] + v[..., 1] * bv[:, None, 1]) / (ps * bspeed[:, None])
        cosang = np.where((ps > 0.5) & (bspeed[:, None] > 0.5), cosang, 0.0)
        relspeed = np.hypot(relv[..., 0], relv[..., 1])
        dn = np.where(np.isfinite(dn), dn, 60.0)
        f = {}
        for j in range(3):
            f[f"d{j + 1}"] = dn[:, j]; f[f"close{j + 1}"] = closing[:, j]; f[f"spd{j + 1}"] = ps[:, j]
            f[f"cos{j + 1}"] = cosang[:, j]
        f["relspd1"] = relspeed[:, 0]
        f["n5"] = (d <= 5.0).sum(1).astype(float); f["n10"] = (d <= 10.0).sum(1).astype(float)
        f["nvis"] = np.isfinite(pos[:, sl, 0]).sum(1).astype(float)
        f["ball_x_att"] = ball[:, 0] * dop[:, k]
        since, until = _touch_times(touches, F, fps, k)
        f["t_since_touch"] = since; f["t_until_touch"] = until
        per_team[tag] = f; d1[tag] = dn[:, 0]
    for name in per_team["H"]:
        for tag in ("H", "A"):
            cols.append(per_team[tag][name]); kinds.append(tag); names.append(f"{name}_{tag}")
    diff = d1["H"] - d1["A"]
    for w_s in (0.5, 1.0, 2.0):
        cols.append(_centred_mean(np.where(match.ball_present, diff, np.nan), int(w_s * fps)))
        kinds.append("N"); names.append(f"d1diff_pm{w_s}s")
    cols.append(diff); kinds.append("N"); names.append("d1diff")
    lt = np.zeros(F)   # team of the last detected touch: +1 H, -1 A, 0 contested / none
    if len(touches):
        tf = touches[:, 0].astype(int); tt = touches[:, 2]
        o = np.argsort(tf); tf, tt = tf[o], tt[o]
        k = np.searchsorted(tf, np.arange(F), side="right") - 1
        ok = k >= 0; lt[ok] = np.where(tt[k[ok]] == 0, 1.0, np.where(tt[k[ok]] == 1, -1.0, 0.0))
    cols.append(lt); kinds.append("N"); names.append("last_touch_team")
    for name, x in (("ball_speed", bspeed), ("ball_acc", bacc), ("ball_absy", np.abs(ball[:, 1])),
                    ("ball_line_dist", distance_to_line(ball, match.pitch))):
        cols.append(x); kinds.append("S"); names.append(name)
    X = np.column_stack(cols).astype(np.float32)
    X[~match.ball_present] = np.nan
    return X, kinds, names


def swap_perm(kinds):
    """(perm, sign): X_swapped = X[:, perm] * sign."""
    idx = list(range(len(kinds))); sign = np.ones(len(kinds), np.float32)
    for i, k in enumerate(kinds):
        if k == "H":
            idx[i], idx[i + 1] = i + 1, i
        if k == "N":
            sign[i] = -1.0
    return np.array(idx), sign


def fit_team_ctrl(train, stride=5):
    Xs, ys = [], []
    kinds = None
    for s in train:
        touches = detect_touches(s.match, TouchParams())
        X, kinds, names = team_ctrl_features(s.match, touches)
        t = s.truth["ctrl_Dfull"]
        sel = s.match.ball_present & np.isin(t, ("H", "A"))
        idx = np.flatnonzero(sel)[::stride]
        Xs.append(X[idx]); ys.append((t[idx] == "H").astype(int))
    X = np.concatenate(Xs); y = np.concatenate(ys)
    perm, sign = swap_perm(kinds)
    Xa = np.concatenate([X, X[:, perm] * sign]); ya = np.concatenate([y, 1 - y])
    m = HistGradientBoostingClassifier(max_iter=300, max_leaf_nodes=31, learning_rate=0.1, min_samples_leaf=200,
                                       random_state=0, early_stopping=False).fit(Xa, ya)
    return {"model": m, "perm": perm, "sign": sign, "names": names, "n_rows": int(len(Xa))}


def predict_team_ctrl(tm, match):
    touches = detect_touches(match, TouchParams())
    X, _, _ = team_ctrl_features(match, touches)
    p = np.full(match.n_frames, np.nan)
    ok = match.ball_present
    if ok.any():
        a = tm["model"].predict_proba(X[ok])[:, 1]
        b = tm["model"].predict_proba(X[ok][:, tm["perm"]] * tm["sign"])[:, 1]
        p[ok] = 0.5 * (a + 1.0 - b)          # swap-antisymmetric
    return p


# ---------------------------------------------------------------- merges
def relabel_ok(lab, half_bounds, s, e, team):
    """Other team present on an adjacent frame (same half)? Returns the other team's label or None."""
    a, b = half_bounds
    other = "A" if team == "H" else "H"
    if (s - 1 >= a and lab[s - 1] == other) or (e < b and lab[e] == other):
        return other
    return None


def half_bounds_of(half, f):
    for _, a, b in half_runs(half):
        if a <= f < b:
            return a, b
    return None


def oracle_merge(lab, gt, half, fixed_point=False):
    lab = np.asarray(lab).astype("<U1").copy()
    cur = spell_f1(gt, to_codes(lab)); n_merges = 0
    while True:
        changed = False
        for s, e, k in spells(to_codes(lab), gt[1]).tolist():
            team = "H" if k == 0 else "A"
            if not np.all(lab[s:e] == team):
                continue
            hb = half_bounds_of(half, s)
            other = relabel_ok(lab, hb, s, e, team) if hb else None
            if other is None:
                continue
            lab[s:e] = other
            f = spell_f1(gt, to_codes(lab))
            if f > cur + 1e-12:
                cur = f; n_merges += 1; changed = True
            else:
                lab[s:e] = team
        if not (fixed_point and changed):
            break
    return lab, n_merges


def length_merge(lab, runs, half, fps, s_max):
    lab = np.asarray(lab).astype("<U1").copy(); n = 0
    sp = spells(to_codes(lab), runs)
    sp = sp[(sp[:, 1] - sp[:, 0]) < s_max * fps]
    for s, e, k in sp[np.argsort(sp[:, 1] - sp[:, 0], kind="stable")].tolist():
        team = "H" if k == 0 else "A"
        if not np.all(lab[s:e] == team):
            continue
        hb = half_bounds_of(half, s)
        other = relabel_ok(lab, hb, s, e, team) if hb else None
        if other is not None:
            lab[s:e] = other; n += 1
    return lab, n


def merge_table(h):
    """Merge candidates of the a3b labels: predicted H/A spells with an other-team neighbour.
    Returns (rows (n, K) float, spans [(s, e, team_code)], y spurious = own-team truth share < 0.5)."""
    pl = to_codes(h["lab"]); runs = h["gt"][1]; tc = team_codes(h["truth"]["ctrl_Dfull"])
    sH, sA = h["strong"][0], h["strong"][1]; bp = h["ball_present"]; fps = h["fps"]
    X, spans, y = [], [], []
    for a, b in runs:
        w = pl[a:b]
        if not len(w):
            continue
        br = np.flatnonzero(w[1:] != w[:-1]) + 1
        st = np.r_[0, br] + a; en = np.r_[br, len(w)] + a; kk = w[st - a]
        for i in range(len(st)):
            k = int(kk[i])
            if k > 1:
                continue
            nb = []
            for j in (i - 1, i + 1):
                if 0 <= j < len(st):
                    nb.append((0.0 if kk[j] == 1 - k else 1.0, (en[j] - st[j]) / fps))
                else:
                    nb.append((2.0, 0.0))
            if not any(n[0] == 0.0 for n in nb):
                continue
            s_, e_ = int(st[i]), int(en[i])
            row = [(e_ - s_) / fps, nb[0][0], nb[0][1], nb[1][0], nb[1][1], float(bp[s_:e_].mean())]
            for name in ("ball", "tag", "pathcrf"):
                mm = margins(h["ev"][name][s_:e_], k); f = np.isfinite(mm)
                row += [float(mm[f].mean()) if f.any() else np.nan, float(mm[f].min()) if f.any() else np.nan]
            own, oth = (sH, sA) if k == 0 else (sA, sH)
            row += [float(own[s_:e_].sum()), float(oth[s_:e_].sum())]
            X.append(row); spans.append((s_, e_, k)); y.append(float((tc[s_:e_] == k).mean()) < 0.5)
    return np.array(X, float).reshape(-1, 14), spans, np.array(y, int)


def apply_merge_model(h, proba, spans, theta):
    lab = np.asarray(h["lab"]).astype("<U1").copy(); n = 0
    for i in np.argsort(-proba, kind="stable"):
        if proba[i] <= theta:
            break
        s, e, k = spans[i]; team = "H" if k == 0 else "A"
        if not np.all(lab[s:e] == team):
            continue
        hb = half_bounds_of(h["half"], s)
        other = relabel_ok(lab, hb, s, e, team) if hb else None
        if other is not None:
            lab[s:e] = other; n += 1
    return lab, n


def merge_model_lomo(halves, thetas=(0.5, 0.6, 0.7, 0.8)):
    """Realistic C1: per held fold, an HGB spell-merge classifier trained on the other folds' out-of-fold a3b labels."""
    tabs = {id(h): merge_table(h) for h in halves}
    folds = sorted({h["held"] for h in halves})
    rows = {t: [] for t in thetas}; nm = {t: 0 for t in thetas}
    if len(folds) < 2:
        return None
    for f in folds:
        tr = [tabs[id(h)] for h in halves if h["held"] != f]
        X = np.concatenate([t[0] for t in tr]); y = np.concatenate([t[2] for t in tr])
        m = HistGradientBoostingClassifier(max_iter=200, max_leaf_nodes=15, min_samples_leaf=40, random_state=0,
                                           early_stopping=False).fit(X, y)
        for h in (h for h in halves if h["held"] == f):
            Xh, spans, _ = tabs[id(h)]
            pr = m.predict_proba(Xh)[:, 1] if len(Xh) else np.zeros(0)
            for t in thetas:
                lab2, k = apply_merge_model(h, pr, spans, t); nm[t] += k
                rows[t].append({"ctrl_Dfull": spell_f1(h["gt"], to_codes(lab2)), "raw_Dfull": spell_f1(h["gt_raw"], to_codes(lab2))})
    n_cand = int(sum(len(t[2]) for t in tabs.values())); n_spur = int(sum(t[2].sum() for t in tabs.values()))
    return {"n_candidates": n_cand, "n_spurious": n_spur,
            "theta": {f"{t:g}": {**mean_sc(rows[t]), "n_merges": nm[t]} for t in thetas}}


# ---------------------------------------------------------------- anatomy
def margins(ev, team):
    """per-frame own-team margin e_own - e_other (NaN where the stream is silent)."""
    m = ev[:, 0].astype(np.float64) - ev[:, 1].astype(np.float64)
    return m if team == 0 else -m


def anatomy_half(h, acc):
    fps = h["fps"]; bp = h["ball_present"]
    tc = team_codes(h["truth"]["ctrl_Dfull"]); vis = bp & (tc >= 0)
    pl = to_codes(h["lab"])
    ball_am = argmax_team(h["ev"]["ball"]); pc_am = argmax_team(h["ev"]["pathcrf"])
    final_err = vis & (pl < 2) & (pl != tc); final_D = vis & (pl == 2)
    ball_err = vis & (ball_am >= 0) & (ball_am != tc)
    pc_err = vis & (pc_am >= 0) & (pc_am != tc)
    a = acc.setdefault("frames", {k: 0 for k in ("vis_alive", "final_team_err", "final_pred_D", "ball_opinion",
                                                  "ball_err", "pathcrf_opinion", "pathcrf_err")})
    a["vis_alive"] += int(vis.sum()); a["final_team_err"] += int(final_err.sum()); a["final_pred_D"] += int(final_D.sum())
    a["ball_opinion"] += int((vis & (ball_am >= 0)).sum()); a["ball_err"] += int(ball_err.sum())
    a["pathcrf_opinion"] += int((vis & (pc_am >= 0)).sum()); a["pathcrf_err"] += int(pc_err.sum())
    acc.setdefault("final_err_runs", []).extend(run_lengths(final_err).tolist())
    acc.setdefault("ball_err_runs", []).extend(run_lengths(ball_err).tolist())
    # spells
    gt = h["gt"]; gs = gt[2]; ps = spells(pl, gt[1])
    ov = overlaps(gs, ps)
    per_gt = np.bincount(ov[1], minlength=len(gs)); per_pred = np.bincount(ov[2], minlength=len(ps))
    mg3, mp3 = greedy_flags(ov, len(gs), len(ps), 0.3); mg7, _ = greedy_flags(ov, len(gs), len(ps), 0.7)
    gvis = bp[gs[:, 0]] if len(gs) else np.zeros(0, bool); pvis = bp[ps[:, 0]] if len(ps) else np.zeros(0, bool)
    d = acc.setdefault("decomp", {})
    for lbl, sel_g, sel_p in (("start_visible", gvis, pvis), ("start_ball_absent", ~gvis, ~pvis)):
        r = d.setdefault(lbl, {k: 0 for k in ("n_gt", "n_pred", "false_splits", "false_merges", "boundary_only",
                                                "matched_07", "unmatched_gt_03", "unmatched_pred_03")})
        r["n_gt"] += int(sel_g.sum()); r["n_pred"] += int(sel_p.sum())
        r["false_splits"] += int(np.maximum(per_gt - 1, 0)[sel_g].sum())
        r["false_merges"] += int(np.maximum(per_pred - 1, 0)[sel_p].sum())
        r["boundary_only"] += int((mg3 & ~mg7)[sel_g].sum()); r["matched_07"] += int(mg7[sel_g].sum())
        r["unmatched_gt_03"] += int((~mg3)[sel_g].sum()); r["unmatched_pred_03"] += int((~mp3)[sel_p].sum())
    # per-spell margins (predicted spells), split matched@0.3 vs false (unmatched@0.3)
    rec = acc.setdefault("spell_margins", [])
    streams = {k: h["ev"][k] for k in ("ball", "tag", "pathcrf")}
    for (s, e, k), m, v in zip(ps.tolist(), mp3.tolist(), pvis.tolist()):
        row = {"matched": bool(m), "visible": bool(v), "len_s": (e - s) / fps}
        comb = np.zeros(e - s); anyf = np.zeros(e - s, bool)
        for name, ev in streams.items():
            mm = margins(ev[s:e], k)
            f = np.isfinite(mm)
            row[name] = float(mm[f].mean()) if f.any() else np.nan
            if name in ("ball", "pathcrf"):
                comb += np.where(f, mm, 0.0); anyf |= f
        row["ball+pathcrf"] = float(comb[anyf].mean()) if anyf.any() else np.nan
        rec.append(row)


def finish_anatomy(acc, fps):
    fr = acc["frames"]
    out = {"frames": fr,
           "final_team_err_rate": fr["final_team_err"] / fr["vis_alive"],
           "final_pred_D_rate": fr["final_pred_D"] / fr["vis_alive"],
           "ball_argmax_err_rate": fr["ball_err"] / fr["ball_opinion"],
           "pathcrf_argmax_err_rate": fr["pathcrf_err"] / fr["pathcrf_opinion"],
           "final_err_runs": len_hist(acc["final_err_runs"], fps),
           "ball_err_runs": len_hist(acc["ball_err_runs"], fps),
           "decomp_by_start": acc["decomp"]}
    rows = acc["spell_margins"]
    sm = {}
    for vis_lbl, vsel in (("visible_start", True), ("all", None)):
        sub = [r for r in rows if vsel is None or r["visible"] == vsel]
        blk = {}
        for cls, msel in (("false", False), ("matched", True)):
            rr = [r for r in sub if r["matched"] == msel]
            blk[cls] = {"n": len(rr), "median_len_s": float(np.median([r["len_s"] for r in rr])) if rr else None}
            for k in ("ball", "tag", "pathcrf", "ball+pathcrf"):
                v = np.array([r[k] for r in rr], float); v = v[np.isfinite(v)]
                blk[cls][k] = {"mean": float(v.mean()) if len(v) else None, "median": float(np.median(v)) if len(v) else None,
                               "share_negative": float((v < 0).mean()) if len(v) else None, "n": int(len(v))}
        med = blk["matched"]["ball+pathcrf"]["median"]
        fv = np.array([r["ball+pathcrf"] for r in sub if not r["matched"]], float); fv = fv[np.isfinite(fv)]
        blk["false_confident_share"] = float((fv > med).mean()) if len(fv) and med is not None else None
        blk["false_wrong_sign_share"] = float((fv < 0).mean()) if len(fv) else None
        sm[vis_lbl] = blk
    out["spell_margins"] = sm
    return out


# ---------------------------------------------------------------- main
def decode_score(h, ev, cfg):
    lab = predict(h["match"], h["tags"], cfg, None, evidence=ev, strong=h["strong"])
    return lab, {"ctrl_Dfull": spell_f1(h["gt"], to_codes(lab)), "raw_Dfull": spell_f1(h["gt_raw"], to_codes(lab))}


def mean_sc(rows, keys=("ctrl_Dfull", "raw_Dfull")):
    return {k: float(np.mean([r[k] for r in rows])) for k in keys}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sim", default=str(ROOT / "runs" / "sim"))
    ap.add_argument("--dir", default=str(ROOT / "runs" / "select" / "a3b"))
    ap.add_argument("--out", default=None, help="output dir for levers.json (default: --dir)")
    ap.add_argument("--folds", type=int, nargs="*", default=None, help="subset of held-out matches (debug)")
    args = ap.parse_args()
    t0 = time.time()
    sel_dir = Path(args.dir)
    cell = json.loads((sel_dir / "best.json").read_text())["cell"]
    keys = [k for k in list_sim_keys(args.sim) if match_id(k) != SEALED_MATCH]
    base = base_for_grid([cell])
    halves, team_model, tm_info = [], None, None
    for held, train_keys, test_keys in lomo_folds(keys):
        if args.folds and held not in args.folds:
            continue
        assert SEALED_MATCH not in {match_id(k) for k in train_keys + test_keys}
        train = load_sims(args.sim, [SEED], train_keys); test = load_sims(args.sim, [SEED], test_keys)
        lengths = estimate_lengths(train, base, train_dead)
        dead = train_dead(train, policy=cell.get("dead_policy", "full"))
        cfg = apply_cell(base, cell); cfg.decoder.lengths = lengths
        if held == 9001:
            tt = time.time(); team_model = fit_team_ctrl(train)
            tm_info = {"train_rows_aug": team_model["n_rows"], "features": team_model["names"],
                       "fit_s": round(time.time() - tt, 1)}
            print(f"team-ctrl model fitted ({team_model['n_rows']} rows, {tm_info['fit_s']} s)", flush=True)
        del train
        for s in test:
            ev = compute_evidence(s.match, s.tags, cfg, dead, cache_key=f"{s.key}_s{s.seed}")
            strong = compute_strong(s.match, cfg)
            lab = predict(s.match, s.tags, cfg, None, evidence=ev, strong=strong)
            h = {"key": s.key, "held": held, "match": s.match, "tags": s.tags, "ev": ev, "strong": strong, "lab": lab,
                 "cfg": cfg, "truth": {v: s.truth[v] for v in ("ctrl_Dfull", "raw_Dfull")}, "fps": s.match.fps,
                 "ball_present": s.match.ball_present, "half": s.match.half,
                 "gt": gt_struct(s.truth["ctrl_Dfull"]), "gt_raw": gt_struct(s.truth["raw_Dfull"])}
            # the fast scorer must equal the official one
            for v, g in (("ctrl_Dfull", h["gt"]), ("raw_Dfull", h["gt_raw"])):
                off = score(s.truth[v], lab)["spell_f1"]; mine = spell_f1(g, to_codes(lab))
                assert abs(off - mine) < 1e-12, (s.key, v, off, mine)
            h["base"] = {"ctrl_Dfull": spell_f1(h["gt"], to_codes(lab)), "raw_Dfull": spell_f1(h["gt_raw"], to_codes(lab))}
            if held == 9001 and team_model is not None:
                h["p_tc"] = predict_team_ctrl(team_model, s.match)
            halves.append(h)
        print(f"fold {held} decoded; base ctrl {np.mean([h['base']['ctrl_Dfull'] for h in halves if h['held'] == held]):.4f}"
              f" ({time.time() - t0:.0f} s)", flush=True)
    fps = halves[0]["fps"]
    baseline = mean_sc([h["base"] for h in halves])
    folds_used = sorted({h["held"] for h in halves})
    print(f"baseline (seed {SEED}, {len(halves)} halves): {baseline}", flush=True)

    # (a) anatomy
    acc = {}
    for h in halves:
        anatomy_half(h, acc)
    ball_runs = np.array(acc["ball_err_runs"], int)
    anat = finish_anatomy(acc, fps)
    print("anatomy done", flush=True)

    # (b) ceiling A: synthetic team-control stream in place of the ball stream on visible frames
    def synth_rows(level, sub, err_runs):
        rows, accs = [], []
        for rep in range(N_REPS):
            for hi, h in enumerate(sub):
                rng = np.random.default_rng([int(level * 1000), rep, hi, h["held"]])
                tc = team_codes(h["truth"]["ctrl_Dfull"]); team = ffill_team(tc, h["half"])
                flip = block_noise(len(team), err_runs, level, rng)
                noisy = np.where(flip, 1 - team, team)
                p_home = np.where(noisy == 0, level, 1.0 - level) if level < 1.0 else np.where(noisy == 0, 1.0, 0.0)
                bp = h["ball_present"]; vis = bp & (tc >= 0)
                accs.append(((noisy == tc) & vis).sum() / max(vis.sum(), 1))
                ev = dict(h["ev"]); ev["ball"] = team_stream(p_home, bp)
                rows.append(decode_score(h, ev, h["cfg"])[1])
        return mean_sc(rows), float(np.mean(accs))

    ceilA = {}
    for level in ACC_LEVELS:
        sc, real_acc = synth_rows(level, halves, ball_runs)
        ceilA[f"{level:.2f}"] = {"realised_vis_acc": real_acc, **sc,
                                 "delta_ctrl": sc["ctrl_Dfull"] - baseline["ctrl_Dfull"],
                                 "delta_raw": sc["raw_Dfull"] - baseline["raw_Dfull"]}
        print(f"ceiling A a={level:.2f}: {ceilA[f'{level:.2f}']}", flush=True)

    learned = None
    f1h = [h for h in halves if h["held"] == 9001]
    if f1h and team_model is not None:
        n = c_tc = c_ball = c_pc = n_ball = n_pc = 0
        for h in f1h:
            tc = team_codes(h["truth"]["ctrl_Dfull"]); vis = h["ball_present"] & (tc >= 0)
            p = h["p_tc"]; pred = np.where(p >= 0.5, 0, 1)
            n += int(vis.sum()); c_tc += int(((pred == tc) & vis).sum())
            bam = argmax_team(h["ev"]["ball"]); okb = vis & (bam >= 0)
            n_ball += int(okb.sum()); c_ball += int(((bam == tc) & okb).sum())
            pam = argmax_team(h["ev"]["pathcrf"]); okp = vis & (pam >= 0)
            n_pc += int(okp.sum()); c_pc += int(((pam == tc) & okp).sum())
        acc_tc = c_tc / n
        base1 = mean_sc([h["base"] for h in f1h])
        combos = {}
        for wb, wt in ((0.0, 1.0), (1.0, 1.0), (0.0, 2.0), (1.0, 2.0)):
            rows = []
            for h in f1h:
                ev = dict(h["ev"]); tcev = team_stream(h["p_tc"], np.isfinite(h["p_tc"]))
                cfg = copy.deepcopy(h["cfg"])
                if wb == 0.0:
                    ev["ball"] = tcev; cfg.decoder.weights["ball"] = wt          # replace: stream takes the ball slot
                else:
                    ev["teamctrl"] = tcev; cfg.decoder.weights["teamctrl"] = wt; cfg.decoder.lengths["teamctrl"] = 1.0
                rows.append(decode_score(h, ev, cfg)[1])
            sc = mean_sc(rows)
            combos[f"w_ball{wb:g}_w_tc{wt:g}"] = {**sc, "delta_ctrl": sc["ctrl_Dfull"] - base1["ctrl_Dfull"],
                                                   "delta_raw": sc["raw_Dfull"] - base1["raw_Dfull"]}
        sc_syn, ra = synth_rows(round(acc_tc, 4), halves, ball_runs)
        sc_syn1, _ = synth_rows(round(acc_tc, 4), f1h, ball_runs)
        learned = {"fold": 9001, "train": "9002-9006 seed 100, stride 5, team-swap augmentation", **(tm_info or {}),
                   "vis_frames": n, "acc_teamctrl": acc_tc, "acc_ball_argmax": c_ball / n_ball,
                   "ball_argmax_coverage": n_ball / n, "acc_pathcrf_argmax": c_pc / n_pc,
                   "baseline_fold": base1, "decode": combos,
                   "spellF1_delta_replace_ball": combos["w_ball0_w_tc1"]["delta_ctrl"],
                   "synthetic_at_model_acc_all_folds": {"acc": acc_tc, "realised": ra, **sc_syn,
                                                        "delta_ctrl": sc_syn["ctrl_Dfull"] - baseline["ctrl_Dfull"],
                                                        "delta_raw": sc_syn["raw_Dfull"] - baseline["raw_Dfull"]},
                   "synthetic_at_model_acc_fold9001": {**sc_syn1, "delta_ctrl": sc_syn1["ctrl_Dfull"] - base1["ctrl_Dfull"]}}
        print(f"learned team-ctrl: acc {acc_tc:.4f} vs ball argmax {c_ball / n_ball:.4f}; combos {combos}", flush=True)

    # (c) ceiling C: merges only
    ceilC = {}
    for name, fn in (("oracle_greedy_one_pass", lambda h: oracle_merge(h["lab"], h["gt"], h["half"])),
                     ("oracle_greedy_fixed_point", lambda h: oracle_merge(h["lab"], h["gt"], h["half"], fixed_point=True)),
                     *[(f"length_lt_{s:g}s", (lambda s_: (lambda h: length_merge(h["lab"], h["gt"][1], h["half"], fps, s_)))(s))
                       for s in MERGE_S]):
        rows, nm, npred0 = [], 0, 0
        for h in halves:
            lab2, k = fn(h); nm += k
            rows.append({"ctrl_Dfull": spell_f1(h["gt"], to_codes(lab2)), "raw_Dfull": spell_f1(h["gt_raw"], to_codes(lab2))})
            npred0 += len(spells(to_codes(h["lab"]), h["gt"][1]))
        sc = mean_sc(rows)
        ceilC[name] = {**sc, "n_merges": nm, "n_pred_spells_before": npred0,
                       "delta_ctrl": sc["ctrl_Dfull"] - baseline["ctrl_Dfull"], "delta_raw": sc["raw_Dfull"] - baseline["raw_Dfull"]}
        print(f"ceiling C {name}: {ceilC[name]}", flush=True)

    c1 = merge_model_lomo(halves)
    if c1:
        for t, r in c1["theta"].items():
            r["delta_ctrl"] = r["ctrl_Dfull"] - baseline["ctrl_Dfull"]; r["delta_raw"] = r["raw_Dfull"] - baseline["raw_Dfull"]
        ceilC["c1_merge_model_lomo"] = c1
        print(f"C1 realistic (LOMO spell merge model): {c1}", flush=True)

    # decision rule (design section 3): dA = synthetic ceiling at the one-fold model's accuracy, dC = oracle merge
    dA = learned["synthetic_at_model_acc_all_folds"]["delta_ctrl"] if learned else None
    dC = ceilC["oracle_greedy_fixed_point"]["delta_ctrl"]
    if dA is None:
        order = None
    elif dA < 0.02 and dC < 0.02:
        order = "E1 + A2 only (both < 0.02)"
    elif dA >= 1.5 * dC:
        order = "T1, T2, T3"
    elif dC >= dA:
        order = "T3, T1, T2"
    else:
        order = "T1, T3, T2 (between rules: dC < dA < 1.5 dC)"
    out = {"cell": cell, "seed": SEED, "folds": folds_used, "n_halves": len(halves), "truth": "ctrl_Dfull",
           "visible": "ball present", "baseline": baseline, "anatomy": anat,
           "ceiling_A_synthetic": {"noise": f"block flips, error runs resampled from the ball-argmax error runs "
                                            f"(n={len(ball_runs)}), correct runs geometric; {N_REPS} reps; "
                                            "visible D-truth frames carry the forward-filled truth team",
                                   "levels": ceilA},
           "learned_one_fold": learned, "ceiling_C_merge": ceilC,
           "decision": {"dA": dA, "dC": dC, "rule": "dA >= 1.5 dC -> T1,T2,T3; dC >= dA -> T3 first; both < 0.02 -> E1+A2",
                        "order": order,
                        "dA_real_one_fold_replace_ball": learned["spellF1_delta_replace_ball"] if learned else None,
                        "dA_real_one_fold_best_combo": max(r["delta_ctrl"] for r in learned["decode"].values()) if learned else None,
                        "dC_real_c1_lomo_best": max(r["delta_ctrl"] for r in c1["theta"].values()) if c1 else None},
           "runtime_s": round(time.time() - t0, 1)}
    out_dir = Path(args.out) if args.out else sel_dir
    (out_dir / "levers.json").write_text(json.dumps(out, indent=1, default=float))

    # compact table
    print("\n=== lever diagnostic (seed 100, folds %s) ===" % folds_used)
    print(f"baseline spellF1 ctrl {baseline['ctrl_Dfull']:.4f} raw {baseline['raw_Dfull']:.4f}")
    print(f"(a) vis-alive team err: final {anat['final_team_err_rate']:.3f}  ball argmax {anat['ball_argmax_err_rate']:.3f}"
          f"  pathcrf argmax {anat['pathcrf_argmax_err_rate']:.3f}; final pred D {anat['final_pred_D_rate']:.3f}")
    for k in ("final_err_runs", "ball_err_runs"):
        r = anat[k]
        print(f"    {k}: n {r['n_runs']} runs<1s {r['lt1s']['share_runs']:.2f} 1-3s {r['1to3s']['share_runs']:.2f}"
              f" >3s {r['gt3s']['share_runs']:.2f} | frames<1s {r['lt1s']['share_frames']:.2f} >3s {r['gt3s']['share_frames']:.2f}")
    sm = anat["spell_margins"]["visible_start"]
    print(f"    vis-start spells: false n {sm['false']['n']} vs matched n {sm['matched']['n']}; mean margin "
          + " ".join(f"{k} {sm['false'][k]['mean']:.2f}/{sm['matched'][k]['mean']:.2f}" for k in ("ball", "tag", "pathcrf", "ball+pathcrf"))
          + f"; false confident share {sm['false_confident_share']:.2f}, wrong-sign {sm['false_wrong_sign_share']:.2f}")
    for lbl, r in anat["decomp_by_start"].items():
        print(f"    {lbl}: {r}")
    print("(b) a     realised  ctrl    d_ctrl  raw     d_raw")
    for k, r in ceilA.items():
        print(f"    {k}  {r['realised_vis_acc']:.3f}    {r['ctrl_Dfull']:.4f} {r['delta_ctrl']:+.4f} {r['raw_Dfull']:.4f} {r['delta_raw']:+.4f}")
    if learned:
        print(f"    learned 9001: acc {learned['acc_teamctrl']:.4f} vs ball argmax {learned['acc_ball_argmax']:.4f}"
              f" (pathcrf {learned['acc_pathcrf_argmax']:.4f})")
        for k, r in learned["decode"].items():
            print(f"      {k}: ctrl {r['ctrl_Dfull']:.4f} ({r['delta_ctrl']:+.4f}) raw {r['raw_Dfull']:.4f} ({r['delta_raw']:+.4f})")
        s6 = learned["synthetic_at_model_acc_all_folds"]
        print(f"      synthetic at a={s6['acc']:.4f} (6 folds): d_ctrl {s6['delta_ctrl']:+.4f}; fold 9001 only "
              f"{learned['synthetic_at_model_acc_fold9001']['delta_ctrl']:+.4f}")
    print("(c) merges")
    for k, r in ceilC.items():
        if k == "c1_merge_model_lomo":
            for t, rr in r["theta"].items():
                print(f"    C1 LOMO merge model theta {t}: d_ctrl {rr['delta_ctrl']:+.4f} d_raw {rr['delta_raw']:+.4f} merges {rr['n_merges']}")
            continue
        print(f"    {k}: d_ctrl {r['delta_ctrl']:+.4f} d_raw {r['delta_raw']:+.4f} merges {r['n_merges']}")
    print(f"decision: dA {dA} dC {dC:+.4f} -> {order}   ({out['runtime_s']} s)")


if __name__ == "__main__":
    main()
