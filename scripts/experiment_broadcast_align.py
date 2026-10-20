"""Exploratory: align broadcast tracking to tactical timeline; census of blackout frames fillable by broadcast ball.
Prints/writes aggregates only."""
import json, time
from pathlib import Path
import numpy as np, pandas as pd
from scipy.signal import fftconvolve
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import pdist, cdist
from mtah.io.metrica_loader import load_tactical, load_broadcast
from mtah.eval.drift import blackout_mask_from_match

D = r"HOME/mtah-data"
OUT = Path(__file__).resolve().parents[1] / "runs" / "broadcast" / "census.json"
FPS = 25; DS = 5  # 5 Hz
H_THR = 0.05; MIN_PL = 8; MIN_RUN_S = 5; NCC_MIN = 0.6; NCC_RATIO = 1.2; COST_MAX = 2.5
NCC_CAP_S = 60

b = load_broadcast(D); t = load_tactical(D)
Fb, Ft = b.n_frames, t.n_frames
graded = t.graded
blackout = blackout_mask_from_match(t)
tb = ~np.isnan(b.pos[..., 0]); tt = ~np.isnan(t.pos[..., 0])
cnt = tb.sum(1)

# ---- 1. segmentation
h = pd.read_csv(Path(D) / "broadcast" / "BAR-ATH_broadcast_homography.csv")
H = h[[f"h{i}{j}" for i in range(3) for j in range(3)]].to_numpy().reshape(-1, 3, 3).astype(float)
with np.errstate(all="ignore"):
    H = H / H[:, 2:3, 2:3]
hv = (h.valid.to_numpy() == 1)[:Fb]
ok = hv & (cnt >= MIN_PL)
dH = np.full(Fb, np.inf)
with np.errstate(all="ignore"):
    dH[1:] = np.linalg.norm((H[1:Fb] - H[:Fb - 1]).reshape(-1, 9), axis=1) / np.linalg.norm(H[:Fb - 1].reshape(-1, 9), axis=1)
dH[~np.isfinite(dH)] = np.inf
split = np.zeros(Fb, bool)
split[1:] = (np.abs(np.diff(cnt)) >= 3) | ((tb[1:] != tb[:-1]).sum(1) >= 3) | (dH[1:] > H_THR)
both = tb[1:] & tb[:-1]
step = np.linalg.norm(b.pos[1:] - b.pos[:-1], axis=-1)
with np.errstate(all="ignore"):
    mstep = np.nanmedian(np.where(both, step, np.nan), axis=1)
split[1:] |= np.nan_to_num(mstep, nan=0) > 1.5
runs = []; i = 0
while i < Fb:
    if not ok[i]:
        i += 1; continue
    j = i + 1
    while j < Fb and ok[j] and not split[j]: j += 1
    runs.append((i, j)); i = j
lens = np.array([(e - s) / FPS for s, e in runs])
print(f"runs {len(runs)}  >=5s: {(lens >= MIN_RUN_S).sum()}  total share {sum(e - s for s, e in runs) / Fb:.3f}", flush=True)
len_q = {str(k): float(np.percentile(lens, k)) for k in (10, 25, 50, 75, 90, 99)}


# ---- 2. signals
def signals(pos, pres, team):
    n = pos.shape[0]; out = np.full((n, 10), np.nan, np.float32)
    for tm in (0, 1):
        sel = np.flatnonzero(team == tm); P = pos[:, sel]; M = pres[:, sel]
        c = M.sum(1)
        with np.errstate(all="ignore"):
            out[:, tm * 5] = np.nanmean(P[..., 0], axis=1)
            out[:, tm * 5 + 1] = np.nanmean(P[..., 1], axis=1)
        for f in np.flatnonzero(c >= 3):
            d = pdist(P[f][M[f]])
            out[f, tm * 5 + 2:tm * 5 + 5] = np.percentile(d, [25, 50, 75])
    return out


def ds5(x):
    n = x.shape[0] // DS
    with np.errstate(all="ignore"):
        return np.nanmean(x[:n * DS].reshape(n, DS, -1), axis=1)


t0 = time.time()
sig_t = ds5(signals(t.pos, tt, t.team)); gt5 = ds5(graded[:, None].astype(np.float32))[:, 0] > 0.99
fin_t = np.isfinite(sig_t).all(1) & gt5
mu_t = np.nanmean(sig_t[fin_t], 0); sd_t = np.nanstd(sig_t[fin_t], 0)
sig_b_full = signals(b.pos, tb, b.team)
print(f"signals {time.time() - t0:.0f}s", flush=True)
Tz = np.where(fin_t[:, None], (sig_t - mu_t) / sd_t, 0.0)
V = fin_t.astype(float)
cs = lambda x: np.concatenate([np.zeros((1,) + x.shape[1:]), np.cumsum(x, 0)])
cT = cs(Tz); cT2 = cs(Tz ** 2); cV = cs(V)


def ncc(sig_run):
    n = len(sig_run); ch = sig_run.shape[1]
    nl = len(Tz) - n + 1
    if nl <= 0: return None
    acc_ = np.zeros(nl); used = 0
    okc = np.isfinite(sig_run).all(0)
    wv = (cV[n:n + nl, 0] - cV[:nl, 0]) >= n if cV.ndim > 1 else None
    wv = (cV[n:n + nl] - cV[:nl]) >= n
    for c in range(ch):
        if not okc[c]: continue
        x = sig_run[:, c].astype(float); sd = x.std()
        if sd < 1e-6: continue
        x = (x - x.mean()) / sd
        num = fftconvolve(Tz[:, c], x[::-1], mode="valid")
        s1 = cT[n:n + nl, c] - cT[:nl, c]; s2 = cT2[n:n + nl, c] - cT2[:nl, c]
        var = s2 / n - (s1 / n) ** 2
        acc_ += num / n / np.sqrt(np.maximum(var, 1e-9)); used += 1
    if used < 6: return None
    r = acc_ / used; r[~wv] = -np.inf
    return r


def run_signal(s, e):
    s5 = s // DS * DS
    seg = sig_b_full[s5:min(e, s5 + NCC_CAP_S * FPS)]
    return s5, ds5(seg)


# ---- refine cost
def frame_cost(bf, tf):
    cs_ = []
    for tm in (0, 1):
        bi = np.flatnonzero(tb[bf] & (b.team == tm)); ti = np.flatnonzero(tt[tf] & (t.team == tm))
        if len(bi) < 3 or len(ti) < 3: continue
        C = cdist(b.pos[bf, bi], t.pos[tf, ti]); r, c = linear_sum_assignment(C)
        cs_.append(C[r, c].mean())
    return np.mean(cs_) if cs_ else np.nan


def refine(s, e, off0):
    fr = np.arange(s, min(e, s + 40 * FPS), 10)

    def cost(off):
        tf = fr + off; m = (tf >= 0) & (tf < Ft)
        v = [frame_cost(f, g) for f, g in zip(fr[m], tf[m]) if graded[g]]
        v = [x for x in v if np.isfinite(x)]
        return np.mean(v) if len(v) >= 5 else np.inf
    cands = {o: cost(o) for o in range(off0 - 25, off0 + 26, 5)}
    ob = min(cands, key=cands.get)
    for o in range(ob - 4, ob + 5):
        if o not in cands: cands[o] = cost(o)
    ob = min(cands, key=cands.get)
    return ob, cands[ob]


# ---- 3. align
acc = []; nrun_eval = 0; nrej_ncc = 0; nrej_cost = 0
cand_runs = [(s, e) for s, e in runs if (e - s) / FPS >= MIN_RUN_S]
for k, (s, e) in enumerate(cand_runs):
    if k % 50 == 0: print(f"progress {k}/{len(cand_runs)} accepted {len(acc)} {time.time() - t0:.0f}s", flush=True)
    s5, sg = run_signal(s, e)
    if len(sg) < 20: continue
    r = ncc(sg)
    if r is None: continue
    nrun_eval += 1
    bi_ = int(np.argmax(r)); best = r[bi_]
    r2 = r.copy(); r2[max(0, bi_ - 5):bi_ + 6] = -np.inf; second = r2.max()
    if not (best >= NCC_MIN and (second <= 0 or best >= NCC_RATIO * second)):
        nrej_ncc += 1; continue
    off0 = bi_ * DS - s5
    ob, cst = refine(s, e, off0)
    if not (cst <= COST_MAX):
        nrej_cost += 1; continue
    acc.append(dict(s=s, e=e, off=int(ob), ncc=float(best), second=float(second), cost=float(cst)))
print(f"align done {time.time() - t0:.0f}s  evaluated {nrun_eval} accepted {len(acc)}", flush=True)

# ---- 3b. global-offset hypothesis: reuse the median accepted offset for every >=5 s run, accept by refine cost only
GLOBAL = None; acc_ncc = list(acc)
if acc:
    GLOBAL = int(np.median([a["off"] for a in acc]))
    have = {a["s"] for a in acc}
    for s, e in cand_runs:
        if s in have: continue
        ob, cst = refine(s, e, GLOBAL)
        if cst <= COST_MAX and abs(ob - GLOBAL) <= 25:
            acc.append(dict(s=s, e=e, off=int(ob), ncc=float("nan"), second=float("nan"), cost=float(cst)))
    acc.sort(key=lambda a: a["s"])
    print(f"global-offset pass {time.time() - t0:.0f}s accepted {len(acc)} (ncc-accepted {len(acc_ncc)})", flush=True)

# ---- 4. census
mapped_b = np.zeros(Fb, bool); cover_t = np.zeros(Ft, bool); ball_ok_t = np.zeros(Ft, bool)
L, W = b.pitch
for a in acc:
    fb = np.arange(a["s"], a["e"]); tf = fb + a["off"]; m = (tf >= 0) & (tf < Ft)
    fb, tf = fb[m], tf[m]
    mapped_b[fb] = True; cover_t[tf] = True
    bp = b.ball[fb]; pres = ~np.isnan(bp[:, 0])
    inside = pres & (np.abs(bp[:, 0]) <= L / 2 + 3) & (np.abs(bp[:, 1]) <= W / 2 + 3)
    near = np.zeros(len(fb), bool)
    for q in np.flatnonzero(inside):
        pp = b.pos[fb[q]][tb[fb[q]]]
        near[q] = pp.shape[0] > 0 and np.min(np.linalg.norm(pp - bp[q], axis=1)) <= 4
    ball_ok_t[tf[near]] = True
G = graded; B = blackout & G
res = {
    "n_runs": len(runs), "n_runs_ge5s": len(cand_runs), "run_len_s_percentiles": len_q,
    "runs_evaluated": nrun_eval, "rejected_ncc": nrej_ncc, "rejected_cost": nrej_cost, "n_accepted_runs": len(acc), "n_ncc_accepted_runs": len(acc_ncc), "global_offset_frames": GLOBAL,
    "a_broadcast_frames_aligned_share": float(mapped_b.sum() / Fb),
    "b_tactical_graded_covered_share": float((cover_t & G).sum() / G.sum()),
    "c_tactical_blackout_covered_share": float((cover_t & B).sum() / B.sum()),
    "d_blackout_covered_with_valid_broadcast_ball_share": float((ball_ok_t & B).sum() / B.sum()),
    "n_blackout_frames": int(B.sum()),
}
if acc:
    offs = np.array([a["off"] for a in acc]) / FPS; cst = np.array([a["cost"] for a in acc])
    res["e_offset_s_percentiles"] = {str(k): float(np.percentile(offs, k)) for k in (0, 10, 25, 50, 75, 90, 100)}
    res["e_cost_m_percentiles"] = {str(k): float(np.percentile(cst, k)) for k in (0, 25, 50, 75, 100)}
    res["e_ncc_median"] = float(np.median([a["ncc"] for a in acc_ncc]))
    tstart = np.array([a["s"] + a["off"] for a in acc])
    res["f_share_negative_offset"] = float((offs < 0).mean())
    res["f_share_out_of_order"] = float((np.diff(tstart) < 0).mean()) if len(acc) > 1 else 0.0
    ko = {}
    for s, e in cand_runs:
        if np.median(cnt[s:e]) >= 12:
            s5, sg = run_signal(s, e); r = ncc(sg)
            if r is not None:
                i1 = int(np.argmax(r)); ob, cst1 = refine(s, e, i1 * DS - s5)
                ko = {"earliest_run_len_s": (e - s) / FPS, "earliest_run_broadcast_start_frame": int(s),
                      "maps_to_tactical_frame": int(s + ob), "refined_cost_m": float(cst1), "ncc": float(r[i1]),
                      "tactical_kickoffs": [11185, 113742]}
            break
    res["kickoff_check"] = ko
    a0 = acc[0]
    res["earliest_accepted_run"] = {"broadcast_start": a0["s"], "maps_to_tactical_frame": a0["s"] + a0["off"]}
OUT.write_text(json.dumps(res, indent=1))
print(json.dumps(res, indent=1))
