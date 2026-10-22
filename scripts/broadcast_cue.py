"""Exploratory (Plan 3r T1): broadcast replay / close-up cue for dead time, label-free validation.

Clock mapping: experiment_broadcast_align.py found tactical = broadcast + 4014, but that holds for the FIRST half only;
a piecewise offset search (2-min windows, Hungarian player cost) gives +26920 for the second half (the broadcast
cuts ~15 min of half-time). The switch broadcast frame is a least-cost changepoint between the two offsets.
Per tactical frame (half 1: broadcast = tactical - 4014; half 2: broadcast = tactical - 26920):
  hom_valid   broadcast homography valid at the aligned broadcast frame
  in_sync     broadcast players match tactical players at the half's offset: the census refine cost (mean Hungarian
              distance per team, >= 3 players per team) at 5 Hz, centred 2 s rolling median (11 samples), <= SYNC_THR
              = 5 m (in-sync mode ~0.5 m; a wrong clock gives 6-10 m in the offset search); undefined counts as not in
              sync. The census per-run threshold (2.5 m, unsmoothed per frame) is kept as the secondary cue
              out_of_sync_strict: it fires on ~25 % of live build-up frames (tracking noise), so it is not used.
  green       share of pitch-green pixels (HSV) in a 192x108 downsample of the broadcast video, one decoded
              frame every 12 (0.48 s), linearly interpolated; cached in runs/broadcast/green.npz
  replay_like = has_broadcast & (~hom_valid | ~in_sync | green < GREEN_THR)
  replay_like_s = centred 2 s (51-frame) moving mean of replay_like > 0.5
GREEN_THR is label-free: Otsu threshold of the sampled green-share histogram (documented in the json).
Writes the per-frame cue to runs/broadcast/cue.npz (gitignored) and aggregates only to
runs/broadcast/cue_validation.json. Never prints frame-level data."""
import argparse, json, os, time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cdist

from mtah.eval.drift import blackout_mask_from_match
from mtah.io.metrica_loader import load_broadcast, load_tactical
from mtah.io.tags import load_tags
from mtah.sim.blackout_sim import runs_of

D = Path(os.environ.get("MTAH_DATA_DIR", r"HOME/mtah-data"))
OUT = Path(__file__).resolve().parents[1] / "runs" / "broadcast"
GT = Path(r"WORKDIR/repo\evaluation\gt\BAR-ATH_possession_gt_public.csv")
CAND = Path(__file__).resolve().parents[1] / "runs" / "candidates" / "a5c.csv"
FPS = 25; OFFSETS = {1: 4014, 2: 26920}; COST_MAX = 2.5; SYNC_THR = 5.0; SYNC_STEP = 5; PIX_STEP = 12; SMOOTH = 51
PUB = (11185, 18685)
SP_TYPES = {"goal_kick": "Goal Kick", "corner": "Corner Kick", "throw_in": "Throw-in", "free_kick": "Free Kick",
            "kick_off": "Kick Off"}


def green_samples(video, step=PIX_STEP):
    import cv2
    cv2.setNumThreads(4)
    cap = cv2.VideoCapture(str(video))
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    idx, val = [], []
    t0 = time.time(); i = 0
    while True:
        if not cap.grab():
            break
        if i % step == 0:
            ok, f = cap.retrieve()
            if ok:
                hsv = cv2.cvtColor(cv2.resize(f, (192, 108), interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2HSV)
                g = (hsv[..., 0] >= 30) & (hsv[..., 0] <= 90) & (hsv[..., 1] >= 50) & (hsv[..., 2] >= 40)
                idx.append(i); val.append(float(g.mean()))
        if i % 25000 == 0:
            print(f"decode {i}/{n} {time.time() - t0:.0f}s", flush=True)
        i += 1
    return np.array(idx), np.array(val, np.float32), i, time.time() - t0


def otsu(x, bins=100):
    h, e = np.histogram(x, bins=bins, range=(0, 1)); c = (e[:-1] + e[1:]) / 2
    w0 = np.cumsum(h); w1 = w0[-1] - w0
    m0 = np.cumsum(h * c) / np.maximum(w0, 1); m1 = (np.sum(h * c) - np.cumsum(h * c)) / np.maximum(w1, 1)
    return float(e[1:][np.argmax(w0 * w1 * (m0 - m1) ** 2)])


def sync_cost(b, t, offset):
    tb = ~np.isnan(b.pos[..., 0]); tt = ~np.isnan(t.pos[..., 0])
    Fb, Ft = b.n_frames, t.n_frames
    cost = np.full(Fb, np.nan, np.float32)
    samp = np.arange(0, Fb, SYNC_STEP)
    for bf in samp:
        tf = bf + offset
        if tf >= Ft or tf < 0:
            continue
        cs = []
        for tm in (0, 1):
            bi = np.flatnonzero(tb[bf] & (b.team == tm)); ti = np.flatnonzero(tt[tf] & (t.team == tm))
            if len(bi) < 3 or len(ti) < 3:
                continue
            C = cdist(b.pos[bf, bi], t.pos[tf, ti]); r, c = linear_sum_assignment(C)
            cs.append(C[r, c].mean())
        if cs:
            cost[bf] = np.mean(cs)
    hold = np.minimum(np.arange(Fb) // SYNC_STEP * SYNC_STEP, Fb - 1)
    med = pd.Series(cost[samp]).rolling(11, center=True, min_periods=3).median().to_numpy()
    cmed = np.full(Fb, np.nan, np.float32); cmed[samp] = med
    return cost[hold], cmed[hold]


def share(x, m):
    m = m & np.isfinite(x) if x.dtype.kind == "f" else m
    return {"share": float(x[m].mean()) if m.any() else None, "n": int(m.sum())}


def runlen_q(mask):
    L = np.array([(e - s) / FPS for s, e in runs_of(mask)])
    if not len(L):
        return {"n": 0}
    return {"n": int(len(L)), "median_s": float(np.median(L)), "p90_s": float(np.percentile(L, 90)),
            "p10_s": float(np.percentile(L, 10)), "max_s": float(L.max())}


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--no-pixel", action="store_true")
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    T0 = time.time(); res = {}
    b = load_broadcast(D); t = load_tactical(D); tags = load_tags(D)
    Fb, Ft = b.n_frames, t.n_frames
    hcsv = pd.read_csv(D / "broadcast" / "BAR-ATH_broadcast_homography.csv", usecols=["frame", "valid"])
    hv_b = np.zeros(Fb, bool); hv_b[:min(Fb, len(hcsv))] = (hcsv.valid.to_numpy() == 1)[:Fb]
    (c1, m1), (c2, m2) = sync_cost(b, t, OFFSETS[1]), sync_cost(b, t, OFFSETS[2])
    # changepoint: broadcast frames < SW use the half-1 offset, >= SW the half-2 offset
    k1 = np.nan_to_num(np.minimum(c1, 10.0), nan=0.0)[::SYNC_STEP]; k2 = np.nan_to_num(np.minimum(c2, 10.0), nan=0.0)[::SYNC_STEP]
    tot = np.concatenate([[0], np.cumsum(k1)]) + (k2.sum() - np.concatenate([[0], np.cumsum(k2)]))
    SW = int(np.argmin(tot)) * SYNC_STEP
    half_b = np.where(np.arange(Fb) < SW, 1, 2)
    cost_b = np.where(half_b == 1, c1, c2); cmed_b = np.where(half_b == 1, m1, m2)
    strict_b = np.nan_to_num(cost_b, nan=np.inf) <= COST_MAX
    sync_b = np.nan_to_num(cmed_b, nan=np.inf) <= SYNC_THR
    res["sync"] = {"offsets": OFFSETS, "switch_broadcast_frame": SW,
                   "out_of_sync_share_of_hom_valid": float(1 - sync_b[hv_b].mean()),
                   "out_of_sync_share_of_hom_valid_cost_defined": float(1 - sync_b[hv_b & np.isfinite(cmed_b)].mean()),
                   "strict_census_out_of_sync_share_of_hom_valid": float(1 - strict_b[hv_b].mean()),
                   "out_of_sync_share_of_hom_valid_by_half": {h: float(1 - sync_b[hv_b & (half_b == h)].mean()) for h in (1, 2)},
                   "single_offset_4014_out_of_sync_share_of_hom_valid": float(1 - (np.nan_to_num(c1, nan=np.inf) <= COST_MAX)[hv_b].mean()),
                   "hom_valid_share_broadcast": float(hv_b.mean())}
    print(f"sync done {time.time() - T0:.0f}s", flush=True)

    # pixel proxy
    green_b = None
    gpath = OUT / "green.npz"
    if not args.no_pixel:
        if gpath.exists():
            z = np.load(gpath); gi, gv, nvid, dsec = z["idx"], z["val"], int(z["nvid"]), float(z["dsec"])
        else:
            gi, gv, nvid, dsec = green_samples(D / "broadcast" / "BAR-ATH_broadcast.mp4")
            np.savez_compressed(gpath, idx=gi, val=gv, nvid=nvid, dsec=dsec)
        green_b = np.interp(np.arange(Fb), gi, gv).astype(np.float32)
        if nvid < Fb:
            green_b[nvid:] = np.nan
        thr = otsu(gv)
        res["pixel"] = {"video_frames": nvid, "tracking_frames": Fb, "decode_s": dsec, "n_samples": int(len(gi)),
                        "green_thr_otsu": thr,
                        "green_q_all": {str(q): float(np.percentile(gv, q)) for q in (5, 25, 50, 75, 95)},
                        "green_q_wide_ref(hom_valid&in_sync)": {str(q): float(np.nanpercentile(green_b[hv_b & sync_b], q)) for q in (1, 5, 25, 50)},
                        "green_q_not_hom_valid": {str(q): float(np.nanpercentile(green_b[~hv_b], q)) for q in (25, 50, 75, 95)},
                        "share_below_thr": float((gv < thr).mean())}
        # alignment sanity: corr(green, hom_valid) over lags (video vs tracking index)
        lags = {}
        hvf = hv_b.astype(float)
        ok = np.isfinite(green_b)
        for lag in (-250, -100, -50, -25, -12, 0, 12, 25, 50, 100, 250):
            a = green_b[max(0, lag):Fb + min(0, lag)]; c = hvf[max(0, -lag):Fb - max(0, lag)]
            m = np.isfinite(a)
            lags[str(lag)] = float(np.corrcoef(a[m], c[m])[0, 1])
        res["pixel"]["corr_green_homvalid_by_lag_frames"] = lags
        print(f"pixel done {time.time() - T0:.0f}s", flush=True)
    else:
        thr = None

    # map to tactical timeline
    tf_all = np.arange(Ft)
    half_t = np.where(t.half > 0, t.half, np.where(tf_all < 100000, 1, 2))
    bf = tf_all - np.where(half_t == 1, OFFSETS[1], OFFSETS[2])
    has_b = (bf >= 0) & (bf < Fb) & np.where(half_t == 1, bf < SW, bf >= SW)
    bfc = np.clip(bf, 0, Fb - 1)
    hom_valid = has_b & hv_b[bfc]
    in_sync = has_b & sync_b[bfc]; in_sync_strict = has_b & strict_b[bfc]
    rl = has_b & (~hom_valid | ~in_sync)
    green = np.full(Ft, np.nan, np.float32)
    if green_b is not None:
        green[has_b] = green_b[bfc[has_b]]
        lowg = has_b & (np.nan_to_num(green, nan=1.0) < thr)
        rl = rl | lowg
    k = np.ones(SMOOTH) / SMOOTH
    rl_mean = np.convolve(rl.astype(float), k, mode="same").astype(np.float32)
    rl_s = has_b & (rl_mean > 0.5)
    np.savez_compressed(OUT / "cue.npz", has_broadcast=has_b, hom_valid=hom_valid, in_sync=in_sync, green=green,
                        replay_like=rl, replay_like_mean2s=rl_mean, replay_like_s=rl_s,
                        sync_cost=np.where(has_b, cost_b[bfc], np.nan).astype(np.float32),
                        sync_cost_med2s=np.where(has_b, cmed_b[bfc], np.nan).astype(np.float32),
                        meta=json.dumps({"offsets": OFFSETS, "switch": SW, "cost_max": COST_MAX, "green_thr": thr, "smooth": SMOOTH}))

    # ---- validation (aggregates only)
    g = t.graded; blk = blackout_mask_from_match(t); vis = t.ball_present
    U = g & has_b
    res["coverage"] = {"graded_with_broadcast_share": float(U.sum() / g.sum()), "n_frames_U": int(U.sum())}
    cues = {"raw": rl, "smooth": rl_s, "hom_invalid": has_b & ~hom_valid, "out_of_sync": has_b & ~in_sync,
            "out_of_sync_strict": has_b & ~in_sync_strict}
    if green_b is not None:
        cues["low_green"] = lowg
    res["overall_share_U"] = {c: float(x[U].mean()) for c, x in cues.items()}
    # (a)
    res["a_blackout"] = {c: {"in_blackout": float(x[U & blk].mean()), "outside_blackout": float(x[U & ~blk].mean()),
                             "visible_ball": float(x[U & vis].mean())} for c, x in cues.items()}
    # (b) 30 s before SET PIECES starts
    sp = tags[tags.code == "SET PIECES"]
    pre_all = np.zeros(Ft, bool); rb = {}
    for key, pref in SP_TYPES.items():
        m = np.zeros(Ft, bool); n = 0
        for s in sp[sp["type"].fillna("").str.startswith(pref)]["start"]:
            a0 = max(0, s - 30 * FPS); m[a0:max(a0, s)] = True; n += 1
        pre_all |= m
        rb[key] = {"n_tags": n, **{c: float(x[U & m].mean()) if (U & m).any() else None for c, x in cues.items()},
                   "n_frames": int((U & m).sum())}
    rb["all_setpieces"] = {c: float(x[U & pre_all].mean()) for c, x in cues.items()}
    rb["elsewhere"] = {c: float(x[U & ~pre_all].mean()) for c, x in cues.items()}
    rb["elsewhere_visible_ball"] = {c: float(x[U & ~pre_all & vis].mean()) for c, x in cues.items()}
    res["b_pre_setpiece_30s"] = rb
    # (c) BUILD UP / PROGRESSION mid-tag (central 50 % of each tag)
    rc = {}
    for code in ("BUILD UP", "PROGRESSION"):
        m = np.zeros(Ft, bool)
        for r in tags[tags.code == code].itertuples(index=False):
            q = (r.end - r.start) // 4; m[r.start + q:r.end - q] = True
        rc[code] = {"n_frames": int((U & m).sum()), **{c: float(x[U & m].mean()) for c, x in cues.items()}}
    res["c_live_midtag"] = rc
    # (d) a5c D vs H/A on visible-ball frames
    a5 = pd.read_csv(CAND)["label"].to_numpy().astype("<U1")
    Vd = U & vis
    res["d_a5c_visible"] = {c: {"P_rl_given_D": float(x[Vd & (a5 == "D")].mean()),
                                "P_rl_given_HA": float(x[Vd & (a5 != "D")].mean()),
                                "n_D": int((Vd & (a5 == "D")).sum()), "n_HA": int((Vd & (a5 != "D")).sum())}
                            for c, x in cues.items()}
    res["d_a5c_visible_lift_raw_smooth"] = {c: res["d_a5c_visible"][c]["P_rl_given_D"] / max(1e-9, res["d_a5c_visible"][c]["P_rl_given_HA"]) for c in ("raw", "smooth")}
    # (e) run lengths
    res["e_runlengths"] = {"raw": runlen_q(rl & g), "smooth": runlen_q(rl_s & g)}
    # (f) public window, aggregates only, primary cue = smooth
    gt = pd.read_csv(GT)["label"].to_numpy().astype("<U1")
    W = np.zeros(Ft, bool); W[PUB[0]:PUB[1]] = True; W &= has_b
    res["f_public_smooth"] = {"truthD_visible_share_rl": float(rl_s[W & (gt == "D") & vis].mean()),
                              "truthHA_share_rl": float(rl_s[W & np.isin(gt, ["H", "A"])].mean())}
    res["runtime_s"] = time.time() - T0
    (OUT / "cue_validation.json").write_text(json.dumps(res, indent=1))
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
