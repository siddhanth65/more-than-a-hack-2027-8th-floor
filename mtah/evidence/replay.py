"""Broadcast replay / close-up cue (real match only) and its D-vs-alive likelihood stream (Plan 3r).

Clock mapping: tactical = broadcast + 4014 holds for the FIRST half only; a piecewise offset search (2-min windows,
Hungarian player cost) gives +26920 for the second half (the broadcast cuts ~15 min of half-time). The switch
broadcast frame is a least-cost changepoint between the two offsets.
Per tactical frame (half 1: broadcast = tactical - 4014; half 2: broadcast = tactical - 26920):
  hom_valid   broadcast homography valid at the aligned broadcast frame
  in_sync     broadcast players match tactical players at the half's offset: mean Hungarian distance per team
              (>= 3 players per team) at 5 Hz, centred 2 s rolling median (11 samples), <= SYNC_THR = 5 m;
              undefined counts as not in sync. The unsmoothed per-frame 2.5 m version (in_sync_strict) is kept
              for diagnostics only.
  green       share of pitch-green pixels (HSV) in a 192x108 downsample of the broadcast video, one decoded frame
              every 12 (0.48 s), linearly interpolated; cached next to the cue as green.npz
  replay_like = has_broadcast & (~hom_valid | ~in_sync | green < GREEN_THR)   (GREEN_THR: Otsu, label-free)
  replay_like_s = centred 2 s (51-frame) moving mean of replay_like > 0.5
`replay_cue` returns replay_like_s as float32 (0/1), NaN where the tactical frame has no broadcast frame.
`replay_evidence` turns it into a D log-likelihood column (H, A = 0) so after the decoder's per-frame centring it is a
pure D-vs-alive term. Per-frame arrays live only under runs/ (gitignored)."""
import json
import math
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
FPS = 25; OFFSETS = {1: 4014, 2: 26920}; COST_MAX = 2.5; SYNC_THR = 5.0; SYNC_STEP = 5; PIX_STEP = 12; SMOOTH = 51
LR_ON = math.log(0.64 / 0.13)     # P(replay_like | D) / P(replay_like | H/A), a5c visible-ball frames, label-free
LR_OFF = math.log(0.36 / 0.87)
DEFAULT_CACHE = "runs/broadcast/cue.npz"


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
    from scipy.optimize import linear_sum_assignment
    from scipy.spatial.distance import cdist
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


def compute_cue(data_dir, cache=DEFAULT_CACHE, pixel=True, b=None, t=None):
    """Per-tactical-frame cue arrays (saved to `cache`, a .npz under runs/) and aggregate diagnostics.
    Returns (arrays, info): arrays has has_broadcast, hom_valid, in_sync, in_sync_strict, green, replay_like,
    replay_like_mean2s, replay_like_s, low_green (None without pixel); info has the "sync"/"pixel" aggregates.
    `pixel` needs the broadcast video under <data_dir>/broadcast (or a cached green.npz next to `cache`)."""
    from mtah.io.metrica_loader import load_broadcast, load_tactical
    D = Path(data_dir); cache = _resolve(cache); cache.parent.mkdir(parents=True, exist_ok=True)
    T0 = time.time(); info = {}
    b = b if b is not None else load_broadcast(D)
    t = t if t is not None else load_tactical(D)
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
    info["sync"] = {"offsets": OFFSETS, "switch_broadcast_frame": SW,
                    "out_of_sync_share_of_hom_valid": float(1 - sync_b[hv_b].mean()),
                    "out_of_sync_share_of_hom_valid_cost_defined": float(1 - sync_b[hv_b & np.isfinite(cmed_b)].mean()),
                    "strict_census_out_of_sync_share_of_hom_valid": float(1 - strict_b[hv_b].mean()),
                    "out_of_sync_share_of_hom_valid_by_half": {h: float(1 - sync_b[hv_b & (half_b == h)].mean()) for h in (1, 2)},
                    "single_offset_4014_out_of_sync_share_of_hom_valid": float(1 - (np.nan_to_num(c1, nan=np.inf) <= COST_MAX)[hv_b].mean()),
                    "hom_valid_share_broadcast": float(hv_b.mean())}
    print(f"sync done {time.time() - T0:.0f}s", flush=True)

    green_b = None; thr = None
    if pixel:
        gpath = cache.parent / "green.npz"
        if gpath.exists():
            z = np.load(gpath); gi, gv, nvid, dsec = z["idx"], z["val"], int(z["nvid"]), float(z["dsec"])
        else:
            video = D / "broadcast" / "BAR-ATH_broadcast.mp4"
            if not video.exists():
                raise FileNotFoundError(f"broadcast video missing: {video} (needed for the replay cue)")
            gi, gv, nvid, dsec = green_samples(video)
            np.savez_compressed(gpath, idx=gi, val=gv, nvid=nvid, dsec=dsec)
        green_b = np.interp(np.arange(Fb), gi, gv).astype(np.float32)
        if nvid < Fb:
            green_b[nvid:] = np.nan
        thr = otsu(gv)
        info["pixel"] = {"video_frames": nvid, "tracking_frames": Fb, "decode_s": dsec, "n_samples": int(len(gi)),
                         "green_thr_otsu": thr,
                         "green_q_all": {str(q): float(np.percentile(gv, q)) for q in (5, 25, 50, 75, 95)},
                         "green_q_wide_ref(hom_valid&in_sync)": {str(q): float(np.nanpercentile(green_b[hv_b & sync_b], q)) for q in (1, 5, 25, 50)},
                         "green_q_not_hom_valid": {str(q): float(np.nanpercentile(green_b[~hv_b], q)) for q in (25, 50, 75, 95)},
                         "share_below_thr": float((gv < thr).mean())}
        lags = {}; hvf = hv_b.astype(float)   # alignment sanity: corr(green, hom_valid) over lags
        for lag in (-250, -100, -50, -25, -12, 0, 12, 25, 50, 100, 250):
            a = green_b[max(0, lag):Fb + min(0, lag)]; c = hvf[max(0, -lag):Fb - max(0, lag)]
            m = np.isfinite(a)
            lags[str(lag)] = float(np.corrcoef(a[m], c[m])[0, 1])
        info["pixel"]["corr_green_homvalid_by_lag_frames"] = lags
        print(f"pixel done {time.time() - T0:.0f}s", flush=True)

    # map to the tactical timeline
    tf_all = np.arange(Ft)
    half_t = np.where(t.half > 0, t.half, np.where(tf_all < 100000, 1, 2))
    bf = tf_all - np.where(half_t == 1, OFFSETS[1], OFFSETS[2])
    has_b = (bf >= 0) & (bf < Fb) & np.where(half_t == 1, bf < SW, bf >= SW)
    bfc = np.clip(bf, 0, Fb - 1)
    hom_valid = has_b & hv_b[bfc]
    in_sync = has_b & sync_b[bfc]; in_sync_strict = has_b & strict_b[bfc]
    rl = has_b & (~hom_valid | ~in_sync)
    green = np.full(Ft, np.nan, np.float32); lowg = None
    if green_b is not None:
        green[has_b] = green_b[bfc[has_b]]
        lowg = has_b & (np.nan_to_num(green, nan=1.0) < thr)
        rl = rl | lowg
    k = np.ones(SMOOTH) / SMOOTH
    rl_mean = np.convolve(rl.astype(float), k, mode="same").astype(np.float32)
    rl_s = has_b & (rl_mean > 0.5)
    np.savez_compressed(cache, has_broadcast=has_b, hom_valid=hom_valid, in_sync=in_sync, green=green,
                        replay_like=rl, replay_like_mean2s=rl_mean, replay_like_s=rl_s,
                        sync_cost=np.where(has_b, cost_b[bfc], np.nan).astype(np.float32),
                        sync_cost_med2s=np.where(has_b, cmed_b[bfc], np.nan).astype(np.float32),
                        meta=json.dumps({"offsets": OFFSETS, "switch": SW, "cost_max": COST_MAX, "green_thr": thr, "smooth": SMOOTH}))
    arrays = {"has_broadcast": has_b, "hom_valid": hom_valid, "in_sync": in_sync, "in_sync_strict": in_sync_strict,
              "green": green, "replay_like": rl, "replay_like_mean2s": rl_mean, "replay_like_s": rl_s, "low_green": lowg}
    return arrays, info


def _resolve(p):
    p = Path(p)
    return p if p.is_absolute() else ROOT / p


def cue_from_arrays(has_broadcast, replay_like_s):
    """(F,) float32: the smoothed cue as 0/1, NaN where there is no broadcast frame."""
    out = np.asarray(replay_like_s, np.float32).copy()
    out[~np.asarray(has_broadcast, bool)] = np.nan
    return out


def replay_cue(data_dir, n_frames, cache=DEFAULT_CACHE):
    """(n_frames,) float32 in [0, 1] (NaN = no broadcast frame), or None when there is no broadcast data under
    `data_dir`. Loads `cache` (relative paths resolve against the repo root); computes and writes it if missing."""
    cp = _resolve(cache)
    if cp.exists():
        z = np.load(cp)
        cue = cue_from_arrays(z["has_broadcast"], z["replay_like_s"])
    else:
        bdir = Path(data_dir) / "broadcast" if data_dir is not None else None
        if bdir is None or not bdir.exists():
            return None
        a, _ = compute_cue(data_dir, cache=cp)
        cue = cue_from_arrays(a["has_broadcast"], a["replay_like_s"])
    if len(cue) != n_frames:
        raise ValueError(f"replay cue has {len(cue)} frames, expected {n_frames}")
    return cue


def replay_evidence(cue, w, lr_on=LR_ON, lr_off=LR_OFF):
    """(F, 3) float32: D column = w * (cue * lr_on + (1 - cue) * lr_off), H and A = 0; NaN rows where cue is NaN."""
    cue = np.asarray(cue, np.float64)
    ev = np.zeros((len(cue), 3), np.float64)
    ev[:, 2] = w * (cue * lr_on + (1.0 - cue) * lr_off)
    ev[~np.isfinite(cue)] = np.nan
    return ev.astype(np.float32)
