"""exp-seq: 5 fps per-step features from a stage-1 half cache (no torch, numpy only).

Base features per 0.2 s step (block of 5 frames at 25 fps; mean for continuous, max for indicators):
  per stream k in (ball, tag, dead, pathcrf, teamctrl): centred H-A margin, centred D margin, missing flag
  (centring as decode.viterbi.cost_matrix; NaN -> 0);
  ball present, blackout (label-free, eval.drift.blackout_mask_from_match), ball speed, |x|, |y| (normalised);
  touch H / A / contested, strong touch H / A, time since last / to next touch, team of last / next touch;
  PathCRF p_home - 0.5, imputed_frac, missing.
`mirror` swaps the teams (H<->A): every column listed in SIGNED flips sign, the H/A column pairs swap.
"""
import numpy as np

STREAMS = ("ball", "tag", "dead", "pathcrf", "teamctrl")
STEP = 5   # frames per step at 25 fps -> 5 fps


def _centred(e):
    e = e.astype(np.float64)
    fin = np.isfinite(e); n = fin.sum(1)
    mean = np.where(fin, e, 0.0).sum(1) / np.maximum(n, 1)
    c = np.where(fin, e - mean[:, None], 0.0)
    return c[:, 0] - c[:, 1], c[:, 2] - 0.5 * (c[:, 0] + c[:, 1]), (n == 0).astype(np.float64)


def _since(ev_mask, fps, cap=30.0):
    """Seconds since the last True (cap where none) and the index of that event (-1 none)."""
    F = len(ev_mask); idx = np.where(ev_mask, np.arange(F), -1)
    last = np.maximum.accumulate(idx)
    dt = np.where(last >= 0, (np.arange(F) - last) / fps, cap)
    return np.minimum(dt, cap), last


def frame_features(z):
    fps = float(z["fps"]); F = len(z["half"])
    cols, names = [], []
    for k in STREAMS:
        ha, dm, miss = _centred(z[f"ev_{k}"])
        cols += [ha, dm, miss]; names += [f"{k}_ha", f"{k}_d", f"{k}_miss"]
    bp = z["ball_present"].astype(np.float64)
    ball = z["ball"].astype(np.float64); pitch = z["pitch"]
    bx = np.nan_to_num(np.abs(ball[:, 0]) / (pitch[0] / 2), nan=0.0)
    by = np.nan_to_num(np.abs(ball[:, 1]) / (pitch[1] / 2), nan=0.0)
    spd = np.clip(np.nan_to_num(z["ball_speed"].astype(np.float64), nan=0.0), 0, 40) / 10.0
    cols += [bp, z["blackout_inf"].astype(np.float64), spd, bx, by]
    names += ["ball_present", "blackout", "ball_speed", "ball_absx", "ball_absy"]
    t = z["touches"]
    tH = np.zeros(F); tA = np.zeros(F); tC = np.zeros(F)
    if len(t):
        fr = t[:, 0].astype(int); team = t[:, 2]
        tH[fr[team == 0]] = 1; tA[fr[team == 1]] = 1; tC[fr[team == -1]] = 1
    anyt = (tH + tA + tC) > 0
    tsign = tH - tA                              # +1 H, -1 A, 0 contested
    since, last = _since(anyt, fps)
    rsince, rlast = _since(anyt[::-1], fps)
    until = rsince[::-1]; nxt = np.where(rlast[::-1] >= 0, F - 1 - rlast[::-1], -1)
    last_team = np.where(last >= 0, tsign[np.maximum(last, 0)], 0.0)
    next_team = np.where(nxt >= 0, tsign[np.maximum(nxt, 0)], 0.0)
    cols += [tH, tA, tC, z["sH"].astype(np.float64), z["sA"].astype(np.float64), np.log1p(since), np.log1p(until),
             last_team, next_team]
    names += ["touch_H", "touch_A", "touch_C", "strong_H", "strong_A", "log_since_touch", "log_until_touch",
              "last_touch_team", "next_touch_team"]
    ph = z["p_home"].astype(np.float64); pm = ~np.isfinite(ph)
    cols += [np.where(pm, 0.0, ph - 0.5), np.nan_to_num(z["imputed_frac"].astype(np.float64), nan=1.0), pm.astype(float)]
    names += ["pcrf_phome", "pcrf_imputed", "pcrf_miss"]
    return np.column_stack(cols).astype(np.float32), names


MAXCOLS = {"touch_H", "touch_A", "touch_C", "strong_H", "strong_A"}
SIGNED = {f"{k}_ha" for k in STREAMS} | {"last_touch_team", "next_touch_team", "pcrf_phome"}
SWAP = [("touch_H", "touch_A"), ("strong_H", "strong_A")]


def to_steps(X, names, half):
    """(T, n) per-step features over the graded frames of each half, plus the frame index of each step start and
    a (T,) segment id (half id) so windows never cross halves. Steps are 5-frame blocks inside each half run."""
    from mtah.io.match import half_runs
    out, starts, seg = [], [], []
    mx = np.array([n in MAXCOLS for n in names])
    for h, s, e in half_runs(half):
        n = (e - s) // STEP * STEP
        if n <= 0:
            continue
        B = X[s:s + n].reshape(-1, STEP, X.shape[1])
        out.append(np.where(mx, B.max(1), B.mean(1)))
        st = s + np.arange(0, n, STEP); starts.append(st); seg.append(np.full(len(st), h))
    return np.concatenate(out).astype(np.float32), np.concatenate(starts), np.concatenate(seg)


def step_labels(truth, starts):
    """Label of the centre frame of each step (codes 0 H, 1 A, 2 D, 3 ungraded)."""
    return truth[np.minimum(starts + STEP // 2, len(truth) - 1)].astype(np.int64)


def mirror(X, names):
    X = X.copy(); ix = {n: i for i, n in enumerate(names)}
    for n in SIGNED:
        X[:, ix[n]] = -X[:, ix[n]]
    for a, b in SWAP:
        X[:, [ix[a], ix[b]]] = X[:, [ix[b], ix[a]]]
    return X


def mirror_y(y):
    y = y.copy(); h = y == 0; a = y == 1; y[h] = 1; y[a] = 0
    return y


def windowed(Xs, seg, widths=(10, 25, 50)):
    """[X, past means over W steps, future means over W steps] per W, within each segment."""
    blocks = [Xs]
    for W in widths:
        P = np.zeros_like(Xs); Fu = np.zeros_like(Xs)
        for h in np.unique(seg):
            ii = np.flatnonzero(seg == h); A = Xs[ii].astype(np.float64); T = len(A)
            cs = np.vstack([np.zeros((1, A.shape[1])), np.cumsum(A, 0)])
            t = np.arange(T)
            lo = np.maximum(t - W, 0); n_p = np.maximum(t - lo, 1)
            P[ii] = ((cs[t] - cs[lo]) / n_p[:, None]).astype(np.float32)
            hi = np.minimum(t + 1 + W, T); n_f = np.maximum(hi - (t + 1), 1)
            Fu[ii] = ((cs[hi] - cs[np.minimum(t + 1, T)]) / n_f[:, None]).astype(np.float32)
        blocks += [P, Fu]
    return np.hstack(blocks)


def upsample(logp_steps, starts, n_frames, half):
    """Per-frame (F, 3) from per-step values: each frame takes its step's value (nearest/hold); frames outside
    the half runs or after the last full step of a half take the nearest step in that half."""
    from mtah.io.match import half_runs
    out = np.full((n_frames, 3), np.nan, np.float32)
    for h, s, e in half_runs(half):
        m = (starts >= s) & (starts < e)
        st = starts[m]; v = logp_steps[m]
        if not len(st):
            continue
        fr = np.arange(s, e); j = np.clip((fr - st[0]) // STEP, 0, len(st) - 1)
        out[s:e] = v[j]
    return out
