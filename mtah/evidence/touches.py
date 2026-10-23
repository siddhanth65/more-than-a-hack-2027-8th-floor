"""Touches as ball-velocity kinks next to a player (football-synthesizer spells_s3, ported)."""
from dataclasses import dataclass

import numpy as np

JUMP_M_PER_FRAME = 3.0


DV_SIGMA_REF = 1.75    # robust sigma (m/s) of the k=3 velocity change on the DFL-derived simulator halves (touch_diagnose)


@dataclass
class TouchParams:
    """mode "abs" (default, what a5c / a10 were selected with): peaks of the k-frame velocity change above the absolute
    theta_dv (m/s), nearest player within r_touch_m AT the peak frame.
    mode "rel" (feed-adaptive): the velocity change at scale k_rel is turned into a robust z-score against each half's
    own noise level (median / MAD over its visible frames); a touch is a local maximum of max(z, direction score) >=
    z_thr, away from ball-gap edges and teleports, with a player within r_rel_m at some frame of +-gate_frames around
    the peak (time-tolerant, the real feed is a smoothed track); weaker peaks within min_gap_s are suppressed and each
    half is capped at max_rate_per_min touches per visible minute (strongest kept). Output column 3 is z * DV_SIGMA_REF
    so theta_hi keeps its meaning (strong touch)."""
    k: int = 3
    theta_dv: float = 5.0
    r_touch_m: float = 1.5
    contest_ratio: float = 1.3
    theta_hi: float = 6.0
    mode: str = "abs"
    k_rel: int = 8
    z_thr: float = 2.5
    z_min: float = 1.5          # rate floor: if a half has fewer than min_rate_per_min touches, accept weaker peaks down to z_min
    min_rate_per_min: float = 6.0
    dv_floor: float = 2.0       # physical floor (m/s) on the velocity change: noise-only paths never fire
    sigma_floor: float = 0.25   # floor on the per-half robust sigma (m/s) so a noise-free half does not inflate z
    ang_deg: float = 45.0       # direction trigger: angle between in/out velocity (needs both speeds >= v_dir_min)
    v_dir_min: float = 2.0
    z_lo: float = 1.5           # the direction trigger still needs this much speed-change evidence
    r_rel_m: float = 1.5
    gate_frames: int = 8        # proximity window half-width (frames)
    guard_frames: int = 10      # no touch within this many frames of a ball gap
    min_gap_s: float = 0.3
    max_rate_per_min: float = 24.0


def velocity_change(ball, k, fps):
    n = len(ball); out = np.full(n, np.nan)
    if n <= 2 * k:
        return out
    b0, b1, b2 = ball[: n - 2 * k], ball[k: n - k], ball[2 * k:]
    vin = (b1 - b0) * fps / k; vout = (b2 - b1) * fps / k
    out[k: n - k] = np.hypot(*(vout - vin).T)
    return out


def _jump_frames(ball):
    d = np.hypot(np.diff(ball[:, 0]), np.diff(ball[:, 1]))
    j = np.zeros(len(ball), bool); j[1:] = d > JUMP_M_PER_FRAME
    return j


def _robust_z(dv, half, vis, sigma_floor=0.25):
    """Per-half robust z of dv (median / 1.4826*MAD over the half's visible finite frames); ungraded frames and halves
    with too few frames use the pooled statistics of all graded halves."""
    ok = np.isfinite(dv) & vis
    z = np.full(len(dv), np.nan)

    def stats(sel):
        x = dv[ok & sel]
        if len(x) < 150:
            return None
        med = float(np.median(x)); return med, max(1.4826 * float(np.median(np.abs(x - med))), sigma_floor)
    pooled = stats(half > 0) or stats(np.ones(len(dv), bool)) or (0.0, DV_SIGMA_REF)   # too little ball: DFL reference
    for h in np.unique(half):
        st = stats(half == h) or pooled
        if st is None:
            continue
        sel = half == h
        z[sel] = (dv[sel] - st[0]) / st[1]
    return z


def _angle_deg(ball, k, fps):
    n = len(ball); out = np.full(n, np.nan); smin = np.full(n, np.nan)
    if n <= 2 * k:
        return out, smin
    b0, b1, b2 = ball[: n - 2 * k], ball[k: n - k], ball[2 * k:]
    vin = (b1 - b0) * fps / k; vout = (b2 - b1) * fps / k
    ni = np.hypot(*vin.T); no = np.hypot(*vout.T)
    cos = (vin * vout).sum(1) / np.maximum(ni * no, 1e-9)
    out[k: n - k] = np.degrees(np.arccos(np.clip(cos, -1, 1))); smin[k: n - k] = np.minimum(ni, no)
    return out, smin


def _detect_rel(match, p):
    ball = match.ball.astype(np.float64); pos = match.pos; team = match.team; fps = match.fps; F = len(ball)
    k = p.k_rel
    vis = ~np.isnan(ball[:, 0])
    dv = velocity_change(ball, k, fps)
    z = _robust_z(dv, match.half, vis & (match.half > 0), p.sigma_floor)
    ang, smin = _angle_deg(ball, k, fps)
    zf = np.where(np.nan_to_num(dv, nan=0.0) >= p.dv_floor, np.nan_to_num(z, nan=-1.0), -1.0)
    dscore = np.where(np.isfinite(ang) & (smin >= p.v_dir_min) & (zf >= p.z_lo), p.z_thr * np.nan_to_num(ang) / p.ang_deg, 0.0)
    score = np.maximum(zf, dscore)
    win_h = max(k, int(round(p.min_gap_s * fps / 2)))
    padded = np.r_[np.full(win_h, -1.0), score, np.full(win_h, -1.0)]
    mx = np.lib.stride_tricks.sliding_window_view(padded, 2 * win_h + 1).max(1)
    peak = (score >= p.z_min) & (score >= mx)
    peak &= ~np.r_[False, peak[:-1] & (score[1:] == score[:-1])]
    # ball-gap guard and teleports
    g = p.guard_frames
    gap = np.r_[np.zeros(g, bool), ~vis, np.zeros(g, bool)]
    peak &= ~np.lib.stride_tricks.sliding_window_view(gap, 2 * g + 1).any(1)
    jumps = _jump_frames(np.nan_to_num(ball, nan=1e6))
    jpad = np.r_[np.zeros(k, bool), jumps, np.zeros(k, bool)]
    peak &= ~np.lib.stride_tricks.sliding_window_view(jpad, 2 * k + 1).any(1)
    fr = np.flatnonzero(peak)
    if not fr.size:
        return np.zeros((0, 4))
    # time-tolerant nearest-player gate: closest (frame, slot) within +-gate_frames
    gf = p.gate_frames; offs = np.arange(-gf, gf + 1)
    idx = np.clip(fr[:, None] + offs[None, :], 0, F - 1)                      # (n, W)
    d = np.hypot(pos[idx, :, 0] - ball[idx, None, 0], pos[idx, :, 1] - ball[idx, None, 1])   # (n, W, S)
    d = np.where(np.isfinite(d), d, np.inf)
    dflat = d.reshape(len(fr), -1)
    am = dflat.argmin(1); dn = dflat[np.arange(len(fr)), am]
    j = am % pos.shape[1]
    ok = dn <= p.r_rel_m
    tj = team[j]
    dt = np.where(team[None, None, :] != tj[:, None, None], d, np.inf).reshape(len(fr), -1).min(1)
    contested = (dt <= p.r_rel_m) & (dt < p.contest_ratio * dn)
    t_team = np.where(contested, -1, tj)
    out = np.c_[fr, j, t_team, score[fr] * DV_SIGMA_REF][ok]
    # per-half rate control on the gated candidates (strongest first): everything >= z_thr, topped up with weaker peaks
    # (down to z_min) until min_rate_per_min, never more than max_rate_per_min touches per visible minute
    keep = np.zeros(len(out), bool)
    hf = match.half[out[:, 0].astype(int)]
    for h in np.unique(hf):
        sel = np.flatnonzero(hf == h)
        vmin = (vis & (match.half == h)).sum() / fps / 60
        order = sel[np.argsort(-out[sel, 3], kind="stable")]
        strong_n = int((out[sel, 3] >= p.z_thr * DV_SIGMA_REF).sum())
        n_keep = min(max(strong_n, int(p.min_rate_per_min * vmin)), int(p.max_rate_per_min * vmin), len(order))
        keep[order[:n_keep]] = True
    return out[keep].astype(float)


def detect_touches(match, p):
    if getattr(p, "mode", "abs") == "rel":
        return _detect_rel(match, p)
    if getattr(p, "mode", "abs") != "abs":
        raise ValueError(f"unknown touch mode {p.mode!r}")
    ball = match.ball.astype(np.float64); pos = match.pos; team = match.team
    dv = velocity_change(ball, p.k, match.fps)
    w = np.nan_to_num(dv, nan=-1.0)
    padded = np.r_[np.full(p.k, -1.0), w, np.full(p.k, -1.0)]
    win = np.lib.stride_tricks.sliding_window_view(padded, 2 * p.k + 1)
    peak = (w >= p.theta_dv) & (w >= win.max(1))
    peak &= ~np.r_[False, peak[:-1] & (w[1:] == w[:-1])]
    jumps = _jump_frames(np.nan_to_num(ball, nan=1e6))
    jpad = np.r_[np.zeros(p.k, bool), jumps, np.zeros(p.k, bool)]
    near_jump = np.lib.stride_tricks.sliding_window_view(jpad, 2 * p.k + 1).any(1)
    peak &= ~near_jump
    fr = np.flatnonzero(peak)
    if not fr.size:
        return np.zeros((0, 4))
    d = np.hypot(pos[fr, :, 0] - ball[fr, None, 0], pos[fr, :, 1] - ball[fr, None, 1])
    d = np.where(np.isfinite(d), d, np.inf)
    j = d.argmin(1); dn = d[np.arange(len(fr)), j]
    ok = dn <= p.r_touch_m
    tj = team[j]
    other = np.where(team[None, :] != tj[:, None], d, np.inf).min(1)
    contested = (other <= p.r_touch_m) & (other < p.contest_ratio * dn)
    t_team = np.where(contested, -1, tj)
    return np.c_[fr, j, t_team, dv[fr]][ok].astype(float)


def strong_touch_frames(touches, team, p, n_frames):
    out = np.zeros(n_frames, bool)
    if len(touches):
        sel = (touches[:, 2] == team) & (touches[:, 3] >= p.theta_hi)
        out[touches[sel, 0].astype(int)] = True
    return out
