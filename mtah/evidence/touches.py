"""Touches as ball-velocity kinks next to a player (football-synthesizer spells_s3, ported)."""
from dataclasses import dataclass

import numpy as np

JUMP_M_PER_FRAME = 3.0


@dataclass
class TouchParams:
    k: int = 3
    theta_dv: float = 5.0
    r_touch_m: float = 1.5
    contest_ratio: float = 1.3
    theta_hi: float = 6.0


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


def detect_touches(match, p):
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
