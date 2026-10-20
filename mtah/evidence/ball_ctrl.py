"""Ball-control evidence: the football-synthesizer carrier/flight/contested model, slot-indexed,
run on continuous-ball segments; forward-backward marginals summed by team."""
from dataclasses import dataclass

import numpy as np

FLIGHT = -1
CONTESTED = -2
CONTEST_MARGIN_M = 1.0


@dataclass
class BallCtrlParams:
    sigma_m: float = 1.0
    c_flight: float = 3.0
    c_contest: float = 3.0
    r_contest_m: float = 2.0
    tau_switch: float = 6.0
    tau_flight: float = 1.0
    w_vel: float = 0.15
    cand_radius_m: float = 5.0
    top_k: int = 4
    alpha_loose: float = 0.5
    min_seg: int = 3
    eps: float = 1e-3


def ball_segments(ball_present, min_seg):
    bp = np.asarray(ball_present, dtype=np.int8)
    edges = np.diff(np.concatenate(([0], bp, [0])))
    return [(int(s), int(e)) for s, e in zip(np.flatnonzero(edges == 1), np.flatnonzero(edges == -1)) if e - s >= min_seg]


def _velocities(xy):
    """Central-difference velocity vectors along axis 0 (NaN at ends / missing)."""
    v = np.full_like(xy, np.nan, dtype=np.float64)
    if len(xy) >= 3:
        v[1:-1] = (xy[2:] - xy[:-2]) / 2.0
    return v


def _frame_states(xy, vel, team, bxy, bv, prev_ctrl, p):
    """(states, costs, teams) for one frame; states: FLIGHT, CONTESTED or slot index."""
    states = [FLIGHT]; costs = [p.c_flight]; teams = [-1]
    on = np.flatnonzero(np.isfinite(xy[:, 0]))
    if not on.size:
        return states, np.array(costs), teams
    d_all = np.full(len(xy), np.inf); d_all[on] = np.hypot(xy[on, 0] - bxy[0], xy[on, 1] - bxy[1])
    order = on[np.argsort(d_all[on])]
    cand = [int(i) for i in order[: p.top_k] if d_all[i] <= p.cand_radius_m]
    for j in prev_ctrl:
        if j not in cand and np.isfinite(xy[j, 0]):
            cand.append(int(j))
    rel_v = np.zeros(len(xy))
    if p.w_vel > 0 and bv is not None and np.all(np.isfinite(bv)):
        ok = np.all(np.isfinite(vel), axis=1)
        rel_v[ok] = np.hypot(*(vel[ok] - bv).T)
    near = {}
    for i in order:
        near.setdefault(int(team[i]), float(d_all[i]))
    for i in cand:
        opp = [v for t, v in near.items() if t != team[i]]
        gap = (min(opp) - d_all[i]) if opp else np.inf
        amb = p.c_contest * max(0.0, 1.0 - gap / CONTEST_MARGIN_M) if np.isfinite(gap) else 0.0
        states.append(int(i)); costs.append(d_all[i] ** 2 / (2 * p.sigma_m ** 2) + p.w_vel * rel_v[i] + amb)
        teams.append(int(team[i]))
    if len(near) >= 2:
        nd = sorted(near.values())
        if nd[0] <= p.r_contest_m and nd[1] <= p.r_contest_m:
            states.append(CONTESTED); costs.append(nd[0] ** 2 / (2 * p.sigma_m ** 2) + p.c_contest / 2); teams.append(-1)
    return states, np.array(costs), teams


def _trans(prev, cur, p):
    a = np.asarray(prev)[:, None]; b = np.asarray(cur)[None, :]
    both = (a >= 0) & (b >= 0)
    t = np.where(both, p.tau_switch, p.tau_flight)
    return np.where(a == b, 0.0, t)


def _lse(x, axis):
    m = np.max(x, axis=axis, keepdims=True)
    return (m + np.log(np.sum(np.exp(x - m), axis=axis, keepdims=True))).squeeze(axis)


def _segment_posteriors(pos, team, ball, p, fps):
    """Forward-backward marginals for one continuous-ball segment: (p_team (n,2), p_loose (n,))."""
    n = len(ball)
    pv = _velocities(pos) * fps    # metres/second so w_vel acts as in the original model
    bv = _velocities(ball) * fps
    seq = []; prev_ctrl = []
    for t in range(n):
        st = _frame_states(pos[t], pv[t], team, ball[t], bv[t] if t and t < n - 1 else None, prev_ctrl, p)
        seq.append(st); prev_ctrl = [s for s in st[0] if s >= 0]
    fwd = [-seq[0][1]]; trans = [None]
    for i in range(1, n):
        tm = _trans(seq[i - 1][0], seq[i][0], p); trans.append(tm)
        fwd.append(_lse(fwd[-1][:, None] - tm, axis=0) - seq[i][1])
    bwd = [None] * n; bwd[-1] = np.zeros(len(seq[-1][0]))
    for i in range(n - 1, 0, -1):
        bwd[i - 1] = _lse(-trans[i] - seq[i][1][None, :] + bwd[i][None, :], axis=1)
    p_team = np.zeros((n, 2), np.float32); p_loose = np.zeros(n, np.float32)
    for i in range(n):
        post = fwd[i] + bwd[i]; post = np.exp(post - _lse(post, axis=0))
        for k, s in enumerate(seq[i][0]):
            tm = seq[i][2][k]
            if s >= 0 and tm in (0, 1):
                p_team[i, tm] += post[k]
            else:
                p_loose[i] += post[k]
    return p_team, p_loose


def ball_ctrl_posteriors(match, params):
    F = match.n_frames
    p_team = np.full((F, 2), np.nan, np.float32); p_loose = np.full(F, np.nan, np.float32)
    for s, e in ball_segments(match.ball_present, params.min_seg):
        pt, pl = _segment_posteriors(match.pos[s:e].astype(np.float64), match.team, match.ball[s:e].astype(np.float64),
                                     params, match.fps)
        p_team[s:e] = pt; p_loose[s:e] = pl
    return p_team, p_loose


def ball_ctrl_evidence(match, params):
    pt, pl = ball_ctrl_posteriors(match, params)
    ev = np.full((match.n_frames, 3), np.nan, np.float32)
    ok = np.isfinite(pl)
    ev[ok, 0] = np.log(pt[ok, 0] + params.alpha_loose * pl[ok] + params.eps)
    ev[ok, 1] = np.log(pt[ok, 1] + params.alpha_loose * pl[ok] + params.eps)
    return ev
