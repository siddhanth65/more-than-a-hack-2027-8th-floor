"""Rule baselines: the bar every candidate must clear."""
import numpy as np

from mtah.io.match import half_runs
from mtah.sim.blackout_sim import runs_of

NOBODY = -1


def nearest_team(match, radius_m=None):
    F = match.n_frames
    out = np.full(F, NOBODY, dtype=np.int8)
    present = match.ball_present
    d = match.pos[present] - match.ball[present][:, None, :]
    dist = np.sqrt((d ** 2).sum(-1))              # (f, S), NaN for missing slots
    dist = np.where(np.isnan(dist), np.inf, dist)
    best = dist.argmin(axis=1)
    bestd = dist[np.arange(len(best)), best]
    team = match.team[best].astype(np.int8)
    team[~np.isfinite(bestd)] = NOBODY
    if radius_m is not None:
        team[bestd > radius_m] = NOBODY
    out[present] = team
    return out


def hold_last(codes, half):
    out = np.asarray(codes, dtype=np.int8).copy()
    # carry within each half only; frames outside every half (pre-match etc.) are left as they are
    spans = [(s, e) for _, s, e in half_runs(half)] or [(0, len(out))]
    for a, b in spans:
        w = out[a:b]
        known = np.flatnonzero(w != NOBODY)
        if not len(known):
            continue
        carry = np.maximum.accumulate(np.where(w != NOBODY, np.arange(len(w)), -1))
        filled = np.where(carry >= 0, w[np.maximum(carry, 0)], NOBODY)
        out[a:b] = filled
    return out


def _to_labels(codes, graded):
    lab = np.where(codes == 0, "H", np.where(codes == 1, "A", "D")).astype("<U1")
    lab[~graded] = "D"
    return lab


def b0_nearest_hold(match):
    return _to_labels(hold_last(nearest_team(match), match.half), match.graded)


def b4_blackout_dead(match, min_gap_s=10.0):
    codes = hold_last(nearest_team(match), match.half)
    gap = int(min_gap_s * match.fps)
    for s, e in runs_of(~match.ball_present):
        if e - s > gap:
            codes[s:e] = NOBODY
    return _to_labels(codes, match.graded)


def smooth_vote(match, window):
    codes = nearest_team(match)
    if window <= 0:
        return _to_labels(codes, match.graded)
    F = len(codes)
    hc = np.concatenate(([0], np.cumsum(codes == 0)))
    ac = np.concatenate(([0], np.cumsum(codes == 1)))
    idx = np.arange(F)
    lo = np.maximum(0, idx - window); hi = np.minimum(F, idx + window + 1)
    home = hc[hi] - hc[lo]; away = ac[hi] - ac[lo]
    out = np.full(F, NOBODY, dtype=np.int8)
    seen = (home + away) > 0
    out[seen] = np.where(home[seen] >= away[seen], 0, 1)
    lab = np.where(out == 0, "H", np.where(out == 1, "A", "D")).astype("<U1")
    return lab   # organisers' baseline writes H/A/D everywhere, including ungraded frames
