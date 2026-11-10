"""Shared pieces of the arbitration experiments (scripts/arb_*.py): the label-free "confident ball control"
pseudo-label rule, the pseudo-blackout mask sampler, and window scoring helpers. Pure numpy (+ the md5-pinned
official scorer wrapper for scoring); no contest rows are ever printed.
"""
import json
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from mtah.sim.blackout_sim import runs_of   # noqa: E402

LAB = np.array(["H", "A", "D"])


# ------------------------------------------------------------------ pseudo-label rule
@dataclass
class RuleParams:
    r_near: float = 1.5      # m: the closest player is within this of the ball
    r_opp: float = 3.0       # m: the closest opponent (other team) is at least this far
    v_rel: float = 2.0       # m/s: speed of the ball relative to the nearest player (over +-0.4 s) at most this
    v_ball_min: float = 0.0  # m/s: the ball itself moves at least this fast (over +-0.4 s): excludes a still (dead) ball
    min_run_s: float = 1.0   # the conditions must hold (same player) for at least this long
    trim_s: float = 0.5      # label only the middle of a run: this much is cut off both ends
    same_player: bool = True

    def to_dict(self):
        return asdict(self)


def _central_vel(x, fps):
    """(T, ..., 2) central-difference velocity in m/s; NaN at the ends and next to a NaN."""
    v = np.full(x.shape, np.nan, np.float32)
    if x.shape[0] >= 3:
        v[1:-1] = (x[2:] - x[:-2]) * (fps / 2.0)
    return v


def frame_geometry(match, chunk=40000, k=10):
    """Per frame: nearest player overall (slot, distance, team), distance of the nearest player of the OTHER team
    (d2), and the speed of the ball relative to that player averaged over +-k frames (|d(ball - player)| / dt, m/s;
    robust to the position jitter that makes instantaneous velocities useless). inf/-1 where undefined. Chunked."""
    F = match.n_frames; fps = match.fps
    slot = np.full(F, -1, np.int16); d1 = np.full(F, np.inf, np.float32); d2 = np.full(F, np.inf, np.float32)
    team = np.asarray(match.team)
    ball = match.ball.astype(np.float32)
    for a in range(0, F, chunk):
        b = min(F, a + chunk)
        d = np.linalg.norm(match.pos[a:b] - ball[a:b, None, :], axis=-1)
        d = np.where(np.isnan(d), np.inf, d).astype(np.float32)
        n = b - a
        dH = np.where(team[None, :] == 0, d, np.inf); dA = np.where(team[None, :] == 1, d, np.inf)
        iH = dH.argmin(1); iA = dA.argmin(1); r = np.arange(n)
        mH = dH[r, iH]; mA = dA[r, iA]
        home = mH <= mA
        s = np.where(home, iH, iA)
        slot[a:b] = np.where(np.isfinite(np.minimum(mH, mA)), s, -1)
        d1[a:b] = np.minimum(mH, mA); d2[a:b] = np.where(home, mA, mH)
    t = np.arange(F); sl = np.maximum(slot, 0)
    tp = np.minimum(t + k, F - 1); tm = np.maximum(t - k, 0)
    rp = ball[tp] - match.pos[tp, sl]; rm = ball[tm] - match.pos[tm, sl]
    rel = np.linalg.norm(rp - rm, axis=-1) / np.maximum((tp - tm) / fps, 1e-6)
    vrel = np.where(np.isnan(rel) | (slot < 0), np.inf, rel).astype(np.float32)
    vb = np.linalg.norm(ball[tp] - ball[tm], axis=-1) / np.maximum((tp - tm) / fps, 1e-6)
    vb = np.where(np.isnan(vb), 0.0, vb).astype(np.float32)
    return {"slot": slot, "d1": d1, "d2": d2, "vrel": vrel, "vball": vb, "team": np.where(slot >= 0, team[np.maximum(slot, 0)], -1)}


def confident_control(match, p=None, geom=None):
    """(F,) int8: 0 = confidently home, 1 = confidently away, -1 = not confident. Uses only the tracking (the
    ball must be visible); label the middle of >= min_run_s runs of constant (team[, player]) with all conditions."""
    p = p or RuleParams()
    g = geom or frame_geometry(match)
    ok = (match.ball_present & match.graded & (g["slot"] >= 0) & (g["d1"] <= p.r_near) & (g["d2"] >= p.r_opp)
          & (g["vrel"] <= p.v_rel) & (g["vball"] >= p.v_ball_min))
    key = np.where(ok, (g["slot"].astype(np.int32) if p.same_player else g["team"].astype(np.int32)), -1)
    out = np.full(match.n_frames, -1, np.int8)
    minrun = int(round(p.min_run_s * match.fps)); trim = int(round(p.trim_s * match.fps))
    if len(key) == 0:
        return out
    chg = np.flatnonzero(np.diff(key) != 0) + 1
    starts = np.concatenate(([0], chg)); stops = np.concatenate((chg, [len(key)]))
    for s, e in zip(starts, stops):
        if key[s] < 0 or e - s < minrun:
            continue
        a, b = s + trim, e - trim
        if b > a:
            out[a:b] = g["team"][s]
    return out


# ------------------------------------------------------------------ pseudo-blackout sampler
def length_pool_from_blackouts(blackout_masks, fps=25.0, min_s=5.0):
    """Run lengths (s) of the simulator's blackout masks (mtah.sim.blackout_sim output = the fitted blackout length
    distribution realised by the corruption code), >= min_s."""
    out = []
    for m in blackout_masks:
        out += [(e - s) / fps for s, e in runs_of(m) if (e - s) / fps >= min_s]
    return np.asarray(out, float)


def sample_mask(match, seed, pool_s, unit_id=0, target=0.30, min_s=5.0, max_tries=20000):
    """(F,) bool pseudo-blackout mask: windows with lengths drawn (with replacement) from `pool_s`, placed ONLY inside
    stretches where the ball is naturally visible (and the frame is graded, within one half), hiding about `target` of
    the visible graded frames. A window drawn longer than the room left in its visible stretch is clipped to it; the
    window is dropped (new draw) when what is left is shorter than min_s. Deterministic in (seed, unit_id)."""
    rng = np.random.default_rng([int(seed), int(unit_id), 4711])
    fps = match.fps; F = match.n_frames
    free = match.ball_present & match.graded
    # a window must stay inside one half (graded halves can be adjacent only through ungraded frames: free is False there)
    mask = np.zeros(F, bool)
    total = int(free.sum()); goal = int(round(target * total)); hidden = 0
    if total == 0 or len(pool_s) == 0:
        return mask
    minf = int(round(min_s * fps))
    # candidate start frames: free frames; we re-check the room at draw time
    cand = np.flatnonzero(free)
    tries = 0
    while hidden < goal and tries < max_tries:
        tries += 1
        s = int(cand[rng.integers(len(cand))])
        if mask[s] or not free[s]:
            continue
        L = int(round(float(pool_s[rng.integers(len(pool_s))]) * fps))
        # room: consecutive free & unmasked frames from s
        e = s
        lim = min(F, s + L)
        while e < lim and free[e] and not mask[e]:
            e += 1
        if e - s < minf:
            continue
        mask[s:e] = True; hidden += e - s
    return mask


def apply_mask(match, mask):
    from mtah.sim.blackout_sim import apply_blackouts
    return apply_blackouts(match, mask)


# ------------------------------------------------------------------ scoring helpers
def windowed(ref, window):
    """ref labels (U1 array) with everything outside `window` set to X."""
    out = np.full(len(ref), "X", dtype="<U1")
    out[window] = np.asarray(ref)[window]
    return out


def score_window(ref, pred, window):
    """(spellF1, macroF1) of `pred` against `ref` restricted to `window` (bool), via the md5-pinned official wrapper.
    NaN, NaN when the reference has no graded frame in the window."""
    from mtah.eval.scorer import score
    g = windowed(ref, window)
    if not (g != "X").any():
        return float("nan"), float("nan")
    r = score(g, pred)
    return r["spell_f1"], r["macro_f1"]


def masked_runs(mask):
    return runs_of(mask)
