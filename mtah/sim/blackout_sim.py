"""Hide the DFL ball the way Metrica's automatic tracker loses it: long blackouts anchored to
dead-ball runs that start near a line, plus occasional live dropouts."""
from dataclasses import asdict, dataclass
from dataclasses import replace as dc_replace

import numpy as np

from mtah.io.match import distance_to_line


@dataclass
class BlackoutParams:
    p_mask_near: float = 0.6
    p_mask_far: float = 0.25
    near_m: float = 5.0
    c_lo: float = 0.3
    c_hi: float = 1.0
    pre_lo_s: float = 0.5
    pre_hi_s: float = 3.0
    post_lo_s: float = 0.5
    post_hi_s: float = 3.0
    live_rate_per_min: float = 0.25
    live_median_s: float = 4.0
    live_sigma_log: float = 0.8
    live_max_s: float = 60.0
    merge_s: float = 1.0
    short_rate_per_min: float = 0.0  # tracker re-acquisition dropouts of 5-10 s, independent of dead runs
    short_lo_s: float = 5.0
    short_hi_s: float = 10.0
    min_s: float = 5.0  # runs shorter than this are not emitted: the real tracker never loses the ball for 0.5-5 s

    def to_dict(self):
        return asdict(self)

    @classmethod
    def from_dict(cls, d):
        return cls(**d)


def runs_of(mask):
    m = np.asarray(mask, dtype=np.int8)
    edges = np.diff(np.concatenate(([0], m, [0])))
    return list(zip(np.flatnonzero(edges == 1).tolist(), np.flatnonzero(edges == -1).tolist()))


def drop_short_runs(mask, fps, min_s):
    out = np.array(mask, dtype=bool, copy=True)
    n = int(min_s * fps)
    for s, e in runs_of(out):
        if e - s < n:
            out[s:e] = False
    return out


def blackout_mask(labels, ball, pitch, fps, params, rng):
    labels = np.asarray(labels); F = len(labels)
    mask = np.zeros(F, dtype=bool)
    # component A: dead-run anchored
    for s, e in runs_of(labels == "D"):
        k = s
        while k < e and np.isnan(ball[k, 0]):
            k += 1
        d = distance_to_line(ball[k], pitch) if k < e else np.inf
        p = params.p_mask_near if d <= params.near_m else params.p_mask_far
        if rng.random() >= p:
            continue
        pre = int(rng.uniform(params.pre_lo_s, params.pre_hi_s) * fps)
        post = int(rng.uniform(params.post_lo_s, params.post_hi_s) * fps)
        c = rng.uniform(params.c_lo, params.c_hi)
        a = max(0, s - pre); b = min(F, s + int(c * (e - s)) + post)
        mask[a:b] = True
    # component B: live dropouts
    alive = np.flatnonzero(labels != "D")
    minutes = F / fps / 60.0
    n_live = rng.poisson(params.live_rate_per_min * minutes) if len(alive) else 0
    for _ in range(n_live):
        s = int(rng.choice(alive))
        dur = float(np.exp(np.log(params.live_median_s) + params.live_sigma_log * rng.standard_normal()))
        dur = min(dur, params.live_max_s)
        mask[s:min(F, s + int(dur * fps))] = True
    # component C: short re-acquisition dropouts
    for _ in range(rng.poisson(params.short_rate_per_min * minutes)):
        s = int(rng.integers(0, F))
        mask[s:min(F, s + int(rng.uniform(params.short_lo_s, params.short_hi_s) * fps))] = True
    # merge gaps shorter than merge_s
    gap = int(params.merge_s * fps)
    runs = runs_of(mask)
    for (s0, e0), (s1, e1) in zip(runs[:-1], runs[1:]):
        if s1 - e0 < gap:
            mask[e0:s1] = True
    return drop_short_runs(mask, fps, params.min_s)


def apply_blackouts(match, mask):
    ball = match.ball.copy(); ball[mask] = np.nan
    bs = match.ball_speed.copy(); bs[mask] = np.nan
    return dc_replace(match, ball=ball, ball_speed=bs)


def blackout_stats(mask, labels, fps, long_s=10.0):
    labels = np.asarray(labels)
    runs = runs_of(mask)
    lens = np.array([e - s for s, e in runs], dtype=float) / fps
    long = lens[lens > long_s]
    F = len(mask); minutes = F / fps / 60.0
    inside = np.zeros(F, bool)
    for (s, e), ln in zip(runs, lens):
        if ln > long_s:
            inside[s:e] = True
    return {
        "n_blackouts": int(len(long)),
        "per_133min": float(len(long) / minutes * 133.0) if minutes else 0.0,
        "masked_share": float(mask.mean()),
        "median_s": float(np.median(long)) if len(long) else 0.0,
        "p90_s": float(np.percentile(long, 90)) if len(long) else 0.0,
        "max_s": float(long.max()) if len(long) else 0.0,
        "dead_share_inside": float((labels[inside] == "D").mean()) if inside.any() else float("nan"),
        "long_share": float(inside.sum() / max(1, mask.sum())),
    }
