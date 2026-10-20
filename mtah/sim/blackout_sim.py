"""Hide the DFL ball the way Metrica's automatic tracker loses it: long blackouts anchored to
dead-ball runs that start near a line, plus occasional live dropouts."""
from dataclasses import asdict, dataclass
from dataclasses import replace as dc_replace

import numpy as np

from mtah.io.match import distance_to_line, speeds_from_positions


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
        # lognormal duration, resampled up to 10 times while shorter than min_s (such runs would be
        # dropped below); if all ten draws are short the dropout is emitted at exactly min_s
        for _ in range(10):
            dur = float(np.exp(np.log(params.live_median_s) + params.live_sigma_log * rng.standard_normal()))
            if dur >= params.min_s:
                break
        else:
            dur = params.min_s
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
    # recompute from the masked track: a speed that needs a hidden neighbour frame is not observable
    return dc_replace(match, ball=ball, ball_speed=speeds_from_positions(ball, match.fps))


def _components(mask, labels, fps, graded, long_s):
    mask = np.asarray(mask, dtype=bool); labels = np.asarray(labels)
    graded = np.ones(len(mask), bool) if graded is None else np.asarray(graded, dtype=bool)
    m = mask & graded
    runs = runs_of(m)
    long_runs = [(s, e) for s, e in runs if (e - s) / fps > long_s]
    inside = np.zeros(len(mask), bool)
    for s, e in long_runs:
        inside[s:e] = True
    return {
        "lengths": [(e - s) / fps for s, e in long_runs],
        "graded_frames": int(graded.sum()), "masked_frames": int(m.sum()),
        "inside_frames": int(inside.sum()), "dead_inside": int((labels[inside] == "D").sum()),
    }


def _finalize(parts, fps):
    lens = np.array([x for c in parts for x in c["lengths"]], dtype=float)
    graded = sum(c["graded_frames"] for c in parts)
    masked = sum(c["masked_frames"] for c in parts)
    inside = sum(c["inside_frames"] for c in parts)
    dead = sum(c["dead_inside"] for c in parts)
    minutes = graded / fps / 60.0
    return {
        "n_blackouts": int(len(lens)),
        "per_graded_min": float(len(lens) / minutes) if minutes else 0.0,
        "masked_share": float(masked / graded) if graded else 0.0,
        "median_s": float(np.median(lens)) if len(lens) else float("nan"),
        "p90_s": float(np.percentile(lens, 90)) if len(lens) else float("nan"),
        "max_s": float(lens.max()) if len(lens) else float("nan"),
        "dead_share_inside": float(dead / inside) if inside else float("nan"),
        "long_share": float(inside / max(1, masked)),
        "lengths_s": lens.tolist(),
    }


def blackout_stats(mask, labels, fps, long_s=10.0, graded=None):
    """Aggregates of one mask. Counts and shares are over graded frames (all frames if graded is None)."""
    return _finalize([_components(mask, labels, fps, graded, long_s)], fps)


def entry_flags(mask, ball, pitch, fps=25.0, long_s=10.0):
    """Per long blackout run: (near_line flags, outward flags) from the last visible ball before it.

    near: that ball is within 5 m of a boundary line. outward: it is closer to the line than the ball
    5 frames earlier (only evaluated when that earlier frame is visible). A frame is visible when it is
    not masked and the ball is not NaN.
    """
    mask = np.asarray(mask, dtype=bool)
    present = ~mask & ~np.isnan(ball[:, 0])
    near, outward = [], []
    for s, e in runs_of(mask):
        if (e - s) / fps <= long_s:
            continue
        k = s - 1
        while k >= 0 and not present[k]:
            k -= 1
        if k < 0:
            continue
        d = distance_to_line(ball[k], pitch)
        near.append(bool(d <= 5.0))
        j = k - 5
        if j >= 0 and present[j]:
            outward.append(bool(d < distance_to_line(ball[j], pitch)))
    return near, outward


def _share(flags):
    return float(np.mean(flags)) if flags else float("nan")


def entry_geometry(mask, ball, pitch, fps=25.0, long_s=10.0):
    """(near_line_share, outward_share) over the long blackout runs of mask; NaN when there are none."""
    near, outward = entry_flags(mask, ball, pitch, fps, long_s)
    return _share(near), _share(outward)


def pooled_blackout_stats(items, long_s=10.0):
    """Stats over several (mask, labels, fps, graded, ball, pitch) inputs, pooling lengths, frames and entries."""
    parts, near, outward = [], [], []
    fps = items[0][2]
    for mask, labels, f, graded, ball, pitch in items:
        parts.append(_components(mask, labels, f, graded, long_s))
        if ball is not None:
            n, o = entry_flags(mask if graded is None else np.asarray(mask, bool) & np.asarray(graded, bool),
                               ball, pitch, f, long_s)
            near += n; outward += o
    out = _finalize(parts, fps)
    out["entry_near_line_share"] = _share(near)
    out["entry_outward_share"] = _share(outward)
    return out


def ks_statistic(lengths_a, lengths_b):
    """Two-sample Kolmogorov-Smirnov distance (max gap between the empirical CDFs)."""
    a = np.sort(np.asarray(lengths_a, dtype=float)); b = np.sort(np.asarray(lengths_b, dtype=float))
    if not len(a) or not len(b):
        return float("nan")
    grid = np.concatenate([a, b])
    cdf_a = np.searchsorted(a, grid, side="right") / len(a)
    cdf_b = np.searchsorted(b, grid, side="right") / len(b)
    return float(np.abs(cdf_a - cdf_b).max())
