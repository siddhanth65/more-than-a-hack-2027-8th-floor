"""The one in-memory format every stage consumes: metres, NaN for missing."""
from dataclasses import dataclass

import numpy as np


@dataclass(repr=False)
class Match:
    fps: float
    pos: np.ndarray        # (F, S, 2) float32 metres, NaN = slot not tracked
    speed: np.ndarray      # (F, S) float32 m/s
    team: np.ndarray       # (S,) int8: 0 = home, 1 = away
    ball: np.ndarray       # (F, 2) float32 metres, NaN = ball absent
    ball_speed: np.ndarray # (F,) float32 m/s
    half: np.ndarray       # (F,) int8: 0 = ungraded, 1 = first half, 2 = second half
    pitch: tuple = (105.3, 68.0)

    def __repr__(self):
        # shapes only: a Match holds competition data and must never print it
        return (f"Match(F={self.n_frames}, S={self.n_slots}, fps={self.fps}, "
                f"pitch={tuple(self.pitch)})")

    @property
    def n_frames(self):
        return int(self.pos.shape[0])

    @property
    def n_slots(self):
        return int(self.pos.shape[1])

    @property
    def graded(self):
        return self.half > 0

    @property
    def ball_present(self):
        return ~np.isnan(self.ball[:, 0])


def speeds_from_positions(pos, fps):
    """Central-difference speed in m/s along axis 0; NaN wherever a neighbour is missing."""
    pos = np.asarray(pos, dtype=np.float32)
    out = np.full(pos.shape[:-1], np.nan, dtype=np.float32)
    if pos.shape[0] < 3:
        return out
    d = pos[2:] - pos[:-2]
    out[1:-1] = np.sqrt((d ** 2).sum(axis=-1)) * (fps / 2.0)
    return out


def half_runs(half):
    """[(half_id, start, stop)] half-open spans of each graded half, in order.

    Each half must be one contiguous block; raises ValueError otherwise.
    """
    half = np.asarray(half)
    out = []
    for h in sorted(set(half[half > 0].tolist())):
        idx = np.flatnonzero(half == h)
        if len(idx) != idx[-1] - idx[0] + 1:
            raise ValueError(f"half {h} is not contiguous")
        out.append((int(h), int(idx[0]), int(idx[-1]) + 1))
    return out


def distance_to_line(xy_m, pitch):
    """Metres from each point to the nearest pitch boundary line (centred coordinates)."""
    xy = np.asarray(xy_m, dtype=np.float64)
    L, W = pitch
    dx = L / 2.0 - np.abs(xy[..., 0])
    dy = W / 2.0 - np.abs(xy[..., 1])
    return np.minimum(dx, dy)
