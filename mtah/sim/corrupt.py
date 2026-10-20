"""Make clean DFL tracking look like automatic tracking: swaps, team flips, dropouts, noise."""
from dataclasses import asdict, dataclass
from dataclasses import replace as dc_replace

import numpy as np

from mtah.io.match import speeds_from_positions
from mtah.sim.blackout_sim import runs_of


@dataclass
class CorruptParams:
    swap_rate_per_team_min: float = 2.0
    swap_dist_m: float = 1.5
    flip_rate_per_min: float = 0.6
    flip_lo_s: float = 0.5
    flip_hi_s: float = 5.0
    dropout_iid: float = 0.03
    edge_dropout_m: float = 8.0
    edge_dropout_s: float = 1.0
    noise_sigma_m: float = 0.3
    noise_rho: float = 0.9

    def to_dict(self):
        return asdict(self)

    @classmethod
    def from_dict(cls, d):
        return cls(**d)


def _swap_rows(pos, a, b, start, stop):
    tmp = pos[start:stop, a].copy()
    pos[start:stop, a] = pos[start:stop, b]
    pos[start:stop, b] = tmp


def corrupt(match, blackout_mask, params, rng):
    F, S = match.n_frames, match.n_slots
    fps = match.fps
    pos = match.pos.copy()
    minutes = F / fps / 60.0
    present = ~np.isnan(pos[..., 0])
    # same-team identity swaps (persist to the end of the half)
    for k in (0, 1):
        slots = np.flatnonzero(match.team == k)
        for _ in range(rng.poisson(params.swap_rate_per_team_min * minutes)):
            f = int(rng.integers(0, F))
            on = slots[present[f, slots]]
            if len(on) < 2:
                continue
            d = np.hypot(pos[f, on, 0][:, None] - pos[f, on, 0][None, :],
                         pos[f, on, 1][:, None] - pos[f, on, 1][None, :])
            np.fill_diagonal(d, np.inf)
            i, j = np.unravel_index(int(d.argmin()), d.shape)
            if d[i, j] <= params.swap_dist_m:
                _swap_rows(pos, int(on[i]), int(on[j]), f, F)
    # team-tag flip runs (cross-team position swap for a short run)
    home = np.flatnonzero(match.team == 0); away = np.flatnonzero(match.team == 1)
    for _ in range(rng.poisson(params.flip_rate_per_min * minutes)):
        f = int(rng.integers(0, F))
        h_on = home[present[f, home]]; a_on = away[present[f, away]]
        if not len(h_on) or not len(a_on):
            continue
        a = int(rng.choice(h_on)); b = int(rng.choice(a_on))
        dur = int(rng.uniform(params.flip_lo_s, params.flip_hi_s) * fps)
        _swap_rows(pos, a, b, f, min(F, f + dur))
    # dropout
    drop = rng.random((F, S)) < params.dropout_iid
    edge = int(params.edge_dropout_s * fps)
    for s, e in runs_of(blackout_mask):
        k = s - 1
        while k >= 0 and np.isnan(match.ball[k, 0]):
            k -= 1
        if k >= 0:
            d = np.hypot(pos[s:min(F, s + edge), :, 0] - match.ball[k, 0], pos[s:min(F, s + edge), :, 1] - match.ball[k, 1])
            drop[s:min(F, s + edge)] |= d <= params.edge_dropout_m
        k = e
        while k < F and np.isnan(match.ball[k, 0]):
            k += 1
        if k < F:
            d = np.hypot(pos[max(0, e - edge):e, :, 0] - match.ball[k, 0], pos[max(0, e - edge):e, :, 1] - match.ball[k, 1])
            drop[max(0, e - edge):e] |= d <= params.edge_dropout_m
    pos[drop] = np.nan
    # AR(1) positional noise
    if params.noise_sigma_m > 0:
        z = rng.standard_normal((F, S, 2)).astype(np.float32) * params.noise_sigma_m
        noise = np.empty_like(z)
        noise[0] = z[0]
        c = np.float32(np.sqrt(1 - params.noise_rho ** 2))
        for t in range(1, F):
            noise[t] = params.noise_rho * noise[t - 1] + c * z[t]
        pos = pos + noise
    speed = speeds_from_positions(pos, fps)
    return dc_replace(match, pos=pos.astype(np.float32), speed=speed)
