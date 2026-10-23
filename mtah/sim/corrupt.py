"""Make clean DFL tracking look like automatic tracking: swaps, team flips, dropouts, player outages, noise."""
from dataclasses import asdict, dataclass
from dataclasses import replace as dc_replace

import numpy as np
from scipy.signal import savgol_filter

from mtah.io.match import speeds_from_positions
from mtah.sim.blackout_sim import runs_of


@dataclass
class CorruptParams:
    # the real tracker has almost no identity switches: teleports > 3 m ~0.02 per player-minute, which the
    # clean DFL tracking already shows; each 0.01 flips/min adds ~0.003 (calibrated, runs/select/players_shift.json)
    swap_rate_per_team_min: float = 0.2
    swap_dist_m: float = 1.5
    flip_rate_per_min: float = 0.02
    flip_lo_s: float = 0.5
    flip_hi_s: float = 5.0
    # no short gaps on the real tracker (missing-run p50 ~8 s inside blackouts), so iid and blackout-edge
    # dropout are off by default; the long gaps come from player_outages
    dropout_iid: float = 0.0
    edge_dropout_m: float = 8.0
    edge_dropout_s: float = 0.0
    # AR(1) noise; implied central-difference speed noise ~0.5 m/s at 25 fps
    # (sigma=0.3, rho=0.9 inflated player speed by ~2 m/s)
    noise_sigma_m: float = 0.1
    noise_rho: float = 0.95
    # long per-slot player outages (see player_outages), much more frequent inside ball blackouts
    outage_rate_out_per_player_min: float = 0.005
    outage_rate_in_per_player_min: float = 0.6
    outage_median_s: float = 5.0
    outage_sigma_log: float = 1.2
    outage_max_s: float = 240.0
    # mass outages: some long blackouts lose a large fraction of the slots at once (bimodal lower tail)
    mass_outage_p: float = 0.2
    mass_outage_frac: float = 0.45
    mass_outage_min_s: float = 5.0
    # dead-ball realism (see dead_ball_realism)
    dead_motion: bool = True
    # real dead ball before restarts: median 4 m/s, p90 14.6, never static for 1 s (Plan 2c)
    dead_v_lo: float = 1.0
    dead_v_hi: float = 12.0
    dead_fast_p: float = 0.15       # share of runs carried (kicked) back fast, at U(dead_fast_lo, dead_fast_hi)
    dead_fast_lo: float = 8.0
    dead_fast_hi: float = 15.0
    dead_hover_lo: float = 2.0
    dead_hover_hi: float = 6.0
    dead_out_keep: float = 0.2      # share of runs left where DFL has the ball (incl. out of pitch)
    dead_jitter_m: float = 0.15
    dead_keep_jitter: bool = True   # kept runs get the jitter too, so no static dead ball remains
    # ball smoothing (see smooth_ball): the real feed's ball track is smooth (robust sigma of the 8-frame |dv| 0.73 vs the
    # unsmoothed sim's 1.75), the DFL-derived sim ball + dead-ball jitter is not; applied last, to visible runs only
    ball_smooth_mode: str = "savgol"   # "none" | "savgol"
    ball_smooth_win: int = 13          # frames (odd); closest of {5..13} to the real feed (scripts/calib_ball_smooth.py)
    ball_smooth_poly: int = 2

    def to_dict(self):
        return asdict(self)

    @classmethod
    def from_dict(cls, d):
        return cls(**d)


def dead_ball_realism(ball, labels, pitch, fps, params, rng):
    """Make the visible ball on dead frames behave like an automatic tracker's: carried/kicked back
    towards the restart spot at U(dead_v_lo, dead_v_hi) m/s (with probability dead_fast_p at
    U(dead_fast_lo, dead_fast_hi)), then hovering, always inside the pitch. A share dead_out_keep of the
    runs keeps the DFL ball position (possibly out of pitch), jittered when dead_keep_jitter (real match:
    ball static on 2 % and out of pitch on 3 % of visible frames, never static for 1 s before a restart;
    DFL raw: 35 % and 17 %)."""
    out = ball.copy(); L, W = pitch; half = np.array([L / 2 - 0.5, W / 2 - 0.5])
    if not params.dead_motion:
        return out
    labels = np.asarray(labels)
    for s, e in runs_of(labels == "D"):
        vis = np.flatnonzero(~np.isnan(ball[s:e, 0])) + s
        if not vis.size:
            continue
        if rng.random() < params.dead_out_keep:
            if params.dead_keep_jitter:
                out[vis] = ball[vis] + rng.normal(0, params.dead_jitter_m, (len(vis), 2))
            continue
        start = np.clip(ball[vis[0]], -half, half)
        after = np.flatnonzero(~np.isnan(ball[e:, 0]))
        target = np.clip(ball[e + after[0]], -half, half) if after.size else np.zeros(2)
        if params.dead_fast_p > 0 and rng.random() < params.dead_fast_p:
            v = rng.uniform(params.dead_fast_lo, params.dead_fast_hi) / fps
        else:
            v = rng.uniform(params.dead_v_lo, params.dead_v_hi) / fps
        pos = start.astype(np.float64).copy(); arrived = False
        for f in range(s, e):
            if np.isnan(ball[f, 0]):
                continue
            if not arrived:
                d = target - pos; dist = np.hypot(*d)
                if dist <= v:
                    pos = target.copy(); arrived = True
                else:
                    pos = pos + d / dist * v
            else:
                ang = rng.uniform(0, 2 * np.pi); step = rng.uniform(params.dead_hover_lo, params.dead_hover_hi) / fps
                pos = pos + step * np.array([np.cos(ang), np.sin(ang)])
                pos = np.where(np.abs(pos) > half, np.sign(pos) * half - (pos - np.sign(pos) * half), pos)
            out[f] = np.clip(pos + rng.normal(0, params.dead_jitter_m, 2), -half, half)
    return out.astype(np.float32)


def smooth_ball(ball, params):
    """Savitzky-Golay smoothing of the ball x, y, per run of consecutive visible (non-NaN) frames; runs shorter than the
    window are left as they are, NaN frames stay NaN. Draws no random numbers. mode "none" returns a copy."""
    out = np.asarray(ball, np.float32).copy()
    mode = getattr(params, "ball_smooth_mode", "none")
    if mode == "none":
        return out
    if mode != "savgol":
        raise ValueError(f"unknown ball_smooth_mode {mode!r}")
    w, po = int(params.ball_smooth_win), int(params.ball_smooth_poly)
    if w % 2 == 0 or w <= po:
        raise ValueError(f"ball_smooth_win must be odd and > ball_smooth_poly (got {w}, {po})")
    for s, e in runs_of(~np.isnan(out[:, 0])):
        if e - s >= w:
            out[s:e] = savgol_filter(ball[s:e].astype(np.float64), w, po, axis=0, mode="interp").astype(np.float32)
    return out


# Plan 2b dead-ball parameters. build_half replays dead_ball_realism with these on the half's stream (and draws the
# current dead ball from a spawned child), so the stream that corrupts the player positions -- hence the
# input-hashed PathCRF cache -- does not change when the dead-ball model does.
PLAN2B_DEAD = CorruptParams(dead_motion=True, dead_v_lo=1.0, dead_v_hi=6.0, dead_fast_p=0.0, dead_hover_lo=0.5,
                            dead_hover_hi=2.0, dead_out_keep=0.1, dead_jitter_m=0.15, dead_keep_jitter=False)


def player_outages(F, S, fps, blackout_mask, params, rng):
    """(F, S) bool: frames on which a slot is lost by the tracker. Per slot, outage starts form a Poisson
    process (per player-minute) at outage_rate_out on non-blackout frames and outage_rate_in on blackout
    frames; each outage lasts min(outage_max_s, lognormal(median=outage_median_s, sigma=outage_sigma_log))
    seconds; overlapping outages merge (real tracker inside blackouts: missing-run p50 ~8 s, p90 ~100 s)."""
    bo = np.asarray(blackout_mask, bool)
    per_frame = np.where(bo, params.outage_rate_in_per_player_min, params.outage_rate_out_per_player_min) / (60.0 * fps)
    starts = rng.random((F, S)) < per_frame[:, None]
    t, j = np.nonzero(starts)
    dur_s = np.minimum(params.outage_max_s,
                       params.outage_median_s * np.exp(params.outage_sigma_log * rng.standard_normal(len(t))))
    n = np.maximum(1, np.round(dur_s * fps).astype(np.int64))
    n = np.minimum(n, max(1, int(round(params.outage_max_s * fps))))
    inc = np.zeros((F + 1, S), np.int32)
    np.add.at(inc, (t, j), 1)
    np.add.at(inc, (np.minimum(F, t + n), j), -1)
    return np.cumsum(inc[:F], axis=0) > 0


def mass_outages(present, F, fps, blackout_mask, params, rng):
    """(F, S) bool: for each blackout run of at least mass_outage_min_s, with probability mass_outage_p lose
    round(mass_outage_frac * n_present) slots, chosen uniformly among the slots present ((S,) bool, or (F, S) read at the
    run start), for the whole run (the real tracker's worst blackouts lose about half the players at once)."""
    present = np.asarray(present, bool)
    out = np.zeros((F, present.shape[-1]), bool)
    if params.mass_outage_p <= 0:
        return out
    for s, e in runs_of(blackout_mask):
        if e - s < params.mass_outage_min_s * fps or rng.random() >= params.mass_outage_p:
            continue
        idx = np.flatnonzero(present[s] if present.ndim == 2 else present)
        k = int(round(params.mass_outage_frac * len(idx)))
        if k:
            out[s:e, rng.choice(idx, size=k, replace=False)] = True
    return out


def _swap_rows(pos, a, b, start, stop):
    tmp = pos[start:stop, a].copy()
    pos[start:stop, a] = pos[start:stop, b]
    pos[start:stop, b] = tmp


def corrupt(match, blackout_mask, params, rng):
    F, S = match.n_frames, match.n_slots
    fps = match.fps
    pos = match.pos.copy()
    minutes = F / fps / 60.0
    # same-team identity swaps (persist to the end of the half)
    for k in (0, 1):
        slots = np.flatnonzero(match.team == k)
        for _ in range(rng.poisson(params.swap_rate_per_team_min * minutes)):
            f = int(rng.integers(0, F))
            present = ~np.isnan(pos[f, :, 0])
            on = slots[present[slots]]
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
        present = ~np.isnan(pos[f, :, 0])
        h_on = home[present[home]]; a_on = away[present[away]]
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
    drop |= player_outages(F, S, fps, blackout_mask, params, rng)
    drop |= mass_outages(~np.isnan(pos[:, :, 0]), F, fps, blackout_mask, params, rng)
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
