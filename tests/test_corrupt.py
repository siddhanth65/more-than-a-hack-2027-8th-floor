from dataclasses import replace

import numpy as np

from mtah.io.match import Match, speeds_from_positions
from mtah.sim.corrupt import CorruptParams, corrupt


NO_OUT = dict(outage_rate_out_per_player_min=0.0, outage_rate_in_per_player_min=0.0)


def grid_match(F=25 * 120, S=6):
    pos = np.zeros((F, S, 2), np.float32)
    for j in range(S):
        pos[:, j, 0] = -20 + 8 * j          # 6 players spaced 8 m along x
    ball = np.zeros((F, 2), np.float32)
    team = np.array([0, 0, 0, 1, 1, 1], np.int8)
    return Match(fps=25.0, pos=pos, speed=speeds_from_positions(pos, 25.0), team=team, ball=ball,
                 ball_speed=speeds_from_positions(ball, 25.0), half=np.ones(F, np.int8), pitch=(105.0, 68.0))


def test_noise_only_is_small_and_seeded():
    m = grid_match()
    p = CorruptParams(swap_rate_per_team_min=0, flip_rate_per_min=0, dropout_iid=0, noise_sigma_m=0.3, **NO_OUT)
    mask = np.zeros(m.n_frames, bool)
    a = corrupt(m, mask, p, np.random.default_rng(0)); b = corrupt(m, mask, p, np.random.default_rng(0))
    assert np.array_equal(np.nan_to_num(a.pos), np.nan_to_num(b.pos))
    err = np.nanstd(a.pos - m.pos)
    assert 0.15 < err < 0.5
    assert not np.isnan(a.pos).any()


def test_flip_run_swaps_across_teams():
    m = grid_match()
    p = CorruptParams(swap_rate_per_team_min=0, flip_rate_per_min=5.0, flip_lo_s=2, flip_hi_s=2,
                      dropout_iid=0, noise_sigma_m=0.0, **NO_OUT)
    out = corrupt(m, np.zeros(m.n_frames, bool), p, np.random.default_rng(1))
    # somewhere a home slot (x < 0 originally) carries an away position (x >= 4) for ~50 frames
    home_x = out.pos[:, :3, 0]
    assert (home_x >= 4).any() and (home_x >= 4).sum() <= 1500
    assert (out.pos[:, 3:, 0] <= -4).any()


def test_dropout_and_edge_dropout():
    m = grid_match()
    p = CorruptParams(swap_rate_per_team_min=0, flip_rate_per_min=0, dropout_iid=0.1, noise_sigma_m=0.0,
                      edge_dropout_m=100.0, edge_dropout_s=1.0, **NO_OUT)
    mask = np.zeros(m.n_frames, bool); mask[1000:1500] = True
    out = corrupt(m, mask, p, np.random.default_rng(2))
    assert 0.05 < np.isnan(out.pos[..., 0]).mean() < 0.2
    assert np.isnan(out.pos[1000:1025, :, 0]).all() and np.isnan(out.pos[1475:1500, :, 0]).all()


def test_same_team_swap_persists():
    m = grid_match()
    pos = m.pos.copy()
    pos[:, 1, 0] = -19.0                      # slots 0 and 1 are 1 m apart, both home
    m = replace(m, pos=pos, speed=speeds_from_positions(pos, 25.0))
    p = CorruptParams(swap_rate_per_team_min=30, swap_dist_m=1.5, flip_rate_per_min=0,
                      dropout_iid=0, noise_sigma_m=0, **NO_OUT)
    out = corrupt(m, np.zeros(m.n_frames, bool), p, np.random.default_rng(0))
    changed = np.flatnonzero(out.pos[:, 0, 0] != m.pos[:, 0, 0])
    assert len(changed) > 0
    f0 = changed[0]
    assert out.pos[f0, 0, 0] == m.pos[f0, 1, 0]
    # with many swaps the pair toggles, so persistence = piecewise-constant runs (no flicker),
    # and the pair of identities is only ever exchanged, never lost
    assert np.all(out.pos[:, 0, 0] + out.pos[:, 1, 0] == -39.0)
    assert np.count_nonzero(np.diff(out.pos[f0:, 0, 0])) <= 150


def test_edge_dropout_only_near_ball():
    m = grid_match()
    ball = m.ball.copy()
    ball[:, 0] = -20.0
    ball[1000:1500] = np.nan
    ball[1500:, 0] = 20.0
    m = replace(m, ball=ball)
    p = CorruptParams(swap_rate_per_team_min=0, flip_rate_per_min=0, dropout_iid=0, noise_sigma_m=0,
                      edge_dropout_m=5.0, edge_dropout_s=1.0, **NO_OUT)
    mask = np.zeros(m.n_frames, bool); mask[1000:1500] = True
    out = corrupt(m, mask, p, np.random.default_rng(0))
    nan = np.isnan(out.pos[..., 0])
    assert nan[1000:1025, 0].all() and nan[1475:1500, 5].all()
    assert not nan[1000:1025, 3].any() and not nan[1475:1500, 3].any()


from mtah.sim.corrupt import dead_ball_realism


def test_dead_ball_realism_moves_ball_inside_pitch():
    F = 25 * 60
    labels = np.full(F, "H", dtype="<U1"); labels[500:1000] = "D"
    ball = np.zeros((F, 2), np.float32); ball[500:1000, 0] = 53.0      # static, 0.5 m outside the goal line
    ball[1000:, 0] = 0.0
    p = CorruptParams(dead_out_keep=0.0)
    out = dead_ball_realism(ball, labels, (105.0, 68.0), 25, p, np.random.default_rng(0))
    assert np.array_equal(out[:500], ball[:500]) and np.array_equal(out[1000:], ball[1000:])
    seg = out[500:1000]
    assert np.abs(seg[:, 0]).max() <= 52.5 and np.abs(seg[:, 1]).max() <= 34.0
    sp = np.hypot(*np.diff(seg, axis=0).T) * 25
    assert np.median(sp) > 0.5                     # no longer static
    assert np.array_equal(out, dead_ball_realism(ball, labels, (105.0, 68.0), 25, p, np.random.default_rng(0)))


def test_dead_ball_realism_keeps_nan_and_out_runs():
    F = 300
    labels = np.full(F, "D", dtype="<U1")
    ball = np.full((F, 2), np.nan, np.float32); ball[:100, 0] = 54.0
    out = dead_ball_realism(ball, labels, (105.0, 68.0), 25, CorruptParams(dead_out_keep=1.0), np.random.default_rng(1))
    assert np.array_equal(np.isnan(out), np.isnan(ball))
    assert np.abs(out[:100, 0] - ball[:100, 0]).max() < 1.0 and np.median(out[:100, 0]) > 52.5   # kept: still out of pitch


def test_kept_dead_run_is_jittered_not_static():
    from mtah.io.match import speeds_from_positions
    F = 25 * 20
    labels = np.full(F, "D", dtype="<U1")
    ball = np.zeros((F, 2), np.float32); ball[:, 0] = 54.0             # static DFL ball, out of pitch
    out = dead_ball_realism(ball, labels, (105.0, 68.0), 25, CorruptParams(dead_out_keep=1.0), np.random.default_rng(2))
    sp = speeds_from_positions(out, 25)[1:-1]
    win = np.lib.stride_tricks.sliding_window_view(sp < 0.5, 25)
    assert not win.all(axis=1).any()                                   # no 1 s static stretch remains


def _carry_step(params, seed):
    F = 25 * 30
    labels = np.full(F, "H", dtype="<U1"); labels[25:25 * 20] = "D"
    ball = np.zeros((F, 2), np.float32); ball[:25 * 20, 0] = -45.0; ball[25 * 20:, 0] = 45.0
    out = dead_ball_realism(ball, labels, (105.0, 68.0), 25, params, np.random.default_rng(seed))
    return float(np.median(np.hypot(*np.diff(out[25:25 + 50], axis=0).T)) * 25)   # m/s over the first 2 s


def test_fast_carry_when_dead_fast_p_is_one():
    fast = [_carry_step(CorruptParams(dead_out_keep=0.0, dead_fast_p=1.0, dead_jitter_m=0.0), s) for s in range(20)]
    assert all(8.0 - 1e-3 <= v <= 15.0 + 1e-3 for v in fast) and max(fast) - min(fast) > 2.0
    slow = [_carry_step(CorruptParams(dead_out_keep=0.0, dead_fast_p=0.0, dead_jitter_m=0.0, dead_v_hi=6.0), s)
            for s in range(20)]
    assert all(1.0 - 1e-3 <= v <= 6.0 + 1e-3 for v in slow)


def test_dead_defaults_and_config_in_sync():
    import json
    from pathlib import Path
    p = CorruptParams()
    assert (p.dead_v_hi, p.dead_fast_p, p.dead_out_keep, p.dead_keep_jitter) == (12.0, 0.15, 0.03, True)
    cfg = json.loads((Path(__file__).resolve().parents[1] / "configs" / "corrupt.json").read_text())
    assert CorruptParams.from_dict(cfg) == p


def _plan2b_dead_ball(ball, labels, pitch, fps, rng):
    """Frozen copy of the Plan 2b dead_ball_realism (default parameters) as the replay reference."""
    out = ball.copy(); L, W = pitch; half = np.array([L / 2 - 0.5, W / 2 - 0.5])
    for s, e in runs_of(np.asarray(labels) == "D"):
        vis = np.flatnonzero(~np.isnan(ball[s:e, 0])) + s
        if not vis.size or rng.random() < 0.1:
            continue
        start = np.clip(ball[vis[0]], -half, half)
        after = np.flatnonzero(~np.isnan(ball[e:, 0]))
        target = np.clip(ball[e + after[0]], -half, half) if after.size else np.zeros(2)
        v = rng.uniform(1.0, 6.0) / fps
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
                ang = rng.uniform(0, 2 * np.pi); step = rng.uniform(0.5, 2.0) / fps
                pos = pos + step * np.array([np.cos(ang), np.sin(ang)])
                pos = np.where(np.abs(pos) > half, np.sign(pos) * half - (pos - np.sign(pos) * half), pos)
            out[f] = np.clip(pos + rng.normal(0, 0.15, 2), -half, half)
    return out.astype(np.float32)


def test_plan2b_params_replay_the_plan2b_stream():
    from mtah.sim.corrupt import PLAN2B_DEAD
    g = np.random.default_rng(7)
    F = 25 * 400
    labels = np.where(g.random(F // 250).repeat(250) < 0.4, "D", "H").astype("<U1")
    ball = (g.uniform(-56, 56, (F // 250, 2)) * [1, 0.65]).repeat(250, axis=0).astype(np.float32)
    ball[g.random(F) < 0.05] = np.nan; ball[3000:3400] = np.nan
    r_old, r_new = np.random.default_rng(11), np.random.default_rng(11)
    a = _plan2b_dead_ball(ball, labels, (105.0, 68.0), 25, r_old)
    b = dead_ball_realism(ball, labels, (105.0, 68.0), 25, PLAN2B_DEAD, r_new)
    assert np.array_equal(a, b, equal_nan=True)
    assert r_old.bit_generator.state == r_new.bit_generator.state


from mtah.sim.blackout_sim import runs_of
from mtah.sim.corrupt import player_outages


def _bo(F, s, e):
    m = np.zeros(F, bool); m[s:e] = True
    return m


def test_player_outages_deterministic_for_seed():
    F, S = 25 * 600, 22
    bo = _bo(F, 3000, 6000)
    p = CorruptParams()
    a = player_outages(F, S, 25.0, bo, p, np.random.default_rng(5))
    b = player_outages(F, S, 25.0, bo, p, np.random.default_rng(5))
    assert a.shape == (F, S) and a.dtype == bool
    assert a.any() and np.array_equal(a, b)


def test_player_outages_zero_rates_all_false():
    F, S = 25 * 600, 22
    p = CorruptParams(outage_rate_out_per_player_min=0.0, outage_rate_in_per_player_min=0.0)
    out = player_outages(F, S, 25.0, _bo(F, 3000, 6000), p, np.random.default_rng(0))
    assert not out.any()


def test_player_outages_only_outside_blackouts_when_inside_rate_zero():
    F, S = 25 * 600, 22
    bo = _bo(F, 3000, 9000)
    p = CorruptParams(outage_rate_out_per_player_min=60.0, outage_rate_in_per_player_min=0.0,
                      outage_median_s=0.04, outage_max_s=0.04)       # one-frame outages
    out = player_outages(F, S, 25.0, bo, p, np.random.default_rng(1))
    assert out[~bo].mean() > 0.02                     # ~1 start per player-second
    assert not out[bo].any()


def test_player_outage_durations_capped():
    F, S = 25 * 600, 50
    p = CorruptParams(outage_rate_out_per_player_min=0.05, outage_rate_in_per_player_min=0.05,
                      outage_median_s=60.0, outage_sigma_log=0.5, outage_max_s=2.0)
    out = player_outages(F, S, 25.0, np.zeros(F, bool), p, np.random.default_rng(2))
    lens = np.array([e - s for j in range(S) for s, e in runs_of(out[:, j])])
    assert len(lens) >= 10
    assert lens.max() <= 50 and np.median(lens) == 50


def test_corrupt_outages_make_long_nan_runs():
    m = grid_match(F=25 * 600)
    p = CorruptParams(swap_rate_per_team_min=0, flip_rate_per_min=0, dropout_iid=0, noise_sigma_m=0,
                      outage_rate_out_per_player_min=0.5, outage_rate_in_per_player_min=0.5)
    out = corrupt(m, np.zeros(m.n_frames, bool), p, np.random.default_rng(3))
    nan = np.isnan(out.pos[..., 0])
    lens = [e - s for j in range(m.n_slots) for s, e in runs_of(nan[:, j])]
    assert max(lens) > 25


# ---- mass outages ----
from mtah.sim.corrupt import mass_outages


def _mo(p, mask, present=None, seed=0, **kw):
    present = np.ones(22, bool) if present is None else present
    prm = CorruptParams(mass_outage_p=p, **kw)
    return mass_outages(present, len(mask), 25.0, mask, prm, np.random.default_rng(seed))


def _mask(F=25 * 60, s=25 * 10, n=25 * 10):
    m = np.zeros(F, bool); m[s:s + n] = True
    return m


def test_mass_outage_p0_all_false():
    assert not _mo(0.0, _mask()).any()


def test_mass_outage_p1_exact_count_on_run_only():
    mask = _mask()
    out = _mo(1.0, mask)
    assert out.shape == (len(mask), 22)
    assert (out[mask].sum(axis=1) == round(0.45 * 22)).all()
    assert (out[mask] == out[mask][0]).all()      # same slots for the whole run
    assert not out[~mask].any()


def test_mass_outage_short_run_untouched():
    assert not _mo(1.0, _mask(n=int(4.9 * 25))).any()


def test_mass_outage_deterministic():
    mask = _mask()
    assert np.array_equal(_mo(1.0, mask, seed=3), _mo(1.0, mask, seed=3))
    assert not np.array_equal(_mo(1.0, mask, seed=3), _mo(1.0, mask, seed=4))


def test_mass_outage_never_picks_absent_slots():
    present = np.ones(22, bool); present[:6] = False
    out = _mo(1.0, _mask(), present=present)
    assert not out[:, :6].any()
    assert out[_mask()].sum(axis=1)[0] == round(0.45 * 16)


def test_spawned_dead_rng_leaves_the_half_stream_alone():
    a, b = np.random.default_rng([100, 3]), np.random.default_rng([100, 3])
    child = a.spawn(1)[0]; child.random(1000)
    assert a.random() == b.random()
