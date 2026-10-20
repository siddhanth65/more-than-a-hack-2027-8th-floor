from dataclasses import replace

import numpy as np

from mtah.io.match import Match, speeds_from_positions
from mtah.sim.corrupt import CorruptParams, corrupt


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
    p = CorruptParams(swap_rate_per_team_min=0, flip_rate_per_min=0, dropout_iid=0, noise_sigma_m=0.3)
    mask = np.zeros(m.n_frames, bool)
    a = corrupt(m, mask, p, np.random.default_rng(0)); b = corrupt(m, mask, p, np.random.default_rng(0))
    assert np.array_equal(np.nan_to_num(a.pos), np.nan_to_num(b.pos))
    err = np.nanstd(a.pos - m.pos)
    assert 0.15 < err < 0.5
    assert not np.isnan(a.pos).any()


def test_flip_run_swaps_across_teams():
    m = grid_match()
    p = CorruptParams(swap_rate_per_team_min=0, flip_rate_per_min=5.0, flip_lo_s=2, flip_hi_s=2,
                      dropout_iid=0, noise_sigma_m=0.0)
    out = corrupt(m, np.zeros(m.n_frames, bool), p, np.random.default_rng(1))
    # somewhere a home slot (x < 0 originally) carries an away position (x >= 4) for ~50 frames
    home_x = out.pos[:, :3, 0]
    assert (home_x >= 4).any() and (home_x >= 4).sum() <= 1500
    assert (out.pos[:, 3:, 0] <= -4).any()


def test_dropout_and_edge_dropout():
    m = grid_match()
    p = CorruptParams(swap_rate_per_team_min=0, flip_rate_per_min=0, dropout_iid=0.1, noise_sigma_m=0.0,
                      edge_dropout_m=100.0, edge_dropout_s=1.0)
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
                      dropout_iid=0, noise_sigma_m=0)
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
                      edge_dropout_m=5.0, edge_dropout_s=1.0)
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
    assert np.array_equal(np.nan_to_num(out), np.nan_to_num(ball))
