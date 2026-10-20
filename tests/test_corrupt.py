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
    assert (home_x >= 4).any() and (home_x >= 4).sum() <= 3 * 60 * 25


def test_dropout_and_edge_dropout():
    m = grid_match()
    p = CorruptParams(swap_rate_per_team_min=0, flip_rate_per_min=0, dropout_iid=0.1, noise_sigma_m=0.0,
                      edge_dropout_m=100.0, edge_dropout_s=1.0)
    mask = np.zeros(m.n_frames, bool); mask[1000:1500] = True
    out = corrupt(m, mask, p, np.random.default_rng(2))
    assert 0.05 < np.isnan(out.pos[..., 0]).mean() < 0.2
    assert np.isnan(out.pos[1000:1025, :, 0]).all() and np.isnan(out.pos[1475:1500, :, 0]).all()
