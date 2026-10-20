import numpy as np
import pytest

from mtah.evidence.ball_ctrl import BallCtrlParams, ball_ctrl_evidence, ball_ctrl_posteriors, ball_segments
from mtah.io.match import Match, speeds_from_positions


def carry_match(F=200):
    """Home slot 0 carries the ball (ball glued 0.3 m ahead) for 100 frames, then away slot 2 does."""
    pos = np.zeros((F, 4, 2), np.float32)
    pos[:, 0, 0] = np.linspace(-20, 0, F); pos[:, 1, 0] = -30
    pos[:, 2, 0] = np.linspace(20, 0, F); pos[:, 3, 0] = 30
    ball = np.zeros((F, 2), np.float32)
    ball[:100, 0] = pos[:100, 0, 0] + 0.3
    ball[100:, 0] = pos[100:, 2, 0] - 0.3
    team = np.array([0, 0, 1, 1], np.int8)
    return Match(fps=25.0, pos=pos, speed=speeds_from_positions(pos, 25.0), team=team, ball=ball,
                 ball_speed=speeds_from_positions(ball, 25.0), half=np.ones(F, np.int8), pitch=(105.0, 68.0))


def test_segments():
    bp = np.array([1, 1, 0, 1, 1, 1, 0, 0, 1], bool)
    assert ball_segments(bp, 2) == [(0, 2), (3, 6)]


def test_posteriors_follow_carrier():
    m = carry_match()
    pt, pl = ball_ctrl_posteriors(m, BallCtrlParams())
    assert pt[20, 0] > 0.8 and pt[150, 1] > 0.8
    assert np.all(np.abs(pt.sum(1) + pl - 1) < 1e-4)


def test_evidence_shape_and_nan_on_gap():
    m = carry_match()
    m.ball[50:60, :] = np.nan
    ev = ball_ctrl_evidence(m, BallCtrlParams())
    assert ev.shape == (200, 3) and ev.dtype == np.float32
    assert np.isnan(ev[55]).all() and np.isnan(ev[:, 2]).all()
    assert ev[20, 0] > ev[20, 1] and ev[150, 1] > ev[150, 0]
