import numpy as np

from mtah.evidence.touches import TouchParams, detect_touches, strong_touch_frames, velocity_change
from mtah.io.match import Match, speeds_from_positions


def kick_match(F=100):
    pos = np.zeros((F, 2, 2), np.float32); pos[:, 0, 0] = 0.0; pos[:, 1, 0] = 30.0
    ball = np.zeros((F, 2), np.float32)
    ball[50:, 0] = np.arange(50) * 0.8     # ball at rest next to slot 0, then kicked at 20 m/s
    team = np.array([0, 1], np.int8)
    return Match(fps=25.0, pos=pos, speed=speeds_from_positions(pos, 25.0), team=team, ball=ball,
                 ball_speed=speeds_from_positions(ball, 25.0), half=np.ones(F, np.int8), pitch=(105.0, 68.0))


def test_velocity_change_peak():
    m = kick_match()
    dv = velocity_change(m.ball, 3, 25.0)
    assert np.nanargmax(dv) in (49, 50, 51, 52) and np.nanmax(dv) > 15


def test_detect_touch_by_home():
    m = kick_match()
    t = detect_touches(m, TouchParams())
    assert len(t) == 1 and t[0, 1] == 0 and t[0, 2] == 0 and t[0, 3] > 15
    strong = strong_touch_frames(t, 0, TouchParams(), m.n_frames)
    assert strong.sum() == 1 and not strong_touch_frames(t, 1, TouchParams(), m.n_frames).any()


def test_continuity_gate_drops_teleport():
    m = kick_match()
    m.ball[52, 0] += 10.0                   # a 10 m jump next to the touch
    assert len(detect_touches(m, TouchParams())) == 0
