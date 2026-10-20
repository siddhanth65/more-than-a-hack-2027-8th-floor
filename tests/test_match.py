import numpy as np
import pytest

from mtah.io.match import Match, distance_to_line, half_runs, speeds_from_positions


def tiny_match():
    F, S = 10, 4
    pos = np.zeros((F, S, 2), np.float32)
    pos[:, 0, 0] = np.arange(F)  # slot 0 moves 1 m/frame along x
    pos[:, 1, :] = np.nan
    ball = np.zeros((F, 2), np.float32)
    ball[5:, :] = np.nan
    half = np.array([0, 0, 1, 1, 1, 1, 2, 2, 2, 0], np.int8)
    return Match(fps=25.0, pos=pos, speed=speeds_from_positions(pos, 25.0),
                 team=np.array([0, 0, 1, 1], np.int8), ball=ball,
                 ball_speed=speeds_from_positions(ball, 25.0), half=half, pitch=(105.3, 68.0))


def test_shapes_and_masks():
    m = tiny_match()
    assert m.n_frames == 10 and m.n_slots == 4
    assert m.graded.tolist() == [False, False, True, True, True, True, True, True, True, False]
    assert m.ball_present.sum() == 5


def test_speed_central_difference():
    m = tiny_match()
    # slot 0: 1 m per frame at 25 fps = 25 m/s in the interior, NaN at the ends
    assert np.isnan(m.speed[0, 0]) and np.isnan(m.speed[-1, 0])
    assert m.speed[5, 0] == pytest.approx(25.0)
    assert np.isnan(m.speed[:, 1]).all()


def test_half_runs():
    m = tiny_match()
    assert half_runs(m.half) == [(1, 2, 6), (2, 6, 9)]


def test_distance_to_line():
    d = distance_to_line(np.array([[0.0, 0.0], [50.0, 0.0], [0.0, 33.0]]), (105.0, 68.0))
    assert d.tolist() == pytest.approx([34.0, 2.5, 1.0])


def test_half_runs_rejects_non_contiguous():
    with pytest.raises(ValueError):
        half_runs(np.array([1, 1, 0, 1, 1], np.int8))


def test_half_runs_empty():
    assert half_runs(np.zeros(5, np.int8)) == []


def test_match_repr_is_data_safe():
    r = repr(tiny_match())
    assert "F=10" in r and "S=4" in r and "array" not in r
