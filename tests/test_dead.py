import pytest
import warnings

import numpy as np
import pandas as pd

from mtah.evidence.dead import DEAD_FEATURES, DeadModel, dead_evidence, dead_features, direction_of_play
from mtah.io.match import Match, speeds_from_positions
from tests.test_baselines import two_player_match


def test_feature_table_shape_and_gap_clock():
    m = two_player_match(F=25 * 60); m.ball[:] = 0.0; m.ball[25 * 20:25 * 30, :] = np.nan
    tags = pd.DataFrame([{"start": 25 * 28, "end": 25 * 40, "code": "SET PIECES", "team": None, "type": "Throw-in", "half": 1}])
    X = dead_features(m, tags)
    assert X.shape == (m.n_frames, len(DEAD_FEATURES)) and X.dtype == np.float32
    f = {n: i for i, n in enumerate(DEAD_FEATURES)}
    assert X[25 * 25, f["ball_present"]] == 0 and abs(X[25 * 25, f["since_gap_start_s"]] - 5.0) < 0.05
    assert abs(X[25 * 25, f["gap_len_s"]] - 10.0) < 0.05 and X[25 * 10, f["gap_len_s"]] == 0
    assert X[25 * 29, f["sp_inside"]] == 1 and X[25 * 29, f["sp_type"]] == 1
    assert abs(X[25 * 26, f["sp_dist_s"]] - 2.0) < 0.05
    assert abs(X[25 * 25, f["until_gap_end_s"]] - (25 * 30 - 1 - 25 * 25) / 25) < 1e-4
    assert abs(X[25 * 30, f["sp_dist_s"]] - (-2.0)) < 0.05      # after the set-piece start: negative


def test_direction_of_play():
    m = two_player_match(F=100)        # home at x=-10, away at x=+10
    assert direction_of_play(m)[50].tolist() == [1, -1]


def test_model_learns_simple_rule():
    rng = np.random.default_rng(0)
    X = np.zeros((2000, len(DEAD_FEATURES)), np.float32)
    X[:, 0] = rng.integers(0, 2, 2000)              # ball_present
    y = (X[:, 0] == 0).astype(int)                   # dead iff ball absent
    model = DeadModel().fit(X, y)
    p = model.predict_proba(X)
    assert (p[y == 1] > 0.8).mean() > 0.95 and (p[y == 0] < 0.2).mean() > 0.95
    ev = dead_evidence(p)
    assert ev.shape == (2000, 3) and np.all(ev[:, 0] == ev[:, 1])


NO_TAGS = pd.DataFrame(columns=["start", "end", "code", "team", "type", "half"])
FI = {n: i for i, n in enumerate(DEAD_FEATURES)}


def make_match(pos, team, ball=None, half=None, pitch=(105.3, 68.0)):
    F = pos.shape[0]
    ball = np.zeros((F, 2), np.float32) if ball is None else ball
    half = np.ones(F, np.int8) if half is None else half
    return Match(fps=25.0, pos=pos.astype(np.float32), speed=speeds_from_positions(pos.astype(np.float32), 25.0),
                 team=np.asarray(team, np.int8), ball=ball.astype(np.float32),
                 ball_speed=speeds_from_positions(ball.astype(np.float32), 25.0), half=half, pitch=pitch)


def test_no_runtime_warnings_with_empty_frames():
    m = two_player_match(F=200); m.pos[50:120] = np.nan; m.speed[50:120] = np.nan
    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)
        dead_features(m, NO_TAGS); direction_of_play(m)


def test_direction_of_play_per_half():
    F = 200; pos = np.zeros((F, 2, 2)); pos[:100, 0, 0] = -10; pos[:100, 1, 0] = 10
    pos[100:, 0, 0] = 10; pos[100:, 1, 0] = -10
    half = np.array([1] * 100 + [2] * 100, np.int8)
    d = direction_of_play(make_match(pos, [0, 1], half=half))
    assert d.shape == (F, 2) and d.dtype == np.int8
    assert d[50].tolist() == [1, -1] and d[150].tolist() == [-1, 1]


def test_player_speed_ratio_nan_not_inf():
    m = two_player_match(F=300)
    assert np.isnan(dead_features(m, NO_TAGS)[:, FI["player_speed_ratio"]]).all()
    pos = np.zeros((300, 2, 2)); pos[:, :, 0] = [[-10, 10]]
    pos[100:, 0, 0] += np.arange(200) * 0.2          # idle, then moves: earlier mean 0
    r = dead_features(make_match(pos, [0, 1]), NO_TAGS)[:, FI["player_speed_ratio"]]
    assert not np.isinf(r).any()
    short = dead_features(make_match(pos[:20], [0, 1]), NO_TAGS)[:, FI["player_speed_ratio"]]
    assert np.isnan(short).all()


def test_entry_line_dist_and_outward():
    F = 400; ball = np.zeros((F, 2)); ball[:, 0] = 0
    ball[194:200, 0] = np.linspace(40, 52.15, 6); ball[200:260] = np.nan
    pos = np.zeros((F, 2, 2)); pos[:, 0, 0] = -10; pos[:, 1, 0] = 10
    X = dead_features(make_match(pos, [0, 1], ball=ball), NO_TAGS)
    assert abs(X[230, FI["entry_line_dist"]] - 0.5) < 0.01 and X[230, FI["entry_outward"]] == 1


def test_touchline_isolated():
    pos = np.zeros((50, 2, 2)); pos[:, 0] = [0, 33.5]; pos[:, 1] = [0, 0]
    assert dead_features(make_match(pos, [0, 1]), NO_TAGS)[10, FI["touchline_isolated"]] == 1
    pos[:, 1] = [0, 31]
    assert dead_features(make_match(pos, [0, 1]), NO_TAGS)[10, FI["touchline_isolated"]] == 0


def test_corner_cluster():
    pos = np.zeros((50, 8, 2)); pos[:, :, 0] = 40 + np.arange(8); pos[:, :, 1] = 30
    X = dead_features(make_match(pos, [0, 0, 0, 0, 1, 1, 1, 1]), NO_TAGS)
    assert X[10, FI["corner_cluster"]] == 8


def test_kickoff_shape():
    pos = np.zeros((50, 16, 2)); pos[:, :8, 0] = -30 + np.arange(8) * 3; pos[:, 8:, 0] = 5 + np.arange(8) * 3
    pos[:, :, 1] = np.linspace(-20, 20, 16)
    team = [0] * 8 + [1] * 8
    assert dead_features(make_match(pos, team), NO_TAGS)[10, FI["kickoff_shape"]] == 1
    pos[:, 0, 0] = 20.0                      # a home player in the opposition half
    assert dead_features(make_match(pos, team), NO_TAGS)[10, FI["kickoff_shape"]] == 0


def test_model_save_load_roundtrip(tmp_path):
    rng = np.random.default_rng(1)
    X = rng.normal(size=(600, len(DEAD_FEATURES))).astype(np.float32); y = (X[:, 1] > 0).astype(int)
    model = DeadModel().fit(X, y); model.save(tmp_path / "m.joblib")
    assert np.array_equal(DeadModel.load(tmp_path / "m.joblib").predict_proba(X), model.predict_proba(X))


def test_player_speed_mean_ignores_nonfinite_speed():
    pos = np.zeros((400, 2, 2)); pos[:, :, 0] = [[-10, 10]]
    m = make_match(pos, [0, 1])
    m.speed[:] = 1.0; m.speed[375:, 0] = 2.0; m.speed[375:, 1] = np.nan      # slot 1 tracked but speed unknown
    r = dead_features(m, NO_TAGS)[:, FI["player_speed_ratio"]]
    assert r[399] == pytest.approx(2.0, rel=1e-3)    # NaN speeds no longer count as 0 m/s


def test_apply_policy_zeroes_columns():
    from mtah.evidence.dead import FEATURE_POLICIES, apply_policy
    X = np.ones((5, len(DEAD_FEATURES)), np.float32)
    Y = apply_policy(X, "no_ball_cues")
    idx = [DEAD_FEATURES.index(n) for n in FEATURE_POLICIES["no_ball_cues"]]
    assert (Y[:, idx] == 0).all() and (np.delete(Y, idx, axis=1) == 1).all() and (X == 1).all()
    assert np.array_equal(apply_policy(X, "full"), X)


def test_dead_model_policy_travels_with_model(tmp_path):
    from mtah.evidence.dead import FEATURE_POLICIES
    rng = np.random.default_rng(0)
    X = rng.normal(size=(600, len(DEAD_FEATURES))).astype(np.float32)
    y = (X[:, 5] > 0).astype(int)
    idx = [DEAD_FEATURES.index(n) for n in FEATURE_POLICIES["no_ball_cues"]]
    m = DeadModel(policy="no_ball_cues").fit(X, y)
    X2 = X.copy(); X2[:, idx] = rng.normal(size=(600, len(idx))) * 50
    assert np.array_equal(m.predict_proba(X), m.predict_proba(X2))
    p = tmp_path / "m.joblib"; m.save(p)
    assert DeadModel.load(p).policy == "no_ball_cues"
    assert np.array_equal(DeadModel.load(p).predict_proba(X2), m.predict_proba(X))


def test_load_old_model_file_defaults_to_full(tmp_path):
    import joblib
    rng = np.random.default_rng(0)
    X = rng.normal(size=(400, len(DEAD_FEATURES))).astype(np.float32)
    m = DeadModel().fit(X, (X[:, 0] > 0).astype(int))
    joblib.dump(m.model, tmp_path / "old.joblib")           # old format: bare sklearn model
    assert DeadModel.load(tmp_path / "old.joblib").policy == "full"
