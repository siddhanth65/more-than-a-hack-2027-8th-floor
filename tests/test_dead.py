import numpy as np
import pandas as pd

from mtah.evidence.dead import DEAD_FEATURES, DeadModel, dead_evidence, dead_features, direction_of_play
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


def test_direction_of_play():
    m = two_player_match(F=100)        # home at x=-10, away at x=+10
    assert direction_of_play(m).tolist() == [1, -1]


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
