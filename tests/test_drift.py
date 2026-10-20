import numpy as np
import pandas as pd

from mtah.eval.drift import drift_report
from tests.test_baselines import two_player_match


def test_drift_keys():
    m = two_player_match(F=25 * 120); m.ball[:] = 0.0
    labels = np.full(m.n_frames, "H", dtype="<U1"); labels[25 * 40:25 * 60] = "A"; labels[25 * 60:25 * 70] = "D"
    tags = pd.DataFrame([{"start": 25 * 41, "end": 25 * 59, "code": "BUILD UP", "team": "A", "type": None, "half": 1},
                         {"start": 25 * 59, "end": 25 * 90, "code": "SET PIECES", "team": None, "type": "Throw-in", "half": 1}])
    mask = np.zeros(m.n_frames, bool); mask[25 * 45:25 * 65] = True
    r = drift_report(labels, m, tags, mask)
    assert r["n_spells"] == 3 and abs(r["tag_agreement"] - 1.0) < 1e-6
    assert r["D_starts_near_sp_share"] == 1.0 and r["share_changes_in_blackout"] == 0.0
    assert 0 < r["pred_D_share"] < 0.2


def test_drift_nan_when_no_changes_or_d_runs():
    m = two_player_match(F=25 * 120); m.ball[:] = 0.0
    labels = np.full(m.n_frames, "H", dtype="<U1")
    tags = pd.DataFrame([{"start": 0, "end": 10, "code": "SET PIECES", "team": None, "type": "Throw-in", "half": 1}])
    r = drift_report(labels, m, tags, np.zeros(m.n_frames, bool))
    assert np.isnan(r["share_changes_in_blackout"]) and np.isnan(r["D_starts_near_sp_share"])
