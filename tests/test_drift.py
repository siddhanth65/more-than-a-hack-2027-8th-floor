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


def test_blackout_mask_from_match():
    from mtah.eval.drift import blackout_mask_from_match
    m = two_player_match(F=25 * 120); m.ball[:] = 0.0
    m.ball[25 * 20:25 * 35] = np.nan      # 15 s gap -> masked
    m.ball[25 * 60:25 * 65] = np.nan      # 5 s gap -> not masked
    mask = blackout_mask_from_match(m)
    assert mask[25 * 20:25 * 35].all() and not mask[25 * 60:25 * 65].any()
    assert mask.sum() == 25 * 15 - int((~m.graded[25 * 20:25 * 35]).sum())


def _implied_setup():
    m = two_player_match(F=25 * 120); m.ball[:] = 0.0
    labels = np.full(m.n_frames, "A", dtype="<U1")
    return m, labels, np.zeros(m.n_frames, bool)


def test_implied_agreement_sign_minus_counts_other_team():
    m, labels, mask = _implied_setup()
    tags = pd.DataFrame([{"start": 25 * 10, "end": 25 * 30, "code": "DEF", "team": "H", "type": None, "half": 1}])
    table = {"DEF": {"sign": -1, "p": 0.8}}
    r = drift_report(labels, m, tags, mask, code_table=table)
    assert abs(r["tag_agreement_implied"] - 1.0) < 1e-9
    assert r["tag_agreement"] == 0.0


def test_implied_agreement_skips_sign_zero_and_unknown():
    m, labels, mask = _implied_setup()
    tags = pd.DataFrame([{"start": 0, "end": 100, "code": "NEUTRAL", "team": "H", "type": None, "half": 1},
                         {"start": 200, "end": 300, "code": "MYSTERY", "team": "H", "type": None, "half": 1}])
    table = {"NEUTRAL": {"sign": 0, "p": 0.5}}
    r = drift_report(labels, m, tags, mask, code_table=table)
    assert np.isnan(r["tag_agreement_implied"])
    assert r["tag_agreement"] == 0.0


def test_implied_agreement_default_table_loaded():
    m, labels, mask = _implied_setup()
    tags = pd.DataFrame([{"start": 0, "end": 100, "code": "BUILD UP", "team": "A", "type": None, "half": 1}])
    r = drift_report(labels, m, tags, mask)
    assert "tag_agreement_implied" in r


def test_d_share_split_by_blackout():
    m, _, _ = _implied_setup()
    labels = np.full(m.n_frames, "H", dtype="<U1")
    labels[25 * 10:25 * 20] = "D"      # outside blackout
    labels[25 * 60:25 * 70] = "D"      # inside blackout
    mask = np.zeros(m.n_frames, bool); mask[25 * 60:25 * 80] = True
    g = m.graded
    r = drift_report(labels, m, pd.DataFrame(columns=["start", "end", "code", "team", "type", "half"]), mask)
    exp_in = (labels[g & mask] == "D").mean(); exp_out = (labels[g & ~mask] == "D").mean()
    assert abs(r["blackout_D_share"] - exp_in) < 1e-9 and abs(r["non_blackout_D_share"] - exp_out) < 1e-9
    assert r["blackout_D_share"] > r["non_blackout_D_share"] > 0


def test_existing_keys_still_present():
    m, labels, mask = _implied_setup()
    r = drift_report(labels, m, pd.DataFrame(columns=["start", "end", "code", "team", "type", "half"]), mask, code_table={})
    for k in ("n_spells", "spells_per_HA_min", "pred_D_share", "spell_len_median_s",
              "share_changes_in_blackout", "D_starts_near_sp_share", "tag_agreement"):
        assert k in r
