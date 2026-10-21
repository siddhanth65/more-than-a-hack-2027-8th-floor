import warnings
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from mtah.evidence.team_ctrl import (FEATURE_POLICIES, MIRROR_PERM, MIRROR_SIGN, TEAM_FEATURES, TeamCtrlModel,
                                     apply_policy, mirror_features, team_ctrl_evidence, team_features)
from tests.test_ball_ctrl import carry_match
from tests.test_select_smoke import _fake_half

FI = {n: i for i, n in enumerate(TEAM_FEATURES)}


def test_feature_shape_dtype_and_nan_rows():
    m = carry_match(F=300); m.ball[200:240] = np.nan
    X = team_features(m)
    assert X.shape == (300, len(TEAM_FEATURES)) and X.dtype == np.float32
    assert np.isnan(X[200:240]).all()                       # ball absent: no opinion at all
    assert not np.isnan(X[np.r_[0:199, 241:300]]).all(1).any()   # ball present: never an all-NaN row
    # home slot 0 carries until frame 100 (ball 0.3 m ahead), then away slot 2 does
    assert abs(X[50, FI["H1_dist"]] - 0.3) < 0.05 and X[50, FI["nearest_team"]] == 1
    assert abs(X[150, FI["A1_dist"]] - 0.3) < 0.05 and X[150, FI["nearest_team"]] == -1
    assert X[50, FI["nearest_gap_m"]] > 0 > X[150, FI["nearest_gap_m"]]
    assert np.isnan(X[50, FI["H3_dist"]])                   # only two home players: 3rd nearest is missing
    assert X[50, FI["H1_dist"]] <= X[50, FI["H2_dist"]]
    t = X[:, [FI["touch_since_any_s"], FI["touch_since_H_s"], FI["touch_since_A_s"]]]
    ok = np.isfinite(X[:, FI["H1_dist"]])
    assert (t[ok] >= 0).all() and (t[ok] <= 5.0).all()


def test_feature_kinematics():
    F = 200; pos = np.zeros((F, 2, 2), np.float32)
    pos[:, 0, 0] = -40 + np.arange(F) * 0.2       # home runs +x at 5 m/s (own half: attacks +x)
    pos[:, 1, 0] = 30.0
    ball = np.zeros((F, 2), np.float32); ball[:, 0] = 0.0     # static ball: home closes at 5 m/s
    from mtah.io.match import Match, speeds_from_positions
    m = Match(fps=25.0, pos=pos, speed=speeds_from_positions(pos, 25.0), team=np.array([0, 1], np.int8), ball=ball,
              ball_speed=speeds_from_positions(ball, 25.0), half=np.ones(F, np.int8), pitch=(105.0, 68.0))
    X = team_features(m)
    assert abs(X[20, FI["H1_dd_dt"]] - (-5.0)) < 1e-3 and abs(X[20, FI["H1_speed"]] - 5.0) < 1e-3
    assert abs(X[20, FI["A1_dd_dt"]]) < 1e-3
    assert np.isnan(X[20, FI["H1_cos"]])            # static ball: the angle is undefined
    assert X[20, FI["touch_since_any_s"]] == 5.0    # no touch at all: capped
    ball2 = ball.copy(); ball2[:, 0] = np.arange(F) * 0.4    # ball runs +x at 10 m/s, same direction as home
    m2 = replace(m, ball=ball2, ball_speed=speeds_from_positions(ball2, 25.0))
    X2 = team_features(m2)
    assert abs(X2[20, FI["H1_cos"]] - 1.0) < 1e-4 and abs(X2[20, FI["ball_speed"]] - 10.0) < 1e-3
    assert abs(X2[20, FI["ball_accel"]]) < 1e-3
    # home lives at x<0 -> attacks +x; ball moving +x is "forward" for home
    assert X2[20, FI["ball_dir_H"]] > 0.99 and X2[20, FI["ball_dir_A"]] < -0.99


def test_mirror_is_an_involution_and_matches_a_team_swap():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(50, len(TEAM_FEATURES))).astype(np.float32); X[3] = np.nan; X[5, 7] = np.nan
    assert np.array_equal(mirror_features(mirror_features(X)), X, equal_nan=True)
    assert sorted(MIRROR_PERM) == list(range(len(TEAM_FEATURES))) and set(np.unique(MIRROR_SIGN)) <= {-1.0, 1.0}
    m = _fake_half(9001, 1, 100, np.random.default_rng(1)).match
    Xs = team_features(replace(m, team=(1 - m.team).astype(np.int8)))
    assert np.allclose(Xs, mirror_features(team_features(m)), equal_nan=True, atol=1e-5)


def test_policy_zeroes_touch_columns():
    X = np.ones((4, len(TEAM_FEATURES)), np.float32)
    Y = apply_policy(X, "no_touch")
    drop = [FI[n] for n in FEATURE_POLICIES["no_touch"]]
    assert drop and (Y[:, drop] == 0).all() and (np.delete(Y, drop, 1) == 1).all() and (X == 1).all()
    assert set(FEATURE_POLICIES) == {"full", "no_touch"}
    assert all(n.startswith("touch_") for n in FEATURE_POLICIES["no_touch"])
    with pytest.raises(KeyError):
        TeamCtrlModel(policy="bogus")


def _toy(n=3000, seed=0):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, len(TEAM_FEATURES))).astype(np.float32)
    X[:, FI["H1_dist"]] = rng.uniform(0, 10, n); X[:, FI["A1_dist"]] = rng.uniform(0, 10, n)
    y = (X[:, FI["H1_dist"]] < X[:, FI["A1_dist"]]).astype(int)   # home controls iff its nearest player is nearer
    return X, y


def test_model_learns_and_is_swap_antisymmetric(tmp_path):
    X, y = _toy()
    m = TeamCtrlModel().fit(X, y)
    Xt, yt = _toy(seed=1)
    p = m.predict_proba(Xt)
    assert p.shape == (len(Xt),) and p.dtype == np.float32
    assert ((p > 0.5) == yt).mean() > 0.9
    assert np.allclose(m.predict_proba(mirror_features(Xt)), 1 - p, atol=1e-5)
    Xn = Xt.copy(); Xn[:10] = np.nan
    pn = m.predict_proba(Xn)
    assert np.isnan(pn[:10]).all() and np.isfinite(pn[10:]).all()
    path = tmp_path / "team.joblib"; m.save(path)
    m2 = TeamCtrlModel.load(path)
    assert m2.policy == "full" and np.array_equal(m2.predict_proba(Xt), p)
    Xc = X.copy(); Xc[:, FI["H3_cos"]] = np.nan; Xc[:, FI["A3_cos"]] = np.nan     # all-NaN training columns
    mc = TeamCtrlModel().fit(Xc, y); mc.save(path)
    assert np.array_equal(TeamCtrlModel.load(path).predict_proba(Xt), mc.predict_proba(Xt))
    m3 = TeamCtrlModel(policy="no_touch").fit(X, y); m3.save(path)
    assert TeamCtrlModel.load(path).policy == "no_touch"
    import joblib
    joblib.dump(m.model, path)                       # old format: bare sklearn model -> policy full
    assert TeamCtrlModel.load(path).policy == "full"


def test_evidence_maths():
    p = np.array([0.9, 0.5, np.nan, 0.0], np.float32)
    ev = team_ctrl_evidence(p, eps=1e-3)
    assert ev.shape == (4, 3) and ev.dtype == np.float32
    assert np.isclose(ev[0, 0], np.log(0.9 + 1e-3)) and np.isclose(ev[0, 1], np.log(0.1 + 1e-3), atol=1e-6)
    assert ev[1, 0] == ev[1, 1] and np.isnan(ev[:, 2]).all() and np.isnan(ev[2]).all()
    assert np.isfinite(ev[3, :2]).all()


def test_no_runtime_warnings_with_empty_frames():
    m = carry_match(F=200); m.pos[50:120] = np.nan; m.speed[50:120] = np.nan; m.ball[150:170] = np.nan
    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)
        X = team_features(m)
    assert np.isnan(X[60, FI["H1_dist"]]) and X[60, FI["nearest_team"]] == 0


SIM = Path(__file__).resolve().parents[1] / "runs" / "sim"


@pytest.mark.dfl
def test_heldout_9001_accuracy_beats_ball_stream():
    """Train on 9002-9006 (seed 100), score 9001 per frame on visible ctrl_Dfull H/A frames (9007 never loaded)."""
    from mtah.evidence.ball_ctrl import BallCtrlParams, ball_ctrl_evidence
    from mtah.select import train_team_ctrl
    from mtah.sim.dataset import load_half_sim
    if not (SIM / "game_9001_H1_s100.json").is_file():
        pytest.skip("runs/sim absent")
    train = [load_half_sim(SIM, f"game_{m}_H{h}", 100) for m in range(9002, 9007) for h in (1, 2)]
    model = train_team_ctrl(train, policy="full")
    del train
    hit = hit_ball = n = 0
    for h in (1, 2):
        s = load_half_sim(SIM, f"game_9001_H{h}", 100)
        truth = s.truth["ctrl_Dfull"]
        p = model.predict_proba(team_features(s.match))
        bev = ball_ctrl_evidence(s.match, BallCtrlParams())
        ok = np.isin(truth, ["H", "A"]) & s.match.ball_present & np.isfinite(p) & np.isfinite(bev[:, 0])
        th = truth[ok] == "H"
        hit += int(((p[ok] > 0.5) == th).sum()); hit_ball += int(((bev[ok, 0] > bev[ok, 1]) == th).sum()); n += int(ok.sum())
    acc, acc_ball = hit / n, hit_ball / n
    print(f"9001 held-out visible H/A frames n={n}: teamctrl acc {acc:.4f}  ball-stream argmax acc {acc_ball:.4f}")
    assert acc >= 0.85 and acc >= acc_ball, (acc, acc_ball)


def test_train_team_ctrl_uses_visible_HA_frames_only(monkeypatch):
    from mtah.select import train_team_ctrl
    sims = [_fake_half(9001, h, 100, np.random.default_rng(h)) for h in (1, 2)]
    sims[0].truth["ctrl_Dfull"][:150] = "D"        # ball visible but truth D: excluded
    seen = {}
    orig = TeamCtrlModel.fit
    def spy(self, X, y, sample_weight=None):
        seen["X"], seen["y"] = X, y
        return orig(self, X, y, sample_weight)
    monkeypatch.setattr(TeamCtrlModel, "fit", spy)
    m = train_team_ctrl(sims, policy="no_touch", stride=5)
    assert m.policy == "no_touch"
    want = sum(len(np.flatnonzero(np.isin(s.truth["ctrl_Dfull"], ["H", "A"]) & s.match.ball_present)[::5]) for s in sims)
    assert len(seen["y"]) == want and set(np.unique(seen["y"])) == {0, 1}
    assert not np.isnan(seen["X"]).all(1).any()
    p = m.predict_proba(team_features(sims[1].match))
    t = sims[1].truth["ctrl_Dfull"]; ok = np.isin(t, ["H", "A"]) & np.isfinite(p)
    assert ((p[ok] > 0.5) == (t[ok] == "H")).mean() > 0.9
