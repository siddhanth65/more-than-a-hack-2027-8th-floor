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


# ---- Plan 2e T1: long-context player-motion features and the long / long_no_ball policies ----
from mtah.evidence.dead import (FEATURE_POLICIES, LONG_FEATURES, LONG_POLICIES, apply_policy,  # noqa: E402
                                dead_features_long, dead_table, long_features, policy_feature_names)

LI = {n: i for i, n in enumerate(LONG_FEATURES)}


def _move_then_stand(F=1000, stop=500, P=12, v=3.0):
    """P players (6 per team) running along x at v m/s until `stop`, then standing still."""
    pos = np.zeros((F, P, 2)); pos[:, :, 1] = np.linspace(-25, 25, P)
    pos[:, :6, 0] = -20.0; pos[:, 6:, 0] = 5.0
    t = np.minimum(np.arange(F), stop) / 25.0
    pos[:, :, 0] += (v * t)[:, None]
    return make_match(pos, [0] * 6 + [1] * 6)


def test_long_feature_names_and_shape():
    assert len(LONG_FEATURES) == 7 * 3 + 2 and len(set(LONG_FEATURES)) == len(LONG_FEATURES)
    assert LONG_FEATURES[:7] == tuple(f"{n}_2s" for n in ("lc_speed", "lc_slow_share", "lc_accel", "lc_team_spread",
                                                          "lc_centroid_dist", "lc_compactness", "lc_n_visible"))
    assert LONG_FEATURES[-2:] == ("lc_speed_trend", "lc_slow_trend")
    m = _move_then_stand()
    L = long_features(m)
    assert L.shape == (m.n_frames, len(LONG_FEATURES)) and L.dtype == np.float32
    X = dead_features_long(m, NO_TAGS)
    assert X.shape == (m.n_frames, len(DEAD_FEATURES) + len(LONG_FEATURES))
    np.testing.assert_array_equal(X[:, :len(DEAD_FEATURES)], dead_features(m, NO_TAGS))
    np.testing.assert_array_equal(X[:, len(DEAD_FEATURES):], L)


def test_long_features_stoppage_onset():
    m = _move_then_stand(F=1000, stop=500)
    L = long_features(m)
    sp, slow = L[:, LI["lc_speed_2s"]], L[:, LI["lc_slow_share_2s"]]
    assert sp[200] == pytest.approx(3.0, abs=0.05) and slow[200] == 0
    assert sp[800] == pytest.approx(0.0, abs=1e-6) and slow[800] == 1
    assert slow[800] > slow[200]
    k = 500 + 40                                       # 1.6 s after the stop
    assert L[k, LI["lc_speed_trend"]] < -0.5 and L[k, LI["lc_slow_trend"]] > 0.2
    assert L[200, LI["lc_speed_trend"]] == pytest.approx(0.0, abs=1e-5)
    assert L[200, LI["lc_n_visible_10s"]] == 12
    assert L[200, LI["lc_centroid_dist_5s"]] == pytest.approx(np.hypot(25.0, 2 * (25 - 2.5 * 50 / 11)), abs=1e-3)
    assert L[200, LI["lc_accel_2s"]] == pytest.approx(0.0, abs=0.5)


def test_long_features_nan_below_six_players_and_per_half():
    m = _move_then_stand(F=1000, stop=1000)
    m.pos[290:810, 5:] = np.nan; m.speed[290:810, 5:] = np.nan    # 5 visible players for 20.8 s
    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)
        L = long_features(m)
    for n in ("lc_speed", "lc_slow_share", "lc_accel", "lc_team_spread", "lc_centroid_dist", "lc_compactness"):
        assert np.isnan(L[550, LI[f"{n}_10s"]]), n
        assert np.isfinite(L[200, LI[f"{n}_10s"]]), n
    assert L[550, LI["lc_n_visible_10s"]] == 5
    assert np.isfinite(L[310, LI["lc_speed_2s"]])      # the window still reaches valid frames before the gap
    # halves: windows do not cross the half boundary; frames outside every half stay NaN
    pos = np.zeros((1000, 12, 2)); pos[:, :, 1] = np.linspace(-25, 25, 12); pos[:, 6:, 0] = 10.0
    pos[:, :, 0] += (3.0 * np.minimum(np.arange(1000), 499) / 25.0)[:, None]      # half 1 moving, half 2 still
    half = np.r_[np.ones(500), np.full(480, 2), np.zeros(20)].astype(np.int8)
    L2 = long_features(make_match(pos, [0] * 6 + [1] * 6, half=half))
    assert L2[501, LI["lc_speed_10s"]] == pytest.approx(0.0, abs=1e-6)
    assert L2[498, LI["lc_speed_10s"]] == pytest.approx(3.0, abs=0.05)
    assert np.isnan(L2[990]).all()


def test_long_policies_tables_and_apply_policy():
    assert {"long", "long_no_ball"} <= LONG_POLICIES and set(LONG_POLICIES) <= set(FEATURE_POLICIES)
    assert FEATURE_POLICIES["long"] == FEATURE_POLICIES["full"]
    assert FEATURE_POLICIES["long_no_ball"] == FEATURE_POLICIES["no_ball_at_all"]
    for p in ("full", "no_ball_cues", "no_ball_at_all"):
        assert policy_feature_names(p) == DEAD_FEATURES
    for p in ("long", "long_no_ball"):
        assert policy_feature_names(p) == DEAD_FEATURES + LONG_FEATURES
    names = DEAD_FEATURES + LONG_FEATURES
    X = np.ones((5, len(names)), np.float32)
    Y = apply_policy(X, "long_no_ball")
    idx = [names.index(n) for n in FEATURE_POLICIES["long_no_ball"]]
    assert (Y[:, idx] == 0).all() and (np.delete(Y, idx, axis=1) == 1).all()
    assert np.array_equal(apply_policy(X, "long"), X)
    with pytest.raises(ValueError):
        apply_policy(X, "full")                        # base policy on the long table
    with pytest.raises(ValueError):
        apply_policy(np.ones((5, len(DEAD_FEATURES)), np.float32), "long")
    m = _move_then_stand(F=300, stop=150)
    assert dead_table(m, NO_TAGS, "long").shape[1] == len(names)
    assert dead_table(m, NO_TAGS, "no_ball_at_all").shape[1] == len(DEAD_FEATURES)


def test_long_model_roundtrip(tmp_path):
    names = DEAD_FEATURES + LONG_FEATURES
    rng = np.random.default_rng(3)
    X = rng.normal(size=(600, len(names))).astype(np.float32)
    y = (X[:, names.index("lc_speed_5s")] > 0).astype(int)
    m = DeadModel(policy="long_no_ball").fit(X, y)
    assert m.feature_names == names
    p = tmp_path / "m.joblib"; m.save(p)
    m2 = DeadModel.load(p)
    assert m2.policy == "long_no_ball" and m2.feature_names == names
    assert np.array_equal(m2.predict_proba(X), m.predict_proba(X))
    X2 = X.copy(); X2[:, names.index("ball_present")] = 7.0          # dropped column: ignored
    assert np.array_equal(m2.predict_proba(X2), m.predict_proba(X))
    mm = _move_then_stand(F=300, stop=150)
    assert m2.features(mm, NO_TAGS).shape == (300, len(names))
    assert DeadModel().features(mm, NO_TAGS).shape == (300, len(DEAD_FEATURES))


def test_compute_evidence_and_train_dead_pick_the_table_by_policy():
    from types import SimpleNamespace
    from mtah.pipeline import PipelineConfig, compute_evidence
    from mtah.select import train_dead
    m = _move_then_stand(F=400, stop=200)

    class Stub:
        def __init__(self, policy):
            self.policy = policy; self.seen = None

        def predict_proba(self, X):
            self.seen = X.shape; return np.full(len(X), 0.5, np.float32)

    for pol, width in (("full", len(DEAD_FEATURES)), ("long", len(DEAD_FEATURES) + len(LONG_FEATURES)),
                       ("long_no_ball", len(DEAD_FEATURES) + len(LONG_FEATURES))):
        cfg = PipelineConfig(); cfg.dead_policy = pol
        stub = Stub(pol)
        ev = compute_evidence(m, NO_TAGS, cfg, stub)
        assert stub.seen == (400, width) and ev["dead"].shape == (400, 3)
    truth = np.array(["H"] * 200 + ["D"] * 200)
    m.ball[:, 0] = np.linspace(-50, 50, 400); m.ball[100:130] = np.nan; m.ball[300:330] = np.nan   # ball gaps:
    m.ball_speed[:] = 1.0                                                  # no all-NaN entry_* columns for HGB
    sims = [SimpleNamespace(match=m, tags=NO_TAGS, truth={"ctrl_Dfull": truth})]
    dm = train_dead(sims, policy="long")
    assert dm.policy == "long" and len(dm.feature_names) == len(DEAD_FEATURES) + len(LONG_FEATURES)
    assert dm.predict_proba(dm.features(m, NO_TAGS)).shape == (400,)


@pytest.mark.dfl
def test_long_policies_heldout_9001_visible_dead():
    """Train dead models on sims 9002-9006 (seed 100, every 5th frame) for policies full / long / long_no_ball and
    score 9001's visible graded frames (ball present) against ctrl_Dfull D: recall / precision at P > 0.5 and
    average precision. 9007 is never loaded. Gate: no AP regression of `long` vs `full` (> 0.01)."""
    from pathlib import Path
    from sklearn.metrics import average_precision_score
    from mtah.sim.dataset import load_half_sim
    sim = Path(__file__).resolve().parents[1] / "runs" / "sim"
    if not (sim / "game_9001_H1_s100.json").is_file():
        pytest.skip("runs/sim absent")
    nb = len(DEAD_FEATURES)
    Xs, ys = [], []
    for mid in range(9002, 9007):
        for h in (1, 2):
            s = load_half_sim(sim, f"game_{mid}_H{h}", 100)
            Xs.append(dead_features_long(s.match, s.tags)[::5])          # base columns == dead_features (tested above)
            ys.append((s.truth["ctrl_Dfull"] == "D").astype(int)[::5])
            del s
    X, y = np.concatenate(Xs), np.concatenate(ys); del Xs, ys
    models = {p: DeadModel(policy=p).fit(X if p in LONG_POLICIES else X[:, :nb], y)
              for p in ("full", "long", "long_no_ball")}
    del X, y
    P = {p: [] for p in models}; Y = []
    for h in (1, 2):
        s = load_half_sim(sim, f"game_9001_H{h}", 100)
        Xt = dead_features_long(s.match, s.tags)
        vis = s.match.ball_present & (s.match.half > 0)
        Y.append((s.truth["ctrl_Dfull"] == "D")[vis])
        for p, m in models.items():
            P[p].append(m.predict_proba(Xt if p in LONG_POLICIES else Xt[:, :nb])[vis])
        del s, Xt
    Y = np.concatenate(Y)
    res = {}
    for p in models:
        pr = np.concatenate(P[p]); hit = pr > 0.5
        res[p] = {"recall": float((hit & Y).sum() / max(Y.sum(), 1)), "precision": float((hit & Y).sum() / max(hit.sum(), 1)),
                  "ap": float(average_precision_score(Y, pr))}
        print(f"9001 visible frames n={len(Y)} dead share {Y.mean():.4f}  policy {p:13s} recall {res[p]['recall']:.4f}  "
              f"precision {res[p]['precision']:.4f}  AP {res[p]['ap']:.4f}")
    assert res["long"]["ap"] >= res["full"]["ap"] - 0.01, res


# ---- Plan 2e T2: per-match standardised long features (long_rel / long_rel_no_ball) ----
from mtah.evidence.dead import (LONG_REL_FEATURES, LONG_REL_POLICIES, _standardise, dead_features_long_rel,  # noqa: E402
                                smooth_player_speed)


def _varying_match(F=1500, P=12, seed=0, half=None):
    """P players (6 per team) running along x with smoothly varying speeds (0.2 .. 4.2 m/s, different phases)."""
    rng = np.random.default_rng(seed)
    t = np.arange(F) / 25.0
    per = rng.uniform(8, 30, P); ph = rng.uniform(0, 2 * np.pi, P)
    v = 2.2 + 2.0 * np.sin(2 * np.pi * t[:, None] / per + ph)
    pos = np.zeros((F, P, 2)); pos[:, :, 1] = np.linspace(-25, 25, P)
    pos[:, :, 0] = np.cumsum(v, axis=0) / 25.0 + np.where(np.arange(P) < 6, -20.0, 5.0)
    return make_match(pos, [0] * 6 + [1] * 6, half=half)


def test_long_rel_names_shape_and_dispatch():
    assert len(LONG_REL_FEATURES) == len(LONG_FEATURES) and all(n.startswith("lr_") for n in LONG_REL_FEATURES)
    assert LONG_REL_FEATURES[0] == "lr_speed_2s" and LONG_REL_FEATURES[-2:] == ("lr_speed_trend", "lr_slow_trend")
    m = _varying_match()
    R = long_features(m, mode="rel")
    assert R.shape == (m.n_frames, len(LONG_REL_FEATURES)) and R.dtype == np.float32
    X = dead_table(m, NO_TAGS, "long_rel")
    np.testing.assert_array_equal(X, dead_features_long_rel(m, NO_TAGS))
    np.testing.assert_array_equal(X[:, :len(DEAD_FEATURES)], dead_features(m, NO_TAGS))
    np.testing.assert_array_equal(X[:, len(DEAD_FEATURES):], R)
    with pytest.raises(ValueError):
        long_features(m, mode="relative")


def test_standardise_is_nan_aware_and_uses_the_reference_rows():
    col = np.arange(100, dtype=float); col[[3, 70]] = np.nan
    ref = np.zeros(100, bool); ref[:50] = True
    z = _standardise(col, ref, eps=0.0)
    v = col[:50][np.isfinite(col[:50])]
    q25, q50, q75 = np.percentile(v, [25, 50, 75])
    assert z[10] == pytest.approx((10 - q50) / (q75 - q25)) and np.isnan(z[3]) and np.isnan(z[70])
    assert np.isfinite(z[90])                                         # non-reference rows are scaled too
    few = np.zeros(100, bool); few[:5] = True                         # < LONG_REL_MIN_REF reference rows: all rows
    z2 = _standardise(col, few, eps=0.0)
    assert np.nanmedian(z2) == pytest.approx(0.0, abs=1e-9)
    assert np.isnan(_standardise(np.full(20, np.nan), np.ones(20, bool), 0.1)).all()


def test_long_rel_standardised_per_half_over_ball_visible_frames():
    half = np.r_[np.ones(1500), np.full(1500, 2)].astype(np.int8)
    m = _varying_match(F=3000, half=half)
    m.ball[2000:2300] = np.nan                                        # ball gap in half 2: not a reference row
    m.pos[400:900, 8:] = np.nan; m.speed[400:900, 8:] = np.nan        # 8 visible: still >= 6
    m.pos[1000:1200, 4:] = np.nan; m.speed[1000:1200, 4:] = np.nan    # 4 visible: NaN quantities
    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)
        R = long_features(m, mode="rel")
    vis = m.ball_present
    for i, n in enumerate(LONG_REL_FEATURES):
        if n.startswith("lr_n_visible"):
            continue
        for h in (1, 2):
            rows = (m.half == h) & vis & np.isfinite(R[:, i])
            assert np.median(R[rows, i]) == pytest.approx(0.0, abs=1e-4), (n, h)
    assert np.isnan(R[1100, LONG_REL_FEATURES.index("lr_speed_2s")])
    # a speed offset in half 2 only changes nothing in either half (each half is standardised on its own)
    m2 = _varying_match(F=3000, half=half)
    m2.ball[2000:2300] = np.nan
    m2.pos[400:900, 8:] = np.nan; m2.speed[400:900, 8:] = np.nan
    m2.pos[1000:1200, 4:] = np.nan; m2.speed[1000:1200, 4:] = np.nan
    m2.speed[1500:] += 1.5
    np.testing.assert_allclose(long_features(m2, mode="rel"), R, atol=1e-4, equal_nan=True)
    assert not np.allclose(long_features(m2)[1600:], long_features(m)[1600:], equal_nan=True)   # abs mode moves


def test_long_rel_invariant_to_a_constant_speed_offset():
    m1 = _varying_match(seed=1)
    m2 = _varying_match(seed=1); m2.speed = (m2.speed + 0.8).astype(np.float32)
    R1, R2 = long_features(m1, mode="rel"), long_features(m2, mode="rel")
    assert np.isfinite(R1[:, :6]).all()
    np.testing.assert_allclose(R2, R1, atol=1e-4, equal_nan=True)
    A1, A2 = long_features(m1), long_features(m2)
    assert np.abs(A2[:, LI["lc_speed_5s"]] - A1[:, LI["lc_speed_5s"]]).min() > 0.7     # abs features shift


def test_smoothing_reduces_jitter_only_speed():
    rng = np.random.default_rng(5)
    F, P = 1000, 12
    pos = np.zeros((F, P, 2)); pos[:, :, 1] = np.linspace(-25, 25, P)
    pos += rng.normal(0, 0.05, pos.shape)                             # standing players, 5 cm tracking jitter
    m = make_match(pos, [0] * 6 + [1] * 6)
    on = np.isfinite(m.pos[..., 0]); on[100:200, 0] = False
    raw = np.where(on, m.speed, np.nan).astype(float)
    sm = smooth_player_speed(m.speed, on, 25.0)
    assert np.isnan(sm[150, 0]) and np.isnan(sm[0]).all()             # untracked / no speed stays NaN
    inner = slice(20, -20)
    assert np.nanstd(sm[inner]) < np.nanstd(raw[inner]) / 3
    d_raw = np.nanmean(np.abs(np.diff(raw[inner], axis=0))); d_sm = np.nanmean(np.abs(np.diff(sm[inner], axis=0)))
    assert d_sm < d_raw / 5                                           # |accel| from smoothed speed: jitter mostly gone
    assert np.nanmean(sm[inner]) == pytest.approx(np.nanmean(raw[inner]), rel=0.05)


def test_long_rel_policies_tables_and_old_models_unaffected(tmp_path):
    assert LONG_REL_POLICIES == {"long_rel", "long_rel_no_ball"} and LONG_REL_POLICIES <= LONG_POLICIES
    assert FEATURE_POLICIES["long_rel"] == FEATURE_POLICIES["full"]
    assert FEATURE_POLICIES["long_rel_no_ball"] == FEATURE_POLICIES["no_ball_at_all"]
    names = DEAD_FEATURES + LONG_REL_FEATURES
    for p in LONG_REL_POLICIES:
        assert policy_feature_names(p) == names
    X = np.ones((5, len(names)), np.float32)
    Y = apply_policy(X, "long_rel_no_ball")
    idx = [names.index(n) for n in FEATURE_POLICIES["long_rel_no_ball"]]
    assert (Y[:, idx] == 0).all() and (np.delete(Y, idx, axis=1) == 1).all()
    with pytest.raises(ValueError):
        apply_policy(np.ones((5, len(DEAD_FEATURES)), np.float32), "long_rel")
    m = _varying_match(F=400)
    for pol, width in (("full", len(DEAD_FEATURES)), ("long", len(DEAD_FEATURES) + len(LONG_FEATURES)),
                       ("long_rel", len(names)), ("long_rel_no_ball", len(names))):
        assert dead_table(m, NO_TAGS, pol).shape == (400, width)
    # old models: the abs table and the long / full models are untouched by the rel variant
    np.testing.assert_array_equal(long_features(m), long_features(m, mode="abs"))
    np.testing.assert_array_equal(dead_table(m, NO_TAGS, "long"), dead_features_long(m, NO_TAGS))
    rng = np.random.default_rng(4)
    Xr = rng.normal(size=(600, len(names))).astype(np.float32)
    dm = DeadModel(policy="long_rel").fit(Xr, (Xr[:, names.index("lr_speed_5s")] > 0).astype(int))
    dm.save(tmp_path / "r.joblib"); dm2 = DeadModel.load(tmp_path / "r.joblib")
    assert dm2.policy == "long_rel" and dm2.feature_names == names
    assert np.array_equal(dm2.predict_proba(Xr), dm.predict_proba(Xr))
    assert dm2.features(m, NO_TAGS).shape == (400, len(names))
    assert DeadModel(policy="long").features(m, NO_TAGS).shape == (400, len(DEAD_FEATURES) + len(LONG_FEATURES))
