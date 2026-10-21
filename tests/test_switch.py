from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from mtah.decode.viterbi import DecoderParams, cost_matrix, decode, transition_matrix, viterbi_path
from mtah.evidence.switch import (FEATURE_POLICIES, SWITCH_FEATURES, TEAMCTRL_COLS, SwitchModel, hazard_to_tau,
                                  switch_evidence, switch_features, switch_labels)
from mtah.pipeline import PipelineConfig, compute_evidence, predict
from mtah.spells.postprocess import PostParams, postprocess, snap_team_boundaries
from tests.test_ball_ctrl import carry_match
from tests.test_select_smoke import _fake_half

TAGS0 = pd.DataFrame(columns=["start", "end", "code", "team", "type", "half"])
FI = {n: i for i, n in enumerate(SWITCH_FEATURES)}


# ---------------------------------------------------------------- labels
def test_labels_direct_and_short_d_changes():
    fps = 25.0
    truth = np.array(list("H" * 50 + "A" * 50 + "D" * 10 + "H" * 50 + "D" * 40 + "A" * 50 + "D" * 5 + "A" * 40), "<U1")
    y = switch_labels(truth, fps, half_s=0.25)
    assert y.shape == truth.shape and y.dtype == np.int8
    assert set(np.unique(y)) <= {-1, 0, 1}
    assert (y[truth == "D"] == -1).all()
    # direct H->A at 50: +-6 frames -> [44, 56)
    assert (y[44:56] == 1).all() and y[43] == 0 and y[56] == 0
    # A -> D(10 frames < 1 s) -> H: change; D run [100, 110) -> [94, 100) and [110, 116)
    assert (y[94:100] == 1).all() and (y[110:116] == 1).all() and y[93] == 0 and y[116] == 0
    # H -> D(40 frames > 1 s) -> A: not a change
    assert (y[150:160] == 0).all() and (y[200:210] == 0).all()
    # A -> D(5) -> A: same team, not a change
    assert (y[244:250] == 0).all() and (y[255:262] == 0).all()
    bp = np.ones(len(truth), bool); bp[46:48] = False
    y2 = switch_labels(truth, fps, ball_present=bp)
    assert (y2[46:48] == -1).all() and (y2[44:46] == 1).all()


# ---------------------------------------------------------------- features
def test_features_shape_nan_rows_and_teamctrl_columns():
    m = carry_match(F=300); m.ball[200:240] = np.nan
    X = switch_features(m)
    assert X.shape == (300, len(SWITCH_FEATURES)) and X.dtype == np.float32
    assert np.isnan(X[200:240]).all() and not np.isnan(X[np.r_[0:199, 241:300]]).all(1).any()
    assert np.isnan(X[:, [FI[n] for n in TEAMCTRL_COLS]]).all()             # no teamctrl stream: NaN columns
    # the carrier changes at frame 100: the nearest-team indicator flips across +-0.25 s there, not at 50
    assert X[100, FI["d_nearest_0.25s"]] == 2.0 and X[50, FI["d_nearest_0.25s"]] == 0.0
    assert X[100, FI["gap_prod_0.25s"]] < 0 < X[50, FI["gap_prod_0.25s"]]
    ts = np.full((300, 3), np.nan, np.float32); ts[:100, 0] = np.log(0.9); ts[:100, 1] = np.log(0.1)
    ts[100:, 0] = np.log(0.2); ts[100:, 1] = np.log(0.8)
    Xt = switch_features(m, ts)
    assert Xt[100, FI["tc_d_margin_0.25s"]] > 3.0 and Xt[50, FI["tc_d_margin_0.25s"]] == 0.0
    assert np.isnan(Xt[220]).all()
    assert all(n.startswith("touch_") for n in FEATURE_POLICIES["no_touch"]) and set(FEATURE_POLICIES) == {"full", "no_touch"}


def test_features_are_team_swap_invariant():
    m = _fake_half(9001, 1, 100, np.random.default_rng(1)).match
    X = switch_features(m)
    Xs = switch_features(replace(m, team=(1 - m.team).astype(np.int8)))
    assert np.allclose(X, Xs, equal_nan=True, atol=1e-5)


# ---------------------------------------------------------------- model
def test_model_learns_saves_and_loads(tmp_path):
    s = _fake_half(9001, 1, 100, np.random.default_rng(3))
    X = switch_features(s.match); y = switch_labels(s.truth["ctrl_Dfull"], s.match.fps, ball_present=s.match.ball_present)
    assert (y == 1).any() and (y == 0).any()
    m = SwitchModel().fit(X, y)
    h = m.predict_hazard(X)
    assert h.shape == (len(X),) and h.dtype == np.float32
    assert np.isnan(h[~s.match.ball_present]).all() and ((h[s.match.ball_present] > 0) & (h[s.match.ball_present] < 1)).all()
    assert h[y == 1].mean() > h[y == 0].mean() and not m.with_teamctrl
    path = tmp_path / "sw.joblib"; m.save(path)
    m2 = SwitchModel.load(path)
    assert m2.policy == "full" and np.array_equal(m2.predict_hazard(X), h, equal_nan=True)
    m3 = SwitchModel(policy="no_touch").fit(X, y); m3.save(path)
    assert SwitchModel.load(path).policy == "no_touch"
    with pytest.raises(KeyError):
        SwitchModel(policy="bogus")


# ---------------------------------------------------------------- tau maths
def test_hazard_to_tau_maths_and_nan():
    h = np.array([0.1, 0.2, 0.4, np.nan, 0.8], np.float32)
    mu = np.mean([0.1, 0.2, 0.4, 0.8])
    tau = hazard_to_tau(h, 30.0, 2.0, 3.0)
    assert tau.dtype == np.float64 and tau.shape == (5,)
    want = np.maximum(3.0, 30.0 - 2.0 * np.log(np.array([0.1, 0.2, 0.4, 0.8], np.float64) / mu))
    np.testing.assert_allclose(tau[[0, 1, 2, 4]], want, rtol=1e-6)
    assert tau[3] == 30.0                                             # NaN hazard: neutral
    assert tau[0] > 30.0 > tau[4]                                    # low hazard raises, high lowers
    assert (hazard_to_tau(h, 30.0, 50.0, 3.0)[[0, 1, 2, 4]] >= 3.0).all() and hazard_to_tau(h, 30.0, 50.0, 3.0)[4] == 3.0
    assert (hazard_to_tau(h, 30.0, 0.0, 3.0) == 30.0).all()
    assert (hazard_to_tau(np.full(4, np.nan), 30.0, 2.0, 3.0) == 30.0).all()
    ev = switch_evidence(h)
    assert ev.shape == (5, 3) and np.array_equal(ev[:, 0], h, equal_nan=True) and np.isnan(ev[:, 1:]).all()


# ---------------------------------------------------------------- decoder
def _old_viterbi(C, T):
    """Verbatim copy of viterbi_path before the per-frame tau (HEAD 4fdd7e0)."""
    n = len(C)
    back = np.zeros((n, 3), np.int8)
    cur = C[0].copy()
    for t in range(1, n):
        tot = cur[:, None] + T
        bi = tot.argmin(0)
        back[t] = bi
        cur = tot[bi, np.arange(3)] + C[t]
    path = np.zeros(n, np.int8); path[-1] = int(cur.argmin())
    for t in range(n - 1, 0, -1):
        path[t - 1] = back[t, path[t]]
    return path


def test_decoder_bit_identical_without_switch():
    rng = np.random.default_rng(0)
    F = 3000
    ev = {"a": rng.normal(size=(F, 3)).astype(np.float32), "b": rng.normal(size=(F, 3)).astype(np.float32)}
    ev["b"][::7, 2] = np.nan
    half = np.ones(F, np.int8); half[1500:] = 2; half[:20] = 0
    p = DecoderParams(tau_HA=3.0, tau_HD=2.0, tau_DH=2.5)
    C = cost_matrix(ev, p); T = transition_matrix(p)
    ref = np.full(F, "D", "<U1")
    for s, e in ((20, 1500), (1500, F)):
        ref[s:e] = np.array(["H", "A", "D"])[_old_viterbi(C[s:e], T)]
    lab = decode(ev, half, p)
    assert np.array_equal(lab, ref)
    assert np.array_equal(decode(ev, half, p, tau_HA_t=np.full(F, 3.0)), ref)        # constant tau = today
    ev2 = dict(ev, switch=switch_evidence(rng.uniform(0.01, 0.99, F)))
    assert np.array_equal(cost_matrix(ev2, p), C)                                     # "switch" is skipped
    assert np.array_equal(decode(ev2, half, p), ref)
    assert np.array_equal(viterbi_path(C[20:1500], T, None), _old_viterbi(C[20:1500], T))
    with pytest.raises(ValueError):
        decode(ev, half, p, tau_HA_t=np.ones(F - 1))


def test_decoder_switches_where_tau_is_low():
    F = 200
    ev = np.full((F, 3), np.nan, np.float32)
    ev[:, 0] = 0.5; ev[:, 1] = 0.0                          # clear H outside [50, 150)
    ev[50:150, 0] = -0.05; ev[50:150, 1] = 0.05          # weak A evidence: total 10 < 2 * tau_HA = 12
    half = np.ones(F, np.int8)
    p = DecoderParams(tau_HA=6.0, tau_HD=100.0, tau_DH=100.0)
    lab0 = decode({"x": ev}, half, p)
    assert set(lab0) == {"H"}
    tau = np.full(F, 6.0); tau[50] = 1.0; tau[150] = 1.0
    lab = decode({"x": ev}, half, p, tau_HA_t=tau)
    assert "".join(lab[:50]) == "H" * 50 and "".join(lab[50:150]) == "A" * 100 and "".join(lab[150:]) == "H" * 50
    tau2 = np.full(F, 6.0); tau2[60] = 1.0; tau2[140] = 1.0                # cheap switches inside: moves the cuts
    lab2 = decode({"x": ev}, half, p, tau_HA_t=tau2)
    assert "".join(lab2[60:140]) == "A" * 80 and lab2[59] == "H" and lab2[140] == "H"


# ---------------------------------------------------------------- snapping
def test_snap_hazard_moves_boundary_to_peak_and_respects_neighbours():
    x = np.array(list("H" * 40 + "A" * 40 + "H" * 40), "<U1"); half = np.ones(len(x), np.int8)
    hz = np.full(len(x), 0.1); hz[33] = 0.9; hz[95] = 0.8
    out = snap_team_boundaries(x, half, np.zeros((0, 4)), 10, mode="hazard", hazard=hz)
    assert "".join(out) == "H" * 33 + "A" * 47 + "H" * 40           # 40 -> 33; peak at 95 out of reach: 80 stays
    hz2 = np.full(len(x), 0.1); hz2[30] = 0.9
    out2 = snap_team_boundaries(x, half, None, 50, mode="hazard", hazard=hz2)
    assert "".join(out2) == "H" * 30 + "A" * 50 + "H" * 40
    hz3 = np.full(len(x), np.nan); hz3[79] = 0.9                      # the A run keeps >= 1 frame; NaN never wins
    out3 = snap_team_boundaries(x, half, None, 50, mode="hazard", hazard=hz3)
    assert "".join(out3) == "H" * 79 + "A" + "H" * 40
    assert np.array_equal(snap_team_boundaries(x, half, None, 10, mode="hazard", hazard=None), x)
    assert np.array_equal(snap_team_boundaries(x, half, None, 10, mode="hazard", hazard=np.full(len(x), np.nan)), x)
    d = np.array(list("H" * 40 + "D" * 40 + "A" * 40), "<U1")           # D boundaries are never moved
    assert np.array_equal(snap_team_boundaries(d, half, None, 10, mode="hazard", hazard=hz), d)
    with pytest.raises(ValueError):
        snap_team_boundaries(x, half, None, 10, mode="bogus")
    pp = PostParams(island_s=0.0, aba_s=0.0, snap_mode="hazard", snap_w=10)
    assert "".join(postprocess(x, half, 25.0, pp, touches=np.zeros((0, 4)), hazard=hz)) == "".join(out)
    assert np.array_equal(postprocess(x, half, 25.0, pp), x)          # no hazard: unchanged


# ---------------------------------------------------------------- pipeline / config
def test_config_roundtrip_and_defaults():
    cfg = PipelineConfig()
    assert cfg.use_switch is False and cfg.switch_policy == "full"
    assert cfg.decoder.w_switch == 0.0 and cfg.decoder.tau_floor == 3.0 and cfg.post.snap_mode == "touch"
    cfg.use_switch = True; cfg.switch_policy = "no_touch"; cfg.decoder.w_switch = 1.5; cfg.decoder.tau_floor = 4.0
    cfg.post.snap_mode = "hazard"
    d = PipelineConfig.from_dict(cfg.to_dict())
    assert d.use_switch and d.switch_policy == "no_touch" and d.decoder.w_switch == 1.5 and d.decoder.tau_floor == 4.0
    assert d.post.snap_mode == "hazard"
    raw = PipelineConfig().to_dict()
    for k in ("use_switch", "switch_policy"):
        raw.pop(k)
    raw["decoder"].pop("w_switch"); raw["decoder"].pop("tau_floor"); raw["post"].pop("snap_mode")
    old = PipelineConfig.from_dict(raw)                                # configs saved before T2
    assert not old.use_switch and old.decoder.w_switch == 0.0 and old.post.snap_mode == "touch"


def test_pipeline_switch_evidence_and_identity_at_w0():
    s = _fake_half(9001, 1, 100, np.random.default_rng(3))
    X = switch_features(s.match); y = switch_labels(s.truth["ctrl_Dfull"], s.match.fps, ball_present=s.match.ball_present)
    model = SwitchModel().fit(X, y)
    cfg = PipelineConfig()
    ev0 = compute_evidence(s.match, TAGS0, cfg, None, switch_model=model)
    assert "switch" not in ev0                                          # use_switch off
    cfg.use_switch = True
    ev = compute_evidence(s.match, TAGS0, cfg, None, switch_model=model)
    assert ev["switch"].shape == (s.match.n_frames, 3) and np.isnan(ev["switch"][:, 1:]).all()
    assert np.array_equal(ev["switch"][:, 0], model.predict_hazard(X), equal_nan=True)   # not centred
    base = predict(s.match, TAGS0, cfg, None, evidence=ev0)
    assert np.array_equal(predict(s.match, TAGS0, cfg, None, evidence=ev), base)          # w_switch 0: identical
    cfg.decoder.w_switch = 1.0
    lab = predict(s.match, TAGS0, cfg, None, evidence=ev)
    assert lab.shape == base.shape
    with pytest.raises(ValueError):
        predict(s.match, TAGS0, cfg, None, evidence=ev0)                 # w_switch > 0 needs a hazard
    cfg.switch_policy = "no_touch"
    with pytest.raises(ValueError):
        compute_evidence(s.match, TAGS0, cfg, None, switch_model=model)
    cfg.switch_policy = "full"; model.with_teamctrl = True
    with pytest.raises(ValueError):
        compute_evidence(s.match, TAGS0, cfg, None, switch_model=model)  # trained with teamctrl, none available


SIM = Path(__file__).resolve().parents[1] / "runs" / "sim"


@pytest.mark.dfl
def test_heldout_9001_hazard_average_precision():
    """Train on 9002-9006 (seed 100, no teamctrl), score 9001 visible labelled frames (9007 never loaded)."""
    from sklearn.metrics import average_precision_score

    from mtah.evidence.switch import train_switch
    from mtah.sim.dataset import load_half_sim
    if not (SIM / "game_9001_H1_s100.json").is_file():
        pytest.skip("runs/sim absent")
    train = [load_half_sim(SIM, f"game_{m}_H{h}", 100) for m in range(9002, 9007) for h in (1, 2)]
    model = train_switch(train, policy="full", stride=2)
    del train
    hs, ys = [], []
    for h in (1, 2):
        s = load_half_sim(SIM, f"game_9001_H{h}", 100)
        y = switch_labels(s.truth["ctrl_Dfull"], s.match.fps, ball_present=s.match.ball_present)
        hz = model.predict_hazard(switch_features(s.match))
        ok = (y >= 0) & np.isfinite(hz)
        hs.append(hz[ok]); ys.append(y[ok])
    hz, y = np.concatenate(hs), np.concatenate(ys)
    ap, base = float(average_precision_score(y, hz)), float(y.mean())
    print(f"9001 held-out labelled visible frames n={len(y)}: hazard AP {ap:.4f}  base rate {base:.4f}")
    assert ap > 2 * base, (ap, base)
