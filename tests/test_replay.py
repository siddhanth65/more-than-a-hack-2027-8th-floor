import math

import numpy as np
import pandas as pd
import pytest

from mtah.decode.viterbi import DecoderParams, cost_matrix
from mtah.evidence.replay import LR_OFF, LR_ON, cue_from_arrays, replay_evidence
from mtah.pipeline import PipelineConfig, compute_evidence, predict
from tests.test_ball_ctrl import carry_match

NO_TAGS = pd.DataFrame(columns=["start", "end", "code", "team", "type", "half"])


def test_replay_evidence_maths():
    cue = np.array([1.0, 0.0, 0.5, np.nan], np.float32)
    ev = replay_evidence(cue, 2.0)
    assert ev.shape == (4, 3) and ev.dtype == np.float32
    assert np.allclose(ev[:3, :2], 0.0)
    assert ev[0, 2] == pytest.approx(2 * math.log(0.64 / 0.13), rel=1e-6)
    assert ev[1, 2] == pytest.approx(2 * math.log(0.36 / 0.87), rel=1e-6)
    assert ev[2, 2] == pytest.approx(2 * 0.5 * (LR_ON + LR_OFF), rel=1e-6)
    assert np.isnan(ev[3]).all()


def test_cue_from_arrays_nan_without_broadcast():
    c = cue_from_arrays(np.array([True, True, False]), np.array([True, False, True]))
    assert c[0] == 1.0 and c[1] == 0.0 and np.isnan(c[2]) and c.dtype == np.float32


def test_off_or_zero_weight_leaves_evidence_unchanged():
    m = carry_match(F=400)
    base = compute_evidence(m, NO_TAGS, PipelineConfig(), None)
    cue = np.ones(400, np.float32)
    on0 = PipelineConfig(use_replay=True, replay_w=0.0)
    for cfg, c in ((PipelineConfig(), None), (PipelineConfig(), cue), (on0, cue)):
        ev = compute_evidence(m, NO_TAGS, cfg, None, replay_cue=c)
        assert set(ev) == set(base) and all(np.array_equal(ev[k], base[k], equal_nan=True) for k in base)
    assert (predict(m, NO_TAGS, on0, None, evidence=compute_evidence(m, NO_TAGS, on0, None, replay_cue=cue))
            == predict(m, NO_TAGS, PipelineConfig(), None)).all()


def test_use_replay_without_cue_raises():
    m = carry_match(F=400)
    for w in (0.0, 1.0):
        with pytest.raises(ValueError, match="replay_cue"):
            compute_evidence(m, NO_TAGS, PipelineConfig(use_replay=True, replay_w=w), None)
    with pytest.raises(ValueError, match="frames"):
        compute_evidence(m, NO_TAGS, PipelineConfig(use_replay=True, replay_w=1.0), None, replay_cue=np.ones(10))


def test_replay_stream_added_when_on():
    m = carry_match(F=400)
    cue = np.zeros(400, np.float32); cue[100:200] = 1.0
    ev = compute_evidence(m, NO_TAGS, PipelineConfig(use_replay=True, replay_w=1.5), None, replay_cue=cue)
    assert np.array_equal(ev["replay"], replay_evidence(cue, 1.5), equal_nan=True)


def test_config_round_trip_and_old_configs_default_off():
    cfg = PipelineConfig(use_replay=True, replay_w=1.5)
    d = cfg.to_dict()
    assert d["use_replay"] is True and d["replay_w"] == 1.5
    r = PipelineConfig.from_dict(d)
    assert r.use_replay is True and r.replay_w == 1.5
    old = PipelineConfig().to_dict(); old.pop("use_replay"); old.pop("replay_w")
    o = PipelineConfig.from_dict(old)
    assert o.use_replay is False and o.replay_w == 0.0


def test_cost_matrix_D_advantage():
    w = 1.5
    cue = np.array([1.0, 0.0, np.nan], np.float32)
    base = np.array([[0.3, -0.2, 0.1]] * 3, np.float32)
    p = DecoderParams(weights={}, lengths={})
    C0 = cost_matrix({"ball": base}, p)
    C = cost_matrix({"ball": base, "replay": replay_evidence(cue, w)}, p)
    adv = (C[:, 0] - C[:, 2]) - (C0[:, 0] - C0[:, 2])          # D advantage over H gained from the stream
    assert adv[0] - adv[1] == pytest.approx(w * (LR_ON - LR_OFF))
    assert adv[0] == pytest.approx(w * LR_ON) and adv[1] == pytest.approx(w * LR_OFF)
    assert np.allclose(C[:, 0] - C[:, 1], C0[:, 0] - C0[:, 1])  # H vs A untouched
    assert np.allclose(C[2], C0[2])                              # no cue = no opinion
