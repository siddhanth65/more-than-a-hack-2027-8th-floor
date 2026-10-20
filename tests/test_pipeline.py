import pandas as pd

from mtah.pipeline import PipelineConfig, compute_evidence, compute_strong, predict
from tests.test_ball_ctrl import carry_match


def test_predict_end_to_end_without_dead_model():
    m = carry_match(F=400)
    tags = pd.DataFrame(columns=["start", "end", "code", "team", "type", "half"])
    cfg = PipelineConfig()
    ev = compute_evidence(m, tags, cfg, None)
    assert set(ev) == {"ball", "tag"} and ev["ball"].shape == (400, 3)
    lab = predict(m, tags, cfg, None)
    assert "".join(lab[20:90]) == "H" * 70 and "".join(lab[120:]) == "A" * 280
    d = PipelineConfig.from_dict(cfg.to_dict())
    assert d.decoder.tau_HA == cfg.decoder.tau_HA


def test_predict_with_precomputed_strong_matches():
    m = carry_match(F=400)
    tags = pd.DataFrame(columns=["start", "end", "code", "team", "type", "half"])
    cfg = PipelineConfig()
    strong = compute_strong(m, cfg)
    assert (predict(m, tags, cfg, None, strong=strong) == predict(m, tags, cfg, None)).all()


def test_compute_strong_returns_touches_and_old_pair_tolerated():
    m = carry_match(F=400)
    tags = pd.DataFrame(columns=["start", "end", "code", "team", "type", "half"])
    cfg = PipelineConfig()
    sH, sA, touches = compute_strong(m, cfg)
    assert touches.ndim == 2 and touches.shape[1] == 4
    assert (predict(m, tags, cfg, None, strong=(sH, sA)) == predict(m, tags, cfg, None, strong=(sH, sA, touches))).all()


def test_from_dict_defaults_code_table():
    d = PipelineConfig().to_dict()
    d["tag"]["code_table"] = None
    cfg = PipelineConfig.from_dict(d)
    assert cfg.tag.code_table and "DEFENDING IN MIDDLE THIRD" in cfg.tag.code_table


def test_dead_policy_config_roundtrip_and_default():
    cfg = PipelineConfig(); assert cfg.dead_policy == "full"
    cfg.dead_policy = "no_ball_at_all"
    assert PipelineConfig.from_dict(cfg.to_dict()).dead_policy == "no_ball_at_all"
    d = cfg.to_dict(); d.pop("dead_policy")
    assert PipelineConfig.from_dict(d).dead_policy == "full"


def test_compute_evidence_rejects_policy_mismatch():
    import pytest
    from mtah.evidence.dead import DeadModel
    cfg = PipelineConfig(); cfg.dead_policy = "no_ball_cues"
    with pytest.raises(ValueError):
        compute_evidence(carry_match(), pd.DataFrame(columns=["start", "end", "code", "team", "type", "half"]),
                         cfg, DeadModel(policy="full"))
