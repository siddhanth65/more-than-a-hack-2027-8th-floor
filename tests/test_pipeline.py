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


TAGS0 = pd.DataFrame(columns=["start", "end", "code", "team", "type", "half"])


def _toy_team_model(policy="full"):
    import numpy as np
    from mtah.evidence.team_ctrl import TEAM_FEATURES, TeamCtrlModel
    rng = np.random.default_rng(0)
    X = rng.normal(size=(2000, len(TEAM_FEATURES))).astype(np.float32)
    i = TEAM_FEATURES.index("nearest_gap_m")
    return TeamCtrlModel(policy=policy).fit(X, (X[:, i] > 0).astype(int))


def test_teamctrl_config_roundtrip_and_old_configs():
    cfg = PipelineConfig()
    assert cfg.use_teamctrl is False and cfg.team_policy == "full" and cfg.decoder.weights["teamctrl"] == 1.0
    cfg.use_teamctrl = True; cfg.team_policy = "no_touch"; cfg.decoder.weights["teamctrl"] = 2.0
    r = PipelineConfig.from_dict(cfg.to_dict())
    assert r.use_teamctrl is True and r.team_policy == "no_touch" and r.decoder.weights["teamctrl"] == 2.0
    d = PipelineConfig().to_dict(); d.pop("use_teamctrl"); d.pop("team_policy"); d["decoder"]["weights"].pop("teamctrl")
    old = PipelineConfig.from_dict(d)
    assert old.use_teamctrl is False and old.team_policy == "full"


def test_compute_evidence_teamctrl_gating_and_policy_mismatch():
    import numpy as np
    import pytest
    m = carry_match(F=300); m.ball[250:] = np.nan
    team = _toy_team_model()
    cfg = PipelineConfig()
    assert "teamctrl" not in compute_evidence(m, TAGS0, cfg, None, team_model=team)      # use_teamctrl False
    cfg.use_teamctrl = True
    with pytest.raises(ValueError, match="team_model"):
        compute_evidence(m, TAGS0, cfg, None)                                            # weighted stream, no model
    cfg.decoder.weights["teamctrl"] = 0.0
    assert "teamctrl" not in compute_evidence(m, TAGS0, cfg, None)                       # weight 0, no model: absent
    cfg.decoder.weights["teamctrl"] = 1.0
    ev = compute_evidence(m, TAGS0, cfg, None, team_model=team)["teamctrl"]
    assert ev.shape == (300, 3) and ev.dtype == np.float32
    assert np.isnan(ev[250:]).all() and np.isnan(ev[:, 2]).all() and np.isfinite(ev[5:240, :2]).all()
    assert (ev[50, 0] > ev[50, 1]) and (ev[150, 1] > ev[150, 0])     # home carries, then away
    lab = predict(m, TAGS0, cfg, None, team_model=team)
    assert len(lab) == 300
    cfg.team_policy = "no_touch"
    with pytest.raises(ValueError):
        compute_evidence(m, TAGS0, cfg, None, team_model=team)
    cfg.use_teamctrl = False
    compute_evidence(m, TAGS0, cfg, None, team_model=team)          # stream off: no policy check
