import numpy as np
import pandas as pd

from mtah.evidence.tag_prior import TagPriorParams, tag_prior_evidence


def test_tag_prior_columns():
    tags = pd.DataFrame([{"start": 10, "end": 20, "code": "BUILD UP", "team": "H", "type": None, "half": 1},
                         {"start": 30, "end": 40, "code": "SET PIECES", "team": None, "type": "Throw-in", "half": 1},
                         {"start": 50, "end": 60, "code": "SET PIECES", "team": "A", "type": "Corner Kick", "half": 1}])
    ev = tag_prior_evidence(tags, 100, TagPriorParams(p_tag=0.85, p_sp_team=0.8))
    assert ev.shape == (100, 3) and np.isnan(ev[:, 2]).all()
    assert np.isclose(ev[15, 0], np.log(0.85)) and np.isclose(ev[15, 1], np.log(0.15))
    assert np.isnan(ev[35]).all()                      # throw-in without team: no opinion
    assert np.isclose(ev[55, 1], np.log(0.8)) and np.isnan(ev[5]).all()


def _tags(code, team="H"):
    return pd.DataFrame([{"start": 10, "end": 20, "code": code, "team": team, "type": None, "half": 1}])


def test_defensive_code_inverted():
    table = {"DEFENDING IN MIDDLE THIRD": {"sign": -1, "p": 0.7}}
    ev = tag_prior_evidence(_tags("DEFENDING IN MIDDLE THIRD"), 50, TagPriorParams(code_table=table))
    assert np.isclose(ev[15, 1], np.log(0.7)) and np.isclose(ev[15, 0], np.log(0.3))


def test_sign_zero_no_opinion():
    table = {"GOALS": {"sign": 0, "p": 0.5}}
    ev = tag_prior_evidence(_tags("GOALS"), 50, TagPriorParams(code_table=table))
    assert np.isnan(ev).all()


def test_unknown_code_falls_back_to_p_tag():
    ev = tag_prior_evidence(_tags("WEIRD"), 50, TagPriorParams(p_tag=0.9, code_table={"GOALS": {"sign": 0, "p": 0.5}}))
    assert np.isclose(ev[15, 0], np.log(0.9)) and np.isclose(ev[15, 1], np.log(0.1))


def test_json_loads_14_codes():
    from mtah.evidence.tag_prior import load_tag_codes
    t = load_tag_codes()
    assert len(t) == 14 and t["DEFENSIVE TRANSITION"] == {"sign": -1, "p": 0.62} and t["GOALS"]["sign"] == 0


def test_missing_json_returns_empty(tmp_path):
    from mtah.evidence.tag_prior import load_tag_codes
    assert load_tag_codes(tmp_path / "nope.json") == {}


def test_pipeline_config_roundtrips_table():
    from mtah.pipeline import PipelineConfig
    cfg = PipelineConfig()
    assert len(cfg.tag.code_table) == 14
    assert PipelineConfig.from_dict(cfg.to_dict()).tag.code_table == cfg.tag.code_table


def _sp_tags():
    return pd.DataFrame([{"start": 100, "end": 400, "code": "SET PIECES", "team": None, "type": "Throw-in", "half": 1},
                         {"start": 60, "end": 120, "code": "BUILD UP", "team": "H", "type": None, "half": 1}])


def test_sp_dead_bonus_zero_leaves_evidence_unchanged():
    base = tag_prior_evidence(_sp_tags(), 500, TagPriorParams(p_tag=0.85))
    same = tag_prior_evidence(_sp_tags(), 500, TagPriorParams(p_tag=0.85, sp_dead_bonus=0.0), fps=25.0)
    assert np.array_equal(base, same, equal_nan=True)


def test_sp_dead_bonus_is_a_d_advantage_on_the_window_only():
    p = TagPriorParams(p_tag=0.85, sp_dead_bonus=3.0, sp_dead_before_s=2.0, sp_dead_after_s=3.0)
    ev = tag_prior_evidence(_sp_tags(), 500, p, fps=25.0)
    base = tag_prior_evidence(_sp_tags(), 500, TagPriorParams(p_tag=0.85), fps=25.0)
    win = np.zeros(500, bool); win[100 - 50:100 + 75 + 1] = True
    assert np.array_equal(ev[~win], base[~win], equal_nan=True)            # untouched outside [start-2 s, start+3 s]
    nan_rows = win & np.isnan(base).all(1)
    assert nan_rows.sum() == 126 - 61 and np.allclose(ev[nan_rows], [0.0, 0.0, 3.0])
    team_rows = win & ~np.isnan(base[:, 0])                                  # BUILD UP H overlaps frames 60..120
    assert np.allclose(ev[team_rows, :2], base[team_rows, :2]) and np.allclose(ev[team_rows, 2], 3.0)
    c = ev[nan_rows] - ev[nan_rows].mean(1, keepdims=True)                  # what the decoder sees after centring
    assert np.allclose(c[:, 2] - c[:, 0], 3.0) and np.allclose(c[:, 0], c[:, 1])


def test_sp_dead_window_clipped_and_overlaps_not_doubled():
    tags = pd.DataFrame([{"start": 10, "end": 50, "code": "SET PIECES", "team": "A", "type": "Corner", "half": 1},
                         {"start": 30, "end": 80, "code": "SET PIECES", "team": "A", "type": "Corner", "half": 1}])
    ev = tag_prior_evidence(tags, 60, TagPriorParams(sp_dead_bonus=2.0), fps=25.0)
    assert np.allclose(ev[:, 2], 2.0)                                        # [0, 59] covered once, clipped at both ends


def test_pipeline_config_roundtrips_sp_dead_fields_and_old_configs_default_to_zero():
    from mtah.pipeline import PipelineConfig
    cfg = PipelineConfig()
    assert cfg.tag.sp_dead_bonus == 0.0
    cfg.tag.sp_dead_bonus = 3.0; cfg.tag.sp_dead_before_s = 1.5; cfg.tag.sp_dead_after_s = 4.0
    back = PipelineConfig.from_dict(cfg.to_dict()).tag
    assert (back.sp_dead_bonus, back.sp_dead_before_s, back.sp_dead_after_s) == (3.0, 1.5, 4.0)
    d = cfg.to_dict()
    for k in ("sp_dead_bonus", "sp_dead_before_s", "sp_dead_after_s"):
        d["tag"].pop(k)
    assert PipelineConfig.from_dict(d).tag.sp_dead_bonus == 0.0
