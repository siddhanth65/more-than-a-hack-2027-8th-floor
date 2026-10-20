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
