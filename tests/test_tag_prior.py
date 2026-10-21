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
    # legacy flat +-2/3 s window (Plan 2c): no per-type windows
    p = TagPriorParams(p_tag=0.85, sp_dead_bonus=3.0, sp_dead_before_s=2.0, sp_dead_after_s=3.0, sp_dead_before_by_type={})
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


# --- Plan 2f T1: type-specific, ramped pre-restart dead prior ---------------------------------------------------------

def _sp(start, sp_type, team=None, end=None):
    return pd.DataFrame([{"start": start, "end": end if end is not None else start + 50, "code": "SET PIECES",
                          "team": team, "type": sp_type, "half": 1}])


def test_goal_kick_ramp_values():
    s = 1000; n = 2000; b = 4.0
    ev = tag_prior_evidence(_sp(s, "Goal Kick"), n, TagPriorParams(sp_dead_bonus=b), fps=25.0)
    # window [s - 25 s, s + 3 s]; ramp 0 at s - 625 -> full at s - 75
    assert np.isnan(ev[s - 626]).all() and np.isnan(ev[s - 625]).all()      # ramp weight 0: untouched
    for off, frac in ((-550, (625 - 550) / 550), (-350, (625 - 350) / 550), (-100, (625 - 100) / 550)):
        assert np.allclose(ev[s + off], [0, 0, b * frac], atol=1e-5), off
    for off in (-75, -10, 0, 75):
        assert np.allclose(ev[s + off], [0, 0, b])
    assert np.isnan(ev[s + 76]).all()                                       # after-window unchanged (3 s)


def test_throw_in_ramp_and_sim_names():
    s = 500; b = 2.0
    for name in ("Throw-in", "throw-in"):
        ev = tag_prior_evidence(_sp(s, name), 1000, TagPriorParams(sp_dead_bonus=b), fps=25.0)
        # 6 s window: ramp over [s - 150, s - 75]
        assert np.isnan(ev[s - 150]).all()
        assert np.allclose(ev[s - 113, 2], b * (150 - 113) / 75, atol=1e-5)
        assert np.allclose(ev[s - 75], [0, 0, b]) and np.allclose(ev[s + 75], [0, 0, b]) and np.isnan(ev[s + 76]).all()
    # sim type names map like dead._sp_code: "Goal kick" -> goal (25 s), "Corner" -> corner (20 s), "Kick-off" -> kick
    for name, before in (("Goal kick", 25), ("Corner", 20), ("Corner Kick", 20), ("Free kick", 10), ("Kick-off", 15),
                         ("Kick Off (1st half)", 15)):
        ev = tag_prior_evidence(_sp(1000, name), 2000, TagPriorParams(sp_dead_bonus=1.0), fps=25.0)
        assert np.isnan(ev[1000 - before * 25]).all() and ev[1000 - before * 25 + 1, 2] > 0, name


def test_unknown_type_falls_back_flat():
    s = 500; b = 3.0
    for name in ("Penalty", None):
        ev = tag_prior_evidence(_sp(s, name), 1000, TagPriorParams(sp_dead_bonus=b), fps=25.0)
        # sp_dead_before_s 2 s < the 3 s full-bonus span: flat bonus on [s - 50, s + 75]
        assert np.isnan(ev[s - 51]).all() and np.allclose(ev[s - 50], [0, 0, b]) and np.allclose(ev[s + 75], [0, 0, b])
        assert np.isnan(ev[s + 76]).all()


def test_ramp_off_is_flat():
    s = 1000
    ev = tag_prior_evidence(_sp(s, "Goal Kick"), 2000, TagPriorParams(sp_dead_bonus=2.0, sp_dead_ramp=False), fps=25.0)
    assert np.isnan(ev[s - 626]).all() and np.allclose(ev[s - 625], [0, 0, 2.0]) and np.allclose(ev[s - 300], [0, 0, 2.0])


def test_bonus_zero_identical():
    tags = pd.concat([_sp(1000, "Goal Kick", team="H"), _sp(1500, "Throw-in"),
                      pd.DataFrame([{"start": 900, "end": 1200, "code": "BUILD UP", "team": "A", "type": None, "half": 1}])],
                     ignore_index=True)
    a = tag_prior_evidence(tags, 3000, TagPriorParams(sp_dead_bonus=0.0))
    b = tag_prior_evidence(tags, 3000, TagPriorParams(sp_dead_bonus=0.0, sp_dead_ramp=False,
                                                       sp_dead_before_by_type={}))
    assert a.tobytes() == b.tobytes() and np.isnan(a[:, 2]).all()


def test_nan_rows_become_zero_and_team_rows_keep_scores():
    s = 1000; b = 6.0
    tags = pd.concat([_sp(s, "Goal Kick"),
                      pd.DataFrame([{"start": s - 700, "end": s - 200, "code": "BUILD UP", "team": "H", "type": None,
                                     "half": 1}])], ignore_index=True)
    ev = tag_prior_evidence(tags, 2000, TagPriorParams(p_tag=0.85, sp_dead_bonus=b), fps=25.0)
    assert np.allclose(ev[s - 100], [0, 0, b * 525 / 550], atol=1e-5)                 # no team opinion: [0, 0, b*w]
    assert np.allclose(ev[s - 300], [np.log(0.85), np.log(0.15), b * 325 / 550], atol=1e-5)   # team row: H/A kept
    assert np.allclose(ev[s - 650, :2], [np.log(0.85), np.log(0.15)]) and np.isnan(ev[s - 650, 2])  # outside window


def test_overlapping_windows_take_max():
    tags = pd.concat([_sp(1000, "Goal Kick"), _sp(1200, "Throw-in")], ignore_index=True)
    ev = tag_prior_evidence(tags, 2000, TagPriorParams(sp_dead_bonus=1.0), fps=25.0)
    # goal kick full on [925, 1075]; throw-in ramp on [1050, 1125], full on [1125, 1275]
    assert np.allclose(ev[925:1076, 2], 1.0) and np.allclose(ev[1076, 2], 26 / 75, atol=1e-5)
    assert np.allclose(ev[1125:1276, 2], 1.0) and np.nanmax(ev[375:1276, 2]) <= 1.0 + 1e-6 and np.isnan(ev[1276]).all()


def test_config_round_trip_by_type():
    from mtah.pipeline import PipelineConfig
    cfg = PipelineConfig(); cfg.tag.sp_dead_before_by_type = {"goal": 30.0, "throw": 4.0}; cfg.tag.sp_dead_ramp = False
    d = PipelineConfig.from_dict(cfg.to_dict())
    assert d.tag.sp_dead_before_by_type == {"goal": 30.0, "throw": 4.0} and d.tag.sp_dead_ramp is False
    old = PipelineConfig().to_dict(); old["tag"].pop("sp_dead_before_by_type"); old["tag"].pop("sp_dead_ramp")
    o = PipelineConfig.from_dict(old)
    assert o.tag.sp_dead_before_by_type == TagPriorParams().sp_dead_before_by_type and o.tag.sp_dead_ramp is True


def test_grid9():
    from mtah.pipeline import PipelineConfig
    from mtah.select import GRID8, GRID9, GRIDS, apply_cell, evidence_key
    assert GRIDS["grid9"] is GRID9 and len(GRID9) == 4
    assert [c["sp_dead_bonus"] for c in GRID9] == [0.0, 2.0, 4.0, 6.0]
    assert all({k: v for k, v in c.items() if k != "sp_dead_bonus"} == {k: v for k, v in GRID8[0].items() if k != "sp_dead_bonus"}
               for c in GRID9)
    assert GRID9[0]["dead_policy"] == "long_rel" and GRID9[0]["w_dead"] == 4.0 and GRID9[0]["dead_offset"] == -2.0
    assert [apply_cell(PipelineConfig(), c).tag.sp_dead_bonus for c in GRID9] == [0.0, 2.0, 4.0, 6.0]
    assert len({evidence_key(c) for c in GRID9}) == 4
