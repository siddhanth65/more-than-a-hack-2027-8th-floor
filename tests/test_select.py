import pytest
import numpy as np
import pandas as pd

from types import SimpleNamespace

from mtah.select import GRID, apply_cell, mean_rank, score_cell, estimate_lengths
from mtah.pipeline import PipelineConfig, compute_evidence, predict
from tests.test_ball_ctrl import carry_match

TAGS = pd.DataFrame(columns=["start", "end", "code", "team", "type", "half"])
V4 = ["raw_Dfull", "ctrl_Dfull", "raw_Dnarrow", "ctrl_Dnarrow"]


def test_grid_and_apply_cell():
    assert len(GRID) == 144
    cfg = apply_cell(PipelineConfig(), GRID[0])
    assert cfg.decoder.tau_HD == cfg.decoder.tau_DH == GRID[0]["tau_D"] and cfg.post.aba_s == GRID[0]["aba_s"]
    assert cfg.decoder.weights["dead"] == GRID[0]["w_dead"]


def test_mean_rank_selection():
    t = pd.DataFrame({"cell": [0, 1, 2], "raw_Dfull": [0.5, 0.6, 0.55], "ctrl_Dfull": [0.5, 0.58, 0.6],
                      "raw_Dnarrow": [0.4, 0.5, 0.45], "ctrl_Dnarrow": [0.4, 0.5, 0.5]})
    ranked = mean_rank(t, ["raw_Dfull", "ctrl_Dfull", "raw_Dnarrow", "ctrl_Dnarrow"])
    assert ranked.iloc[0]["cell"] == 1
    mr = dict(zip(ranked["cell"], ranked["mean_rank"]))
    assert mr[1] == 1.375 and mr[2] == 1.625 and mr[0] == 3.0


def test_mean_rank_tie_break_ctrl_dfull():
    t = pd.DataFrame({"cell": [0, 1], "raw_Dfull": [0.6, 0.5], "ctrl_Dfull": [0.5, 0.6],
                      "raw_Dnarrow": [0.6, 0.5], "ctrl_Dnarrow": [0.5, 0.6]})
    ranked = mean_rank(t, V4)
    assert ranked["mean_rank"].nunique() == 1
    assert list(ranked["cell"]) == [1, 0]


def test_evidence_reuse_across_cells():
    m = carry_match(); cfg = PipelineConfig()
    ev = compute_evidence(m, TAGS, apply_cell(cfg, GRID[0]), None)
    for cell in (GRID[0], GRID[40], GRID[80]):
        c = apply_cell(cfg, cell)
        assert np.array_equal(predict(m, TAGS, c, None, evidence=ev), predict(m, TAGS, c, None))


def test_score_cell_one_half_two_variants():
    m = carry_match(); cfg = PipelineConfig()
    lab = predict(m, TAGS, cfg, None)
    truth = {"a": lab.copy(), "b": lab.copy()}
    sim = SimpleNamespace(key="game_9001_1", seed=1, match=m, tags=TAGS, truth=truth)
    res, rows = score_cell([sim], [compute_evidence(m, TAGS, cfg, None)], cfg, ("a", "b"))
    assert res["a"]["spell_f1"] == res["b"]["spell_f1"] and res["a"]["macro_f1"] == res["b"]["macro_f1"]
    assert list(rows["variant"]) == ["a", "b"] and len(rows) == 2
    assert {"key", "seed", "variant", "spell_f1", "macro_f1", "n_gt", "n_pred"} <= set(rows.columns)



def _fake_sims(noise_kind, n=6000, seed=0):
    from mtah.sim.dataset import SimHalf
    rng = np.random.default_rng(seed)
    truth = np.array((["H"] * 300 + ["A"] * 300) * (n // 600), dtype="<U1")
    if noise_kind == "white":
        noise = rng.normal(size=n)
    else:
        noise = np.convolve(rng.normal(size=n + 24), np.ones(25) / 5.0, mode="valid")
    m = np.where(truth == "H", 3.0, -3.0) + noise
    ev = np.zeros((n, 3)); ev[:, 0] = m / 2; ev[:, 1] = -m / 2
    dead = np.zeros((n, 3)); dead[:, 2] = np.where(truth == "D", 1.0, -1.0) + noise
    s = SimpleNamespace(key="game_9001_1", seed=1, match=None, tags=TAGS, truth={"ctrl_Dfull": truth}, ev=(ev, dead))
    return [s]


@pytest.mark.parametrize("kind,lo,hi", [("white", 0.9, 2.0), ("ma", 15.0, 40.0)])
def test_estimate_lengths_uses_errors_not_signal(monkeypatch, kind, lo, hi):
    import mtah.select as sel
    sims = _fake_sims(kind)
    def fake_ev(match, tags, cfg, dead, cache_key=None):
        e, d = sims[0].ev
        return {"ball": e, "tag": e, "dead": d}
    monkeypatch.setattr(sel, "compute_evidence", fake_ev)
    L = estimate_lengths(sims, PipelineConfig(), lambda s: None, mode="residual")
    for k in ("ball", "tag"):
        assert lo <= L[k] <= hi, (k, L[k])


def test_apply_cell_sets_dead_policy():
    from mtah.pipeline import PipelineConfig
    from mtah.select import GRID, apply_cell
    assert apply_cell(PipelineConfig(), GRID[0]).dead_policy == "full"
    assert apply_cell(PipelineConfig(), {**GRID[0], "dead_policy": "no_ball_cues"}).dead_policy == "no_ball_cues"


def test_grid2_size_and_axes():
    from mtah.select import GRID2, GRIDS
    assert len(GRID2) == 64 and GRIDS["grid1"] is GRID and GRIDS["grid2"] is GRID2
    assert {c["dead_policy"] for c in GRID2} == {"full", "no_ball_at_all"}
    assert {c["w_pathcrf"] for c in GRID2} == {0.0, 0.5, 1.0, 2.0}
    assert {c["tau_HA"] for c in GRID2} == {10.0, 30.0} and {c["tau_D"] for c in GRID2} == {30.0, 100.0}
    assert {c["w_dead"] for c in GRID2} == {1.0, 2.0} and {c["aba_s"] for c in GRID2} == {1.0}
    assert len({tuple(sorted(c.items())) for c in GRID2}) == 64


def test_apply_cell_grid2_sets_policy_and_pathcrf_weight():
    from mtah.select import GRID2
    cell = next(c for c in GRID2 if c["dead_policy"] == "no_ball_at_all" and c["w_pathcrf"] == 0.5)
    base = PipelineConfig(); base.use_pathcrf = True
    cfg = apply_cell(base, cell)
    assert cfg.dead_policy == "no_ball_at_all" and cfg.decoder.weights["pathcrf"] == 0.5
    assert cfg.use_pathcrf is True and cfg.post.aba_s == 1.0 and cfg.decoder.tau_HD == cell["tau_D"]
    assert base.decoder.weights["pathcrf"] == 1.0 and base.dead_policy == "full"   # base untouched


def _ranked(cells):
    return pd.DataFrame({"cell": cells, "mean_rank": np.arange(1, len(cells) + 1, dtype=float)})


def test_is_real_plausible_bounds():
    from mtah.select import REAL_SPELL_RATE_BAND, is_real_plausible, real_band_checks
    assert REAL_SPELL_RATE_BAND == (2.7, 4.3)
    assert is_real_plausible(0.10, 3.4) and is_real_plausible(0.18, 3.4) and is_real_plausible(0.14, 3.4)
    assert not is_real_plausible(0.099, 3.4) and not is_real_plausible(0.181, 3.4) and not is_real_plausible(float("nan"), 3.4)
    # both bands must hold: the spell rate gate
    assert is_real_plausible(0.14, 2.7) and is_real_plausible(0.14, 4.3)
    assert not is_real_plausible(0.14, 2.69) and not is_real_plausible(0.14, 4.31) and not is_real_plausible(0.14, 5.2)
    assert not is_real_plausible(0.14, float("nan")) and not is_real_plausible(0.05, 5.2)
    assert real_band_checks(0.14, 5.2) == {"d_share_ok": True, "spell_rate_ok": False, "real_plausible": False}
    assert real_band_checks(0.05, 3.0) == {"d_share_ok": False, "spell_rate_ok": True, "real_plausible": False}


def test_choose_plausible_picks_highest_ranked_plausible():
    from mtah.select import choose_plausible
    plaus = {7: {"real_plausible": False}, 3: {"real_plausible": True}, 5: {"real_plausible": True}}
    assert choose_plausible(_ranked([7, 3, 5, 1]), plaus) == 3


def test_choose_plausible_falls_back_to_rank_one(capsys):
    from mtah.select import choose_plausible
    plaus = {7: {"real_plausible": False}, 3: {"real_plausible": False}}
    assert choose_plausible(_ranked([7, 3, 5]), plaus) == 7
    assert "WARNING" in capsys.readouterr().out


def test_choose_plausible_ignores_unchecked_cells():
    from mtah.select import choose_plausible
    # cell 5 was never checked on the real match (not in the top-k): it cannot be chosen over the fallback
    assert choose_plausible(_ranked([7, 5]), {7: {"real_plausible": False}}) == 7
    assert choose_plausible(_ranked([7, 5]), {}) == 7


def test_unit_lengths_cover_every_stream(monkeypatch):
    """Unit mode (the default): L_k = 1.0 for ball, tag, dead, pathcrf and for every stream the evidence carries."""
    import mtah.pipeline as pl
    import mtah.select as sel
    from mtah.evidence.dead import DeadModel
    assert sel.LENGTHS_MODE == "unit" and set(sel.EVIDENCE_STREAMS) == {"ball", "tag", "dead", "pathcrf"}
    from tests.test_select_smoke import _fake_half
    h = _fake_half(9001, 1, 100, np.random.default_rng(0)); m = h.match
    monkeypatch.setattr(pl, "pathcrf_posterior", lambda match, key, params: {
        "p_home": np.full(match.n_frames, 0.6, np.float32), "p_none": np.zeros(match.n_frames, np.float32),
        "imputed_frac": np.zeros(match.n_frames, np.float32)})
    cfg = PipelineConfig(); cfg.use_pathcrf = True
    cfg.decoder.weights["extra"] = 0.5   # a stream the decoder weights beyond EVIDENCE_STREAMS
    dead = DeadModel()
    from mtah.evidence.dead import dead_features
    dead.fit(dead_features(m, h.tags), (h.truth["ctrl_Dfull"] == "D").astype(int))
    ev = compute_evidence(m, h.tags, cfg, dead, cache_key="k")
    assert {"ball", "tag", "dead", "pathcrf"} <= set(ev)
    ev["extra"] = ev["ball"]
    called = []
    L = estimate_lengths([], cfg, lambda sims: called.append(1))   # unit mode never trains or computes evidence
    assert not called
    for k in ("ball", "tag", "dead", "pathcrf", "extra", *ev):
        assert L[k] == 1.0, (k, L)
    assert estimate_lengths([], None, None) == {k: 1.0 for k in sel.EVIDENCE_STREAMS}
    with pytest.raises(ValueError):
        estimate_lengths([], cfg, None, mode="bogus")


def test_grid3_axes_and_apply_cell_sets_sp_dead_bonus():
    from mtah.select import GRID3, GRIDS
    assert len(GRID3) == 64 and GRIDS["grid3"] is GRID3
    keys = {"tau_HA", "tau_D", "w_dead", "w_pathcrf", "dead_policy", "sp_dead_bonus", "aba_s"}
    assert all(set(c) == keys for c in GRID3)
    assert {c["tau_HA"] for c in GRID3} == {30.0} and {c["tau_D"] for c in GRID3} == {30.0, 100.0}
    assert {c["w_dead"] for c in GRID3} == {2.0, 3.0, 4.0, 6.0} and {c["w_pathcrf"] for c in GRID3} == {0.5, 1.0}
    assert {c["dead_policy"] for c in GRID3} == {"full", "no_ball_at_all"}
    assert {c["sp_dead_bonus"] for c in GRID3} == {0.0, 3.0} and {c["aba_s"] for c in GRID3} == {1.0}
    assert len({tuple(sorted(c.items())) for c in GRID3}) == 64
    cell = next(c for c in GRID3 if c["sp_dead_bonus"] == 3.0 and c["dead_policy"] == "no_ball_at_all")
    base = PipelineConfig()
    cfg = apply_cell(base, cell)
    assert cfg.tag.sp_dead_bonus == 3.0 and cfg.dead_policy == "no_ball_at_all"
    assert cfg.decoder.weights["dead"] == cell["w_dead"] and cfg.decoder.weights["pathcrf"] == cell["w_pathcrf"]
    assert not hasattr(base.tag, "sp_dead_bonus") or base.tag.sp_dead_bonus != 3.0   # base untouched


def test_grid4_axes_and_apply_cell_sets_team_fields():
    from mtah.select import GRID4, GRIDS
    assert len(GRID4) == 96 and GRIDS["grid4"] is GRID4
    keys = {"tau_HA", "tau_D", "w_dead", "w_pathcrf", "dead_policy", "sp_dead_bonus", "w_teamctrl", "w_ball",
            "team_policy", "aba_s", "merge_theta"}
    assert all(set(c) == keys for c in GRID4)
    assert {c["tau_HA"] for c in GRID4} == {30.0, 60.0} and {c["w_teamctrl"] for c in GRID4} == {0.0, 1.0, 2.0}
    assert {c["w_ball"] for c in GRID4} == {1.0} and {c["w_dead"] for c in GRID4} == {3.0, 4.0}
    assert {c["merge_theta"] for c in GRID4} == {1.0, 0.7} and GRID4[0]["merge_theta"] == 1.0
    assert {c["team_policy"] for c in GRID4} == {"full", "no_touch"} and {c["aba_s"] for c in GRID4} == {1.0, 2.0}
    assert {c["w_pathcrf"] for c in GRID4} == {1.0} and {c["tau_D"] for c in GRID4} == {100.0}
    assert {c["dead_policy"] for c in GRID4} == {"full"} and {c["sp_dead_bonus"] for c in GRID4} == {0.0}
    assert len({tuple(sorted(c.items())) for c in GRID4}) == 96
    # a3b's cell 52 (tau_HA 30, w_dead 4, w_pathcrf 1, full, no bonus, aba 1) with the stream off is in the grid
    assert any(c["tau_HA"] == 30.0 and c["w_dead"] == 4.0 and c["w_teamctrl"] == 0.0 and c["w_ball"] == 1.0
               and c["aba_s"] == 1.0 and c["merge_theta"] == 1.0 for c in GRID4)
    cell = next(c for c in GRID4 if c["w_teamctrl"] == 2.0 and c["team_policy"] == "no_touch" and c["merge_theta"] == 0.7
                and c["aba_s"] == 2.0 and c["tau_HA"] == 60.0)
    base = PipelineConfig()
    cfg = apply_cell(base, cell)
    assert cfg.post.merge_theta == 0.7 and base.post.merge_theta == 1.0
    assert apply_cell(base, {**cell, "merge_theta": 1.0}).post.merge_theta == 1.0
    assert cfg.decoder.weights["teamctrl"] == 2.0 and cfg.decoder.weights["ball"] == 1.0 and cfg.team_policy == "no_touch"
    assert cfg.post.aba_s == 2.0 and cfg.decoder.tau_HA == 60.0 and cfg.decoder.tau_HD == cfg.decoder.tau_DH == 100.0
    assert cfg.use_teamctrl is False      # the stream switch belongs to the base config (select_lomo.base_for_grid)
    assert base.decoder.weights["ball"] == 1.0 and base.team_policy == "full"   # base untouched
    old = apply_cell(base, GRID[0])
    assert old.post.merge_theta == 1.0
    assert old.decoder.weights["ball"] == 1.0 and old.team_policy == "full" and old.decoder.weights["teamctrl"] == 1.0


def test_merge_ref_cells_spread_and_strip_theta():
    from mtah.select import GRID4, merge_cells, merge_ref_cells
    assert len(merge_cells(GRID4)) == 48 and merge_ref_cells([c for c in GRID4 if c["merge_theta"] == 1.0]) == []
    refs = merge_ref_cells(GRID4)
    assert len(refs) == 4 and all(c["merge_theta"] == 1.0 for c in refs)
    assert len({tuple(sorted(c.items())) for c in refs}) == 4
    assert {c["tau_HA"] for c in refs} == {30.0, 60.0}
    assert merge_ref_cells(GRID4, k=1) == [{**merge_cells(GRID4)[0], "merge_theta": 1.0}]
