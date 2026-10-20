import numpy as np
import pandas as pd

from types import SimpleNamespace

from mtah.select import GRID, apply_cell, mean_rank, score_cell, estimate_lengths
from mtah.pipeline import PipelineConfig, compute_evidence, predict
from tests.test_ball_ctrl import carry_match

TAGS = pd.DataFrame(columns=["start", "end", "code", "team", "type", "half"])
V4 = ["raw_Dfull", "ctrl_Dfull", "raw_Dnarrow", "ctrl_Dnarrow"]


def test_grid_and_apply_cell():
    assert len(GRID) == 81
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

