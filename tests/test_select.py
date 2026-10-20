import numpy as np
import pandas as pd

from mtah.select import GRID, apply_cell, mean_rank
from mtah.pipeline import PipelineConfig


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
