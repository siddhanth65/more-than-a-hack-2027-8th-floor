import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

import mtah.select as sel
from mtah.io.match import Match, speeds_from_positions
from mtah.sim.dataset import SimHalf, save_half

ROOT = Path(__file__).resolve().parents[1]
F = 1500


def _fake_half(match_id, half, seed, rng):
    """Tiny synthetic half: 150-frame blocks of home / away carry with a dead block (ball absent) every third."""
    pos = np.zeros((F, 4, 2), np.float32)
    ball = np.zeros((F, 2), np.float32)
    labels = np.empty(F, "<U1")
    for b in range(F // 150):
        s, e = b * 150, (b + 1) * 150
        lab = "D" if b % 3 == 2 else ("H" if b % 2 == 0 else "A")
        labels[s:e] = lab
        t = np.linspace(-20, 20, e - s)
        pos[s:e, 0, 0] = t; pos[s:e, 1, 0] = -30; pos[s:e, 2, 0] = -t; pos[s:e, 3, 0] = 30
        if lab == "H":
            ball[s:e, 0] = pos[s:e, 0, 0] + 0.3
        elif lab == "A":
            ball[s:e, 0] = pos[s:e, 2, 0] - 0.3
        else:
            ball[s:e] = np.nan
    pos += rng.normal(0, 0.05, pos.shape).astype(np.float32)
    ball += rng.normal(0, 0.05, ball.shape).astype(np.float32)
    m = Match(fps=25.0, pos=pos, speed=speeds_from_positions(pos, 25.0), team=np.array([0, 0, 1, 1], np.int8),
              ball=ball, ball_speed=speeds_from_positions(ball, 25.0), half=np.full(F, half, np.int8), pitch=(105.0, 68.0))
    tags = pd.DataFrame({"start": [0, 450], "end": [140, 590], "code": ["PROGRESSION", "PROGRESSION"],
                         "team": ["H", "A"], "type": [None, None], "half": [1, 1]})
    events = pd.DataFrame({"frame": [300], "team": ["H"], "cls": [8], "poss_only": [False]})
    truth = {v: labels.copy() for v in ("raw_Dfull", "ctrl_Dfull", "raw_Dnarrow", "ctrl_Dnarrow")}
    return SimHalf(key=f"game_{match_id}_{half}", seed=seed, match=m, tags=tags,
                   blackout=np.isnan(ball[:, 0]), truth=truth, events=events)


def _load_script():
    spec = importlib.util.spec_from_file_location("select_lomo_script", ROOT / "scripts" / "select_lomo.py")
    mod = importlib.util.module_from_spec(spec); sys.modules["select_lomo_script"] = mod
    spec.loader.exec_module(mod)
    return mod


def test_run_selection_smoke(tmp_path, monkeypatch):
    rng = np.random.default_rng(0)
    sim_dir = tmp_path / "sim"
    seeds = [100, 101]
    for mid in (9001, 9002, 9003):
        for half in (1, 2):
            for seed in seeds:
                save_half(_fake_half(mid, half, seed, rng), sim_dir)
    grid = [{"tau_HA": 3.0, "tau_D": 3.0, "w_dead": 1.0, "aba_s": 0.5},
            {"tau_HA": 10.0, "tau_D": 10.0, "w_dead": 1.0, "aba_s": 0.5}]
    monkeypatch.setattr(sel, "GRID", grid)
    mod = _load_script()
    out = tmp_path / "out"
    best = mod.run_selection(sim_dir, seeds, out)
    assert best in grid
    for name in ("summary.csv", "best.json", "registration.json", "drift_reference.json"):
        assert (out / name).exists(), name
    for fold in (9001, 9002, 9003):
        assert (out / f"fold_{fold}_halves.csv").exists() and (out / f"fold_{fold}_baselines.csv").exists()
    b = json.loads((out / "best.json").read_text())
    bar = b["spec_bar"]
    assert {"margins_vs_best_baseline", "required_margin", "seed_sd", "baseline_used", "folds_won_ctrl_Dfull",
            "count_ratios", "bar_pass"} <= set(bar)
    assert "seed_sd" in b and "baseline_used" in b
    assert json.loads((out / "drift_reference.json").read_text())
    # resume: a second run reuses the fold tables
    best2 = mod.run_selection(sim_dir, seeds, out)
    assert best2 == best
