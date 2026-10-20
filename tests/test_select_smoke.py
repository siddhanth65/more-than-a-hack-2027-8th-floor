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


def _fake_posterior(calls):
    def fake(match, cache_key, params):
        calls.append(cache_key)
        F = match.n_frames
        p = np.where(np.isnan(match.ball[:, 0]), np.nan, 0.6).astype(np.float32)
        return {"p_home": p, "p_none": np.zeros(F, np.float32), "imputed_frac": np.zeros(F, np.float32)}
    return fake


def test_run_selection_two_policies_pathcrf_and_real(tmp_path, monkeypatch):
    """grid2-style cells: per-policy dead models (no policy-mismatch error), PathCRF stream, real-match plausibility."""
    import mtah.pipeline as pl
    rng = np.random.default_rng(1)
    sim_dir = tmp_path / "sim"
    seeds = [100, 101]
    for mid in (9001, 9002, 9003):
        for half in (1, 2):
            for seed in seeds:
                save_half(_fake_half(mid, half, seed, rng), sim_dir)
    grid = [{"tau_HA": 10.0, "tau_D": 30.0, "w_dead": 1.0, "w_pathcrf": 1.0, "dead_policy": "no_ball_at_all", "aba_s": 1.0},
            {"tau_HA": 10.0, "tau_D": 30.0, "w_dead": 1.0, "w_pathcrf": 0.0, "dead_policy": "full", "aba_s": 1.0},
            {"tau_HA": 30.0, "tau_D": 100.0, "w_dead": 2.0, "w_pathcrf": 0.5, "dead_policy": "full", "aba_s": 1.0}]
    calls = []
    monkeypatch.setattr(pl, "pathcrf_posterior", _fake_posterior(calls))
    mod = _load_script()
    policies = []
    orig = mod.train_dead
    monkeypatch.setattr(mod, "train_dead", lambda sims, variant="ctrl_Dfull", policy="full": policies.append(policy) or orig(sims, variant, policy))
    real = _fake_half(9999, 1, 0, rng)
    monkeypatch.setattr(mod, "load_tactical", lambda d: real.match)
    monkeypatch.setattr(mod, "load_tags", lambda d: real.tags)
    ev_policy = {}   # id(evidence element) -> dead_policy it was computed for (elements kept alive in ev_keep)
    ev_keep = []
    orig_ev = mod.compute_evidence
    def spy_ev(match, tags, cfg, dead, **kw):
        r = orig_ev(match, tags, cfg, dead, **kw); ev_policy[id(r)] = cfg.dead_policy; ev_keep.append(r); return r
    monkeypatch.setattr(mod, "compute_evidence", spy_ev)
    scored = []   # (cell policy, {policies of the evidence it was scored with})
    orig_sc = mod.score_cell
    def spy_sc(test, ev, cfg, *a, **kw):
        scored.append((cfg.dead_policy, {ev_policy[id(e)] for e in ev})); return orig_sc(test, ev, cfg, *a, **kw)
    monkeypatch.setattr(mod, "score_cell", spy_sc)
    out = tmp_path / "out"
    best = mod.run_selection(sim_dir, seeds, out, grid=grid, real_data_dir="fake-data-dir")
    assert best in grid
    assert scored and {p for p, _ in scored} == {"full", "no_ball_at_all"}
    assert all(used == {p} for p, used in scored), scored   # every cell scored with its own policy's evidence only
    assert {"full", "no_ball_at_all"} <= set(policies)
    assert any(k.startswith("game_9001_1_s") for k in calls) and "real_tactical" in calls
    summ = pd.read_csv(out / "summary.csv")
    assert set(summ["cell"]) == {0, 1, 2}
    real_tab = pd.read_csv(out / "summary_real.csv")
    assert {"cell", "mean_rank", "non_blackout_D_share", "tag_agreement_implied", "pred_D_share",
            "spells_per_HA_min", "real_plausible"} <= set(real_tab.columns)
    assert set(real_tab["cell"]) == {0, 1, 2}
    b = json.loads((out / "best.json").read_text()); reg = json.loads((out / "registration.json").read_text())
    assert set(b["real_plausibility"]) == {"0", "1", "2"} and reg["real_plausibility"] == b["real_plausibility"]
    want = sel.choose_plausible(summ, {int(k): v for k, v in b["real_plausibility"].items()})
    assert b["cell"] == grid[want] == best
    for v in b["real_plausibility"].values():
        assert v["real_plausible"] == sel.is_real_plausible(v["non_blackout_D_share"])


def test_base_for_grid_enables_pathcrf_only_when_weighted():
    mod = _load_script()
    assert mod.base_for_grid(sel.GRID).use_pathcrf is False
    assert mod.base_for_grid(sel.GRID2).use_pathcrf is True
    assert mod.base_for_grid([{**sel.GRID2[0], "w_pathcrf": 0.0}]).use_pathcrf is False
    assert mod.base_for_grid(sel.GRID2, threads=8).pathcrf.threads == 8


def test_sealed_bar_gate(tmp_path):
    import pytest
    mod = _load_script()
    p = tmp_path / "best.json"
    p.write_text(json.dumps({"spec_bar": {"bar_pass": False}}))
    with pytest.raises(SystemExit):
        mod.check_bar_gate(p, None)
    assert mod.check_bar_gate(p, "registered exploratory read") is False
    p.write_text(json.dumps({"spec_bar": {"bar_pass": True}}))
    assert mod.check_bar_gate(p, None) is True


def test_resume_rejects_fold_csv_from_other_grid(tmp_path):
    import pytest
    mod = _load_script()
    grid = [{"tau_HA": 10.0, "tau_D": 30.0, "w_dead": 1.0, "w_pathcrf": 0.0, "dead_policy": "full", "aba_s": 1.0},
            {"tau_HA": 30.0, "tau_D": 100.0, "w_dead": 2.0, "w_pathcrf": 0.5, "dead_policy": "no_ball_at_all", "aba_s": 1.0}]
    df = pd.DataFrame([{"cell": i, **c} for i, c in enumerate(grid)] + [{"cell": "smooth50"}])
    mod.check_resume_grid(df, grid, tmp_path)   # same grid: fine
    for col, val in (("tau_D", 31.0), ("dead_policy", "full")):
        bad = df.copy(); bad.loc[1, col] = val
        with pytest.raises(SystemExit, match="different grid"):
            mod.check_resume_grid(bad, grid, tmp_path)
    with pytest.raises(SystemExit, match="different grid"):
        mod.check_resume_grid(df.iloc[[0, 2]], grid, tmp_path)   # a cell missing


def test_sealed_refuses_seed_mismatch(tmp_path, monkeypatch):
    import pytest
    mod = _load_script()
    (tmp_path / "sim").mkdir()
    (tmp_path / "best.json").write_text(json.dumps({"cell": sel.GRID[0], "spec_bar": {"bar_pass": True}}))
    (tmp_path / "registration.json").write_text(json.dumps({"seeds": [100, 101]}))
    monkeypatch.setattr(sys, "argv", ["select_lomo", "--sim", str(tmp_path / "sim"), "--seeds", "100", "102", "--sealed",
                                      "--best-from", str(tmp_path), "--out", str(tmp_path)])
    with pytest.raises(SystemExit, match="registered seeds"):
        mod.main()


def test_data_dir_unset_exits():
    import pytest
    mod = _load_script()
    with pytest.raises(SystemExit, match="MTAH_DATA_DIR"):
        mod.data_dir_or_exit(None)
    assert mod.data_dir_or_exit("x") == "x"
