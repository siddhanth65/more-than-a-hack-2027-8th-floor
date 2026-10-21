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
    r1 = mod.run_selection(sim_dir, seeds, out)
    best = r1["best_cell"]
    assert best in grid and r1["resumed_folds"] == []
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
    assert (out / "config.json").exists() and (out / "dead_model.joblib").exists()   # candidate artefacts at registration
    assert json.loads((out / "registration.json").read_text())["lengths_mode"] == sel.LENGTHS_MODE
    # resume: a second run reuses the fold tables
    r2 = mod.run_selection(sim_dir, seeds, out)
    assert r2["best_cell"] == best and sorted(r2["resumed_folds"]) == [9001, 9002, 9003]
    assert b["seeds"] == seeds and b["git"]
    # a changed grid must not silently resume
    import pytest
    monkeypatch.setattr(sel, "GRID", grid[:1])
    with pytest.raises(SystemExit):
        mod.run_selection(sim_dir, seeds, out)


def _fake_posterior(calls):
    def fake(match, cache_key, params):
        calls.append(cache_key)
        F = match.n_frames
        p = np.where(np.isnan(match.ball[:, 0]), np.nan, 0.6).astype(np.float32)
        return {"p_home": p, "p_none": np.zeros(F, np.float32), "imputed_frac": np.zeros(F, np.float32)}
    return fake


def test_run_selection_two_policies_pathcrf_and_real(tmp_path, monkeypatch):
    """grid2/grid4-style cells: per-policy dead and team-control models (no policy-mismatch error), PathCRF and teamctrl
    streams, evidence routed by (dead_policy, sp_dead_bonus, team_policy), real-match plausibility, team artefact."""
    import mtah.pipeline as pl
    rng = np.random.default_rng(1)
    sim_dir = tmp_path / "sim"
    seeds = [100, 101]
    for mid in (9001, 9002, 9003):
        for half in (1, 2):
            for seed in seeds:
                save_half(_fake_half(mid, half, seed, rng), sim_dir)
    grid = [{"tau_HA": 10.0, "tau_D": 30.0, "w_dead": 1.0, "w_pathcrf": 1.0, "dead_policy": "no_ball_at_all", "sp_dead_bonus": 0.0, "aba_s": 1.0},
            {"tau_HA": 10.0, "tau_D": 30.0, "w_dead": 1.0, "w_pathcrf": 0.0, "dead_policy": "full", "sp_dead_bonus": 0.0, "aba_s": 1.0},
            {"tau_HA": 30.0, "tau_D": 100.0, "w_dead": 2.0, "w_pathcrf": 0.5, "dead_policy": "full", "sp_dead_bonus": 3.0, "aba_s": 1.0},
            {"tau_HA": 30.0, "tau_D": 100.0, "w_dead": 2.0, "w_pathcrf": 0.5, "dead_policy": "full", "sp_dead_bonus": 0.0,
             "w_teamctrl": 2.0, "w_ball": 0.5, "team_policy": "no_touch", "aba_s": 1.0}]
    calls = []
    monkeypatch.setattr(pl, "pathcrf_posterior", _fake_posterior(calls))
    mod = _load_script()
    policies = []
    orig = mod.train_dead
    monkeypatch.setattr(mod, "train_dead", lambda sims, variant="ctrl_Dfull", policy="full": policies.append(policy) or orig(sims, variant, policy))
    team_policies = []
    orig_tt = mod.train_team_ctrl
    touch_seen = []
    monkeypatch.setattr(mod, "train_team_ctrl", lambda sims, variant="ctrl_Dfull", policy="full", **kw: team_policies.append(policy)
                        or touch_seen.append(kw.get("touch_params")) or orig_tt(sims, variant, policy, **kw))
    real = _fake_half(9999, 1, 0, rng)
    monkeypatch.setattr(mod, "load_tactical", lambda d: real.match)
    monkeypatch.setattr(mod, "load_tags", lambda d: real.tags)
    ev_policy = {}   # id(evidence element) -> dead_policy it was computed for (elements kept alive in ev_keep)
    ev_keep = []
    orig_ev = mod.compute_evidence
    def spy_ev(match, tags, cfg, dead, **kw):
        tm = kw.get("team_model")
        assert tm is None or tm.policy == cfg.team_policy
        r = orig_ev(match, tags, cfg, dead, **kw)
        assert "teamctrl" in r   # some cell weights it: every evidence set carries the stream
        ev_policy[id(r)] = (cfg.dead_policy, getattr(cfg.tag, "sp_dead_bonus", 0.0), cfg.team_policy); ev_keep.append(r); return r
    monkeypatch.setattr(mod, "compute_evidence", spy_ev)
    scored = []   # (cell policy, {policies of the evidence it was scored with})
    orig_sc = mod.score_cell
    def spy_sc(test, ev, cfg, *a, **kw):
        scored.append(((cfg.dead_policy, getattr(cfg.tag, "sp_dead_bonus", 0.0), cfg.team_policy), {ev_policy[id(e)] for e in ev})); return orig_sc(test, ev, cfg, *a, **kw)
    monkeypatch.setattr(mod, "score_cell", spy_sc)
    out = tmp_path / "out"
    best = mod.run_selection(sim_dir, seeds, out, grid=grid, real_data_dir="fake-data-dir")["best_cell"]
    assert best in grid
    assert scored and {p for p, _ in scored} == {("full", 0.0, "full"), ("full", 3.0, "full"), ("no_ball_at_all", 0.0, "full"),
                                                 ("full", 0.0, "no_touch")}
    assert all(used == {p} for p, used in scored), scored   # every cell scored with its own (policy, bonus, team) evidence only
    assert {"full", "no_ball_at_all"} <= set(policies)
    assert {"full", "no_touch"} <= set(team_policies)
    assert touch_seen and all(tp is not None for tp in touch_seen)   # base.touch reaches every team-model fit
    assert any(k.startswith("game_9001_1_s") for k in calls) and "real_tactical" in calls
    summ = pd.read_csv(out / "summary.csv")
    assert set(summ["cell"]) == {0, 1, 2, 3}
    real_tab = pd.read_csv(out / "summary_real.csv")
    assert {"cell", "mean_rank", "non_blackout_D_share", "tag_agreement_implied", "pred_D_share",
            "spells_per_graded_min", "spells_per_HA_min", "d_share_ok", "spell_rate_ok", "real_plausible"} <= set(real_tab.columns)
    assert set(real_tab["cell"]) == {0, 1, 2, 3}
    b = json.loads((out / "best.json").read_text()); reg = json.loads((out / "registration.json").read_text())
    assert set(b["real_plausibility"]) == {"0", "1", "2", "3"} and reg["real_plausibility"] == b["real_plausibility"]
    want = sel.choose_plausible(summ, {int(k): v for k, v in b["real_plausibility"].items()})
    assert b["cell"] == grid[want] == best
    for v in b["real_plausibility"].values():
        assert v["real_plausible"] == sel.is_real_plausible(v["non_blackout_D_share"], v["spells_per_graded_min"])
        assert v["spell_rate_ok"] == (sel.REAL_SPELL_RATE_BAND[0] <= v["spells_per_graded_min"] <= sel.REAL_SPELL_RATE_BAND[1])
        assert v["real_plausible"] == (v["d_share_ok"] and v["spell_rate_ok"])
    assert b["real_bands"] == {"non_blackout_D_share": list(sel.REAL_D_SHARE_BAND),
                               "spells_per_graded_min": list(sel.REAL_SPELL_RATE_BAND)}
    uses_team = best.get("w_teamctrl", 0.0) > 0
    assert (out / "team_model.joblib").exists() == uses_team and reg["use_teamctrl"] == uses_team
    assert reg["team_policy"] == (best["team_policy"] if uses_team else None)


def test_base_for_grid_enables_pathcrf_only_when_weighted():
    mod = _load_script()
    assert mod.base_for_grid(sel.GRID4).use_teamctrl is True and mod.base_for_grid(sel.GRID3).use_teamctrl is False
    assert mod.base_for_grid([c for c in sel.GRID4 if c["w_teamctrl"] == 0]).use_teamctrl is False
    assert mod.ev_key(sel.GRID4[0]) == ("full", 0.0, sel.GRID4[0]["team_policy"], 0.0) and mod.ev_key(sel.GRID[0]) == ("full", 0.0, "full", 0.0)
    assert mod.base_for_grid(sel.GRID).use_pathcrf is False
    assert mod.base_for_grid(sel.GRID2).use_pathcrf is True
    assert mod.base_for_grid([{**sel.GRID2[0], "w_pathcrf": 0.0}]).use_pathcrf is False
    assert mod.base_for_grid(sel.GRID2, threads=8).pathcrf.threads == 8


def test_sealed_bar_gate(tmp_path):
    import pytest
    mod = _load_script()
    best = {"spec_bar": {"bar_pass": False}}
    with pytest.raises(SystemExit):
        mod.check_bar_gate(best, None)
    assert mod.check_bar_gate(best, "registered exploratory read") == ["bar"]   # forced
    assert mod.check_bar_gate({"spec_bar": {"bar_pass": True}}, None) == []


def test_resume_rejects_fold_csv_from_other_grid(tmp_path):
    import pytest
    mod = _load_script()
    grid = [{"tau_HA": 10.0, "tau_D": 30.0, "w_dead": 1.0, "w_pathcrf": 0.0, "dead_policy": "full", "sp_dead_bonus": 0.0, "aba_s": 1.0},
            {"tau_HA": 30.0, "tau_D": 100.0, "w_dead": 2.0, "w_pathcrf": 0.5, "dead_policy": "no_ball_at_all", "sp_dead_bonus": 0.0, "aba_s": 1.0}]
    df = pd.DataFrame([{"cell": i, **c} for i, c in enumerate(grid)] + [{"cell": "smooth50"}])
    fp = mod.run_fingerprint([100, 101], grid, "grid2")
    meta = dict(fp)
    mod.check_resume(meta, df, grid, fp, tmp_path)   # same grid and seeds: fine
    for col, val in (("tau_D", 31.0), ("dead_policy", "full"), ("sp_dead_bonus", 3.0)):
        bad = df.copy(); bad.loc[1, col] = val
        with pytest.raises(SystemExit, match="different grid"):
            mod.check_resume(meta, bad, grid, fp, tmp_path)
    with pytest.raises(SystemExit, match="different grid"):
        mod.check_resume(meta, df.iloc[[0, 2]], grid, fp, tmp_path)   # a cell missing
    with pytest.raises(SystemExit, match="different grid"):
        mod.check_resume({**meta, "grid_hash": "x"}, df, grid, fp, tmp_path)
    with pytest.raises(SystemExit, match="seeds"):
        mod.check_resume({**meta, "seeds": [100, 102]}, df, grid, fp, tmp_path)
    with pytest.raises(SystemExit, match="fingerprint"):
        mod.check_resume(None, df, grid, fp, tmp_path)


def test_sealed_refuses_seed_mismatch(tmp_path, monkeypatch):
    import pytest
    mod = _load_script()
    (tmp_path / "sim").mkdir()
    (tmp_path / "best.json").write_text(json.dumps({"cell": sel.GRID[0], "spec_bar": {"bar_pass": True},
                                                    "git": "abc", "seeds": [100, 102]}))
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


def test_sealed_main_defaults_out_to_best_from_and_guards_registration(tmp_path, monkeypatch):
    """--sealed --best-from X without --out writes config/dead model into X; the read is logged per registration
    in the (tracked) sealed log, a repeat read is refused unless forced, and an implausible chosen cell is gated."""
    import pytest
    from mtah.io import readlog
    from mtah.sim.dataset import SEALED_MATCH
    monkeypatch.setattr(readlog, "READS_DIR", tmp_path / "reads")
    rng = np.random.default_rng(2)
    sim_dir = tmp_path / "sim"; seeds = [100, 101]
    for mid in (9001, 9002, 9003, SEALED_MATCH):
        for half in (1, 2):
            for seed in seeds:
                save_half(_fake_half(mid, half, seed, rng), sim_dir)
    grid = [{"tau_HA": 3.0, "tau_D": 3.0, "w_dead": 1.0, "aba_s": 0.5},
            {"tau_HA": 10.0, "tau_D": 10.0, "w_dead": 1.0, "aba_s": 0.5}]
    monkeypatch.setattr(sel, "GRID", grid)
    mod = _load_script()
    out = tmp_path / "sel"
    r = mod.run_selection(sim_dir, seeds, out)
    assert r["chosen_real_plausible"] is None
    b = json.loads((out / "best.json").read_text())
    assert "chosen_real_plausible" in b and b["chosen_real_plausible"] is None
    b["spec_bar"]["bar_pass"] = True; (out / "best.json").write_text(json.dumps(b))
    argv = ["select_lomo", "--sim", str(sim_dir), "--seeds", "100", "101", "--sealed", "--best-from", str(out)]
    monkeypatch.setattr(sys, "argv", argv)
    before = ((out / "config.json").read_bytes(), (out / "dead_model.joblib").read_bytes())
    trained = []
    orig_td = mod.train_dead
    monkeypatch.setattr(mod, "train_dead", lambda *a, **k: trained.append(1) or orig_td(*a, **k))
    mod.main()
    assert trained == []   # the sealed path loads the registered artefacts, it never retrains
    assert before == ((out / "config.json").read_bytes(), (out / "dead_model.joblib").read_bytes())
    assert (out / "config.json").exists() and (out / "dead_model.joblib").exists()   # M3: --out defaulted to --best-from
    log = readlog.read_log("sealed")
    reg = json.loads((out / "registration.json").read_text())
    assert len(log) == 1 and log[0]["registration"] == mod.registration_id(reg) and log[0]["match"] == SEALED_MATCH
    assert log[0]["select_dir"] == out.name   # outside the repo: basename only
    assert log[0]["forced"] is False and log[0]["force_reason"] is None and set(log[0]["sealed"]) == set(b["scores"])
    with pytest.raises(SystemExit, match="already read"):
        mod.main()
    monkeypatch.setattr(sys, "argv", argv + ["--force-sealed", "smoke re-read"])
    mod.main()
    log = readlog.read_log("sealed")
    assert len(log) == 2 and log[1]["forced"] is True and log[1]["forced_over"] == ["repeat_read"]
    assert log[1]["force_reason"] == "smoke re-read"
    b["chosen_real_plausible"] = False; (out / "best.json").write_text(json.dumps(b))
    monkeypatch.setattr(sys, "argv", argv)
    with pytest.raises(SystemExit, match="real_plausible"):
        mod.main()
    assert len(readlog.read_log("sealed")) == 2


def test_candidate_artifacts_with_team_model_and_sealed_path(tmp_path, monkeypatch):
    """A teamctrl cell writes team_model.joblib at registration; the sealed path loads it (never retrains) and
    feeds it to compute_evidence; a teamctrl config without the file is refused."""
    import pytest
    from mtah.io import readlog
    from mtah.pipeline import PipelineConfig
    from mtah.sim.dataset import SEALED_MATCH
    monkeypatch.setattr(readlog, "READS_DIR", tmp_path / "reads")
    rng = np.random.default_rng(3)
    sim_dir = tmp_path / "sim"; seeds = [100]
    for mid in (9001, 9002, 9003, SEALED_MATCH):
        for half in (1, 2):
            save_half(_fake_half(mid, half, seeds[0], rng), sim_dir)
    grid = [{"tau_HA": 10.0, "tau_D": 10.0, "w_dead": 1.0, "w_teamctrl": 2.0, "w_ball": 0.5, "team_policy": "no_touch",
             "aba_s": 0.5}]
    mod = _load_script()
    out = tmp_path / "sel"
    mod.run_selection(sim_dir, seeds, out, grid=grid, base=mod.base_for_grid(grid))
    assert (out / "team_model.joblib").exists()
    cfg, dead, lengths = mod.load_candidate_artifacts(out)
    assert cfg.use_teamctrl and cfg.team_policy == "no_touch" and cfg.decoder.weights["teamctrl"] == 2.0
    assert mod.load_team_model(out, cfg).policy == "no_touch"
    reg = json.loads((out / "registration.json").read_text())
    assert reg["use_teamctrl"] is True and reg["team_policy"] == "no_touch"
    b = json.loads((out / "best.json").read_text()); b["spec_bar"]["bar_pass"] = True
    (out / "best.json").write_text(json.dumps(b))
    trained, used = [], []
    monkeypatch.setattr(mod, "train_team_ctrl", lambda *a, **k: trained.append(1))
    orig_ev = mod.compute_evidence
    def spy_ev(match, tags, cfg, dead, **kw):
        used.append(kw.get("team_model")); return orig_ev(match, tags, cfg, dead, **kw)
    monkeypatch.setattr(mod, "compute_evidence", spy_ev)
    monkeypatch.setattr(sys, "argv", ["select_lomo", "--sim", str(sim_dir), "--seeds", "100", "--sealed", "--best-from", str(out)])
    mod.main()
    assert trained == [] and used and all(t is not None and t.policy == "no_touch" for t in used)
    assert readlog.read_log("sealed")[-1]["match"] == SEALED_MATCH
    (out / "team_model.joblib").unlink()
    with pytest.raises(SystemExit, match="team_model"):
        mod.load_team_model(out, cfg)
    assert mod.load_team_model(out, PipelineConfig()) is None    # stream off: nothing needed


def test_resume_rejects_changed_team_columns(tmp_path):
    import pytest
    mod = _load_script()
    grid = [dict(c) for c in sel.GRID4[:3]]
    df = pd.DataFrame([{"cell": i, **c} for i, c in enumerate(grid)] + [{"cell": "smooth50"}])
    fp = mod.run_fingerprint([100], grid, "grid4")
    mod.check_resume(dict(fp), df, grid, fp, tmp_path)
    for col, val in (("w_teamctrl", 9.0), ("w_ball", 9.0), ("team_policy", "x")):
        bad = df.copy(); bad.loc[1, col] = val
        with pytest.raises(SystemExit, match="different grid"):
            mod.check_resume(dict(fp), bad, grid, fp, tmp_path)


def test_run_selection_merge_cells_route_model_and_artifacts(tmp_path, monkeypatch):
    """merge_theta < 1 cells: one merge model per fold, trained on an inner split of the fold's TRAIN matches only,
    passed to score_cell for merge cells only; the registered merge_model.joblib is written, reloaded by the sealed
    path (never retrained) and refused when missing; merge_theta joins the resume columns."""
    import pytest
    from mtah.io import readlog
    from mtah.sim.dataset import SEALED_MATCH, match_id
    from mtah.spells.merge import MergeModel
    monkeypatch.setattr(readlog, "READS_DIR", tmp_path / "reads")
    rng = np.random.default_rng(4)
    sim_dir = tmp_path / "sim"; seeds = [100]
    for mid in (9001, 9002, 9003, SEALED_MATCH):
        for half in (1, 2):
            save_half(_fake_half(mid, half, seeds[0], rng), sim_dir)
    grid = [{"tau_HA": 10.0, "tau_D": 10.0, "w_dead": 1.0, "aba_s": 0.5, "merge_theta": 0.7},
            {"tau_HA": 10.0, "tau_D": 10.0, "w_dead": 1.0, "aba_s": 0.5, "merge_theta": 1.0}]
    mod = _load_script()
    trained_on = []
    orig_tmt = sel.train_merge_model
    def spy_tmt(train, base, g, lengths, **kw):
        trained_on.append(sorted({match_id(s.key) for s in train}))
        return orig_tmt(train, base, g, lengths, **kw)
    monkeypatch.setattr(sel, "train_merge_model", spy_tmt)
    inner_held = []
    orig_mtt = sel.merge_training_table
    def spy_mtt(train, *a, **kw):
        df, info = orig_mtt(train, *a, **kw); inner_held.append(info["held_match"]); return df, info
    monkeypatch.setattr(sel, "merge_training_table", spy_mtt)
    scored = []
    orig_sc = mod.score_cell
    def spy_sc(test, ev, cfg, *a, **kw):
        scored.append((cfg.post.merge_theta, kw.get("merge_model"))); return orig_sc(test, ev, cfg, *a, **kw)
    monkeypatch.setattr(mod, "score_cell", spy_sc)
    # force the merge cell to be chosen: rank by a summary in which cell 0 wins
    out = tmp_path / "sel"
    monkeypatch.setattr(mod, "mean_rank", lambda t, v: t.assign(mean_rank=t["cell"].astype(float) + 1.0)
                        .sort_values("mean_rank").reset_index(drop=True))
    r = mod.run_selection(sim_dir, seeds, out, grid=grid)
    assert r["best_cell"] == grid[0]
    folds = {9001: [9002, 9003], 9002: [9001, 9003], 9003: [9001, 9002]}
    assert trained_on[:3] == [folds[h] for h in (9001, 9002, 9003)]          # fold models: train matches only
    assert all(SEALED_MATCH not in t for t in trained_on)
    assert inner_held[:3] == [9003, 9003, 9002]                               # inner held-out = last train match
    for h in (9001, 9002, 9003):
        info = json.loads((out / f"fold_{h}_merge.json").read_text())
        assert info["n_halves"] == 2 and info["n_ref_cells"] == 1 and "label_seconds" in info
    fold_scores = scored[:6]
    assert all((m is not None) == (th < 1.0) for th, m in fold_scores) and {th for th, _ in fold_scores} == {0.7, 1.0}
    assert all(isinstance(m, MergeModel) for th, m in fold_scores if th < 1.0)
    assert (out / "merge_model.joblib").exists()
    reg = json.loads((out / "registration.json").read_text())
    assert reg["use_merge"] is True and reg["merge_model"]["held_match"] == 9003   # registration: inner split of all non-sealed
    cfg, dead, lengths = mod.load_candidate_artifacts(out)
    assert cfg.post.merge_theta == 0.7 and isinstance(mod.load_merge_model(out, cfg), MergeModel)
    df = pd.read_csv(out / "fold_9001.csv")
    assert "merge_theta" in df.columns and "merge_theta" in mod.GRID_PARAM_COLS
    bad = df.copy(); bad.loc[0, "merge_theta"] = 1.0
    fp = mod.run_fingerprint(seeds, grid, "custom")
    with pytest.raises(SystemExit, match="different grid"):
        mod.check_resume(dict(fp), bad, grid, fp, tmp_path)
    # sealed path: loads the registered merge model, never retrains it
    b = json.loads((out / "best.json").read_text()); b["spec_bar"]["bar_pass"] = True; b["chosen_real_plausible"] = None
    (out / "best.json").write_text(json.dumps(b))
    retrained = []
    monkeypatch.setattr(sel, "train_merge_model", lambda *a, **k: retrained.append(1))
    scored.clear()
    monkeypatch.setattr(sys, "argv", ["select_lomo", "--sim", str(sim_dir), "--seeds", "100", "--sealed", "--best-from", str(out)])
    mod.main()
    assert retrained == [] and scored and all(isinstance(m, MergeModel) for _, m in scored)
    (out / "merge_model.joblib").unlink()
    with pytest.raises(SystemExit, match="merge_model"):
        mod.load_merge_model(out, cfg)
    from mtah.pipeline import PipelineConfig
    assert mod.load_merge_model(out, PipelineConfig()) is None


def test_run_selection_switch_cells_route_model_and_artifacts(tmp_path, monkeypatch):
    """w_switch / snap_mode cells: one switch model per fold on the fold's TRAIN halves (the merge model's inner split
    trains its own on the inner matches), the hazard rides in every evidence set, w_switch = 0 cells decode exactly
    as without it; the registered switch_model.joblib is written, reloaded by the sealed path (never retrained) and
    refused when missing; w_switch / snap_mode join the resume columns."""
    import pytest
    from mtah.evidence.switch import SwitchModel
    from mtah.io import readlog
    from mtah.pipeline import PipelineConfig, predict
    from mtah.sim.dataset import SEALED_MATCH, match_id
    monkeypatch.setattr(readlog, "READS_DIR", tmp_path / "reads")
    rng = np.random.default_rng(5)
    sim_dir = tmp_path / "sim"; seeds = [100]
    for mid in (9001, 9002, 9003, SEALED_MATCH):
        for half in (1, 2):
            save_half(_fake_half(mid, half, seeds[0], rng), sim_dir)
    c0 = {"tau_HA": 10.0, "tau_D": 10.0, "w_dead": 1.0, "aba_s": 0.5, "w_switch": 0.0, "snap_mode": "touch"}
    grid = [{**c0, "w_switch": 2.0, "snap_mode": "hazard"}, c0, {**c0, "w_switch": 2.0, "snap_mode": "hazard", "merge_theta": 0.7}]
    mod = _load_script()
    base = mod.base_for_grid(grid)
    assert base.use_switch and base.switch_policy == sel.SWITCH_POLICY == "no_team"
    assert not mod.base_for_grid([c0]).use_switch and mod.base_for_grid([{**c0, "snap_mode": "hazard"}]).use_switch
    switch_calls = []   # (sorted training match ids, policy)
    orig_ts = mod.train_switch
    def spy_ts(sims, **kw):
        switch_calls.append((sorted({match_id(s.key) for s in sims}), kw.get("policy")))
        return orig_ts(sims, **kw)
    monkeypatch.setattr(mod, "train_switch", spy_ts)
    orig_ev = mod.compute_evidence
    def spy_ev(match, tags, cfg, dead, **kw):
        r = orig_ev(match, tags, cfg, dead, **kw)
        assert "switch" in r and isinstance(kw.get("switch_model"), SwitchModel)
        return r
    monkeypatch.setattr(mod, "compute_evidence", spy_ev)
    scored, identical = [], []
    orig_sc = mod.score_cell
    def spy_sc(test, ev, cfg, *a, **kw):
        scored.append((cfg.decoder.w_switch, cfg.post.snap_mode))
        if cfg.decoder.w_switch == 0 and cfg.post.snap_mode == "touch" and cfg.post.merge_theta >= 1.0:
            for s, e in zip(test, ev):
                assert "switch" in e
                with_h = predict(s.match, s.tags, cfg, None, evidence=e)
                without = predict(s.match, s.tags, cfg, None, evidence={k: v for k, v in e.items() if k != "switch"})
                identical.append(bool(np.array_equal(with_h, without)))
        return orig_sc(test, ev, cfg, *a, **kw)
    monkeypatch.setattr(mod, "score_cell", spy_sc)
    monkeypatch.setattr(mod, "mean_rank", lambda t, v: t.assign(mean_rank=t["cell"].astype(float) + 1.0)
                        .sort_values("mean_rank").reset_index(drop=True))   # cell 0 (switch, no merge) wins
    out = tmp_path / "sel"
    r = mod.run_selection(sim_dir, seeds, out, grid=grid, base=base)
    assert r["best_cell"] == grid[0]
    folds = {9001: [9002, 9003], 9002: [9001, 9003], 9003: [9001, 9002]}
    fold_calls = [c for c in switch_calls if len(c[0]) == 2]
    assert [c[0] for c in fold_calls[:3]] == [folds[h] for h in (9001, 9002, 9003)]   # once per fold, train matches
    assert len([c for c in switch_calls[:6] if len(c[0]) == 2]) == 3                 # ... and only once
    assert all(len(c[0]) == 1 for c in switch_calls[:6] if len(c[0]) != 2)            # merge inner split: 1 match
    assert all(SEALED_MATCH not in c[0] and c[1] == "no_team" for c in switch_calls)
    assert identical and all(identical)                                               # w_switch 0: hazard unused
    assert {(0.0, "touch"), (2.0, "hazard")} <= set(scored)
    assert (out / "switch_model.joblib").exists()
    reg = json.loads((out / "registration.json").read_text())
    assert reg["use_switch"] is True and reg["switch_policy"] == "no_team"
    cfg, dead, lengths = mod.load_candidate_artifacts(out)
    assert cfg.use_switch and cfg.switch_policy == "no_team" and cfg.decoder.w_switch == 2.0 and cfg.post.snap_mode == "hazard"
    sw = mod.load_switch_model(out, cfg)
    assert isinstance(sw, SwitchModel) and sw.policy == "no_team" and not sw.with_teamctrl
    df = pd.read_csv(out / "fold_9001.csv")
    assert {"w_switch", "snap_mode"} <= set(df.columns) and {"w_switch", "snap_mode"} <= set(mod.GRID_PARAM_COLS)
    fp = mod.run_fingerprint(seeds, grid, "custom")
    for col, val in (("w_switch", 0.0), ("snap_mode", "touch")):
        bad = df.copy(); bad.loc[0, col] = val
        with pytest.raises(SystemExit, match="different grid"):
            mod.check_resume(dict(fp), bad, grid, fp, tmp_path)
    # sealed path: loads the registered switch model, never retrains it, feeds it to compute_evidence
    b = json.loads((out / "best.json").read_text()); b["spec_bar"]["bar_pass"] = True; b["chosen_real_plausible"] = None
    (out / "best.json").write_text(json.dumps(b))
    switch_calls.clear()
    monkeypatch.setattr(sys, "argv", ["select_lomo", "--sim", str(sim_dir), "--seeds", "100", "--sealed", "--best-from", str(out)])
    mod.main()
    assert switch_calls == [] and readlog.read_log("sealed")[-1]["match"] == SEALED_MATCH
    (out / "switch_model.joblib").unlink()
    with pytest.raises(SystemExit, match="switch_model"):
        mod.load_switch_model(out, cfg)
    assert mod.load_switch_model(out, PipelineConfig()) is None    # use_switch off: nothing needed


def test_sealed_match_never_reaches_model_training(tmp_path, monkeypatch):
    """Teamctrl + switch + merge cells with the sealed match present in the sim dir: no dead/team/switch/merge training
    set (folds, merge inner splits, real-plausibility, registration) ever contains SEALED_MATCH. Also: label-identical
    duplicate cells (w_teamctrl 0, team_policy no_touch, no merge) take no real-match slot."""
    from mtah.sim.dataset import SEALED_MATCH, match_id
    rng = np.random.default_rng(6)
    sim_dir = tmp_path / "sim"; seeds = [100]
    for mid in (9001, 9002, 9003, SEALED_MATCH):
        for half in (1, 2):
            save_half(_fake_half(mid, half, seeds[0], rng), sim_dir)
    c0 = {"tau_HA": 10.0, "tau_D": 10.0, "w_dead": 1.0, "aba_s": 0.5, "w_teamctrl": 0.0, "team_policy": "full",
          "w_switch": 0.0, "snap_mode": "touch", "merge_theta": 1.0}
    grid = [{**c0, "w_teamctrl": 2.0, "w_switch": 2.0, "snap_mode": "hazard", "merge_theta": 0.7},
            c0, {**c0, "team_policy": "no_touch"}, {**c0, "team_policy": "no_touch", "merge_theta": 0.7}]
    assert sel.label_duplicate_cells(grid) == {2}
    mod = _load_script()
    seen = []   # (trainer, sorted match ids)
    def spy(name, fn):
        def f(sims, *a, **kw):
            seen.append((name, sorted({match_id(s.key) for s in sims}))); return fn(sims, *a, **kw)
        return f
    monkeypatch.setattr(mod, "train_dead", spy("dead", mod.train_dead))
    monkeypatch.setattr(mod, "train_team_ctrl", spy("team", mod.train_team_ctrl))
    monkeypatch.setattr(mod, "train_switch", spy("switch", mod.train_switch))
    monkeypatch.setattr(sel, "merge_training_table", spy("merge", sel.merge_training_table))
    real = _fake_half(9999, 1, 0, rng)
    monkeypatch.setattr(mod, "load_tactical", lambda d: real.match)
    monkeypatch.setattr(mod, "load_tags", lambda d: real.tags)
    out = tmp_path / "sel"
    mod.run_selection(sim_dir, seeds, out, grid=grid, real_data_dir="fake-data-dir")
    assert {n for n, _ in seen} == {"dead", "team", "switch", "merge"}
    assert all(SEALED_MATCH not in mids and mids for _, mids in seen), seen
    real_tab = pd.read_csv(out / "summary_real.csv")
    assert set(real_tab["cell"]) == {0, 1, 3}          # cell 2 repeats cell 1's labels: skipped


def test_write_candidate_artifacts_deletes_stale_models(tmp_path):
    from mtah.pipeline import PipelineConfig
    mod = _load_script()
    class Dummy:
        def save(self, p):
            Path(p).write_bytes(b"x")
    for name in (mod.TEAM_MODEL_FILE, mod.MERGE_MODEL_FILE, mod.SWITCH_MODEL_FILE):
        (tmp_path / name).write_bytes(b"stale")
    cell = {"tau_HA": 10.0, "tau_D": 10.0, "w_dead": 1.0, "aba_s": 0.5}
    mod.write_candidate_artifacts(tmp_path, PipelineConfig(), cell, {"ball": 1.0}, Dummy())
    assert (tmp_path / "dead_model.joblib").exists() and (tmp_path / "config.json").exists()
    assert not any((tmp_path / n).exists() for n in (mod.TEAM_MODEL_FILE, mod.MERGE_MODEL_FILE, mod.SWITCH_MODEL_FILE))
    base = PipelineConfig(); base.use_switch = True; base.switch_policy = sel.SWITCH_POLICY
    mod.write_candidate_artifacts(tmp_path, base, {**cell, "merge_theta": 0.7, "w_switch": 2.0}, {"ball": 1.0}, Dummy(),
                                  merge=Dummy(), switch=Dummy())
    assert (tmp_path / mod.MERGE_MODEL_FILE).exists() and (tmp_path / mod.SWITCH_MODEL_FILE).exists()
    assert not (tmp_path / mod.TEAM_MODEL_FILE).exists()


def _load_make_candidate():
    spec = importlib.util.spec_from_file_location("make_candidate_script", ROOT / "scripts" / "make_candidate.py")
    mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
    return mod


def test_merge_cell_without_teamctrl_weight_registers_stream_and_reproduces_labels(tmp_path, monkeypatch):
    """T3 review CRITICAL #2: the grid has a w_teamctrl > 0 cell, so selection decodes every cell with the teamctrl
    stream; the chosen merge cell has w_teamctrl 0 but its merge features read that stream. Registration keeps
    use_teamctrl (weight 0), writes team_model.joblib, and make_candidate's loader + prediction reproduce the
    selection-time (real-plausibility) labels bit-for-bit."""
    from mtah.spells.merge import match_spell_table
    from mtah.pipeline import compute_strong
    rng = np.random.default_rng(7)
    sim_dir = tmp_path / "sim"; seeds = [100]
    for mid in (9001, 9002, 9003):
        for half in (1, 2):
            save_half(_fake_half(mid, half, seeds[0], rng), sim_dir)
    c0 = {"tau_HA": 10.0, "tau_D": 10.0, "w_dead": 1.0, "aba_s": 0.5, "w_teamctrl": 0.0, "team_policy": "no_touch",
          "merge_theta": 0.7}
    grid = [c0, {**c0, "w_teamctrl": 2.0, "merge_theta": 1.0}]
    mod = _load_script()
    real = _fake_half(9999, 1, 0, rng)
    monkeypatch.setattr(mod, "load_tactical", lambda d: real.match)
    monkeypatch.setattr(mod, "load_tags", lambda d: real.tags)
    monkeypatch.setattr(mod, "choose_plausible", lambda ranked, plaus: 0)     # the merge cell is registered
    sel_labels = []
    orig_pr = mod.predict
    def spy_pr(match, tags, cfg, dead, **kw):
        lab = orig_pr(match, tags, cfg, dead, **kw)
        if match is real.match and cfg.post.merge_theta < 1.0:
            assert "teamctrl" in kw["evidence"]
            sel_labels.append(lab.copy())
        return lab
    monkeypatch.setattr(mod, "predict", spy_pr)
    out = tmp_path / "sel"
    r = mod.run_selection(sim_dir, seeds, out, grid=grid, real_data_dir="fake-data-dir")
    assert r["best_cell"] == c0 and len(sel_labels) == 1
    reg = json.loads((out / "registration.json").read_text())
    assert reg["use_teamctrl"] is True and reg["team_policy"] == "no_touch" and reg["use_merge"] is True
    assert (out / "team_model.joblib").exists() and (out / "merge_model.joblib").exists()
    mc = _load_make_candidate()
    cfg, dead, lengths, team, merge, switch = mc.load_selection(out)
    assert cfg.use_teamctrl and cfg.decoder.weights["teamctrl"] == 0.0 and cfg.post.merge_theta == 0.7
    assert team is not None and team.policy == "no_touch" and merge is not None and switch is None
    assert mod.load_team_model(out, cfg) is not None                         # the sealed path loads it too
    lab = mc.candidate_labels(real.match, real.tags, cfg, dead, team, merge, switch)
    assert np.array_equal(lab, sel_labels[0])
    ev = mc.compute_evidence(real.match, real.tags, cfg, dead, cache_key="real_tactical", team_model=team)
    tab = match_spell_table(lab, real.match, ev, compute_strong(real.match, cfg))
    assert np.isfinite(tab["teamctrl_mean"].to_numpy(float)).any()          # merge features see the stream
