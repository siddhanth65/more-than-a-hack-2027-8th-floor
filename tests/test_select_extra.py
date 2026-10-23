"""Training-only extra halves (SkillCorner) in the LOMO selection: weights, grouping, never in test / held sets."""
import json
from dataclasses import replace

import numpy as np
import pytest

import mtah.select as sel
from mtah.evidence import dead as dead_mod
from mtah.sim.dataset import load_extra_halves, save_half
from tests.test_select_smoke import _fake_half, _load_script

SEEDS = [100]


def _write(tmp_path, dfl=(9001, 9002, 9003), extra=(9101, 9102)):
    rng = np.random.default_rng(0)
    sim_dir, ex_dir = tmp_path / "sim", tmp_path / "sim_extra"
    for mid in dfl:
        for half in (1, 2):
            save_half(_fake_half(mid, half, 100, rng), sim_dir)
    for mid in extra:
        for half in (1, 2):
            save_half(_fake_half(mid, half, 100, rng), ex_dir)
    return sim_dir, ex_dir


def _extra(ex_dir, w=1.0):
    return sel.ExtraSet(load_extra_halves(ex_dir, SEEDS), w, ex_dir)


CELL = {"tau_HA": 3.0, "tau_D": 3.0, "w_dead": 1.0, "aba_s": 0.5}


def test_grid13_shape():
    g = sel.GRIDS["grid13"]
    assert len(g) == 9 and g is sel.GRID13
    assert sorted({c["extra_weight"] for c in g}) == [0.0, 0.5, 1.0]
    assert sorted({c["dead_offset"] for c in g}) == [-2.0, -1.0, 0.0]
    for c in g:   # everything else is a5c's cell (GRID8[0])
        assert {k: v for k, v in c.items() if k not in ("extra_weight", "dead_offset")} == \
               {k: v for k, v in sel.GRID8[0].items() if k != "dead_offset"}
    assert {"extra_weight": 0.0, "dead_offset": -2.0}.items() <= {k: g[0][k] for k in ("extra_weight", "dead_offset")}.items()
    assert sel.evidence_key(g[0]) != sel.evidence_key(g[1])   # the offset is evidence; extra_weight is a model-group axis


def test_extra_weight_groups_and_defaults(tmp_path):
    sim_dir, ex_dir = _write(tmp_path)
    grid = [{**CELL}, {**CELL, "extra_weight": 0.5}, {**CELL, "extra_weight": 0.0}, {**CELL, "extra_weight": 0.5}]
    ex = _extra(ex_dir, 0.7)
    assert sel.extra_weight_groups(grid, ex) == {0.0: [2], 0.5: [1, 3], 0.7: [0]}   # cell without one: the default W
    with pytest.raises(SystemExit):
        sel.extra_weight_groups(grid, None)   # weight > 0 without an extra set
    assert sel.extra_weight_groups([CELL], None) == {0.0: [0]}


def test_extraset_views_and_loader(tmp_path):
    _, ex_dir = _write(tmp_path)
    ex = _extra(ex_dir)
    assert len(ex.sims) == 4 and all(s.extra and s.weight == 1.0 for s in ex.sims)
    assert ex.at(0.0) == [] and [s.weight for s in ex.at(0.5)] == [0.5] * 4 and all(s.extra for s in ex.at(0.5))
    assert ex.info()["n_halves"] == 4 and ex.info()["keys_sha"]
    with pytest.raises(SystemExit):
        load_extra_halves(ex_dir, [999])


def test_trainers_use_weights_and_skip_zero(tmp_path, monkeypatch):
    sim_dir, ex_dir = _write(tmp_path)
    mod = _load_script()
    base = mod.load_sims(sim_dir, SEEDS, ["game_9001_1", "game_9001_2"])
    ex = _extra(ex_dir)
    seen = []
    orig = dead_mod.DeadModel.fit
    monkeypatch.setattr(dead_mod.DeadModel, "fit",
                        lambda self, X, y, sample_weight=None: seen.append((len(y), sample_weight)) or orig(self, X, y, sample_weight))
    sel.train_dead(base)
    assert seen[-1][1] is None                                  # plain sims: the unweighted fit
    sel.train_dead(base + ex.at(0.5))
    n, w = seen[-1]
    assert len(w) == n and set(np.unique(w)) == {0.5, 1.0}
    nb = sum(len(s.truth["ctrl_Dfull"]) for s in base)
    assert (w == 1.0).sum() == nb and (w == 0.5).sum() == n - nb
    sel.train_dead(base + ex.at(0.0))                           # weight 0: the extras are left out entirely
    assert seen[-1][0] == nb and seen[-1][1] is None


def test_team_and_switch_trainers_take_weights(tmp_path, monkeypatch):
    from mtah.evidence import switch as switch_mod, team_ctrl as tc
    sim_dir, ex_dir = _write(tmp_path)
    mod = _load_script()
    base = mod.load_sims(sim_dir, SEEDS, ["game_9001_1"]); ex = _extra(ex_dir).at(0.25)
    got = {}
    ot, os_ = tc.TeamCtrlModel.fit, switch_mod.SwitchModel.fit
    monkeypatch.setattr(tc.TeamCtrlModel, "fit", lambda self, X, y, sample_weight=None:
                        got.__setitem__("team", (len(y), sample_weight)) or ot(self, X, y, sample_weight))
    monkeypatch.setattr(switch_mod.SwitchModel, "fit", lambda self, X, y, sample_weight=None:
                        got.__setitem__("switch", (len(y), sample_weight)) or os_(self, X, y, sample_weight))
    sel.train_team_ctrl(base + ex); sel.train_switch(base + ex, policy="no_team")
    for k in ("team", "switch"):
        n, w = got[k]
        assert len(w) == n and set(np.unique(w)) == {0.25, 1.0}, k
    sel.train_team_ctrl(base); sel.train_switch(base, policy="no_team")
    assert got["team"][1] is None and got["switch"][1] is None


def test_inner_held_split_never_holds_an_extra(tmp_path):
    sim_dir, ex_dir = _write(tmp_path)
    mod = _load_script()
    base = mod.load_sims(sim_dir, SEEDS, ["game_9001_1", "game_9001_2", "game_9002_1", "game_9002_2"])
    train = base + _extra(ex_dir).at(1.0)
    inner, held = sel.inner_held_split(train, "t", "x")
    assert {s.key.split("_")[1] for s in held} == {"9002"} and not any(s.extra for s in held)
    assert sum(s.extra for s in inner) == 4 and len(inner) + len(held) == len(train)
    # the extras do not count as train matches: one native match -> in-sample fallback on it
    inner, held = sel.inner_held_split(base[:2] + _extra(ex_dir).at(1.0), "t", "x")
    assert {s.key for s in held} == {"game_9001_1", "game_9001_2"}


def test_selection_extras_train_only(tmp_path, monkeypatch):
    sim_dir, ex_dir = _write(tmp_path)
    mod = _load_script()
    grid = [{**CELL, "extra_weight": 0.0}, {**CELL, "extra_weight": 1.0}, {**CELL, "tau_HA": 10.0, "extra_weight": 0.5}]
    log = {"dead": [], "scored": [], "decoded": []}
    od, osc, oev = mod.train_dead, mod.score_cell, mod.compute_evidence
    monkeypatch.setattr(mod, "train_dead", lambda sims, variant="ctrl_Dfull", policy="full": log["dead"].append(
        sorted((s.key, getattr(s, "weight", 1.0)) for s in sims if getattr(s, "extra", False))) or od(sims, variant, policy))
    monkeypatch.setattr(mod, "score_cell", lambda sims, *a, **k: log["scored"].append([s.key for s in sims]) or osc(sims, *a, **k))
    monkeypatch.setattr(mod, "compute_evidence", lambda m, t, cfg, d, cache_key=None, **k:
                        log["decoded"].append(cache_key) or oev(m, t, cfg, d, cache_key=cache_key, **k))
    out = tmp_path / "out"
    r = mod.run_selection(sim_dir, SEEDS, out, grid=grid, extra=_extra(ex_dir))
    assert r["best_cell"] in grid
    # folds: DFL matches only; extras never tested nor decoded
    assert all(not k.startswith("game_91") for ks in log["scored"] for k in ks)
    assert all(not k.startswith("game_91") for k in log["decoded"])
    ew_seen = sorted({tuple(sorted({w for _, w in call})) for call in log["dead"] if call})
    assert ew_seen == [(0.5,), (1.0,)]                              # weight-0 group trains on the DFL halves alone
    assert any(not call for call in log["dead"])                    # ... and exists (the control)
    for call in log["dead"]:
        assert not call or len(call) == 4                           # all 4 extra halves, every time they are used
    reg = json.loads((out / "registration.json").read_text())
    assert reg["train_extra"]["n_halves"] == 4 and reg["train_extra"]["default_weight"] == 1.0
    assert reg["extra_weight"] == r["best_cell"]["extra_weight"]
    for fold in (9001, 9002, 9003):                                 # test folds are DFL matches only
        h = json.loads((out / f"fold_{fold}.meta.json").read_text())
        assert h["extra"]["n_halves"] == 4
    summ = __import__("pandas").read_csv(out / "summary.csv")
    assert len(summ) == 3
    # resume with the same extra set works; with another one it is refused
    r2 = mod.run_selection(sim_dir, SEEDS, out, grid=grid, extra=_extra(ex_dir))
    assert sorted(r2["resumed_folds"]) == [9001, 9002, 9003]
    other = sel.ExtraSet(load_extra_halves(ex_dir, SEEDS)[:2], 1.0, ex_dir)
    with pytest.raises(SystemExit):
        mod.run_selection(sim_dir, SEEDS, out, grid=grid, extra=other)
    with pytest.raises(SystemExit):
        mod.run_selection(sim_dir, SEEDS, out, grid=grid)           # an extra-set run is never resumed without it


def test_selection_without_extras_unchanged_and_rejects_extra_cells(tmp_path):
    sim_dir, _ = _write(tmp_path)
    mod = _load_script()
    with pytest.raises(SystemExit):
        mod.run_selection(sim_dir, SEEDS, tmp_path / "o1", grid=[{**CELL, "extra_weight": 1.0}])
    r = mod.run_selection(sim_dir, SEEDS, tmp_path / "o2", grid=[CELL])
    reg = json.loads((tmp_path / "o2" / "registration.json").read_text())
    assert reg["train_extra"] is None and reg["extra_weight"] == 0.0 and r["best_cell"] == CELL
