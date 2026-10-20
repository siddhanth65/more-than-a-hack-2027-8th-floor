import importlib.util
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


def _mod():
    spec = importlib.util.spec_from_file_location("select_lomo_gate", ROOT / "scripts" / "select_lomo.py")
    mod = importlib.util.module_from_spec(spec); sys.modules["select_lomo_gate"] = mod
    spec.loader.exec_module(mod)
    return mod


def _best(tmp, bar_pass=True, seeds=(100, 101), git="abc"):
    (tmp / "best.json").write_text(json.dumps({"cell": {"a": 1}, "git": git, "seeds": list(seeds),
                                               "spec_bar": {"bar_pass": bar_pass, "x": 1}}))
    return tmp


def test_gate_refuses_when_bar_fails(tmp_path, capsys):
    mod = _mod()
    best = json.loads((_best(tmp_path, bar_pass=False) / "best.json").read_text())
    with pytest.raises(SystemExit):
        mod.check_bar_gate(best, None)
    assert '"bar_pass": false' in capsys.readouterr().out
    assert mod.check_bar_gate(best, "exploratory") == ["bar"]
    assert mod.check_bar_gate({"spec_bar": {"bar_pass": True}}, None) == []


def test_best_from_seed_mismatch_refuses(tmp_path):
    mod = _mod()
    _best(tmp_path, seeds=(100, 101))
    assert mod.load_best(tmp_path, [100, 101])["cell"] == {"a": 1}
    with pytest.raises(SystemExit):
        mod.load_best(tmp_path, [100, 102])


def test_best_from_missing_fields_refuses(tmp_path):
    mod = _mod()
    (tmp_path / "best.json").write_text(json.dumps({"cell": {}, "spec_bar": {"bar_pass": True}}))
    with pytest.raises(SystemExit):
        mod.load_best(tmp_path, [100])


def test_best_from_git_mismatch_warns(tmp_path, capsys):
    mod = _mod()
    _best(tmp_path, git="zzzzzzz")
    mod.load_best(tmp_path, [100, 101])
    assert "WARNING" in capsys.readouterr().out


def test_gate_refuses_implausible_chosen_cell(capsys):
    mod = _mod()
    ok_bar = {"spec_bar": {"bar_pass": True}}
    assert mod.check_bar_gate({**ok_bar, "chosen_real_plausible": True}, None) == []
    assert mod.check_bar_gate({**ok_bar, "chosen_real_plausible": None}, None) == []   # --no-real
    assert mod.check_bar_gate(ok_bar, None) == []   # older best.json without the field
    with pytest.raises(SystemExit, match="real_plausible"):
        mod.check_bar_gate({**ok_bar, "chosen_real_plausible": False}, None)
    assert mod.check_bar_gate({**ok_bar, "chosen_real_plausible": False}, "exploratory") == ["real_plausible"]
    assert mod.check_bar_gate({"spec_bar": {"bar_pass": False}, "chosen_real_plausible": False}, "x") == ["bar", "real_plausible"]


def _bar_inputs(ratios, n_folds=6, won=None):
    """allf/halves/best for compute_bar: the cell beats both baselines by 0.2 on every variant in `won` folds."""
    import pandas as pd
    V = ("raw_Dfull", "ctrl_Dfull", "raw_Dnarrow", "ctrl_Dnarrow")
    won = n_folds if won is None else won
    rows = []
    for k, held in enumerate(range(9001, 9001 + n_folds)):
        rows.append({"cell": "0", "held": held, **{v: (0.6 if k < won else 0.4) for v in V}})
        rows.append({"cell": "smooth50", "held": held, **{v: 0.4 for v in V}})
        rows.append({"cell": "b4", "held": held, **{v: 0.3 for v in V}})
    allf = pd.DataFrame(rows)
    halves = pd.DataFrame([{"cell": "0", "seed": s, "variant": v, "spell_f1": 0.6} for s in (100, 101) for v in V])
    best = {**{v: 0.7 for v in V}, **{f"{v}_ratio": ratios[v] for v in V}}
    return allf, halves, best


def test_compute_bar_uses_ctrl_ratios_only():
    mod = _mod()
    raw_off = {"raw_Dfull": 0.6, "raw_Dnarrow": 0.55, "ctrl_Dfull": 1.0, "ctrl_Dnarrow": 1.1}
    bar, _ = mod.compute_bar(*_bar_inputs(raw_off), "0")
    assert bar["bar_pass"] is True
    assert bar["count_ratios"] == raw_off   # all four recorded unchanged
    assert set(bar["count_ratios_ok"]) == {"ctrl_Dfull", "ctrl_Dnarrow"}
    ctrl_off = {**raw_off, "ctrl_Dnarrow": 0.7}
    bar, _ = mod.compute_bar(*_bar_inputs(ctrl_off), "0")
    assert bar["bar_pass"] is False and bar["count_ratios_ok"]["ctrl_Dnarrow"] is False


def test_compute_bar_folds_won_threshold_from_n_folds():
    mod = _mod()
    ok = {"raw_Dfull": 0.6, "raw_Dnarrow": 0.6, "ctrl_Dfull": 1.0, "ctrl_Dnarrow": 1.0}
    for n, won, want in ((6, 5, True), (6, 4, False), (5, 4, True), (5, 3, False)):
        bar, _ = mod.compute_bar(*_bar_inputs(ok, n_folds=n, won=won), "0")
        assert bar["n_folds"] == n and bar["folds_won_needed"] == n - 1 and bar["folds_won_ctrl_Dfull"] == won
        assert bar["bar_pass"] is want, (n, won, bar)


def test_guard_skips_legacy_entries_without_match(monkeypatch):
    mod = _mod()
    reg = {"grid": "g", "seeds": [100], "sha256": "x"}
    monkeypatch.setattr(mod.readlog, "read_log", lambda name: [{"registration": reg, "time": "t"}])   # legacy: no match
    assert mod.check_sealed_guard(reg, None) == []
    monkeypatch.setattr(mod.readlog, "read_log",
                        lambda name: [{"registration": reg, "match": mod.SEALED_MATCH, "time": "t"}])
    with pytest.raises(SystemExit, match="already read"):
        mod.check_sealed_guard(reg, None)
