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
    assert mod.check_bar_gate(best, "exploratory") is True
    assert mod.check_bar_gate({"spec_bar": {"bar_pass": True}}, None) is False


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
