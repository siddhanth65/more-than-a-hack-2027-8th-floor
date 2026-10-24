import importlib.util
import json
import sys
from pathlib import Path

import pytest

from mtah.io import readlog

ROOT = Path(__file__).resolve().parents[1]


def _load(name):
    spec = importlib.util.spec_from_file_location(name + "_t", ROOT / "scripts" / f"{name}.py")
    mod = importlib.util.module_from_spec(spec); sys.modules[name + "_t"] = mod
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture
def logdir(tmp_path, monkeypatch):
    monkeypatch.setattr(readlog, "READS_DIR", tmp_path)
    return tmp_path


def test_append_and_read(logdir):
    assert readlog.read_log("sealed") == []
    readlog.append_read("sealed", {"cell": {"a": 1}, "seeds": [1]})
    readlog.append_read("sealed", {"cell": {"a": 2}, "seeds": [1]})
    got = readlog.read_log("sealed")
    assert [g["cell"]["a"] for g in got] == [1, 2]
    assert readlog.read_log("public") == []
    assert (logdir / "sealed.jsonl").read_text().count("\n") == 2


def test_bad_kind(logdir):
    with pytest.raises(ValueError):
        readlog.append_read("other", {})


def test_sealed_guard_per_registration(logdir):
    """A second sealed read of the same registration is refused unless a force reason is given."""
    mod = _load("select_lomo")
    reg = {"grid": "grid2", "seeds": [100, 101], "grid_size": 64, "best_cell": {"tau_HA": 10.0}, "git": "abc"}
    rid = mod.registration_id(reg)
    assert rid == mod.registration_id({**reg, "git": "other", "selection": "x"})   # identity ignores other fields
    assert rid != mod.registration_id({**reg, "seeds": [100, 102]})
    assert rid != mod.registration_id({**reg, "best_cell": {"tau_HA": 30.0}})
    assert rid != mod.registration_id({**reg, "grid": "grid1"})
    assert mod.check_sealed_guard(rid) == []   # first read: allowed
    # a legacy record (no registration) and another registration do not block
    readlog.append_read("sealed", {"cell": {"tau_HA": 10.0}, "seeds": [100, 101], "git": "d85c166"})
    readlog.append_read("sealed", {"match": 9007, "registration": mod.registration_id({**reg, "seeds": [1]})})
    assert mod.check_sealed_guard(rid) == []
    readlog.append_read("sealed", {"match": 9007, "registration": rid, "time": "t1"})
    assert (logdir / "sealed.jsonl").exists()
    with pytest.raises(SystemExit, match="already read"):
        mod.check_sealed_guard(rid)
    prev = mod.check_sealed_guard(rid, force_reason="re-read after scorer fix")
    assert len(prev) == 1 and prev[0]["time"] == "t1"
    assert mod.check_sealed_guard(rid, match=9006) == []   # other match


def test_public_guard(logdir):
    mod = _load("public_read")
    for note in "vwxyz":   # budget 6 total: 5 logged + 1 done by hand
        mod.check_budget()
        readlog.append_read("public", {"note": note})
    with pytest.raises(SystemExit):
        mod.check_budget()
