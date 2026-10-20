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


def test_sealed_guard_keyed_on_cell_and_seeds(logdir):
    mod = _load("select_lomo")
    readlog.append_read("sealed", {"cell": {"a": 1}, "seeds": [100, 101], "git": "abc"})
    with pytest.raises(SystemExit):
        mod.check_sealed_guard({"a": 1}, [100, 101])
    mod.check_sealed_guard({"a": 1}, [100, 102])  # different seeds: allowed
    mod.check_sealed_guard({"a": 2}, [100, 101])  # different cell: allowed


def test_public_guard(logdir):
    mod = _load("public_read")
    mod.check_budget()
    readlog.append_read("public", {"note": "x"})
    mod.check_budget()
    readlog.append_read("public", {"note": "y"})
    with pytest.raises(SystemExit):
        mod.check_budget()
