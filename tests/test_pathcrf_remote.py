"""scripts/pathcrf_remote_assemble.py: prepare -> (remote wrapper run, faked) -> assemble == pathcrf_posterior's cache."""
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pytest

import mtah.evidence.pathcrf as pc
from mtah.sim.dataset import load_half_sim, save_half
from tests.test_select_smoke import _fake_half

ROOT = Path(__file__).resolve().parents[1]


def _script():
    spec = importlib.util.spec_from_file_location("pathcrf_remote_assemble", ROOT / "scripts" / "pathcrf_remote_assemble.py")
    mod = importlib.util.module_from_spec(spec); sys.modules["pathcrf_remote_assemble"] = mod
    spec.loader.exec_module(mod)
    return mod


def _fake_out(inp_path, out_path, n_ok=None):
    with np.load(inp_path) as z:
        F = len(z["pos"])
    rng = np.random.default_rng(F)
    p = rng.uniform(0.05, 0.95, F).astype(np.float32); p[:7] = np.nan
    np.savez(out_path, p_home=p, p_none=rng.uniform(0, 0.3, F).astype(np.float32),
             imputed_frac=rng.uniform(0, 0.5, F).astype(np.float32), chunk_id=np.zeros(F, np.int32),
             n_chunks=np.int32(10), n_ok=np.int32(10 if n_ok is None else n_ok))


def test_prepare_assemble_matches_pathcrf_posterior(tmp_path, monkeypatch):
    mod = _script()
    sim = tmp_path / "sim"
    rng = np.random.default_rng(0)
    for h in (1, 2):
        save_half(_fake_half(9101, h, 100, rng), sim)
    params = pc.PathCRFParams(cache_dir=str(tmp_path / "cache_remote"))
    work = tmp_path / "remote"
    man = mod.prepare(sim, [100], work, params)
    assert len(man["entries"]) == 2 and (work / "run_all.sh").read_text().startswith("#!/bin/bash")
    assert "--threads 4" in (work / "run_all.sh").read_text() and "\r" not in (work / "run_all.sh").read_text()
    (work / "out").mkdir()
    for f in (work / "in").glob("*.in.npz"):
        _fake_out(f, work / "out" / f.name.replace(".in.npz", ".out.npz"))
    mod.assemble(work, tmp_path / "cache_remote")
    # the normal path with the same (faked) wrapper outputs
    for key in ("game_9101_1", "game_9101_2"):
        m = load_half_sim(sim, key, 100).match
        ref_params = pc.PathCRFParams(cache_dir=str(tmp_path / "cache_ref"))
        monkeypatch.setattr(pc, "_run_half", lambda match, a, b, p, tmp, key=key: pc.read_half_output(
            work / "out" / f"{key}_s100_h0.out.npz", a))
        ref = pc.pathcrf_posterior(m, f"{key}_s100", ref_params)
        with np.load(tmp_path / "cache_remote" / f"{key}_s100.npz") as z, np.load(tmp_path / "cache_ref" / f"{key}_s100.npz") as r:
            assert sorted(z.files) == sorted(r.files) == sorted(["ckpt", "threads", "input_sha", "wrapper_sha", "p_home", "p_none", "imputed_frac"])
            for k in r.files:
                assert (str(z[k]) == str(r[k])) if k.endswith("_sha") else np.array_equal(z[k], r[k], equal_nan=True), k
        # and pathcrf_posterior accepts the assembled file as a cache hit (no wrapper call)
        monkeypatch.setattr(pc, "_run_half", lambda *a, **k: (_ for _ in ()).throw(AssertionError("recomputed")))
        hit = pc.pathcrf_posterior(m, f"{key}_s100", pc.PathCRFParams(cache_dir=str(tmp_path / "cache_remote")))
        assert np.array_equal(hit["p_home"], ref["p_home"], equal_nan=True)


def test_assemble_refuses_stale_or_bad_outputs(tmp_path):
    mod = _script()
    sim = tmp_path / "sim"
    rng = np.random.default_rng(1)
    save_half(_fake_half(9101, 1, 100, rng), sim)
    work = tmp_path / "remote"
    mod.prepare(sim, [100], work, pc.PathCRFParams())
    (work / "out").mkdir()
    with pytest.raises(SystemExit):           # remote output missing
        mod.assemble(work, tmp_path / "c")
    f = next((work / "in").glob("*.in.npz"))
    _fake_out(f, work / "out" / f.name.replace(".in.npz", ".out.npz"), n_ok=2)
    with pytest.raises(RuntimeError):         # under half of the windows predicted
        mod.assemble(work, tmp_path / "c")
    _fake_out(f, work / "out" / f.name.replace(".in.npz", ".out.npz"))
    man = json.loads((work / "manifest.json").read_text()); man["entries"][0]["input_sha"] = "0" * 64
    (work / "manifest.json").write_text(json.dumps(man))
    with pytest.raises(SystemExit):           # the sim changed since prepare
        mod.assemble(work, tmp_path / "c")
