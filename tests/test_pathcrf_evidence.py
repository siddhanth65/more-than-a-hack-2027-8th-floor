from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from mtah.evidence.pathcrf import PathCRFParams, pathcrf_evidence
from mtah.pipeline import PipelineConfig, compute_evidence
from tests.test_ball_ctrl import carry_match

NO_TAGS = pd.DataFrame(columns=["start", "end", "code", "team", "type", "half"])


def test_evidence_values_and_contract():
    p = np.array([0.9, 0.9, np.nan], np.float32)
    imp = np.array([0.0, 1.0, 0.0], np.float32)
    e = pathcrf_evidence(p, imp, eps=1e-3, conf_alpha=1.0)
    assert e.dtype == np.float32 and e.shape == (3, 3)
    assert e[0, 0] == pytest.approx(np.log(0.901), abs=1e-5)
    assert e[0, 1] == pytest.approx(np.log(0.101), abs=1e-5)
    assert np.isnan(e[:, 2]).all()
    assert e[1, 0] == 0 and e[1, 1] == 0              # fully imputed nodes: zero confidence, still finite
    assert np.isnan(e[2]).all()                       # no prediction -> no opinion


def test_evidence_confidence_scales_linearly():
    p = np.full(2, 0.2, np.float32)
    e = pathcrf_evidence(p, np.array([0.0, 0.25], np.float32), eps=1e-3, conf_alpha=2.0)
    np.testing.assert_allclose(e[1, :2], 0.5 * e[0, :2], rtol=1e-6)


def test_params_env_defaults(monkeypatch):
    monkeypatch.setenv("MTAH_PATHCRF_REPO", "X:/repo")
    monkeypatch.setenv("MTAH_PATHCRF_PY", "X:/py.exe")
    p = PathCRFParams()
    assert (p.repo, p.python, p.ckpt, p.eps, p.conf_alpha) == ("X:/repo", "X:/py.exe", 140, 1e-3, 1.0)


def test_config_roundtrip_and_old_dicts():
    cfg = PipelineConfig()
    assert cfg.use_pathcrf is False and cfg.decoder.weights["pathcrf"] == 1.0
    cfg.use_pathcrf = True; cfg.pathcrf.conf_alpha = 0.5
    d = PipelineConfig.from_dict(cfg.to_dict())
    assert d.use_pathcrf is True and d.pathcrf.conf_alpha == 0.5 and isinstance(d.pathcrf, PathCRFParams)
    old = PipelineConfig().to_dict(); old.pop("use_pathcrf"); old.pop("pathcrf")
    old["decoder"]["weights"].pop("pathcrf"); old["decoder"]["lengths"].pop("pathcrf")
    o = PipelineConfig.from_dict(old)
    assert o.use_pathcrf is False and o.pathcrf.ckpt == 140


def test_compute_evidence_without_pathcrf_has_no_stream():
    assert "pathcrf" not in compute_evidence(carry_match(F=200), NO_TAGS, PipelineConfig(), None)


def test_compute_evidence_with_pathcrf(monkeypatch):
    import mtah.pipeline as pl
    seen = {}

    def fake(match, cache_key, params):
        seen["key"] = cache_key
        F = match.n_frames
        return {"p_home": np.full(F, 0.7, np.float32), "imputed_frac": np.zeros(F, np.float32)}
    monkeypatch.setattr(pl, "pathcrf_posterior", fake)
    cfg = PipelineConfig(); cfg.use_pathcrf = True
    ev = compute_evidence(carry_match(F=200), NO_TAGS, cfg, None, cache_key="k_s1")
    assert seen["key"] == "k_s1"
    e = ev["pathcrf"]
    assert e.shape == (200, 3) and e.dtype == np.float32
    assert e[0, 0] == pytest.approx(np.log(0.701), abs=1e-5) and np.isnan(e[:, 2]).all()
    with pytest.raises(ValueError):
        compute_evidence(carry_match(F=200), NO_TAGS, cfg, None)


def test_apply_cell_w_pathcrf_only_sets_weight():
    from mtah.select import GRID, apply_cell
    base = PipelineConfig()
    assert apply_cell(base, GRID[0]).use_pathcrf is False
    off = apply_cell(base, {**GRID[0], "w_pathcrf": 0.0})
    assert off.use_pathcrf is False and off.decoder.weights["pathcrf"] == 0.0
    base.use_pathcrf = True
    on = apply_cell(base, {**GRID[0], "w_pathcrf": 0.0})
    assert on.use_pathcrf is True and on.decoder.weights["pathcrf"] == 0.0     # caller owns use_pathcrf
    assert base.decoder.weights["pathcrf"] == 1.0


def test_zero_weight_equals_no_stream():
    from mtah.decode.viterbi import DecoderParams, cost_matrix
    rng = np.random.default_rng(0)
    a = rng.normal(size=(50, 3)).astype(np.float32)
    b = rng.normal(size=(50, 3)).astype(np.float32); b[::7] = np.nan
    p = DecoderParams(weights={"ball": 1.0, "pathcrf": 0.0}, lengths={"ball": 1.0, "pathcrf": 1.0})
    np.testing.assert_array_equal(cost_matrix({"ball": a, "pathcrf": b}, p), cost_matrix({"ball": a}, p))


def test_config_omits_and_ignores_runtime_params(monkeypatch):
    monkeypatch.setenv("MTAH_PATHCRF_REPO", "X:/env_repo")
    monkeypatch.setenv("MTAH_PATHCRF_PY", "X:/env_py.exe")
    d = PipelineConfig().to_dict()
    assert set(d["pathcrf"]) == {"ckpt", "eps", "conf_alpha"}
    d["pathcrf"].update(repo="C:/old/repo", python="C:/old/py.exe", cache_dir="C:/old/cache", threads=99, ckpt=141)
    p = PipelineConfig.from_dict(d).pathcrf
    assert (p.repo, p.python, p.ckpt) == ("X:/env_repo", "X:/env_py.exe", 141)
    assert p.threads == 4 and p.cache_dir == PathCRFParams().cache_dir


def test_default_cache_dir_is_repo_anchored():
    cd = Path(PathCRFParams().cache_dir)
    assert cd.is_absolute() and cd.parts[-3:] == ("runs", "cache", "pathcrf")
    assert (cd.parents[2] / "mtah").is_dir()


def test_cache_invalidation_and_atomic_write(monkeypatch, tmp_path):
    import mtah.evidence.pathcrf as pc
    calls = []

    def fake(match, a, b, params, tmp):
        calls.append(a)
        n = b - a
        return {"p_home": np.full(n, 0.6, np.float32), "p_none": np.zeros(n, np.float32),
                "imputed_frac": np.zeros(n, np.float32)}
    monkeypatch.setattr(pc, "_run_half", fake)
    m = carry_match(F=120)
    params = PathCRFParams(cache_dir=str(tmp_path))
    pc.pathcrf_posterior(m, "k", params); n1 = len(calls)
    assert n1 >= 1 and [p.name for p in tmp_path.iterdir()] == ["k.npz"]       # no temp leftovers
    pc.pathcrf_posterior(m, "k", params); assert len(calls) == n1               # hit
    m.pos = m.pos.copy(); m.pos[3, 0, 0] += 1.0                                 # input changed, same frame count
    pc.pathcrf_posterior(m, "k", params); assert len(calls) == 2 * n1
    pc.pathcrf_posterior(m, "k", params); assert len(calls) == 2 * n1
    p141 = PathCRFParams(cache_dir=str(tmp_path), ckpt=141)
    pc.pathcrf_posterior(m, "k", p141); assert len(calls) == 3 * n1
    p2 = PathCRFParams(cache_dir=str(tmp_path), ckpt=141, threads=2)
    pc.pathcrf_posterior(m, "k", p2); assert len(calls) == 4 * n1
    pc.pathcrf_posterior(m, "k", p2); assert len(calls) == 4 * n1
    monkeypatch.setattr(pc, "_wrapper_sha", lambda: "changed")
    pc.pathcrf_posterior(m, "k", p2); assert len(calls) == 5 * n1


def test_run_half_raises_on_failed_chunks(monkeypatch, tmp_path):
    import mtah.evidence.pathcrf as pc

    def fake_subprocess(cmd, **kw):
        out = cmd[cmd.index("--out") + 1]
        np.savez(out, p_home=np.full(100, np.nan, np.float32), p_none=np.full(100, np.nan, np.float32),
                 imputed_frac=np.zeros(100, np.float32), n_chunks=np.int32(10), n_ok=np.int32(4))
    monkeypatch.setattr(pc.subprocess, "run", fake_subprocess)
    with pytest.raises(RuntimeError, match="0.40"):
        pc._run_half(carry_match(F=100), 0, 100, PathCRFParams(), str(tmp_path))


def _wrapper_module():
    import importlib.util
    path = Path(__file__).resolve().parents[1] / "third_party" / "pathcrf" / "run_infer.py"
    spec = importlib.util.spec_from_file_location("pathcrf_run_infer", path)
    mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
    return mod


def test_impute_extrapolates_one_second_then_holds():
    w = _wrapper_module()
    pos = np.full((120, 1, 2), np.nan, np.float32)
    pos[:50, 0, 0] = np.arange(50) * 0.2          # 5 m/s along x
    pos[:50, 0, 1] = 1.0
    out = w.impute(pos, 25.0)
    np.testing.assert_allclose(out[:50], pos[:50])
    assert out[49 + 25, 0, 0] == pytest.approx(9.8 + 5.0, abs=1e-4)      # 1 s of extrapolation at 5 m/s
    assert out[49 + 25, 0, 1] == pytest.approx(1.0)
    assert out[119, 0, 0] == pytest.approx(out[49 + 25, 0, 0])            # then hold
    assert not np.isnan(out).any()


def test_impute_backfills_start_and_clips():
    w = _wrapper_module()
    pos = np.full((30, 2, 2), np.nan, np.float32)
    pos[10:, 0] = (3.0, -4.0)
    pos[:, 1, 0] = 80.0; pos[:, 1, 1] = -50.0          # outside 105 x 68
    out = w.impute(pos, 25.0)
    assert (out[:10, 0] == (3.0, -4.0)).all()
    assert (out[:, 1, 0] == 52.5).all() and (out[:, 1, 1] == -34.0).all()
    assert np.isnan(w.impute(np.full((5, 1, 2), np.nan, np.float32), 25.0)).all()   # never tracked stays NaN


@pytest.mark.parametrize("k", [1, 3])
@pytest.mark.parametrize("r", [0, 124, 125, 249])
def test_chunks_of_remainder_merging(k, r):
    w = _wrapper_module()
    F = k * 250 + r
    ch = w.chunks_of(F, 250)
    assert ch[0][0] == 0 and ch[-1][1] == F
    assert all(ch[i][1] == ch[i + 1][0] for i in range(len(ch) - 1))
    sizes = [b - a for a, b in ch]
    if r < 125:
        assert len(ch) == k and sizes[-1] == 250 + r      # short remainder merged into the last window
    else:
        assert len(ch) == k + 1 and sizes[-1] == r


DFL = Path(__file__).resolve().parents[1] / "runs" / "dfl_extra"
SIM = Path(__file__).resolve().parents[1] / "runs" / "sim"


def _pathcrf_available():
    p = PathCRFParams()
    return Path(p.python).is_file() and Path(p.repo).is_dir()


def _proxy_helper():
    """Label-free nearest-player proxy of scripts/precompute_pathcrf.py (no 9007 truth is consulted)."""
    import importlib.util
    path = Path(__file__).resolve().parents[1] / "scripts" / "precompute_pathcrf.py"
    spec = importlib.util.spec_from_file_location("precompute_pathcrf_helper", path)
    mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
    return mod.nearest_player_proxy


@pytest.mark.dfl
def test_integration_clean_9007_h1_agreement(tmp_path):
    from mtah.evidence.pathcrf import pathcrf_posterior
    from mtah.io.dfl_proxy import load_half
    npz, census = DFL / "game_9007_H1.npz", DFL / "fp1d_census.json"
    if not (npz.is_file() and census.is_file() and _pathcrf_available()):
        pytest.skip("DFL proxy data or the PathCRF venv is absent")
    m, _labels = load_half(npz, census)  # labels deliberately unused: 9007 is the sealed match
    r = pathcrf_posterior(m, "test_clean_9007_H1", PathCRFParams(cache_dir=str(tmp_path)))
    agree, n = _proxy_helper()(m, r["p_home"])
    assert n > 0
    assert agree >= 0.85, agree


@pytest.mark.dfl
def test_integration_sim_9007_h1_seed100(tmp_path):
    from mtah.evidence.pathcrf import pathcrf_posterior
    from mtah.sim.dataset import load_half_sim
    if not ((SIM / "game_9007_H1_s100.json").is_file() and _pathcrf_available()):
        pytest.skip("sim data or the PathCRF venv is absent")
    sim = load_half_sim(SIM, "game_9007_H1", 100)  # truth deliberately unused: 9007 is the sealed match
    r = pathcrf_posterior(sim.match, "test_sim_9007_H1_s100", PathCRFParams(cache_dir=str(tmp_path)))
    nan_share = float(np.isnan(r["p_home"]).mean())
    agree, n = _proxy_helper()(sim.match, r["p_home"])
    assert nan_share <= 0.05, nan_share
    assert n > 0
    assert agree >= 0.70, agree
