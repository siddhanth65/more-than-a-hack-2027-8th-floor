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


def test_apply_cell_w_pathcrf():
    from mtah.select import GRID, apply_cell
    base = PipelineConfig()
    assert apply_cell(base, GRID[0]).use_pathcrf is False
    off = apply_cell(base, {**GRID[0], "w_pathcrf": 0.0})
    assert off.use_pathcrf is False and off.decoder.weights["pathcrf"] == 0.0
    on = apply_cell(base, {**GRID[0], "w_pathcrf": 2.0})
    assert on.use_pathcrf is True and on.decoder.weights["pathcrf"] == 2.0
    assert base.use_pathcrf is False and base.decoder.weights["pathcrf"] == 1.0
