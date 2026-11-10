"""PathCRF input pre-smoothing (mtah.evidence.pathcrf.smooth_positions / smooth_s cache key)."""
import numpy as np

from mtah.evidence import pathcrf as pc


def test_smooth_keeps_nan_and_reduces_noise():
    rng = np.random.default_rng(0)
    t = np.arange(500)[:, None, None]
    base = np.concatenate([0.1 * t, 0.0 * t], axis=-1).astype(np.float32).repeat(3, axis=1)
    pos = base + rng.normal(0, 0.2, base.shape).astype(np.float32)
    pos[100:150, 1] = np.nan
    out = pc.smooth_positions(pos, 25.0, 1.0)
    assert np.isnan(out[100:150, 1]).all() and np.isfinite(out[:100]).all()
    inner = slice(20, 480)
    ok = np.isfinite(pos[inner, 0, 0])
    assert np.abs(out[inner, 0] - base[inner, 0]).mean() < 0.5 * np.abs(pos[inner, 0] - base[inner, 0]).mean()
    assert np.abs(out[inner, 0, 0] - base[inner, 0, 0]).max() < 0.2        # a linear ramp is preserved (no bias)


def test_smooth_off_is_identity():
    pos = np.random.default_rng(1).normal(size=(50, 4, 2)).astype(np.float32)
    assert pc.smooth_positions(pos, 25.0, 0.0) is pos
    assert pc.smooth_positions(pos, 25.0, 0.01) is pos


def test_cache_key_includes_smooth(tmp_path):
    from mtah.io.match import Match
    F = 30
    m = Match(fps=25.0, pos=np.random.default_rng(0).normal(size=(F, 4, 2)).astype(np.float32), speed=np.zeros((F, 4), np.float32),
              team=np.array([0, 0, 1, 1], np.int8), ball=np.zeros((F, 2), np.float32), ball_speed=np.zeros(F, np.float32),
              half=np.ones(F, np.int8))
    p0 = pc.PathCRFParams(cache_dir=str(tmp_path))
    res = {k: np.zeros(F, np.float32) for k in ("p_home", "p_none", "imputed_frac")}
    pc.write_cache(tmp_path / "k.npz", p0, dict(input_sha=pc._input_sha(m), wrapper_sha=pc._wrapper_sha()), res)
    assert pc.pathcrf_posterior(m, "k", p0)["p_home"].shape == (F,)           # same smooth_s: cache hit
    p1 = pc.PathCRFParams(cache_dir=str(tmp_path), smooth_s=1.0)
    import pytest
    with pytest.raises(Exception):                                            # different smooth_s: miss -> would run PathCRF
        pc._run_half = lambda *a, **k: (_ for _ in ()).throw(RuntimeError("miss"))
        pc.pathcrf_posterior(m, "k", p1)
