"""PathCRF (Kim et al. 2026, MPL-2.0) players-only team posterior as an evidence stream.

PathCRF runs in its own venv (torch) as an external tool: `third_party/pathcrf/run_infer.py` is called per
half via subprocess; nothing from PathCRF is imported here. See third_party/pathcrf/SETUP.md.
"""
import os
import subprocess
import tempfile
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from mtah.io.match import half_runs

WRAPPER = Path(__file__).resolve().parents[2] / "third_party" / "pathcrf" / "run_infer.py"
DEFAULT_REPO = r"HOME/mtah-ext\pathcrf"
DEFAULT_PY = r"HOME/mtah-ext\pathcrf-venv\Scripts\python.exe"


@dataclass
class PathCRFParams:
    repo: str = field(default_factory=lambda: os.environ.get("MTAH_PATHCRF_REPO", DEFAULT_REPO))
    python: str = field(default_factory=lambda: os.environ.get("MTAH_PATHCRF_PY", DEFAULT_PY))
    ckpt: int = 140
    cache_dir: str = "runs/cache/pathcrf"
    eps: float = 1e-3
    conf_alpha: float = 1.0      # confidence = 1 - conf_alpha * imputed_frac (imputed nodes are the main damage)
    threads: int = 4


def _run_half(match, a, b, params, tmp):
    stem = Path(tmp) / f"h{a}"
    inp, out = f"{stem}.in.npz", f"{stem}.out.npz"
    np.savez(inp, pos=match.pos[a:b].astype(np.float32), team=match.team.astype(np.int8),
             fps=np.float32(match.fps), pitch=np.asarray(match.pitch, np.float32))
    env = dict(os.environ, TQDM_DISABLE="1")
    subprocess.run([params.python, str(WRAPPER), "--in", inp, "--out", out, "--ckpt", str(params.ckpt),
                    "--repo", params.repo, "--threads", str(params.threads)], check=True, env=env)
    with np.load(out) as z:
        return {k: z[k] for k in ("p_home", "p_none", "imputed_frac")}


def pathcrf_posterior(match, cache_key, params):
    """Per-frame PathCRF team posterior: dict(p_home, p_none, imputed_frac), each (F,) float32.

    p_home = P(home | a player owns the ball) (NaN where PathCRF gave no prediction, incl. ungraded frames);
    p_none = posterior mass on the out-of-pitch nodes; imputed_frac = share of the 22 nodes whose position
    was imputed. One wrapper run per half (halves are never crossed). The merged result is cached as
    <cache_dir>/<cache_key>.npz and reused when its checkpoint and frame count match; the per-half
    hand-off files live in a temporary directory that is removed afterwards.
    """
    cache = Path(params.cache_dir) / f"{cache_key}.npz"
    F = match.n_frames
    if cache.is_file():
        with np.load(cache) as z:
            if int(z["ckpt"]) == params.ckpt and len(z["p_home"]) == F:
                return {k: z[k] for k in ("p_home", "p_none", "imputed_frac")}
    res = {k: np.full(F, np.nan, np.float32) for k in ("p_home", "p_none", "imputed_frac")}
    with tempfile.TemporaryDirectory(prefix="pathcrf_") as tmp:
        for _, a, b in half_runs(match.half):
            for k, v in _run_half(match, a, b, params, tmp).items():
                res[k][a:b] = v
    cache.parent.mkdir(parents=True, exist_ok=True)
    np.savez(cache, ckpt=np.int32(params.ckpt), **res)
    return res


def pathcrf_evidence(p_home, imputed_frac, eps=1e-3, conf_alpha=1.0):
    """(F,3) float32: H = c*log(p+eps), A = c*log(1-p+eps), D = NaN, c = clip(1 - conf_alpha*imputed_frac, 0, 1).

    Rows are NaN (no opinion) where p_home is NaN."""
    p = np.asarray(p_home, np.float64)
    c = np.clip(1.0 - conf_alpha * np.asarray(imputed_frac, np.float64), 0.0, 1.0)
    out = np.full((len(p), 3), np.nan, np.float32)
    ok = np.isfinite(p)
    out[ok, 0] = c[ok] * np.log(p[ok] + eps)
    out[ok, 1] = c[ok] * np.log(1.0 - p[ok] + eps)
    return out
