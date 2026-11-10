"""Learned sequence streams (exp-seq; candidate a10): per-frame H/A/D log-probabilities from Model A (windowed HGB,
this venv's pinned sklearn) and Model B (BiGRU, run in the PathCRF venv via subprocess: mtah/evidence/seq_b_infer.py),
used as EXTRA cost streams in the Viterbi decoder.

Inputs are the five evidence streams of the selection (ball, tag, dead, pathcrf, teamctrl), the strong touches, the
PathCRF posterior and the ball (mtah.evidence.seq_feats.frame_features), at 5-frame steps (25 fps). The streams are
team-symmetrised and upsampled to frames (each frame takes its step's value; NaN outside the graded halves).
`d_offset` is added to the D column of every learned stream (dead calibration; 0 = as trained).

Selection-config block ("seq_stream" in config.json; absent = no learned stream):
  {"weights": {"seqA": 4.0, "seqB": 12.0}, "lengths": {...}, "d_offset": 0.0, "modelA": "seq_modelA.joblib",
   "modelB": "seq_modelB.pt", "step_features": [...]}
"""
import os, subprocess, tempfile
from pathlib import Path

import joblib
import numpy as np

from mtah.eval.drift import blackout_mask_from_match
from mtah.evidence import seq_feats as fx
from mtah.evidence.pathcrf import pathcrf_posterior

WRAPPER = Path(__file__).resolve().parent / "seq_b_infer.py"
DEFAULT_PY = os.environ.get("MTAH_PATHCRF_PY", os.path.join(os.path.expanduser("~"), "mtah-ext", "pathcrf-venv", "Scripts", "python.exe"))
TORCH_PY = os.environ.get("MTAH_PATHCRF_PY", DEFAULT_PY)   # Model B runs in the PathCRF venv (torch)
STREAMS = ("ball", "tag", "dead", "pathcrf", "teamctrl")
LAST_INFO = {}


def feature_inputs(match, ev, strong, pathcrf_params, cache_key):
    """The per-frame dict exp_seq_feats.frame_features reads (identical to the experiment's half caches)."""
    r = pathcrf_posterior(match, cache_key, pathcrf_params)
    return {**{f"ev_{k}": ev[k].astype(np.float32) for k in STREAMS}, "touches": strong[2], "sH": strong[0],
            "sA": strong[1], "half": match.half, "ball_present": match.ball_present,
            "blackout_inf": blackout_mask_from_match(match), "ball": match.ball.astype(np.float32),
            "ball_speed": np.asarray(match.ball_speed, np.float32), "pitch": np.asarray(match.pitch, np.float32),
            "fps": np.float32(match.fps), "p_home": r["p_home"], "imputed_frac": r["imputed_frac"]}


def _predict_sym(model, W, wn):
    p1 = model.predict_proba(W)
    p2 = model.predict_proba(fx.mirror(W, wn))[:, [1, 0, 2]]
    return 0.5 * (p1 + p2)


def model_a_logp(path, Xs, seg, names):
    m = joblib.load(path)
    wn = names + [f"{a}_{d}{w}" for w in (10, 25, 50) for d in ("p", "f") for a in names]
    return np.log(np.clip(_predict_sym(m, fx.windowed(Xs, seg), wn), 1e-4, 1)).astype(np.float32)


def model_b_logp(path, Xs, seg, names, python=None, threads=4):
    """Model B in the PathCRF venv (subprocess, like PathCRF); records the torch version in LAST_INFO."""
    with tempfile.TemporaryDirectory() as tmp:
        inp, out = os.path.join(tmp, "x.npz"), os.path.join(tmp, "lp.npz")
        np.savez(inp, X=Xs.astype(np.float32), seg=seg, names=np.array(names))
        env = dict(os.environ, CUDA_VISIBLE_DEVICES="", OMP_NUM_THREADS=str(threads))
        subprocess.run([python or TORCH_PY, str(WRAPPER), "--in", inp, "--ckpt", str(path), "--out", out,
                        "--threads", str(threads)], check=True, env=env, timeout=30 * 60)
        with np.load(out) as z:
            lp = z["logp"].copy(); LAST_INFO["torch_version"] = str(z["torch_version"])
    LAST_INFO["torch_python"] = python or TORCH_PY
    return lp


def compute_streams(z, spec, model_dir):
    """{stream name: (F, 3) float32 log-probs} for spec["weights"] (keys seqA / seqB), d_offset NOT applied."""
    if abs(float(z["fps"]) - 25.0) > 1e-6:
        raise ValueError(f"learned streams assume 25 fps (5-frame steps); got {float(z['fps'])}")
    X, names = fx.frame_features(z)
    if spec.get("step_features") is not None and names != list(spec["step_features"]):
        raise ValueError("step feature names differ from the config's")
    Xs, starts, seg = fx.to_steps(X, names, z["half"])
    F = len(z["half"]); out = {}
    for k in spec["weights"]:
        if k == "seqA":
            lp = model_a_logp(Path(model_dir) / spec["modelA"], Xs, seg, names)
        elif k == "seqB":
            lp = model_b_logp(Path(model_dir) / spec["modelB"], Xs, seg, names)
        else:
            raise ValueError(f"unknown learned stream {k!r}")
        out[k] = fx.upsample(lp, starts, F, z["half"])
    return out


def with_d_offset(streams, d_offset):
    """Copy of the streams with `d_offset` added to the D column (NaN stays NaN)."""
    out = {}
    for k, v in streams.items():
        v = v.copy(); v[:, 2] = v[:, 2] + np.float32(d_offset); out[k] = v
    return out


def add_to_decoder(cfg, ev, streams, spec):
    """(cfg copy, evidence dict) with the learned streams added at spec weights/lengths and spec d_offset."""
    import copy
    cfg = copy.deepcopy(cfg); ev = dict(ev)
    for k, v in with_d_offset(streams, float(spec.get("d_offset", 0.0))).items():
        ev[k] = v
        cfg.decoder.weights[k] = float(spec["weights"][k])
        cfg.decoder.lengths[k] = float(spec.get("lengths", {}).get(k, 1.0))
    return cfg, ev
