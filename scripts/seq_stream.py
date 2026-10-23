"""Learned sequence stream(s) of the exp-seq final variant for one match (a simulator half or the real match).

Inputs: the a5c selection artefacts (config, dead/team models: the five a5c evidence streams the features read) and
runs/select/exp_seq/final/ (config.json, modelA.joblib and/or modelB.pt, optional merge_model.joblib). Output: a
.npy array (n_streams, F, 3) float32 of per-frame H/A/D log-probabilities in config "stream_weights" order (NaN
outside the graded halves), written with np.save (byte-reproducible). Deterministic: HGB inference, CPU torch with
4 threads, torch.use_deterministic_algorithms(True), fixed seed; PathCRF is read from its cache only.

  <venv python> scripts/seq_stream.py --sim game_9001_H1 100 --out x.npy
  <venv python> scripts/seq_stream.py --real C:/.../mtah-data --out real.npy [--drift]

--drift (real match only): label-free drift_report aggregates of a5c and of a5c + the stream(s) (+ merge model).
"""
import argparse, copy, hashlib, json, os, sys
from pathlib import Path

os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["OMP_NUM_THREADS"] = "4"
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "scripts"))

import joblib
import numpy as np

try:
    import torch
except ModuleNotFoundError:   # project venv (sklearn of the a5c artefacts) has no torch: borrow the CPU-capable
    # Python 3.12 user-site torch, appended AFTER the venv so numpy/sklearn stay the venv's (no install)
    sys.path.append(str(Path.home() / "AppData" / "Roaming" / "Python" / "Python312" / "site-packages"))
    import torch

torch.set_num_threads(4)
torch.use_deterministic_algorithms(True)
torch.manual_seed(0)

import exp_seq_feats as fx
import mtah.evidence.pathcrf as pcrf
from mtah.decode.viterbi import decode
from mtah.eval.drift import blackout_mask_from_match, drift_report
from mtah.pipeline import compute_evidence, compute_strong, predict
from mtah.sim.dataset import SEALED_MATCH, load_half_sim
from mtah.spells.merge import MergeModel, apply_merge_model
from mtah.spells.postprocess import postprocess

MAIN = Path(r"WORKDIR/possession")
STREAMS = ("ball", "tag", "dead", "pathcrf", "teamctrl")


def _no_run(*a, **k):
    raise RuntimeError("PathCRF cache miss: refusing to run PathCRF (read-only cache)")


pcrf._run_half = _no_run


def load_a5c(sel_dir):
    from make_candidate import load_selection
    cfg, dead, lengths, team, merge, switch = load_selection(sel_dir)
    assert not cfg.use_switch and not cfg.use_replay
    cfg.pathcrf.cache_dir = str(MAIN / "runs" / "cache" / "pathcrf")
    return cfg, dead, team, merge


def match_inputs(m, tags, cfg, dead, team, cache_key):
    """(z, ev, strong): the per-frame inputs of exp_seq_feats.frame_features, the a5c evidence and strong touches."""
    ev = compute_evidence(m, tags, cfg, dead, cache_key=cache_key, team_model=team)
    strong = compute_strong(m, cfg)
    r = pcrf.pathcrf_posterior(m, cache_key, cfg.pathcrf)
    z = {**{f"ev_{k}": ev[k].astype(np.float32) for k in STREAMS}, "touches": strong[2], "sH": strong[0],
         "sA": strong[1], "half": m.half, "ball_present": m.ball_present, "blackout_inf": blackout_mask_from_match(m),
         "ball": m.ball.astype(np.float32), "ball_speed": np.asarray(m.ball_speed, np.float32),
         "pitch": np.asarray(m.pitch, np.float32), "fps": np.float32(m.fps), "p_home": r["p_home"],
         "imputed_frac": r["imputed_frac"]}
    return z, ev, strong


def _predict_sym(model, W, wn):
    p1 = model.predict_proba(W)
    p2 = model.predict_proba(fx.mirror(W, wn))[:, [1, 0, 2]]
    return 0.5 * (p1 + p2)


def compute_stream(z, final_dir):
    """{stream name: (F, 3) float32 log-probs} for the final variant's learned model(s)."""
    import exp_seq_gru as g
    final_dir = Path(final_dir); conf = json.loads((final_dir / "config.json").read_text())
    if abs(float(z["fps"]) - 25.0) > 1e-6:
        raise ValueError(f"features assume 25 fps (5-frame steps); got {float(z['fps'])}")
    X, names = fx.frame_features(z)
    if names != conf["feature_spec"]["step_features"]:
        raise ValueError("feature names differ from the final config's feature_spec")
    Xs, starts, seg = fx.to_steps(X, names, z["half"])
    F = len(z["half"]); out = {}
    for k in conf["stream_weights"]:
        model = k[len("seq"):]
        if model == "A":
            m = joblib.load(final_dir / "modelA.joblib")
            wn = names + [f"{a}_{d}{w}" for w in (10, 25, 50) for d in ("p", "f") for a in names]
            p = _predict_sym(m, fx.windowed(Xs, seg), wn)
            lp = np.log(np.clip(p, 1e-4, 1)).astype(np.float32)
        elif model == "B":
            ck = torch.load(final_dir / "modelB.pt", weights_only=False)
            net = g.Net(ck["nf"], ck["hidden"]); net.load_state_dict(ck["state_dict"]); net.eval()
            mu, sd = ck["mu"].numpy(), ck["sd"].numpy()
            lp = np.zeros((len(Xs), 3), np.float32)
            for h in np.unique(seg):              # one sequence per half (windows never cross halves)
                ii = np.flatnonzero(seg == h)
                lp[ii] = g.infer(net, Xs[ii], mu, sd, ck["names"])
        else:
            raise ValueError(k)
        out[k] = fx.upsample(lp, starts, F, z["half"])
    return out, conf


def decode_variant(m, ev, strong, seq, conf, cfg, final_dir):
    dcfg = copy.deepcopy(cfg.decoder); ev_dec = dict(ev)
    for k, w in conf["stream_weights"].items():
        ev_dec[k] = seq[k]; dcfg.weights[k] = float(w); dcfg.lengths[k] = float(conf["stream_lengths"][k])
    lab = postprocess(decode(ev_dec, m.half, dcfg), m.half, m.fps, cfg.post, strong_H=strong[0], strong_A=strong[1],
                      touches=strong[2])
    if conf["merge"]:
        mm = MergeModel.load(Path(final_dir) / conf["merge_model"])
        lab = apply_merge_model(lab, m, ev, strong, mm, float(conf["merge_theta"]))
    return lab


def main():
    ap = argparse.ArgumentParser()
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--sim", nargs=2, metavar=("KEY", "SEED")); g.add_argument("--real", metavar="DATA_DIR")
    ap.add_argument("--select", default=str(MAIN / "runs" / "select" / "a5c"))
    ap.add_argument("--final", default=str(MAIN / "runs" / "select" / "exp_seq" / "final"))
    ap.add_argument("--out", required=True); ap.add_argument("--drift", action="store_true")
    a = ap.parse_args()
    cfg, dead, team, merge_a5c = load_a5c(a.select)
    if a.sim:
        key, seed = a.sim[0], int(a.sim[1])
        if int(key.split("_")[1]) == SEALED_MATCH:
            raise SystemExit("sealed match refused")
        s = load_half_sim(MAIN / "runs" / "sim", key, seed); m, tags, ck = s.match, s.tags, f"{key}_s{seed}"
    else:
        from mtah.io.metrica_loader import load_tactical
        from mtah.io.tags import load_tags
        m = load_tactical(a.real); tags = load_tags(a.real); ck = "real_tactical"
    z, ev, strong = match_inputs(m, tags, cfg, dead, team, ck)
    seq, conf = compute_stream(z, a.final)
    arr = np.stack([seq[k] for k in conf["stream_weights"]]).astype(np.float32)
    np.save(a.out, arr)
    h = hashlib.sha256(Path(a.out).read_bytes()).hexdigest()
    print(json.dumps({"out": a.out, "streams": list(conf["stream_weights"]), "shape": list(arr.shape), "sha256": h}))
    if a.drift:
        if a.sim:
            raise SystemExit("--drift is for the real match")
        mask = blackout_mask_from_match(m)
        lab_a5c = predict(m, tags, cfg, dead, evidence=ev, strong=strong, merge_model=merge_a5c)
        lab_new = decode_variant(m, ev, strong, seq, conf, cfg, a.final)
        keys = ("non_blackout_D_share", "spells_per_graded_min", "tag_agreement_implied", "pred_D_share", "n_spells",
                "spell_len_median_s")
        rep = {}
        for name, lab in (("a5c", lab_a5c), (conf["variant"], lab_new)):
            d = drift_report(lab, m, tags, mask, code_table=cfg.tag.code_table)
            rep[name] = {k: d[k] for k in keys}
            v, r = d["non_blackout_D_share"], d["spells_per_graded_min"]
            rep[name]["in_bands"] = bool(0.10 <= v <= 0.18 and 2.7 <= r <= 4.3)
        print(json.dumps(rep, indent=1))
        Path(a.out).with_suffix(".drift.json").write_text(json.dumps(rep, indent=1))


if __name__ == "__main__":
    main()
