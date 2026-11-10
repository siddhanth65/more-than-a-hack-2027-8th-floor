"""Learned sequence stream(s) for one match (a simulator half or the real match); run in the mtah venv.

Model A (HGB) runs in this venv (pinned sklearn); Model B (BiGRU) runs in the PathCRF venv via subprocess
(mtah.evidence.seqstream / seq_b_infer.py); no torch is imported here. The a5c-style evidence comes from the
selection's artefacts (scripts/make_candidate.load_selection); PathCRF is read from its cache only.
Output: .npy (n_streams, F, 3) float32 log-probabilities (d_offset NOT applied), np.save = byte-reproducible.

  python scripts/seq_stream.py --sim game_9001_H1 100 --out x.npy                         (final dir + a13)
  python scripts/seq_stream.py --real <DATA_DIR> --out real.npy --select runs/select/a10  (registered selection)
"""
import argparse, hashlib, json, os, sys
from pathlib import Path

os.environ["OMP_NUM_THREADS"] = "4"
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "scripts"))

import numpy as np

import mtah.evidence.pathcrf as pcrf
import mtah.evidence.seqstream as sq
from mtah.pipeline import compute_evidence, compute_strong
from mtah.sim.dataset import SEALED_MATCH, load_half_sim

MAIN = Path(__file__).resolve().parents[1]


def _no_run(*a, **k):
    raise RuntimeError("PathCRF cache miss: refusing to run PathCRF (read-only cache)")


pcrf._run_half = _no_run


def spec_from_final(final_dir):
    c = json.loads((Path(final_dir) / "config.json").read_text())
    return {"weights": c["stream_weights"], "lengths": c["stream_lengths"], "d_offset": 0.0,
            "modelA": "modelA.joblib", "modelB": "modelB.pt", "step_features": c["feature_spec"]["step_features"]}


def load_match(sim=None, real=None):
    if sim:
        key, seed = sim[0], int(sim[1])
        if int(key.split("_")[1]) == SEALED_MATCH:
            raise SystemExit("sealed match refused")
        s = load_half_sim(MAIN / "runs" / "sim", key, seed)
        return s.match, s.tags, f"{key}_s{seed}"
    from mtah.io.metrica_loader import load_tactical
    from mtah.io.tags import load_tags
    return load_tactical(real), load_tags(real), "real_tactical"


def main():
    ap = argparse.ArgumentParser()
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--sim", nargs=2, metavar=("KEY", "SEED")); g.add_argument("--real", metavar="DATA_DIR")
    ap.add_argument("--select", default=None, help="selection dir with a seq_stream block (default: a13 + --final)")
    ap.add_argument("--final", default=str(MAIN / "runs" / "select" / "exp_seq2" / "final"))
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    from make_candidate import load_selection
    if a.select:
        cfg, dead, lengths, team, merge, switch, spec = load_selection(a.select, with_seq=True)
        model_dir = Path(a.select)
    else:
        cfg, dead, lengths, team, merge, switch, _ = load_selection(MAIN / "runs" / "select" / "a13r", with_seq=True)
        spec, model_dir = spec_from_final(a.final), Path(a.final)
    cfg.pathcrf.cache_dir = str(MAIN / "runs" / "cache" / "pathcrf")
    m, tags, ck = load_match(a.sim, a.real)
    ev = compute_evidence(m, tags, cfg, dead, cache_key=ck, team_model=team)
    z = sq.feature_inputs(m, ev, compute_strong(m, cfg), cfg.pathcrf, ck)
    st = sq.compute_streams(z, spec, model_dir)
    arr = np.stack([st[k] for k in spec["weights"]]).astype(np.float32)
    np.save(a.out, arr)
    print(json.dumps({"out": a.out, "streams": list(spec["weights"]), "shape": list(arr.shape),
                      "sha256": hashlib.sha256(Path(a.out).read_bytes()).hexdigest(), **sq.LAST_INFO}))


if __name__ == "__main__":
    main()
