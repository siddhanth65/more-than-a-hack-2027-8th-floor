"""Plan 7 T2 (a14): sim-vs-real adversarial validation of the learned streams' INPUTS on the corrected build.

Same machinery as scripts/touch_adversarial.py (time-blocked 5-fold CV, HGB, team-mirror augmentation; sim = fold-F test
halves of 9001-9006, seed 100, with the out-of-sample evidence of the exp-seq2 `fold{F}` caches vs the real match), but
  * sim side  = runs/select/exp_seq2/lomo/feat/fold{F}/*_s100.npz (step features of the NEW build, a13-cell evidence, rel
                touches; the Model A/B training inputs),
  * real side = the registered a13 selection (config + dead / team models) run on the real match: the SAME step features
                (mtah.evidence.seqstream.feature_inputs -> seq_feats.frame_features -> to_steps).
auc_base = AUC on the 5 fps step features (Model B's input); auc_windowed = AUC on Model A's windowed inputs.
Aggregates only; the sealed match and the ground truth are never loaded.         -> runs/select/exp_seq2/adversarial.json
"""
import json, os, sys, time
os.environ["OMP_NUM_THREADS"] = "4"
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
from pathlib import Path

import touch_adversarial as ta
import mtah.evidence.pathcrf as pcrf
import mtah.evidence.seqstream as sq
from make_candidate import load_selection
from touch_common import MAIN, MATCHES, SEED, load_real
from mtah.evidence import seq_feats as fx
from mtah.io.tags import load_tags
from mtah.pipeline import compute_evidence, compute_strong

P = MAIN / "runs" / "select" / "exp_seq2"


def _no_run(*a, **k):
    raise RuntimeError("PathCRF cache miss: refusing to run PathCRF (read-only cache)")


pcrf._run_half = _no_run


def real_steps():
    cfg, dead, lengths, team, merge, switch, seq = load_selection(Path(__file__).resolve().parents[1] / "runs" / "select" / "a13r", with_seq=True)
    cfg.pathcrf.cache_dir = str(MAIN / "runs" / "cache" / "pathcrf")
    m = load_real(); tags = load_tags(os.environ.get("MTAH_DATA_DIR", r"HOME/mtah-data"))
    ev = compute_evidence(m, tags, cfg, dead, cache_key="real_tactical", team_model=team)
    z = sq.feature_inputs(m, ev, compute_strong(m, cfg), cfg.pathcrf, "real_tactical")
    X, names = fx.frame_features(z); Xs, _, seg = fx.to_steps(X, names, m.half)
    return [("real", Xs, seg)], names


def main():
    t0 = time.time()
    sim = []
    for F in MATCHES:
        for h in ("H1", "H2"):
            z = np.load(P / "lomo" / "feat" / f"fold{F}" / f"game_{F}_{h}_s{SEED}.npz")
            sim.append((f"game_{F}_{h}", z["X"], z["seg"])); names = [str(n) for n in z["names"]]
    real, rn = real_steps(); assert rn == names
    X, y, blocks = ta.assemble(sim, real, names); folds = ta.fold_of(blocks, y)
    auc, fa = ta.cv_auc(X, y, folds); sa = ta.single_auc(X, y, names)
    ti = [names.index(n) for n in ta.TOUCH_NAMES]; oi = [i for i in range(len(names)) if i not in ti]
    res = {"auc_base": auc, "auc_base_folds": fa, "auc_touch_group_only": ta.cv_auc(X[:, ti], y, folds)[0],
           "auc_without_touch_group": ta.cv_auc(X[:, oi], y, folds)[0],
           "top_single_auc": dict(sorted(sa.items(), key=lambda kv: -kv[1])[:8]),
           "reference": {"a10_abs_base": 0.9970312742891425, "a10_abs_windowed": 0.9954552530834226,
                         "a13_rebuild_base": 0.951794771319997, "a13_rebuild_windowed": 0.9854530864769107}}
    print(json.dumps(res, indent=1), f"({time.time() - t0:.0f}s)", flush=True)
    del X
    Xw, yw, bw = ta.assemble(sim, real, names, windowed=True)
    res["auc_windowed"] = ta.cv_auc(Xw, yw, ta.fold_of(bw, yw), sub=6)[0]
    print("windowed", res["auc_windowed"], f"({time.time() - t0:.0f}s)")
    (P / "adversarial.json").write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
