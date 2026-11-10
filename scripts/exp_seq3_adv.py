"""exp-seq3 lever-1 gate: sim-vs-real adversarial AUC of Model B's step features under each normalisation mode.
Label-free (real side: features only). Same machinery and sim side as scripts/seq2_adversarial.py.
  python scripts/exp_seq3_adv.py           -> exp_seq3/real_steps.npz (cached), exp_seq3/adversarial_norm_halfcv.json"""
import json, os, sys, time
os.environ["OMP_NUM_THREADS"] = "4"
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np

import seq3_common as c3
import touch_adversarial as ta
from touch_common import MATCHES, SEED

P = c3.EXP3.parent / "exp_seq2"


def real_cached():
    f = c3.EXP3 / "real_steps.npz"
    if not f.exists():
        import seq2_adversarial as sa
        parts, names = sa.real_steps()
        _, Xs, seg = parts[0]
        np.savez_compressed(f, X=Xs, seg=seg, names=np.array(names))
    z = np.load(f)
    return z["X"], z["seg"], [str(n) for n in z["names"]]


def half_cv_auc(sim, real, names):
    """Leave-halves-out 2-fold CV: fold 0 = real H1 + sim halves 0-5, fold 1 = real H2 + sim halves 6-11. Per-half
    fingerprints (atoms a per-half transform leaves) cannot transfer across folds, so this measures sim-vs-real shift,
    not half identity (the block CV of touch_adversarial does not control for that)."""
    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.metrics import roc_auc_score
    from mtah.evidence import seq_feats as fx
    parts = []
    for i, (_, X, seg) in enumerate(sim):
        parts.append((X, 0, 0 if i < len(sim) // 2 else 1))
    _, Xr, segr = real[0]
    for j, h in enumerate(np.unique(segr)):
        parts.append((Xr[segr == h], 1, j))
    Xs, ys, fs = [], [], []
    for X, lab, f in parts:
        for M in (X, fx.mirror(X, names)):
            Xs.append(M); ys.append(np.full(len(X), lab)); fs.append(np.full(len(X), f))
    X, y, f = np.concatenate(Xs), np.concatenate(ys), np.concatenate(fs); oof = np.zeros(len(y))
    for k in (0, 1):
        tr = np.flatnonzero(f != k)[::4]; te = f == k
        oof[te] = HistGradientBoostingClassifier(**ta.HP).fit(X[tr], y[tr]).predict_proba(X[te])[:, 1]
    return float(roc_auc_score(y, oof))


def main():
    c3.EXP3.mkdir(parents=True, exist_ok=True); t0 = time.time()
    Xr, segr, rn = real_cached()
    sim = []
    for F in MATCHES:
        for h in ("H1", "H2"):
            z = np.load(P / "lomo" / "feat" / f"fold{F}" / f"game_{F}_{h}_s{SEED}.npz")
            sim.append((f"game_{F}_{h}", z["X"], z["seg"])); names = [str(n) for n in z["names"]]
    assert rn == names
    jf = c3.EXP3 / "adversarial_norm_halfcv.json"
    res = json.loads(jf.read_text()) if jf.exists() else {}   # merge: never overwrite earlier modes
    for mode in (sys.argv[1:] or c3.MODES):
        s = [(k, c3.normalise(X, names, seg, mode), seg) for k, X, seg in sim]
        r = [("real", c3.normalise(Xr, names, segr, mode), segr)]
        X, y, blocks = ta.assemble(s, r, names); sa = ta.single_auc(X, y, names); del X
        res[mode] = {"auc_half_cv": half_cv_auc(s, r, names), "top_single_auc": dict(sorted(sa.items(), key=lambda kv: -kv[1])[:8])}
        print(mode, json.dumps(res[mode]), f"({time.time() - t0:.0f}s)", flush=True)
        jf.write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
