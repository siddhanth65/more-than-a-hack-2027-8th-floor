"""Step 3b (label-free): does the feed-adaptive touch detector close the sim-to-real gap of the learned streams' inputs?
Re-runs the exp-robust adversarial validation (scripts/robust_adversarial.py of the exp-robust branch; the AUC machinery is
copied here unchanged: time-blocked 5-fold CV, HGB, team-mirror augmentation, sim 9001-9006 seed 100 vs the real match) with
the 9 touch columns of the 5 fps step features recomputed from `abs` (reproduces the cached features) or `rel` touches, on
BOTH sides. Everything else in the features (the five evidence streams incl. teamctrl, ball geometry, PathCRF posterior)
stays as cached, so the change isolates the touch features.  Aggregates only; sealed 9007 / ground truth never touched.
  python scripts/touch_adversarial.py [--windowed] [--only cached,abs,rel]               -> runs/touch/adversarial.json"""
import argparse, json, os, sys, time
os.environ["OMP_NUM_THREADS"] = "4"
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score

from touch_common import MAIN, MATCHES, OUT, SEED, load_real, sim_halves
from mtah.evidence import seq_feats as fx
from mtah.evidence.touches import TouchParams, detect_touches, strong_touch_frames

LOMO = MAIN / "runs" / "select" / "exp_seq" / "lomo"
CACHE = MAIN / "runs" / "select" / "exp_robust" / "real_cache.npz"
TOUCH_NAMES = ["touch_H", "touch_A", "touch_C", "strong_H", "strong_A", "log_since_touch", "log_until_touch",
               "last_touch_team", "next_touch_team"]
BLOCK = 1500; NFOLD = 5
HP = dict(max_iter=200, learning_rate=0.1, max_leaf_nodes=31, min_samples_leaf=200, max_bins=63, random_state=0,
          class_weight="balanced")


def touch_cols(touches, sH, sA, F, fps):
    """The touch columns of seq_feats.frame_features (same code), (F, 9)."""
    tH = np.zeros(F); tA = np.zeros(F); tC = np.zeros(F)
    if len(touches):
        fr = touches[:, 0].astype(int); team = touches[:, 2]
        tH[fr[team == 0]] = 1; tA[fr[team == 1]] = 1; tC[fr[team == -1]] = 1
    anyt = (tH + tA + tC) > 0
    tsign = tH - tA
    since, last = fx._since(anyt, fps)
    rsince, rlast = fx._since(anyt[::-1], fps)
    until = rsince[::-1]; nxt = np.where(rlast[::-1] >= 0, F - 1 - rlast[::-1], -1)
    last_team = np.where(last >= 0, tsign[np.maximum(last, 0)], 0.0)
    next_team = np.where(nxt >= 0, tsign[np.maximum(nxt, 0)], 0.0)
    return np.column_stack([tH, tA, tC, sH.astype(float), sA.astype(float), np.log1p(since), np.log1p(until), last_team,
                            next_team]).astype(np.float32)


def swap_touch(X, names, match, params):
    """Replace the touch columns of cached step features X by those from `params` touches."""
    t = detect_touches(match, params); F = match.n_frames
    sH = strong_touch_frames(t, 0, params, F); sA = strong_touch_frames(t, 1, params, F)
    cols = touch_cols(t, sH, sA, F, match.fps)
    Xs, _, _ = fx.to_steps(cols, TOUCH_NAMES, match.half)
    assert len(Xs) == len(X), (len(Xs), len(X))
    X = X.copy()
    for i, n in enumerate(TOUCH_NAMES):
        X[:, names.index(n)] = Xs[:, i]
    return X, len(t)


def load_sim(params):
    out = []; nt = []
    for m in MATCHES:
        for key, s in sim_halves([m]):
            z = np.load(LOMO / "feat" / f"fold{m}" / f"{key}_s{SEED}.npz")
            X, seg, names = z["X"], z["seg"], [str(n) for n in z["names"]]
            if params is not None:
                X, n = swap_touch(X, names, s.match, params)
                nt.append(n / ((s.match.ball_present & s.match.graded).sum() / 25 / 60))
            out.append((f"{key}_s{SEED}", X, seg))
    return out, names, (float(np.mean(nt)) if nt else None)


def load_real_steps(params):
    z = np.load(CACHE); X, seg, names = z["X"], z["seg"], [str(n) for n in z["names"]]
    rate = None
    if params is not None:
        m = load_real(); X, n = swap_touch(X, names, m, params); rate = n / ((m.ball_present & m.graded).sum() / 25 / 60)
    return [("real", X, seg)], names, rate


def assemble(sim, real, names, windowed=False):
    Xs, ys, bs = [], [], []; bid = 0
    for lab, parts in ((0, sim), (1, real)):
        for _, X, seg in parts:
            W = fx.windowed(X, seg) if windowed else X
            wn = (names + [f"{a}_{d}{w}" for w in (10, 25, 50) for d in ("p", "f") for a in names]) if windowed else names
            for h in np.unique(seg):
                ii = np.flatnonzero(seg == h)
                blk = bid + np.arange(len(ii)) // BLOCK; bid = int(blk.max()) + 1
                for M in (W[ii], fx.mirror(W[ii], wn)):
                    Xs.append(M); ys.append(np.full(len(ii), lab)); bs.append(blk)
    return np.concatenate(Xs), np.concatenate(ys), np.concatenate(bs)


def fold_of(blocks, y):
    f = np.zeros(len(blocks), int)
    for lab in (0, 1):
        ub = np.unique(blocks[y == lab]); rng = np.random.default_rng(lab)
        perm = rng.permutation(len(ub)); fmap = {b: i % NFOLD for b, i in zip(ub[perm], range(len(ub)))}
        msk = y == lab; f[msk] = [fmap[b] for b in blocks[msk]]
    return f


def cv_auc(X, y, folds, sub=4):
    oof = np.zeros(len(y))
    for k in range(NFOLD):
        tr = np.flatnonzero(folds != k)[::sub]; te = np.flatnonzero(folds == k)
        oof[te] = HistGradientBoostingClassifier(**HP).fit(X[tr], y[tr]).predict_proba(X[te])[:, 1]
    return float(roc_auc_score(y, oof)), [float(roc_auc_score(y[folds == k], oof[folds == k])) for k in range(NFOLD)]


def single_auc(X, y, names):
    out = {}
    for i, n in enumerate(names):
        a = roc_auc_score(y, X[:, i]); out[n] = float(max(a, 1 - a))
    return out


def run(tag, params, windowed):
    t0 = time.time()
    sim, names, sr = load_sim(params); real, _, rr = load_real_steps(params)
    X, y, blocks = assemble(sim, real, names); folds = fold_of(blocks, y)
    auc, fa = cv_auc(X, y, folds)
    sa = single_auc(X, y, names)
    ti = [names.index(n) for n in TOUCH_NAMES]; oi = [i for i in range(len(names)) if i not in ti]
    auc_touch_only, _ = cv_auc(X[:, ti], y, folds)
    auc_no_touch, _ = cv_auc(X[:, oi], y, folds)
    S, R = X[y == 0], X[y == 1]
    med = {n: {"sim_median": float(np.median(S[:, names.index(n)])), "real_median": float(np.median(R[:, names.index(n)])),
               "sim_mean": float(S[:, names.index(n)].mean()), "real_mean": float(R[:, names.index(n)].mean())}
           for n in TOUCH_NAMES}
    res = {"tag": tag, "touch_rate_per_visible_min": {"sim_mean": sr, "real": rr}, "auc_base": auc, "auc_base_folds": fa,
           "auc_touch_group_only": auc_touch_only, "auc_without_touch_group": auc_no_touch,
           "single_auc_touch_cols": {n: sa[n] for n in TOUCH_NAMES},
           "top_single_auc_other": dict(sorted(((n, v) for n, v in sa.items() if n not in TOUCH_NAMES), key=lambda kv: -kv[1])[:5]),
           "touch_col_summary": med}
    print(f"[{tag}] base {auc:.4f} touch-only {auc_touch_only:.4f} no-touch {auc_no_touch:.4f} since {sa['log_since_touch']:.3f} "
          f"({time.time() - t0:.0f}s)", flush=True)
    if windowed:
        del X
        Xw, yw, bw = assemble(sim, real, names, windowed=True)
        res["auc_windowed"] = cv_auc(Xw, yw, fold_of(bw, yw), sub=6)[0]
        print(f"[{tag}] windowed {res['auc_windowed']:.4f} ({time.time() - t0:.0f}s)", flush=True)
    return res


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--windowed", action="store_true"); ap.add_argument("--only", default=None)
    a = ap.parse_args()
    runs = {"cached": None, "abs": TouchParams(), "rel": TouchParams(mode="rel")}
    out = {}
    p = OUT / "adversarial.json"
    if p.exists():
        out = json.loads(p.read_text())
    for tag, prm in runs.items():
        if a.only and tag not in a.only.split(","):
            continue
        out[tag] = run(tag, prm, a.windowed)
        p.write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
