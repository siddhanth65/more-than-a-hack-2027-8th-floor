"""exp-seq stage 2: 5 fps feature caches for all 24 halves + Model A (HistGradientBoosting on windowed features).

Inner model: train 9002-9005 (team-swap mirror augmentation), iterations chosen by 9006 log-loss -> 9006 step
log-probs (used by stage 3 to choose the decoder weight). Final model: 9002-9006 with the chosen iterations -> 9001
step log-probs. Predictions are team-symmetrised: 0.5 * (p(x) + swap(p(mirror x))).
"""
import json, os, sys, time
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "4")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "scripts"))

import joblib
import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import log_loss

import exp_seq_feats as fx

OUT = Path(r"WORKDIR/possession\runs\select\exp_seq")
SEEDS = [100, 101]
INNER_TRAIN = [9002, 9003, 9004, 9005]; INNER_VAL = [9006]; FINAL_TRAIN = [9002, 9003, 9004, 9005, 9006]; TEST = [9001]
TRAIN_STRIDE = 2


def names_for(ms):
    return [f"game_{m}_{h}_s{s}" for m in ms for h in ("H1", "H2") for s in SEEDS]


def build_feats(log):
    (OUT / "feat").mkdir(exist_ok=True)
    for n in names_for(FINAL_TRAIN + TEST):
        p = OUT / "feat" / f"{n}.npz"
        if p.exists():
            continue
        t0 = time.time()
        z = dict(np.load(OUT / "halves" / f"{n}.npz"))
        X, names = fx.frame_features(z)
        Xs, starts, seg = fx.to_steps(X, names, z["half"])
        np.savez_compressed(p, X=Xs, starts=starts, seg=seg, y=fx.step_labels(z["truth_ctrl"], starts),
                            n_frames=np.int64(len(z["half"])), names=np.array(names))
        log(f"feat {n} {Xs.shape} {time.time() - t0:.1f}s")


def load_feat(n):
    z = np.load(OUT / "feat" / f"{n}.npz")
    return z["X"], z["starts"], z["seg"], z["y"], [str(s) for s in z["names"]]


def design(ms, stride=1, aug=True, labelled=True):
    Xs, ys = [], []
    for n in names_for(ms):
        X, _, seg, y, names = load_feat(n)
        W = fx.windowed(X, seg)
        wn = names + [f"{a}_{d}{w}" for w in (10, 25, 50) for d in ("p", "f") for a in names]
        keep = (y < 3) if labelled else np.ones(len(y), bool)
        idx = np.flatnonzero(keep)[::stride]
        Xs.append(W[idx]); ys.append(y[idx])
        if aug:
            Xs.append(fx.mirror(W[idx], wn)); ys.append(fx.mirror_y(y[idx]))
    return np.concatenate(Xs), np.concatenate(ys), wn


def predict_sym(model, W, wn):
    p1 = model.predict_proba(W)
    p2 = model.predict_proba(fx.mirror(W, wn))[:, [1, 0, 2]]
    return 0.5 * (p1 + p2)


def save_preds(model, ms, tag, wn_ref):
    d = OUT / f"pred_A"; d.mkdir(exist_ok=True)
    for n in names_for(ms):
        X, starts, seg, y, names = load_feat(n)
        W = fx.windowed(X, seg)
        p = predict_sym(model, W, wn_ref)
        np.savez_compressed(d / f"{n}.npz", logp=np.log(np.clip(p, 1e-4, 1)).astype(np.float32), starts=starts,
                            y=y, model=np.array(tag))


def main():
    logf = open(OUT / "stage2.log", "a")

    def log(msg):
        line = f"[{time.strftime('%H:%M:%S')}] {msg}"; print(line, flush=True); logf.write(line + "\n"); logf.flush()

    build_feats(log)
    res = {}
    # ---- inner model: 9002-9005, iterations by 9006 log-loss -------------------------------------------------------
    Xtr, ytr, wn = design(INNER_TRAIN, stride=TRAIN_STRIDE)
    Xva, yva, _ = design(INNER_VAL, stride=1, aug=False)
    log(f"inner train {Xtr.shape}, val {Xva.shape}")
    t0 = time.time()
    hp = dict(learning_rate=0.08, max_leaf_nodes=31, min_samples_leaf=200, l2_regularization=1.0, max_bins=63)
    m = HistGradientBoostingClassifier(max_iter=50, warm_start=True, early_stopping=False, random_state=0, **hp)
    curve = []
    for it in range(50, 801, 50):
        m.set_params(max_iter=it); m.fit(Xtr, ytr)
        ll = log_loss(yva, predict_sym(m, Xva, wn), labels=[0, 1, 2]); curve.append((it, ll))
        log(f"  iter {it}: val logloss {ll:.4f} ({time.time() - t0:.0f}s)")
        if len(curve) >= 4 and min(c[1] for c in curve[-3:]) > min(c[1] for c in curve[:-3]):
            break
    best_it = min(curve, key=lambda c: c[1])[0]
    t_inner = time.time() - t0
    m_inner = HistGradientBoostingClassifier(max_iter=best_it, early_stopping=False, random_state=0, **hp).fit(Xtr, ytr)
    save_preds(m_inner, INNER_VAL, "inner", wn)
    acc = (predict_sym(m_inner, Xva, wn).argmax(1) == yva).mean()
    log(f"inner best_it {best_it}, val acc {acc:.4f}, curve {curve}")
    del Xtr, ytr, Xva, yva
    # ---- final model: 9002-9006 ---------------------------------------------------------------------------------
    Xtr, ytr, _ = design(FINAL_TRAIN, stride=TRAIN_STRIDE)
    t0 = time.time()
    m_final = HistGradientBoostingClassifier(max_iter=best_it, early_stopping=False, random_state=0, **hp).fit(Xtr, ytr)
    t_final = time.time() - t0
    joblib.dump(m_final, OUT / "models" / "modelA_final.joblib")
    t0 = time.time(); save_preds(m_final, TEST, "final", wn); t_inf = time.time() - t0
    Xte, yte, _ = design(TEST, aug=False)
    acc_te = (predict_sym(m_final, Xte, wn).argmax(1) == yte).mean()
    res = {"hp": hp, "best_it": best_it, "curve": curve, "inner_seconds": round(t_inner, 1),
           "final_train_seconds": round(t_final, 1), "test_inference_seconds_4_halves": round(t_inf, 1),
           "n_features": len(wn), "train_rows_final": int(len(ytr)), "test_step_acc": float(acc_te),
           "val_step_acc": float(acc)}
    (OUT / "modelA.json").write_text(json.dumps(res, indent=1))
    log(f"MODEL A done {res}")


if __name__ == "__main__":
    main()
