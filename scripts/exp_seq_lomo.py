"""exp-seq T2: 6-fold LOMO of the learned sequence stream(s) as EXTRA decoder evidence on top of a5c.

Folds: 9001-9006 held out in turn (train on the other five, seeds 100 101; 9007 is never loaded). Per fold F:
  * test halves of F: evidence from dead/team models trained on all five train matches (src "fold{F}");
  * train halves of m != F: CROSS-FITTED evidence from models trained on the other four train matches, i.e. on
    {9001..9006} minus {F, m} (src "pair{a}_{b}", shared by folds a and b);
  * inner validation match V = max(train) (as the prototype and select.merge_training_table): Model A iterations /
    Model B epochs are chosen on V with inner models trained on train minus V; the inner models' predictions on V
    train the per-decoder merge model (same 3 GRID8 reference cells as a5c's fold merge model, honest inner split);
  * final fold models (all five train matches) predict the test halves.
Decoders: a5c, A extra (w 2 4 8), B extra (w 4 8 12 16), A+B (w_A 4, w_B 8 12); each with merge OFF and merge ON
(theta 0.7, merge model retrained on that decoder's inner predictions). Scored on the 4 spellF1 variants.

Stages (each resumable; files are per fold / per half checkpoints):
  seed      copy the fold-9001 prototype caches/predictions into the LOMO layout (+ p_dead)
  evidence  [--folds ...] stream models + half caches
  feat      5 fps step features for every half cache
  A         [--folds ...] Model A (HGB) inner + final per fold
  decode    [--folds ...] variants per fold -> lomo/decode_f{F}.csv
  summary   6-fold table, mean rank, per-fold wins vs a5c -> lomo/summary.csv / summary.json
  finalA    Model A on all six matches (fold{m} caches) -> final/modelA.joblib
  finalize  merge model + config.json for the chosen variant
Model B (BiGRU, PathCRF venv torch) lives in scripts/exp_seq_lomo_gru.py.
Reads the main tree's runs/sim and runs/cache/pathcrf (read-only); writes only under runs/select/exp_seq/.
"""
import argparse, copy, json, os, shutil, sys, time
from pathlib import Path

os.environ["OMP_NUM_THREADS"] = "4"
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "scripts"))

import joblib
import numpy as np
import pandas as pd

import exp_seq_feats as fx
import mtah.evidence.pathcrf as pcrf
import mtah.select as sel
from mtah.decode.viterbi import decode
from mtah.eval.drift import blackout_mask_from_match
from mtah.eval.scorer import score
from mtah.evidence.dead import DeadModel, dead_evidence, dead_table
from mtah.evidence.team_ctrl import TeamCtrlModel
from mtah.pipeline import PipelineConfig, compute_evidence, compute_strong
from mtah.sim.dataset import SEALED_MATCH, list_sim_keys, load_half_sim, match_id
from mtah.spells.merge import MergeModel, apply_merge_model, match_spell_table, merge_labels
from mtah.spells.postprocess import postprocess

MAIN = Path(r"WORKDIR/possession")
SIM = MAIN / "runs" / "sim"
P = MAIN / "runs" / "select" / "exp_seq"
L = P / "lomo"
FINAL = P / "final"
MATCHES = [9001, 9002, 9003, 9004, 9005, 9006]
SEEDS = [100, 101]
STREAMS = ("ball", "tag", "dead", "pathcrf", "teamctrl")
VARS = ("ctrl_Dfull", "raw_Dfull", "ctrl_Dnarrow", "raw_Dnarrow")
LAB = {"H": 0, "A": 1, "D": 2}
THETA = 0.7
DECODERS = {"a5c": {}, "A2": {"A": 2.0}, "A4": {"A": 4.0}, "A8": {"A": 8.0},
            "B4": {"B": 4.0}, "B8": {"B": 8.0}, "B12": {"B": 12.0}, "B16": {"B": 16.0},
            "AB4_8": {"A": 4.0, "B": 8.0}, "AB4_12": {"A": 4.0, "B": 12.0}}
HP_A = dict(learning_rate=0.08, max_leaf_nodes=31, min_samples_leaf=200, l2_regularization=1.0, max_bins=63)
TRAIN_STRIDE = 2


def _no_run(*a, **k):
    raise RuntimeError("PathCRF cache miss: refusing to run PathCRF (read-only cache)")


pcrf._run_half = _no_run
_LOGF = None


def log(msg):
    global _LOGF
    if _LOGF is None:
        L.mkdir(parents=True, exist_ok=True); _LOGF = open(L / "lomo.log", "a")
    line = f"[{time.strftime('%H:%M:%S')}] {msg}"; print(line, flush=True); _LOGF.write(line + "\n"); _LOGF.flush()


# ---- layout ---------------------------------------------------------------------------------------------------------
def split(F):
    train = [m for m in MATCHES if m != F]
    V = max(train)
    return train, V, [m for m in train if m != V]


def src_fold(F):
    return f"fold{F}"


def src_pair(a, b):
    a, b = sorted((a, b)); return f"pair{a}_{b}"


def src_of(F, m):
    """Evidence source of match m's halves in fold F (test match: fold models; train match: cross-fitted pair)."""
    return src_fold(F) if m == F else src_pair(F, m)


def halves_of(m):
    return [(f"game_{m}_{h}", s) for h in ("H1", "H2") for s in SEEDS]


def hpath(src, key, seed):
    return L / "halves" / src / f"{key}_s{seed}.npz"


def fpath(src, key, seed):
    return L / "feat" / src / f"{key}_s{seed}.npz"


def ppath(model, F, key, seed):
    """Predictions of Model A/B in fold F (inner model on V's halves, final fold model on F's halves); F = 'all'
    is unused (the final model has no held-out sim predictions)."""
    return L / f"pred_{model}" / f"f{F}" / f"{key}_s{seed}.npz"


def base_cfg():
    base = PipelineConfig()
    base.use_pathcrf = True; base.use_teamctrl = True; base.use_switch = False
    base.pathcrf.cache_dir = str(MAIN / "runs" / "cache" / "pathcrf")
    return base


def a5c_cfg(base=None):
    cfg = sel.apply_cell(base or base_cfg(), sel.GRID8[0])
    cfg.decoder.lengths = sel.estimate_lengths([], cfg, None)   # unit lengths (LENGTHS_MODE "unit")
    return cfg


def keys_of(ms):
    return [k for k in list_sim_keys(SIM) if match_id(k) in ms and match_id(k) != SEALED_MATCH]


def load_sims(ms):
    out = []
    for k in keys_of(ms):
        assert match_id(k) != SEALED_MATCH
        for s in SEEDS:
            out.append(load_half_sim(SIM, k, s))
    return out


def load_sim(key, seed):
    assert int(key.split("_")[1]) != SEALED_MATCH
    return load_half_sim(SIM, key, seed)


# ---- stage: evidence ------------------------------------------------------------------------------------------------
def get_models(src, train_ms, base):
    md = L / "models"; md.mkdir(parents=True, exist_ok=True)
    dp, tp = md / f"dead_{src}.joblib", md / f"team_{src}.joblib"
    if dp.exists() and tp.exists():
        return DeadModel.load(dp), TeamCtrlModel.load(tp)
    assert SEALED_MATCH not in train_ms
    t0 = time.time(); sims = load_sims(train_ms)
    dead = sel.train_dead(sims, policy="long_rel"); dead.save(dp)
    team = sel.train_team_ctrl(sims, policy="full", touch_params=base.touch); team.save(tp)
    log(f"models {src} on {train_ms}: {time.time() - t0:.0f}s")
    return dead, team


def save_half(s, cfg, dead, team, path):
    p_dead = dead.predict_proba(dead_table(s.match, s.tags, dead.policy))
    ev = compute_evidence(s.match, s.tags, cfg, None, cache_key=f"{s.key}_s{s.seed}", team_model=team)
    ev["dead"] = dead_evidence(p_dead, logit_offset=cfg.dead_offset)
    strong = compute_strong(s.match, cfg)
    r = pcrf.pathcrf_posterior(s.match, f"{s.key}_s{s.seed}", cfg.pathcrf)
    m = s.match
    tr = {v: np.array([LAB.get(c, 3) for c in s.truth[v]], np.int8) for v in ("ctrl_Dfull", "raw_Dfull")}
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp.npz")
    np.savez_compressed(tmp, **{f"ev_{k}": ev[k].astype(np.float32) for k in STREAMS},
                        touches=strong[2], sH=strong[0], sA=strong[1], half=m.half, ball_present=m.ball_present,
                        graded=m.graded, blackout_inf=blackout_mask_from_match(m), ball=m.ball.astype(np.float32),
                        ball_speed=np.asarray(m.ball_speed, np.float32), pitch=np.asarray(m.pitch, np.float32),
                        fps=np.float32(m.fps), p_home=r["p_home"], imputed_frac=r["imputed_frac"],
                        truth_ctrl=tr["ctrl_Dfull"], truth_raw=tr["raw_Dfull"], p_dead=np.asarray(p_dead, np.float32))
    os.replace(tmp, path)


def ensure_halves(src, train_ms, ms, base, cfg):
    todo = [(k, s) for m in ms for k, s in halves_of(m) if not hpath(src, k, s).exists()]
    if not todo:
        return
    dead, team = get_models(src, train_ms, base)
    for k, sd in todo:
        t0 = time.time(); s = load_sim(k, sd)
        save_half(s, cfg, dead, team, hpath(src, k, sd)); log(f"half {src}/{k}_s{sd} {time.time() - t0:.0f}s")


def stage_evidence(folds):
    base = base_cfg(); cfg = a5c_cfg(base)
    for F in folds:
        train, V, _ = split(F)
        ensure_halves(src_fold(F), train, [F], base, cfg)
        for m in train:
            ensure_halves(src_pair(F, m), [x for x in MATCHES if x not in (F, m)], [m], base, cfg)
        log(f"evidence fold {F} complete")


# ---- stage: seed from the fold-9001 prototype ---------------------------------------------------------------------
def stage_seed():
    """Prototype (fold 9001) halves -> lomo layout (+ p_dead, checked against the cached dead evidence); models,
    feature-independent predictions A/B (inner 9006, test 9001) and the model json."""
    base = base_cfg(); cfg = a5c_cfg(base)
    md = L / "models"; md.mkdir(parents=True, exist_ok=True)
    for m in MATCHES:
        src_old = "fold9001" if m == 9001 else f"cf{m}"
        src = src_fold(9001) if m == 9001 else src_pair(9001, m)
        for kind in ("dead", "team"):
            a, b = P / "models" / f"{kind}_{src_old}.joblib", md / f"{kind}_{src}.joblib"
            if not b.exists():
                shutil.copy2(a, b)
        dead = DeadModel.load(md / f"dead_{src}.joblib")
        for k, sd in halves_of(m):
            dst = hpath(src, k, sd)
            if dst.exists():
                continue
            z = dict(np.load(P / "halves" / f"{k}_s{sd}.npz"))
            s = load_sim(k, sd)
            p_dead = dead.predict_proba(dead_table(s.match, s.tags, dead.policy))
            ev = dead_evidence(p_dead, logit_offset=cfg.dead_offset)
            assert np.array_equal(ev.astype(np.float32), z["ev_dead"]), f"p_dead mismatch {k}_s{sd}"
            z["p_dead"] = np.asarray(p_dead, np.float32)
            dst.parent.mkdir(parents=True, exist_ok=True); tmp = dst.with_suffix(".tmp.npz")
            np.savez_compressed(tmp, **z); os.replace(tmp, dst); log(f"seeded {src}/{k}_s{sd}")
    for model in ("A", "B"):
        d = L / f"pred_{model}" / "f9001"; d.mkdir(parents=True, exist_ok=True)
        for m in (9001, 9006):
            for k, sd in halves_of(m):
                if not (d / f"{k}_s{sd}.npz").exists():
                    shutil.copy2(P / f"pred_{model}" / f"{k}_s{sd}.npz", d / f"{k}_s{sd}.npz")
    ja = json.loads((P / "modelA.json").read_text())
    (L / "modelA_f9001.json").write_text(json.dumps({"best_it": ja["best_it"], "curve": ja["curve"], "seeded": True,
                                                    "val_step_acc": ja["val_step_acc"],
                                                    "test_step_acc": ja["test_step_acc"]}, indent=1))
    jb = json.loads((P / "modelB.json").read_text())
    (L / "modelB_f9001.json").write_text(json.dumps({"best_epoch": jb["best_epoch"], "inner_val_nll": jb["inner_val_nll"],
                                                    "seeded": True}, indent=1))
    log("seed done")


# ---- stage: features ------------------------------------------------------------------------------------------------
def stage_feat():
    n = 0
    for hp in sorted((L / "halves").glob("*/*.npz")):
        if hp.name.endswith(".tmp.npz"):
            continue
        src = hp.parent.name; out = L / "feat" / src / hp.name
        if out.exists():
            continue
        z = dict(np.load(hp))
        X, names = fx.frame_features(z)
        Xs, starts, seg = fx.to_steps(X, names, z["half"])
        out.parent.mkdir(parents=True, exist_ok=True); tmp = out.with_suffix(".tmp.npz")
        np.savez_compressed(tmp, X=Xs, starts=starts, seg=seg, y=fx.step_labels(z["truth_ctrl"], starts),
                            n_frames=np.int64(len(z["half"])), names=np.array(names))
        os.replace(tmp, out); n += 1
    log(f"feat: {n} new caches")


# ---- stage: Model A -------------------------------------------------------------------------------------------------
def load_feat(path):
    z = np.load(path)
    return z["X"], z["starts"], z["seg"], z["y"], [str(s) for s in z["names"]]


def wnames(names):
    return names + [f"{a}_{d}{w}" for w in (10, 25, 50) for d in ("p", "f") for a in names]


def design(paths, stride=1, aug=True):
    Xs, ys = [], []
    for p in paths:
        X, _, seg, y, names = load_feat(p)
        W = fx.windowed(X, seg); wn = wnames(names)
        idx = np.flatnonzero(y < 3)[::stride]
        Xs.append(W[idx]); ys.append(y[idx])
        if aug:
            Xs.append(fx.mirror(W[idx], wn)); ys.append(fx.mirror_y(y[idx]))
    return np.concatenate(Xs), np.concatenate(ys), wn


def predict_sym(model, W, wn):
    p1 = model.predict_proba(W)
    p2 = model.predict_proba(fx.mirror(W, wn))[:, [1, 0, 2]]
    return 0.5 * (p1 + p2)


def a_logp(model, X, seg, names):
    W = fx.windowed(X, seg)
    return np.log(np.clip(predict_sym(model, W, wnames(names)), 1e-4, 1)).astype(np.float32)


def feats_for(F, ms):
    return [fpath(src_of(F, m), k, s) for m in ms for k, s in halves_of(m)]


def save_preds_A(model, F, m):
    for k, sd in halves_of(m):
        X, starts, seg, y, names = load_feat(fpath(src_of(F, m), k, sd))
        out = ppath("A", F, k, sd); out.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(out, logp=a_logp(model, X, seg, names), starts=starts, y=y)


def stage_A(folds):
    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.metrics import log_loss
    for F in folds:
        js = L / f"modelA_f{F}.json"
        if js.exists():
            log(f"A fold {F}: done"); continue
        train, V, inner = split(F)
        t0 = time.time()
        Xtr, ytr, wn = design(feats_for(F, inner), stride=TRAIN_STRIDE)
        Xva, yva, _ = design(feats_for(F, [V]), aug=False)
        m = HistGradientBoostingClassifier(max_iter=50, warm_start=True, early_stopping=False, random_state=0, **HP_A)
        curve = []
        for it in range(50, 801, 50):
            m.set_params(max_iter=it); m.fit(Xtr, ytr)
            ll = log_loss(yva, predict_sym(m, Xva, wn), labels=[0, 1, 2]); curve.append((it, ll))
            log(f"  A f{F} iter {it}: val logloss {ll:.4f} ({time.time() - t0:.0f}s)")
            if len(curve) >= 4 and min(c[1] for c in curve[-3:]) > min(c[1] for c in curve[:-3]):
                break
        best_it = min(curve, key=lambda c: c[1])[0]
        m_inner = HistGradientBoostingClassifier(max_iter=best_it, early_stopping=False, random_state=0, **HP_A).fit(Xtr, ytr)
        save_preds_A(m_inner, F, V)
        acc = float((predict_sym(m_inner, Xva, wn).argmax(1) == yva).mean())
        t_inner = time.time() - t0
        del Xtr, ytr, Xva, yva, m, m_inner
        t0 = time.time()
        Xtr, ytr, _ = design(feats_for(F, train), stride=TRAIN_STRIDE)
        m_fin = HistGradientBoostingClassifier(max_iter=best_it, early_stopping=False, random_state=0, **HP_A).fit(Xtr, ytr)
        del Xtr, ytr
        save_preds_A(m_fin, F, F)
        Xte, yte, _ = design(feats_for(F, [F]), aug=False)
        acc_te = float((predict_sym(m_fin, Xte, wn).argmax(1) == yte).mean())
        res = {"best_it": best_it, "curve": curve, "val_step_acc": acc, "test_step_acc": acc_te,
               "inner_seconds": round(t_inner, 1), "final_seconds": round(time.time() - t0, 1), "V": V}
        js.write_text(json.dumps(res, indent=1))
        log(f"A fold {F} done: {res}")


# ---- stage: decode --------------------------------------------------------------------------------------------------
class Half:
    def __init__(self, F, m, key, seed):
        self.s = load_sim(key, seed)
        z = np.load(hpath(src_of(F, m), key, seed))
        self.ev = {k: z[f"ev_{k}"] for k in STREAMS}
        self.strong = (z["sH"], z["sA"], z["touches"])
        self.p_dead = z["p_dead"]
        self.seq = {}
        for model in ("A", "B"):
            p = np.load(ppath(model, F, key, seed))
            self.seq[model] = fx.upsample(p["logp"], p["starts"], self.s.match.n_frames, self.s.match.half)


def decode_labels(h, ev, wts, cfg):
    dcfg = copy.deepcopy(cfg.decoder); ev_dec = dict(ev)
    for model, w in wts.items():
        k = f"seq{model}"; ev_dec[k] = h.seq[model]; dcfg.weights[k] = float(w); dcfg.lengths[k] = 1.0
    lab = decode(ev_dec, h.s.match.half, dcfg)
    return postprocess(lab, h.s.match.half, h.s.match.fps, cfg.post, strong_H=h.strong[0], strong_A=h.strong[1],
                       touches=h.strong[2])


def train_merge_for(halves, wts, base):
    """select.merge_training_table on cached inner-split evidence: the 3 GRID8 reference cells (merge off), the
    decoder plus the learned stream(s) at `wts`, spell table + merge labels vs ctrl_Dfull, MergeModel fit."""
    refs = sel.merge_ref_cells(sel.GRID8)
    tabs = []; t0 = time.time()
    for h in halves:
        for ci, c in enumerate(refs):
            cfg = sel.apply_cell(base, c); cfg.decoder.lengths = sel.estimate_lengths([], cfg, None)
            ev = dict(h.ev); ev["dead"] = dead_evidence(h.p_dead, logit_offset=cfg.dead_offset)
            lab = decode_labels(h, ev, wts, cfg)
            tab = match_spell_table(lab, h.s.match, ev, h.strong)
            tab["y"] = merge_labels(h.s.truth[sel.MERGE_TRUTH], lab, tab)
            tab["key"] = h.s.key; tab["seed"] = h.s.seed; tab["ref_cell"] = ci
            tabs.append(tab[np.isfinite(tab["y"].to_numpy(float))])
    df = pd.concat(tabs, ignore_index=True)
    mm = MergeModel().fit(df, df["y"].to_numpy(float))
    mm.info.update(n_ref_cells=len(refs), n_halves=len(halves), held_match=int(match_id(halves[0].s.key)),
                   total_seconds=round(time.time() - t0, 1))
    return mm


def score_rows(h, lab, tags):
    out = []
    for v in VARS:
        r = score(h.s.truth[v], lab)
        out.append({**tags, "key": h.s.key, "seed": h.s.seed, "variant": v, "spell_f1": r["spell_f1"],
                    "macro_f1": r["macro_f1"], "n_gt": r["n_gt_spells"], "n_pred": r["n_pred_spells"]})
    return out


def stage_decode(folds):
    base = base_cfg(); cfg = a5c_cfg(base)
    (L / "merge").mkdir(exist_ok=True)
    for F in folds:
        out = L / f"decode_f{F}.csv"
        if out.exists():
            log(f"decode fold {F}: done"); continue
        train, V, _ = split(F)
        t0 = time.time()
        val = [Half(F, V, k, s) for k, s in halves_of(V)]
        test = [Half(F, F, k, s) for k, s in halves_of(F)]
        rows, minfo = [], {}
        for d, wts in DECODERS.items():
            mp = L / "merge" / f"f{F}_{d}.joblib"
            if mp.exists():
                mm = MergeModel.load(mp)
            else:
                mm = train_merge_for(val, wts, base); mm.save(mp)
            minfo[d] = mm.info
            for h in test:
                lab = decode_labels(h, h.ev, wts, cfg)
                rows += score_rows(h, lab, {"fold": F, "decoder": d, "merge": "off"})
                lab_m = apply_merge_model(lab, h.s.match, h.ev, h.strong, mm, THETA)
                rows += score_rows(h, lab_m, {"fold": F, "decoder": d, "merge": "on"})
            dd = pd.DataFrame(rows); dd = dd[(dd.decoder == d) & (dd.variant == "ctrl_Dfull")]
            log(f"f{F} {d}: ctrl_Dfull off {dd[dd['merge'] == 'off'].spell_f1.mean():.4f} "
                f"on {dd[dd['merge'] == 'on'].spell_f1.mean():.4f}  merge {mm.info}")
        (L / f"merge_info_f{F}.json").write_text(json.dumps(minfo, indent=1))
        tmp = out.with_suffix(".tmp.csv"); pd.DataFrame(rows).to_csv(tmp, index=False); os.replace(tmp, out)
        log(f"decode fold {F} done {time.time() - t0:.0f}s")


# ---- stage: summary -------------------------------------------------------------------------------------------------
def fold_table():
    d = pd.concat([pd.read_csv(p) for p in sorted(L.glob("decode_f*.csv"))], ignore_index=True)
    d["name"] = d["decoder"] + "/" + d["merge"]
    sf = d.pivot_table(index=["fold", "name"], columns="variant", values="spell_f1", aggfunc="mean")
    mc = d[d.variant == "ctrl_Dfull"].groupby(["fold", "name"])[["macro_f1", "n_pred", "n_gt"]].mean()
    mr = d[d.variant == "raw_Dfull"].groupby(["fold", "name"])[["macro_f1"]].mean().rename(columns={"macro_f1": "macro_f1_raw"})
    return sf.join(mc).join(mr).reset_index()


def stage_summary():
    ft = fold_table()
    folds = sorted(ft.fold.unique())
    mean = ft.groupby("name")[list(VARS) + ["macro_f1", "macro_f1_raw", "n_pred", "n_gt"]].mean()
    ref = ft[ft.name == "a5c/on"].set_index("fold")["ctrl_Dfull"]
    diff = ft.assign(delta=ft.ctrl_Dfull - ft.fold.map(ref))
    mean["wins"] = diff.groupby("name")["delta"].apply(lambda x: int((x >= 0.02).sum()))
    mean["losses"] = diff.groupby("name")["delta"].apply(lambda x: int((x <= -0.02).sum()))
    mean["min_delta"] = diff.groupby("name")["delta"].min()
    t = sel.mean_rank(mean.reset_index(), list(VARS))
    t.to_csv(L / "summary.csv", index=False)
    per_fold = ft.pivot_table(index="name", columns="fold", values="ctrl_Dfull")
    per_fold.to_csv(L / "summary_per_fold_ctrl_Dfull.csv")
    pd.set_option("display.width", 250)
    log(f"folds {folds}\n" + t.round(4).to_string(index=False) + "\n\nper-fold ctrl_Dfull\n" + per_fold.round(4).to_string())
    (L / "summary.json").write_text(json.dumps({"folds": [int(f) for f in folds], "best": t.iloc[0]["name"],
                                                "table": t.to_dict(orient="records")}, indent=1))


# ---- stage: final ---------------------------------------------------------------------------------------------------
def final_feat_paths():
    """All six matches, each with out-of-sample evidence from models trained on the other five (fold{m} caches)."""
    return [fpath(src_fold(m), k, s) for m in MATCHES for k, s in halves_of(m)]


def stage_finalA():
    from sklearn.ensemble import HistGradientBoostingClassifier
    FINAL.mkdir(parents=True, exist_ok=True)
    out = FINAL / "modelA.joblib"
    if out.exists():
        log("finalA: exists"); return
    its = [json.loads((L / f"modelA_f{F}.json").read_text())["best_it"] for F in MATCHES]
    it = int(round(float(np.median(its)) / 50) * 50)   # median fold choice on the 50-iteration grid
    t0 = time.time()
    Xtr, ytr, wn = design(final_feat_paths(), stride=TRAIN_STRIDE)
    m = HistGradientBoostingClassifier(max_iter=it, early_stopping=False, random_state=0, **HP_A).fit(Xtr, ytr)
    joblib.dump(m, out)
    (FINAL / "modelA.json").write_text(json.dumps({"max_iter": it, "fold_best_its": its, "hp": HP_A,
                                                   "train_stride": TRAIN_STRIDE, "rows": int(len(ytr)),
                                                   "n_features": len(wn), "seconds": round(time.time() - t0, 1)}, indent=1))
    log(f"finalA: iters {it} (folds {its}) {time.time() - t0:.0f}s")


def stage_finalize(variant):
    """Final merge model (inner split of all six: stream models + learned models on 9001-9005 = fold 9006's, decoded
    on 9006's halves, i.e. fold 9006's test caches/predictions) and the JSON config."""
    d, mg = variant.split("/")
    wts = DECODERS[d]
    base = base_cfg()
    if mg == "on":
        hs = [Half(9006, 9006, k, s) for k, s in halves_of(9006)]
        mm = train_merge_for(hs, wts, base); mm.save(FINAL / "merge_model.joblib")
        log(f"final merge model {mm.info}")
    feat_names = [str(s) for s in np.load(final_feat_paths()[0])["names"]]
    summ = json.loads((L / "summary.json").read_text())
    row = next(r for r in summ["table"] if r["name"] == variant)
    cfg = {"variant": variant, "decoder": d, "stream_weights": {f"seq{k}": v for k, v in wts.items()},
           "stream_lengths": {f"seq{k}": 1.0 for k in wts}, "merge": mg == "on", "merge_theta": THETA if mg == "on" else 1.0,
           "merge_model": "merge_model.joblib" if mg == "on" else None,
           "base_selection": "runs/select/a5c",
           "models": {k: {"A": "modelA.joblib", "B": "modelB.pt"}[k] for k in wts},
           "feature_spec": {"module": "scripts/exp_seq_feats.py", "step_frames": fx.STEP, "streams": list(fx.STREAMS),
                            "step_features": feat_names, "windows_steps": [10, 25, 50],
                            "A_input": "windowed(X): [X, past mean W, future mean W] for W in 10 25 50 within each half",
                            "B_input": "standardised X (modelB.pt mu/sd), 600-step windows stride 300, centre kept",
                            "team_symmetrised": True, "logp_clip": 1e-4,
                            "upsample": "each frame takes its 5-frame step's value (exp_seq_feats.upsample)"},
           "training": {"matches": MATCHES, "seeds": SEEDS, "evidence": "per match: a5c streams from dead/team models "
                        "trained on the other five matches (out-of-sample, as fold test caches)"},
           "lomo_6fold": {k: row[k] for k in list(VARS) + ["macro_f1", "n_pred", "wins", "losses", "mean_rank"]}}
    (FINAL / "config.json").write_text(json.dumps(cfg, indent=1))
    log(f"final config written: {variant}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("stage"); ap.add_argument("--folds", type=int, nargs="*", default=MATCHES)
    ap.add_argument("--variant")
    a = ap.parse_args()
    assert SEALED_MATCH not in a.folds
    {"seed": lambda: stage_seed(), "evidence": lambda: stage_evidence(a.folds), "feat": stage_feat,
     "A": lambda: stage_A(a.folds), "decode": lambda: stage_decode(a.folds), "summary": stage_summary,
     "finalA": stage_finalA, "finalize": lambda: stage_finalize(a.variant)}[a.stage]()


if __name__ == "__main__":
    main()
