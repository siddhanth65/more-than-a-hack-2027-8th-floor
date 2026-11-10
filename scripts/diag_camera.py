"""Diagnostic 1: camera motion (from homography files only) as a cue. Label-free kill test.
Never reads contest ground truth; aggregates only. See docs/diag/REPORT.md.
Usage: OMP_NUM_THREADS=2 python -I scripts/diag_camera.py [--stage all|tactical|broadcast]"""
import argparse, json, os, sys, time
from pathlib import Path

for k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(k, "2")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import Ridge
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler

from mtah.io.metrica_loader import load_tactical

D = Path(os.environ.get("MTAH_DATA_DIR", os.path.join(os.path.expanduser("~"), "mtah-data")))
OUT = ROOT / "runs" / "diag"
CAND = ROOT / "runs" / "candidates"
FPS = 25
L, W = 105.3, 68.0
OFFSETS = {1: 4014, 2: 26920}
STEP = 5          # frame subsample for model fits
CANDS = ("a5c", "a14")
CAMF = ("cx", "cy", "logw", "pan", "zrate", "pan4", "zrate4")


# ---------------------------------------------------------------- camera features
def footprint(csvpath):
    """Per-frame pitch footprint of the image from the homography (pitch 0-1 -> pixels).
    centre = pitch point under image centre; width = pitch length of the mid-row image span (10%..90% of width);
    depth = pitch length of the vertical span (30%..70% of height). Metres, centred pitch coordinates."""
    h = pd.read_csv(csvpath)
    n = int(h.frame.max()) + 1
    H = h[[f"h{i}{j}" for i in range(3) for j in range(3)]].to_numpy().reshape(-1, 3, 3)
    valid = (h.valid.to_numpy() == 1)
    pts = np.array([[960, 540, 1], [192, 540, 1], [1728, 540, 1], [960, 324, 1], [960, 756, 1]], float).T
    out = np.full((n, 4), np.nan)   # cx, cy, width, depth
    idx = h.frame.to_numpy()
    with np.errstate(all="ignore"):
        for s in range(0, len(h), 20000):
            sl = slice(s, s + 20000)
            try:
                Hi = np.linalg.inv(H[sl])
            except np.linalg.LinAlgError:
                Hi = np.stack([np.linalg.pinv(m) for m in H[sl]])
            P = Hi @ pts
            P = P[:, :2, :] / P[:, 2:3, :]
            X = (P[:, 0, :] - 0.5) * L; Y = (P[:, 1, :] - 0.5) * W
            wid = np.hypot(X[:, 2] - X[:, 1], Y[:, 2] - Y[:, 1])
            dep = np.hypot(X[:, 4] - X[:, 3], Y[:, 4] - Y[:, 3])
            o = np.stack([X[:, 0], Y[:, 0], wid, dep], 1)
            bad = (~valid[sl]) | ~np.isfinite(o).all(1) | (np.abs(o[:, 0]) > 120) | (np.abs(o[:, 1]) > 100) | (wid > 400) | (wid < 3)
            o[bad] = np.nan
            out[idx[sl]] = o
    return out


def smooth_nan(x, w):
    """centred moving average ignoring NaN (window w frames)."""
    k = np.ones(w)
    v = np.isfinite(x)
    num = np.convolve(np.where(v, x, 0.0), k, "same"); den = np.convolve(v.astype(float), k, "same")
    with np.errstate(all="ignore"):
        r = num / den
    r[den < max(2, w // 3)] = np.nan
    return r


def cam_feats(fp, w_s=1.0):
    """smoothed camera features per frame: cx, cy, logw, pan (m/s), zrate (|d log width|/s), 4 s means"""
    w = int(w_s * FPS) | 1
    cx, cy, lw = smooth_nan(fp[:, 0], w), smooth_nan(fp[:, 1], w), smooth_nan(np.log(fp[:, 2]), w)
    d = int(FPS * 0.5)

    def dd(x):
        o = np.full_like(x, np.nan); o[d:-d] = (x[2 * d:] - x[:-2 * d]) / (2 * d / FPS); return o
    pan = np.hypot(dd(cx), dd(cy)); zr = dd(lw)
    return {"cx": cx, "cy": cy, "logw": lw, "pan": pan, "zrate": np.abs(zr),
            "pan4": smooth_nan(pan, 4 * FPS | 1), "zrate4": smooth_nan(np.abs(zr), 4 * FPS | 1)}


def q(x, qs=(1, 5, 25, 50, 75, 95, 99)):
    x = x[np.isfinite(x)]
    if not len(x):
        return {"n": 0}
    return {"n": int(len(x)), "mean": float(x.mean()), "sd": float(x.std()), **{f"p{p}": float(np.percentile(x, p)) for p in qs}}


# ---------------------------------------------------------------- player summaries
def player_feats(t, idx):
    team = t.team
    P = t.pos[idx].astype(np.float64)      # (n,S,2)
    S = t.speed[idx].astype(np.float64)
    with np.errstate(all="ignore"):
        d = np.maximum(idx - FPS, 0)
        V = (P - t.pos[d].astype(np.float64)) / ((idx - d)[:, None, None] / FPS + 1e-9)
        allm = np.nanmean(P, 1)
        cen, spr = [], []
        for k in (0, 1):
            Pk = np.where((team == k)[None, :, None], P, np.nan)
            cen.append(np.nanmean(Pk, 1)); spr.append(np.nanstd(Pk, 1))
        left = (cen[0][:, 0] <= cen[1][:, 0])[:, None]      # side-invariant: left team first
        cl = np.where(left, cen[0], cen[1]); cr = np.where(left, cen[1], cen[0])
        sl = np.where(left, spr[0], spr[1]); sr = np.where(left, spr[1], spr[0])
        vm = np.nanmean(V, 1)
        spm = np.nanmean(S, 1); sps = np.nanstd(S, 1); lowf = np.nanmean(np.where(np.isfinite(S), S < 1.0, np.nan), 1)
        sdx, sdy = np.nanstd(P[..., 0], 1), np.nanstd(P[..., 1], 1)
    return pd.DataFrame({"all_x": allm[:, 0], "all_y": allm[:, 1], "cl_x": cl[:, 0], "cl_y": cl[:, 1], "cr_x": cr[:, 0],
                         "cr_y": cr[:, 1], "sl_x": sl[:, 0], "sl_y": sl[:, 1], "sr_x": sr[:, 0], "sr_y": sr[:, 1],
                         "sdx": sdx, "sdy": sdy, "vx": vm[:, 0], "vy": vm[:, 1], "spd_mean": spm, "spd_sd": sps,
                         "spd_low": lowf})


# ---------------------------------------------------------------- helpers
def err_stats(e):
    e = e[np.isfinite(e)]
    return {"median": float(np.median(e)), "p90": float(np.percentile(e, 90)), "n": int(len(e))}


def ball_errors(pred, ball):
    d = pred - ball
    return {"along": err_stats(np.abs(d[:, 0])), "across": err_stats(np.abs(d[:, 1])), "euclid": err_stats(np.hypot(d[:, 0], d[:, 1]))}


def ridge_fit_pred(Xtr, ytr, Xte, alpha=10.0):
    sc = StandardScaler().fit(Xtr)
    m = Ridge(alpha=alpha).fit(sc.transform(Xtr), ytr)
    return m.predict(sc.transform(Xte))


def auc(y, s):
    ok = np.isfinite(s)
    if len(np.unique(y[ok])) < 2:
        return float("nan")
    return float(roc_auc_score(y[ok], s[ok]))


def gbt(Xtr, ytr, Xte, seed=0):
    m = HistGradientBoostingClassifier(max_depth=3, max_iter=150, learning_rate=0.08, min_samples_leaf=40,
                                       l2_regularization=1.0, random_state=seed)
    m.fit(Xtr, ytr)
    return m.predict_proba(Xte)[:, 1]


def load_labels(name, n):
    s = pd.read_csv(CAND / f"{name}.csv")
    lab = np.full(n, "", object)
    lab[s.frame.to_numpy()] = s.label.to_numpy()
    return lab


# ---------------------------------------------------------------- analysis for one camera
def analyse(name, t, cf, usable, labels, blk_long, res):
    n = t.n_frames
    half = t.half
    vis = t.ball_present & t.graded
    camok = usable & np.isfinite(cf["cx"]) & np.isfinite(cf["pan"]) & np.isfinite(cf["zrate"])
    sub = np.zeros(n, bool); sub[::STEP] = True
    idx = np.flatnonzero(camok & vis & sub)
    pf = player_feats(t, idx)
    cam = pd.DataFrame({k: cf[k][idx] for k in CAMF})
    ball = t.ball[idx].astype(np.float64)
    hh = half[idx]
    r = {"n_visible_usable_sub": int(len(idx)), "n_per_half": {str(h): int((hh == h).sum()) for h in (1, 2)}}
    # ---- (a) ball-position proxy
    r["a_camera_centre_as_ball"] = {str(h): ball_errors(cam[["cx", "cy"]].to_numpy()[hh == h], ball[hh == h]) for h in (1, 2)}
    r["a_players_mean_position"] = {str(h): ball_errors(pf[["all_x", "all_y"]].to_numpy()[hh == h], ball[hh == h]) for h in (1, 2)}
    X_p = pf.to_numpy(); X_c = cam.to_numpy(); X_pc = np.hstack([X_p, X_c])
    fin = np.isfinite(X_pc).all(1) & np.isfinite(ball).all(1)
    r["a_ridge"] = {}
    for tr, te in ((1, 2), (2, 1)):
        mtr, mte = fin & (hh == tr), fin & (hh == te)
        o = {"n_train": int(mtr.sum()), "n_test": int(mte.sum())}
        pr = {}
        for nm, X in (("players", X_p), ("camera", X_c), ("players+camera", X_pc)):
            pr[nm] = ridge_fit_pred(X[mtr], ball[mtr], X[mte])
            o[nm] = ball_errors(pr[nm], ball[mte])
        b = ball[mte]
        e_p = np.hypot(*(pr["players"] - b).T); e_pc = np.hypot(*(pr["players+camera"] - b).T)
        o["reduction_median_euclid"] = float(1 - np.median(e_pc) / np.median(e_p))
        o["reduction_mean_euclid"] = float(1 - e_pc.mean() / e_p.mean())
        o["reduction_p90_euclid"] = float(1 - np.percentile(e_pc, 90) / np.percentile(e_p, 90))
        for ax, j in (("along", 0), ("across", 1)):
            a_p = np.abs(pr["players"][:, j] - b[:, j]); a_pc = np.abs(pr["players+camera"][:, j] - b[:, j])
            o[f"reduction_median_{ax}"] = float(1 - np.median(a_pc) / np.median(a_p))
        r["a_ridge"][f"train{tr}_test{te}"] = o
    # ---- (b) dead-ball proxy
    r["b"] = {}
    dcols = ["spd_mean", "spd_sd", "spd_low", "vx", "vy", "sdx", "sdy", "all_x", "all_y"]
    Xp = pf[dcols].to_numpy(); ballspd = t.ball_speed[idx].astype(float)
    Xsets = {"players": Xp, "players+camera": np.hstack([Xp, X_c]),
             "players+ballspeed": np.column_stack([Xp, ballspd]), "players+ballspeed+camera": np.column_stack([Xp, ballspd, X_c])}
    for cand in CANDS:
        lab = labels[cand][idx]
        ok = np.isin(lab, ["H", "A", "D"])
        y = (lab == "D").astype(int)
        rb = {"dead_share_visible": float(y[ok].mean()), "n": int(ok.sum()), "single_feature_auc": {}, "gbt": {}}
        for h in (1, 2):
            mm = ok & (hh == h)
            rb["single_feature_auc"][str(h)] = {f: auc(y[mm], cam[f].to_numpy()[mm]) for f in ("pan", "pan4", "logw", "zrate", "zrate4")}
        for tr, te in ((1, 2), (2, 1)):
            mtr, mte = ok & (hh == tr), ok & (hh == te)
            o = {nm: auc(y[mte], gbt(X[mtr], y[mtr], X[mte])) for nm, X in Xsets.items()}
            o["camera_only"] = auc(y[mte], gbt(X_c[mtr], y[mtr], X_c[mte]))
            o["increment_camera_over_players"] = o["players+camera"] - o["players"]
            o["increment_camera_over_players+ballspeed"] = o["players+ballspeed+camera"] - o["players+ballspeed"]
            rb["gbt"][f"train{tr}_test{te}"] = o
        r["b"][cand] = rb
    # ---- (c) blackouts
    nb = camok & t.graded & ~t.ball_present
    vv = camok & vis
    featn = ("pan", "pan4", "logw", "zrate", "zrate4")

    def dist(mask):
        mask = mask & sub
        return {f: q(cf[f][mask], (10, 25, 50, 75, 90)) for f in featn}
    r["c"] = {"n_blackout_sub": int((nb & sub).sum()), "n_long_blackout_sub": int((nb & blk_long & sub).sum()),
              "blackout_all": dist(nb), "blackout_long(>10s)": dist(nb & blk_long)}
    for cand in CANDS:
        lab = labels[cand]
        live = np.isin(lab, ["H", "A"]); dead = lab == "D"
        c = {"visible_live": dist(vv & live), "visible_dead": dist(vv & dead),
             "blackout_called_live": dist(nb & live), "blackout_called_dead": dist(nb & dead),
             "blackout_dead_share": float(dead[nb].mean()),
             "n_blackout_called_live_sub": int((nb & live & sub).sum()), "n_blackout_called_dead_sub": int((nb & dead & sub).sum())}
        ci = np.flatnonzero(vv & sub & np.isin(lab, ["H", "A", "D"]))
        bi = np.flatnonzero(nb & sub & np.isin(lab, ["H", "A", "D"]))
        Xc = np.column_stack([cf[k][ci] for k in CAMF]); Xb = np.column_stack([cf[k][bi] for k in CAMF])
        yc = (lab[ci] == "D").astype(int); yb = (lab[bi] == "D").astype(int)
        pr = np.full(len(bi), np.nan)
        for tr in (1, 2):      # train on visible frames of one half, score blackouts of the other
            mtr = half[ci] == tr; mte = half[bi] == 3 - tr
            if mte.any() and mtr.any():
                pr[mte] = gbt(Xc[mtr], yc[mtr], Xb[mte])
        okp = np.isfinite(pr)
        c["camera_dead_posterior_in_blackouts"] = {
            "mean_p_dead_where_called_live": float(pr[okp & (yb == 0)].mean()), "mean_p_dead_where_called_dead": float(pr[okp & (yb == 1)].mean()),
            "auc_called_dead_vs_live": auc(yb[okp], pr[okp]), "base_rate_called_dead": float(yb[okp].mean())}
        r["c"][cand] = c
    # onset profile: is the camera frozen from the moment the ball track is lost? (pan around the start of long blackouts)
    from mtah.sim.blackout_sim import runs_of
    wins = [(-5, -2), (-2, 0), (0, 2), (2, 5), (5, 10)]
    prof = {f"{a}..{b}s": [] for a, b in wins}
    for s0, e0 in runs_of(blk_long):
        if e0 - s0 < 10 * FPS or s0 < 5 * FPS:
            continue
        for a, b in wins:
            v = cf["pan"][s0 + a * FPS: s0 + b * FPS]
            v = v[np.isfinite(v)]
            if len(v) > 10:
                prof[f"{a}..{b}s"].append(np.median(v))
    r["c_onset_profile_pan_median_over_blackouts"] = {k: {"n": len(v), "median": float(np.median(v)) if v else None,
                                                          "p25": float(np.percentile(v, 25)) if v else None,
                                                          "p75": float(np.percentile(v, 75)) if v else None} for k, v in prof.items()}
    res[name] = r


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--stage", default="all"); args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    T0 = time.time()
    t = load_tactical(D); n = t.n_frames
    from mtah.eval.drift import blackout_mask_from_match
    blk_all = ~t.ball_present & t.graded
    blk_long = blackout_mask_from_match(t)
    labels = {c: load_labels(c, n) for c in CANDS}
    jp = OUT / "diag_camera.json"
    res = json.loads(jp.read_text()) if jp.exists() else {}
    res["meta"] = {"step": STEP, "ridge_alpha": 10.0, "pitch_m": [L, W], "cand_labels": list(CANDS)}
    g = t.graded

    if args.stage in ("all", "tactical"):
        fpt = footprint(D / "tactical" / "BAR-ATH_tactical_homography.csv")
        cft = cam_feats(fpt)
        mv = {f: q(cft[f][g]) for f in ("cx", "cy", "logw", "pan", "zrate")}
        mv["width_m"] = q(fpt[g, 2]); mv["depth_m"] = q(fpt[g, 3])
        static = bool(max(mv["cx"]["sd"], mv["cy"]["sd"]) < 1.0 and mv["logw"]["sd"] < 0.02)
        res["tactical_motion"] = {"graded_valid_share": float(np.isfinite(fpt[g, 0]).mean()), "stats": mv, "static": static}
        print("tactical motion done", f"{time.time()-T0:.0f}s", "static=", static, flush=True)
        jp.write_text(json.dumps(res, indent=1))
        if not static:
            analyse("tactical", t, cft, np.isfinite(fpt[:, 0]), labels, blk_long, res)
            jp.write_text(json.dumps(res, indent=1))
            print("tactical analysis done", f"{time.time()-T0:.0f}s", flush=True)
        del fpt, cft

    if args.stage in ("all", "broadcast"):
        fpb = footprint(D / "broadcast" / "BAR-ATH_broadcast_homography.csv")
        cfb = cam_feats(fpb)
        cue = np.load(ROOT / "runs" / "broadcast" / "cue.npz", allow_pickle=True)
        hb, rl, insync = cue["has_broadcast"], cue["replay_like_s"], cue["in_sync"]
        Fb = len(fpb)
        half_t = np.where(t.half > 0, t.half, np.where(np.arange(n) < 100000, 1, 2))
        bf = np.arange(n) - np.where(half_t == 1, OFFSETS[1], OFFSETS[2])
        ok = hb & (bf >= 0) & (bf < Fb)
        bfc = np.clip(bf, 0, Fb - 1)
        cfm = {k: np.where(ok, v[bfc], np.nan) for k, v in cfb.items()}
        usable = ok & insync & ~rl & np.isfinite(cfm["cx"])
        res["broadcast_mapping"] = {
            "graded_share_mapped": float(ok[g].mean()), "graded_share_usable(valid,in_sync,not_replay)": float(usable[g].mean()),
            "visible_ball_share_usable": float(usable[g & t.ball_present].mean()), "blackout_share_usable": float(usable[blk_all].mean()),
            "motion": {f: q(cfm[f][usable & g]) for f in ("cx", "cy", "logw", "pan", "zrate")}}
        analyse("broadcast", t, cfm, usable, labels, blk_long, res)
        jp.write_text(json.dumps(res, indent=1))
        print("broadcast done", f"{time.time()-T0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
