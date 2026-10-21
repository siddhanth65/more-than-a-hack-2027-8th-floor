"""Real-vs-sim shift of the long-context dead features (Plan 2e T2 step 1). Aggregates only.

For every LONG_FEATURES column on NON-blackout graded frames: p10/p50/p90 on the real match vs the sims (9001-9006,
seed 100; 9007 = sealed, never loaded). Weak labels: real = the 2 s before SET PIECES tag starts (dead-ish) and the
central 50 % of BUILD UP / PROGRESSION tags (alive-ish); sim = truth ctrl_Dfull D / H-A on ball-visible frames.
Also the raw per-player speed distribution on ball-visible frames and the share of player-frames < 1 m/s, and the
same quantities after a centred 1 s box smoothing of each player's speed (the long_rel candidate), incl. |accel|.

"kind": LEVEL = weak-dead and alive medians move the same way by > 0.5 sim IQR (calibration shift);
SEP = the real dead-vs-alive AUC is > 0.1 closer to 0.5 than the sim one (separation lost).

Writes runs/select/a5/long_shift.json and prints a compact table.
    python scripts/diagnose_long_shift.py
"""
import json
import os
import sys
from pathlib import Path

import numpy as np

os.environ.setdefault("OMP_NUM_THREADS", "4")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from mtah.eval.drift import blackout_mask_from_match  # noqa: E402
from mtah.evidence.dead import LONG_FEATURES, LONG_SLOW_MPS, LONG_WINDOWS_S, _centred_mean, long_features  # noqa: E402
from mtah.io.match import half_runs  # noqa: E402
from mtah.io.metrica_loader import load_tactical  # noqa: E402
from mtah.io.tags import load_tags  # noqa: E402
from mtah.sim.dataset import load_half_sim  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "runs" / "select" / "a5" / "long_shift.json"
SIM_DIR = ROOT / "runs" / "sim"
SIM_KEYS = [f"game_{m}_H{h}" for m in range(9001, 9007) for h in (1, 2)]   # 9007 = sealed: never loaded
SEED = 100
PHASE_CODES = ("BUILD UP", "PROGRESSION")
SUB = 2          # keep every 2nd frame of the large pools (memory)
SM_NAMES = tuple(f"sm_{n}_{w}s" for w in LONG_WINDOWS_S for n in ("speed", "slow_share", "accel"))


def pct(x):
    x = np.asarray(x, np.float64); x = x[np.isfinite(x)]
    if not len(x):
        return None
    return [round(float(v), 4) for v in np.percentile(x, [10, 50, 90])]


def auc(pos, neg):
    """P(pos > neg) + 0.5 P(tie) (Mann-Whitney with average ranks), NaN-aware."""
    pos = pos[np.isfinite(pos)]; neg = neg[np.isfinite(neg)]
    if not len(pos) or not len(neg):
        return None
    allv = np.concatenate([pos, neg])
    u, inv, cnt = np.unique(allv, return_inverse=True, return_counts=True)
    start = np.concatenate([[0], np.cumsum(cnt)[:-1]])
    r = (start + (cnt + 1) / 2.0)[inv]
    U = r[:len(pos)].sum() - len(pos) * (len(pos) + 1) / 2
    return round(float(U / (len(pos) * len(neg))), 4)


def smooth_speed(speed, fps):
    """Centred 1 s box (+-0.5 s) per player, NaN-aware; NaN where the raw speed is NaN."""
    hw = int(round(0.5 * fps)); out = np.full(speed.shape, np.nan)
    for j in range(speed.shape[1]):
        out[:, j] = _centred_mean(speed[:, j], hw)
    out[~np.isfinite(speed)] = np.nan
    return out


def smoothed_cols(match):
    """(F, 9) smoothed-speed versions of lc_speed / lc_slow_share / lc_accel at each window (per half)."""
    fps = match.fps; F = match.n_frames
    X = np.full((F, len(SM_NAMES)), np.nan)
    for _, a, b in half_runs(match.half):
        on = np.isfinite(match.pos[a:b, :, 0])
        sp = smooth_speed(np.where(on, match.speed[a:b], np.nan).astype(np.float64), fps)
        ok = np.isfinite(sp); cnt = ok.sum(1); enough = on.sum(1) >= 6; has = enough & (cnt > 0)
        s0 = np.where(ok, sp, 0.0)
        q = {k: np.full(b - a, np.nan) for k in ("speed", "slow_share", "accel")}
        q["speed"][has] = s0.sum(1)[has] / cnt[has]
        q["slow_share"][has] = (ok & (s0 < LONG_SLOW_MPS)).sum(1)[has] / cnt[has]
        okd = ok[1:] & ok[:-1]; cd = okd.sum(1)
        dsum = np.where(okd, np.abs(s0[1:] - s0[:-1]) * fps, 0.0).sum(1)
        g = (cd > 0) & enough[1:]; acc = q["accel"][1:]; acc[g] = dsum[g] / cd[g]
        i = 0
        for w in LONG_WINDOWS_S:
            for n in ("speed", "slow_share", "accel"):
                X[a:b, i] = _centred_mean(q[n], int(round(w * fps))); i += 1
    return X


def real_weak(tags, F, fps):
    pre = np.zeros(F, bool); phase = np.zeros(F, bool); sp = np.zeros(F, bool)
    for r in tags.itertuples(index=False):
        a = max(0, int(r.start)); b = min(F - 1, int(r.end))
        if b < a:
            continue
        if r.code == "SET PIECES":
            sp[a:b + 1] = True; pre[max(0, a - int(2 * fps)):a] = True
        elif r.code in PHASE_CODES:
            q = (b - a) // 4; phase[a + q:b - q + 1] = True
    pre &= ~sp; phase &= ~sp & ~pre
    return pre, phase


def player_speeds(match, sel, smoothed=False):
    sp = match.speed.astype(np.float64)
    on = np.isfinite(match.pos[..., 0])
    if smoothed:
        sm = np.full(sp.shape, np.nan)
        for _, a, b in half_runs(match.half):
            sm[a:b] = smooth_speed(np.where(on[a:b], sp[a:b], np.nan), match.fps)
        sp = sm
    v = sp[sel][on[sel] & np.isfinite(sp[sel])]
    return v[::7]   # thin for memory


def collect(match, dead_w, alive_w):
    nb = match.graded & ~blackout_mask_from_match(match)
    X = np.hstack([long_features(match).astype(np.float64), smoothed_cols(match)])
    vis = nb & match.ball_present
    return {"all": X[nb][::SUB], "dead": X[dead_w & nb], "alive": X[alive_w & nb][::SUB],
            "ps": player_speeds(match, vis), "ps_sm": player_speeds(match, vis, smoothed=True)}


def fmt(v):
    return "/".join(f"{x:.2f}" for x in v) if v else "-"


def main():
    data_dir = os.environ.get("MTAH_DATA_DIR")
    if not data_dir:
        raise SystemExit("set MTAH_DATA_DIR")
    names = LONG_FEATURES + SM_NAMES
    m = load_tactical(data_dir); tags = load_tags(data_dir)
    pre, phase = real_weak(tags, m.n_frames, m.fps)
    real = collect(m, pre, phase); real_fps = m.fps
    del m
    sim = {k: [] for k in ("all", "dead", "alive", "ps", "ps_sm")}
    sim_fps = None
    for key in SIM_KEYS:
        s = load_half_sim(SIM_DIR, key, SEED)
        t = s.truth["ctrl_Dfull"]; vis = s.match.ball_present
        c = collect(s.match, (t == "D") & vis, np.isin(t, ("H", "A")) & vis)
        for k in sim:
            sim[k].append(c[k])
        sim_fps = s.match.fps
        del s, c
    sim = {k: np.concatenate(v) for k, v in sim.items()}
    out = {"real_fps": real_fps, "sim_fps": sim_fps, "sim_keys": SIM_KEYS, "seed": SEED,
           "weak_labels": {"real_dead": "2 s before SET PIECES starts (outside SP tags)",
                           "real_alive": "central 50 % of BUILD UP / PROGRESSION tags",
                           "sim_dead": "truth ctrl_Dfull D, ball visible", "sim_alive": "truth H/A, ball visible"},
           "n": {d: {k: int(len(v)) for k, v in src.items()} for d, src in (("real", real), ("sim", sim))},
           "player_speed": {}, "features": {}}
    for k, lab in (("ps", "raw"), ("ps_sm", "smoothed_1s")):
        out["player_speed"][lab] = {d: {"p10_p50_p90": pct(src[k]), "share_lt_1mps": round(float((src[k] < 1).mean()), 4)}
                                    for d, src in (("real", real), ("sim", sim))}
        print(f"player speed {lab:12s} real {out['player_speed'][lab]['real']}  sim {out['player_speed'][lab]['sim']}")
    print(f"{'feature':22s} {'real all p10/50/90':>20s} {'sim all p10/50/90':>20s} {'rD50':>6s} {'sD50':>6s} "
          f"{'rA50':>6s} {'sA50':>6s} {'shD':>5s} {'shA':>5s} {'aucR':>5s} {'aucS':>5s} kind")
    for i, n in enumerate(names):
        r = {k: real[k][:, i] for k in ("all", "dead", "alive")}
        s = {k: sim[k][:, i] for k in ("all", "dead", "alive")}
        med = {d: {k: float(np.nanmedian(v)) if np.isfinite(v).any() else np.nan for k, v in src.items()}
               for d, src in (("r", r), ("s", s))}
        sq = np.nanpercentile(s["all"], [25, 75]); iqr = max(float(sq[1] - sq[0]), 1e-9)
        sh = {k: (med["r"][k] - med["s"][k]) / iqr for k in ("all", "dead", "alive")}
        ar, as_ = auc(r["dead"], r["alive"]), auc(s["dead"], s["alive"])
        level = bool(np.sign(sh["dead"]) == np.sign(sh["alive"]) and min(abs(sh["dead"]), abs(sh["alive"])) > 0.5)
        sep = ar is not None and as_ is not None and abs(ar - 0.5) < abs(as_ - 0.5) - 0.1
        kind = "+".join(k for k, f in (("LEVEL", level), ("SEP", sep)) if f) or "-"
        out["features"][n] = {"real": {k: pct(v) for k, v in r.items()}, "sim": {k: pct(v) for k, v in s.items()},
                              "shift_median_in_sim_IQR": {k: round(v, 3) for k, v in sh.items()},
                              "auc_dead_vs_alive": {"real": ar, "sim": as_}, "kind": kind}
        print(f"{n:22s} {fmt(pct(r['all'])):>20s} {fmt(pct(s['all'])):>20s} {med['r']['dead']:6.2f} {med['s']['dead']:6.2f} "
              f"{med['r']['alive']:6.2f} {med['s']['alive']:6.2f} {sh['dead']:5.2f} {sh['alive']:5.2f} "
              f"{ar if ar is not None else float('nan'):5.3f} {as_ if as_ is not None else float('nan'):5.3f} {kind}")
    OUT.write_text(json.dumps(out, indent=1))
    print(f"wrote {OUT.relative_to(ROOT)}  n={out['n']}")


if __name__ == "__main__":
    main()
