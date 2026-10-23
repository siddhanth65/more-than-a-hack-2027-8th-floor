"""Step 1 (label-free, aggregates only): why `detect_touches` (mode abs) fires ~900x per sim half but ~15x on the real match.
Per-frame ball speed, |d speed|, |d direction|, the thresholded statistic dv (k=3 velocity change, m/s^2), the nearest
player distance, raw peak counts before/after each gate; real vs sims 9001-9006 seed 100.   -> runs/touch/diagnosis.json"""
import json, os, sys
os.environ.setdefault("OMP_NUM_THREADS", "4")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
from touch_common import OUT, load_real, sim_halves
from mtah.evidence.touches import TouchParams, velocity_change, detect_touches, _jump_frames


def q(x, ps=(50, 90, 99)):
    x = np.asarray(x, float); x = x[np.isfinite(x)]
    return [round(float(v), 4) for v in np.percentile(x, ps)] if len(x) else None


def per_half(m, lo=None, hi=None):
    """Arrays over graded visible frames of ONE contiguous span (a sim half or the real match)."""
    fps = m.fps; b = m.ball.astype(np.float64); p = TouchParams()
    vis = ~np.isnan(b[:, 0]) & m.graded
    v = np.diff(b, axis=0) * fps                        # (F-1, 2) m/s
    ok1 = vis[1:] & vis[:-1]
    sp = np.hypot(v[:, 0], v[:, 1])
    ok2 = ok1[1:] & ok1[:-1]
    dsp = np.abs(np.diff(sp))                           # m/s per frame
    ang = np.arctan2(v[:, 1], v[:, 0]); da = np.degrees(np.abs((np.diff(ang) + np.pi) % (2 * np.pi) - np.pi))
    mv = ok2 & (sp[1:] > 2) & (sp[:-1] > 2)
    dv = velocity_change(b, p.k, fps)
    dvv = dv[vis & np.isfinite(dv)]
    # nearest player (any team) over visible frames, sampled
    idx = np.flatnonzero(vis)[::5]
    d = np.hypot(m.pos[idx, :, 0] - b[idx, None, 0], m.pos[idx, :, 1] - b[idx, None, 1])
    dn = np.nanmin(np.where(np.isfinite(d), d, np.inf), 1)
    # peak stages
    w = np.nan_to_num(dv, nan=-1.0); k = p.k
    padded = np.r_[np.full(k, -1.0), w, np.full(k, -1.0)]
    win = np.lib.stride_tricks.sliding_window_view(padded, 2 * k + 1)
    peak = (w >= p.theta_dv) & (w >= win.max(1))
    peak &= ~np.r_[False, peak[:-1] & (w[1:] == w[:-1])]
    jumps = _jump_frames(np.nan_to_num(b, nan=1e6))
    jp = np.r_[np.zeros(k, bool), jumps, np.zeros(k, bool)]
    nj = np.lib.stride_tricks.sliding_window_view(jp, 2 * k + 1).any(1)
    peak_nj = peak & ~nj
    fr = np.flatnonzero(peak_nj)
    dpk = np.hypot(m.pos[fr, :, 0] - b[fr, None, 0], m.pos[fr, :, 1] - b[fr, None, 1]) if len(fr) else np.zeros((0, m.n_slots))
    dpk = np.nanmin(np.where(np.isfinite(dpk), dpk, np.inf), 1) if len(fr) else np.zeros(0)
    t = detect_touches(m, p)
    return {"vis_frames": int(vis.sum()), "vis_min": float(vis.sum() / fps / 60),
            "zero_disp_share": float((sp[ok1] == 0).mean()), "speed": q(sp[ok1]), "dspeed_per_frame": q(dsp[ok2]),
            "ddir_deg_moving": q(da[mv]), "dv_k3": q(dvv, (50, 90, 99, 99.9)), "dv_ge_5_share": float((dvv >= 5).mean()),
            "dv_ge_6_share": float((dvv >= 6).mean()), "dv_mad_sigma": float(1.4826 * np.median(np.abs(dvv - np.median(dvv)))),
            "nearest_player_m": q(dn, (5, 50, 90)), "peaks_dv5": int(peak.sum()), "peaks_after_jump_gate": int(peak_nj.sum()),
            "peaks_nearest_le_1p5m": int((dpk <= 1.5).sum()), "peak_nearest_player_m": q(dpk, (10, 50, 90)) if len(dpk) else None,
            "touches_abs": int(len(t)), "touches_abs_per_vis_min": float(len(t) / max(1e-9, vis.sum() / fps / 60))}


def main():
    res = {"real": per_half(load_real())}
    sims = {}
    for key, s in sim_halves():
        sims[key] = per_half(s.match)
    res["sim_halves"] = sims
    keys = ["zero_disp_share", "touches_abs_per_vis_min", "dv_ge_5_share", "dv_mad_sigma", "peaks_dv5", "touches_abs"]
    res["sim_mean"] = {k: float(np.mean([s[k] for s in sims.values()])) for k in keys}
    for k in ("speed", "dspeed_per_frame", "ddir_deg_moving", "dv_k3", "nearest_player_m"):
        res.setdefault("sim_median_of_halves", {})[k] = [round(float(np.median([s[k][i] for s in sims.values()])), 4)
                                                          for i in range(len(res["real"][k]))]
    (OUT / "diagnosis.json").write_text(json.dumps(res, indent=1))
    r = res["real"]
    print("REAL", json.dumps(r)); print("SIM mean", res["sim_mean"]); print("SIM med", res["sim_median_of_halves"])


if __name__ == "__main__":
    main()
