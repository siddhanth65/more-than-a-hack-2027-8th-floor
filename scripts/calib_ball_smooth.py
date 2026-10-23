"""Label-free calibration of CorruptParams.ball_smooth_win: smooth the (unsmoothed-build) sim ball of 9001-9006 seed 100 with
Savitzky-Golay windows {5..13} and compare with the real tactical feed on the robust sigma of the velocity change |dv|
(k=3 and k=8, 1.4826 MAD over graded visible frames) and the ball-speed p50 / p90 on visible frames. Aggregates only; the
sealed match and the ground truth are never loaded.   -> runs/touch/ball_smooth_calibration.json"""
import json, os, sys
os.environ.setdefault("OMP_NUM_THREADS", "4")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
from touch_common import OUT, load_real, sim_halves
from mtah.evidence.touches import velocity_change
from mtah.io.match import speeds_from_positions
from mtah.sim.corrupt import CorruptParams, smooth_ball

WINS = (5, 7, 9, 11, 13)
EXTRA = (17, 25)      # beyond the grid: informational only (never chosen)


def stats(ball, graded, fps):
    b = ball.astype(np.float64); vis = ~np.isnan(b[:, 0]) & graded
    sp = speeds_from_positions(ball, fps)
    out = {}
    for k in (3, 8):
        dv = velocity_change(b, k, fps); x = dv[vis & np.isfinite(dv)]
        out[f"sigma_k{k}"] = float(1.4826 * np.median(np.abs(x - np.median(x))))
    s = sp[vis & np.isfinite(sp)]
    out["speed_p50"] = float(np.percentile(s, 50)); out["speed_p90"] = float(np.percentile(s, 90))
    return out


def main():
    real = load_real(); R = stats(real.ball, real.graded, real.fps)
    print("real", {k: round(v, 3) for k, v in R.items()})
    halves = list(sim_halves())
    table = {}
    for tag in ("none",) + WINS + EXTRA:
        rows = []
        for _, s in halves:
            m = s.match
            b = m.ball if tag == "none" else smooth_ball(m.ball, CorruptParams(ball_smooth_mode="savgol", ball_smooth_win=tag))
            rows.append(stats(b, m.graded, m.fps))
        agg = {k: float(np.median([r[k] for r in rows])) for k in rows[0]}
        agg["score"] = float(abs(agg["sigma_k3"] / R["sigma_k3"] - 1) + abs(agg["speed_p50"] / R["speed_p50"] - 1)
                             + abs(agg["speed_p90"] / R["speed_p90"] - 1))
        table[str(tag)] = agg
        print(f"win {tag:>4}: sigma_k3 {agg['sigma_k3']:.3f} sigma_k8 {agg['sigma_k8']:.3f} p50 {agg['speed_p50']:.2f} "
              f"p90 {agg['speed_p90']:.2f} score {agg['score']:.3f}", flush=True)
    best = min(WINS, key=lambda w: table[str(w)]["score"])
    print("closest window:", best)
    (OUT / "ball_smooth_calibration.json").write_text(json.dumps({"real": R, "sim_median_over_halves": table, "chosen": best}, indent=1))


if __name__ == "__main__":
    main()
