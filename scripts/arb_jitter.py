"""Position jitter per half: median/p90 of |second difference| (m) of visible players, and acceleration p99 (m/s^2)."""
import os, sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
from mtah.io.match import half_runs

def jitter(pos, fps=25.0):
    d2 = pos[2:] - 2 * pos[1:-1] + pos[:-2]
    n = np.linalg.norm(d2, axis=-1); n = n[np.isfinite(n)]
    acc = n * fps * fps
    return float(np.median(n)), float(np.percentile(n, 90)), float(np.percentile(acc, 99))

if __name__ == "__main__":
    from mtah.io.metrica_loader import load_tactical
    m = load_tactical(os.environ["MTAH_DATA_DIR"])
    for h, a, b in half_runs(m.half):
        print("real H%d: median %.4f p90 %.4f m, accel p99 %.1f m/s2" % ((h,) + jitter(m.pos[a:b])))
    from mtah.sim.dataset import load_half_sim
    for mt in (9001, 9003, 9005):
        for h in ("H1", "H2"):
            s = load_half_sim(ROOT / "runs/sim", f"game_{mt}_{h}", 100)
            print("sim %d %s: median %.4f p90 %.4f m, accel p99 %.1f m/s2" % ((mt, h) + jitter(s.match.pos)))
