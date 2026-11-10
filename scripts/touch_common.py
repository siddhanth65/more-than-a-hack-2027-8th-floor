"""Touch-detector work (fix-touch): shared paths and loaders. Aggregates only; 9007 (sealed) and the public ground
truth are never loaded. Reads the main tree's runs/ (read-only), writes under this tree's runs/touch/."""
import os
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
MAIN = Path(__file__).resolve().parents[1]
SIM = MAIN / "runs" / "sim"
OUT = ROOT / "runs" / "touch"
DATA = Path(os.environ.get("MTAH_DATA_DIR", os.path.join(os.path.expanduser("~"), "mtah-data")))
MATCHES = [9001, 9002, 9003, 9004, 9005, 9006]
SEED = 100


def load_real():
    """Real tactical Match (cached as an npz under runs/touch/ after the first slow CSV parse)."""
    from mtah.io.match import Match
    from mtah import N_FRAMES  # noqa: F401
    OUT.mkdir(parents=True, exist_ok=True)
    p = OUT / "real_match.npz"
    if p.exists():
        z = np.load(p)
        return Match(fps=float(z["fps"]), pos=z["pos"], speed=z["speed"], team=z["team"], ball=z["ball"],
                     ball_speed=z["ball_speed"], half=z["half"], pitch=tuple(z["pitch"]))
    from mtah.io.metrica_loader import load_tactical
    m = load_tactical(DATA)
    np.savez(p, fps=m.fps, pos=m.pos, speed=m.speed, team=m.team, ball=m.ball, ball_speed=m.ball_speed, half=m.half,
             pitch=np.asarray(m.pitch))
    return m


def sim_halves(matches=MATCHES, seed=SEED):
    from mtah.sim.dataset import SEALED_MATCH, load_half_sim
    for k in matches:
        assert k != SEALED_MATCH
        for h in ("H1", "H2"):
            yield f"game_{k}_{h}", load_half_sim(SIM, f"game_{k}_{h}", seed)
