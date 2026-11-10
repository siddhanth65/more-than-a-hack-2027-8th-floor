"""Native update structure of the real tracking: where do position kinks (|2nd difference| large) fall modulo P frames, per half."""
import os, sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
from mtah.io.match import half_runs
from mtah.io.metrica_loader import load_tactical
m = load_tactical(os.environ["MTAH_DATA_DIR"])
for h, a, b in half_runs(m.half):
    pos = m.pos[a:b]
    d2 = np.linalg.norm(pos[2:] - 2 * pos[1:-1] + pos[:-2], axis=-1)          # at frame t+1
    big = (d2 > 0.01) & np.isfinite(d2)
    frac = big.sum() / np.isfinite(d2).sum()
    print(f"H{h}: start frame index {a}; share of (player,frame) with kink>1 cm: {frac:.3f}")
    t = np.nonzero(big)[0] + 1                                                  # frame index of the kink, relative to half start
    for P in (5, 25):
        cnt = np.bincount((t % P), minlength=P)
        print(f"   kinks by (frame-in-half mod {P}):", (cnt / cnt.sum()).round(3).tolist() if P == 5 else f"max share {cnt.max()/cnt.sum():.3f} at {int(cnt.argmax())}")
    # absolute-frame phase
    ta = t + a
    cnt = np.bincount(ta % 5, minlength=5); print("   kinks by (absolute frame mod 5):", (cnt / cnt.sum()).round(3).tolist())
