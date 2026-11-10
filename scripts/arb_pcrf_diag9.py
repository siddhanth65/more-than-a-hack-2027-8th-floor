import json, os, sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "scripts"))
import numpy as np
import arb_common as ac
from mtah.io.match import half_runs
from mtah.io.metrica_loader import load_tactical
m = load_tactical(os.environ["MTAH_DATA_DIR"])
conf = ac.confident_control(m, ac.RuleParams(**json.load(open("runs/arb/rule.json"))["rule"]))
ph = np.load("runs/cache/pathcrf/real_tactical.npz")["p_home"]
for h, a, b in half_runs(m.half):
    pos = m.pos[a:b]; step = np.linalg.norm(pos[1:] - pos[:-1], axis=-1)
    big = np.nan_to_num(step) > 0.6; print(f"H{h}: single-frame jumps > 0.6 m: {int(big.sum())} ({big.sum() / (b - a) * 25 * 60:.2f} per minute); > 3 m: {int((np.nan_to_num(step) > 3).sum())}")
    rows = []
    for c0 in range(0, b - a - 125, 250):
        i = np.flatnonzero((conf[a+c0:a+c0+250] >= 0) & np.isfinite(ph[a+c0:a+c0+250])) + a + c0
        if len(i) < 3: continue
        rows.append((((ph[i] > .5) == (conf[i] == 0)).mean(), big[c0:c0+250].sum()))
    r = np.array(rows); k = r[:, 1] > 0
    print(f"   chunks with a jump: n={int(k.sum())} acc {r[k,0].mean():.2f}; without: n={int((~k).sum())} acc {r[~k,0].mean():.2f}")
