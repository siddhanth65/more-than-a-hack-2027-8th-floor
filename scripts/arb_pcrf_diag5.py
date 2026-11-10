"""Chunk-level side check: which team owns the deepest-left / deepest-right player in each 10 s chunk, vs the half-level side, and PathCRF correctness."""
import json, os, sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "scripts"))
import numpy as np
import arb_common as ac
from mtah.io.match import half_runs
from mtah.io.metrica_loader import load_tactical
m = load_tactical(os.environ["MTAH_DATA_DIR"]); team = m.team
conf = ac.confident_control(m, ac.RuleParams(**json.load(open("runs/arb/rule.json"))["rule"]))
ph = np.load("runs/cache/pathcrf/real_tactical.npz")["p_home"]
for h, a, b in half_runs(m.half):
    pos = m.pos[a:b]
    left = 0 if np.nanmean(pos[:, team == 0, 0]) < np.nanmean(pos[:, team == 1, 0]) else 1
    res = []; tl = []
    for c0 in range(0, b - a - 125, 250):
        seg = pos[c0:c0 + 250, :, 0]
        with np.errstate(all="ignore"):
            mx = np.nanmean(seg, axis=0)
        if np.isfinite(mx).sum() < 10: continue
        lo, hi = int(np.nanargmin(mx)), int(np.nanargmax(mx))
        side_ok = (team[lo] == left) and (team[hi] != left)         # deepest-left player is on the half-level left team
        side_flip = (team[lo] != left) and (team[hi] == left)
        i = np.flatnonzero((conf[a + c0:a + c0 + 250] >= 0) & np.isfinite(ph[a + c0:a + c0 + 250])) + a + c0
        if len(i) < 3: continue
        acc = float(((ph[i] > .5) == (conf[i] == 0)).mean())
        res.append((1 if side_ok else (-1 if side_flip else 0), acc))
    r = np.array(res)
    print(f"H{h}: chunks {len(r)}; extremes consistent {np.mean(r[:,0]==1):.2f}, flipped {np.mean(r[:,0]==-1):.2f}, mixed {np.mean(r[:,0]==0):.2f}")
    for v, nm in ((1, "consistent"), (-1, "flipped"), (0, "mixed")):
        k = r[:, 0] == v
        if k.sum(): print(f"   {nm}: n={int(k.sum())} mean acc {r[k,1].mean():.2f}")
