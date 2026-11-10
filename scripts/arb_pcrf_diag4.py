"""Does PathCRF fail when the wrapper's per-chunk goalkeeper pick is not a real keeper? (replicates run_infer node selection)"""
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
    F = b - a; rows = []
    for c0 in range(0, F - 125, 250):
        c1 = min(F, c0 + 250); seg = pos[c0:c1]; miss = np.isnan(seg[:, :, 0]); gks = {}
        for k in (0, 1):
            sl = np.flatnonzero((team == k) & (~np.isnan(pos[:, :, 0]).all(0)))
            cnt = (~miss[:, sl]).sum(0); sel = np.sort(sl[np.argsort(-cnt, kind="stable")[:11]])
            xs = np.array([np.nanmean(seg[:, s, 0]) for s in sel])
            if np.isnan(xs).all():
                gks[k] = False; continue
            g = sel[np.nanargmin(xs)] if k == left else sel[np.nanargmax(xs)]
            gx = np.nanmean(seg[:, g, 0]); gks[k] = abs(gx) > 30 and ((gx < 0) == (k == left))
            # how many selected nodes had presence < half of the chunk
        i = np.flatnonzero((conf[a + c0:a + c1] >= 0) & np.isfinite(ph[a + c0:a + c1])) + a + c0
        if len(i) < 3: continue
        acc = float(((ph[i] > .5) == (conf[i] == 0)).mean())
        rows.append((gks[0], gks[1], acc))
    r = np.array(rows, float)
    print(f"H{h}: chunks {len(r)}; wrapper keeper plausible home/away: {r[:,0].mean():.2f}/{r[:,1].mean():.2f}")
    for nm, msk in (("both keepers ok", (r[:,0]==1)&(r[:,1]==1)), ("home keeper bad", r[:,0]==0), ("away keeper bad", r[:,1]==0)):
        if msk.sum(): print(f"   {nm}: n={int(msk.sum())} mean chunk acc {r[msk,2].mean():.2f}")
