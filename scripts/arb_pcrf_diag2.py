"""Tracking-quality aggregates per half / 5-min bin on the real match vs PathCRF accuracy (aggregates only)."""
import json, os, sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "scripts"))
import numpy as np
import arb_common as ac
from mtah.io.metrica_loader import load_tactical
from mtah.io.match import half_runs
m = load_tactical(os.environ["MTAH_DATA_DIR"]); fps = m.fps
conf = ac.confident_control(m, ac.RuleParams(**json.load(open("runs/arb/rule.json"))["rule"]))
ph = np.load("runs/cache/pathcrf/real_tactical.npz")["p_home"]
team = m.team
for h, a, b in half_runs(m.half):
    step = int(300 * fps)
    print(f"H{h}")
    for s in range(a, min(b, a + 12 * step), step):
        e = min(b, s + step); p = m.pos[s:e]
        vis = np.isfinite(p[:, :, 0])
        d = np.linalg.norm(np.diff(p, axis=0), axis=-1)
        both = vis[1:] & vis[:-1]
        dup = float(((d == 0) & both).sum() / max(both.sum(), 1))
        sp = d[both] * fps
        i = np.flatnonzero((conf[s:e] >= 0) & np.isfinite(ph[s:e])) + s
        acc = float(((ph[i] > .5) == (conf[i] == 0)).mean()) if len(i) > 20 else np.nan
        # per-frame visible per team, jump fraction (>12 m/s), slots appearing (distinct visible slots) per team
        jump = float((sp > 12).mean())
        ns = [int(vis[:, team == k].any(0).sum()) for k in (0, 1)]
        # nearest opponent distance median for visible players (team mixing proxy)
        print(f"  {(s-a)/fps/60:3.0f}m acc={acc:.2f} dup={dup:.3f} speed med/p99 {np.median(sp):.2f}/{np.percentile(sp,99):.1f} jump>12m/s {jump:.4f} vis/team {vis[:,team==0].sum(1).mean():.1f}/{vis[:,team==1].sum(1).mean():.1f} distinct slots {ns}")
