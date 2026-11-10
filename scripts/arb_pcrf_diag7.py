"""Per-chunk correlates of PathCRF correctness (real match, per half)."""
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
z = np.load("runs/cache/pathcrf/real_tactical.npz"); ph = z["p_home"]; imp = z["imputed_frac"]; pn = z["p_none"]
for h, a, b in half_runs(m.half):
    rows = []
    for c0 in range(0, b - a - 125, 250):
        s, e = a + c0, a + c0 + 250
        i = np.flatnonzero((conf[s:e] >= 0) & np.isfinite(ph[s:e])) + s
        if len(i) < 3: continue
        acc = float(((ph[i] > .5) == (conf[i] == 0)).mean())
        p = m.pos[s:e]; vis = np.isfinite(p[:, :, 0])
        d = np.linalg.norm(np.diff(p, axis=0), axis=-1); both = vis[1:] & vis[:-1]
        rows.append((acc, float(np.nanmean(imp[s:e])), float(np.nanmean(pn[s:e])), float(((d == 0) & both).sum() / max(both.sum(), 1)),
                     float(np.nanmean(ph[s:e])), float(m.ball_present[s:e].mean()), float(np.nanstd(np.nan_to_num(ph[s:e])))))
    r = np.array(rows); names = ["imputed_frac", "p_none", "dup", "mean p_home", "ball visible", "std p_home"]
    print(f"H{h} n={len(r)}: corr(acc, x):", {n: round(float(np.corrcoef(r[:, 0], r[:, k + 1])[0, 1]), 2) for k, n in enumerate(names)},
          "| mean imputed", round(float(r[:, 1].mean()), 3), "mean p_none", round(float(r[:, 2].mean()), 3), "mean p_home overall", round(float(np.nanmean(ph[a:b])), 3))
