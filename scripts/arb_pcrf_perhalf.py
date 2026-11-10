"""PathCRF accuracy on the pseudo-label frames, per half, for sim halves (cache) - label-free rule vs sim pseudo-labels."""
import json, sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "scripts"))
import numpy as np
import arb_common as ac
from mtah.sim.dataset import load_half_sim
rule = ac.RuleParams(**json.load(open(ROOT / "runs/arb/rule.json"))["rule"])
res = {"H1": [], "H2": []}
for mt in range(9001, 9007):
    for h in ("H1", "H2"):
        for s in (100, 101):
            k = f"game_{mt}_{h}_s{s}"
            sh = load_half_sim(ROOT / "runs/sim", f"game_{mt}_{h}", s)
            c = ac.confident_control(sh.match, rule); ph = np.load(ROOT / f"runs/cache/pathcrf/{k}.npz")["p_home"]
            i = np.flatnonzero((c >= 0) & np.isfinite(ph))
            res[h].append((len(i), float(((ph[i] > 0.5) == (c[i] == 0)).mean())))
for h, v in res.items():
    n = np.array([a for a, _ in v]); a = np.array([b for _, b in v])
    print(h, "pooled acc", float((n * a).sum() / n.sum()), "per-half-instance min/median/max", a.min().round(3), np.median(a).round(3), a.max().round(3), "n", int(n.sum()))
