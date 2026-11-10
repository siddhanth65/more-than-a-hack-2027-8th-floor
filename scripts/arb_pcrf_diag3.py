"""PathCRF accuracy on pseudo-label frames by pseudo-team side and ball side, per half; chunk-level bimodality."""
import json, os, sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "scripts"))
import numpy as np
import arb_common as ac
from mtah.io.metrica_loader import load_tactical
m = load_tactical(os.environ["MTAH_DATA_DIR"]); team = m.team
conf = ac.confident_control(m, ac.RuleParams(**json.load(open("runs/arb/rule.json"))["rule"]))
ph = np.load("runs/cache/pathcrf/real_tactical.npz")["p_home"]; ok = np.isfinite(ph)
for h in (1, 2):
    sel = (m.half == h)
    a, b = np.flatnonzero(sel)[[0, -1]]
    mx = [np.nanmean(m.pos[a:b, team == k, 0]) for k in (0, 1)]; left = 0 if mx[0] < mx[1] else 1
    i = np.flatnonzero(sel & (conf >= 0) & ok); y = conf[i]; right_pred = ph[i] > .5; corr = right_pred == (y == 0)
    side = np.where(y == left, "L", "R"); bx = m.ball[i, 0] < 0
    print(f"H{h} left_team={'home' if left == 0 else 'away'}")
    for s in "LR":
        for bs, nm in ((True, "ball-left"), (False, "ball-right")):
            j = (side == s) & (bx == bs); print(f"   pseudo-team on {s} side, {nm}: n={int(j.sum()):4d} acc={corr[j].mean():.2f} says-home={right_pred[j].mean():.2f}")
    ch = (i - a) // 250; accs = [corr[ch == c].mean() for c in np.unique(ch) if (ch == c).sum() >= 3]
    print("   chunk acc histogram (n chunks, bins <.2,.2-.8,>.8):", sum(x < .2 for x in accs), sum(.2 <= x <= .8 for x in accs), sum(x > .8 for x in accs))
