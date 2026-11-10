"""Label-free diagnosis of PathCRF on the real match, per half and per 5-minute bin (aggregates only)."""
import json, os, sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "scripts"))
import numpy as np
import arb_common as ac
from mtah.io.metrica_loader import load_tactical
from mtah.io.match import half_runs

m = load_tactical(os.environ["MTAH_DATA_DIR"])
rule = ac.RuleParams(**json.load(open("runs/arb/rule.json"))["rule"])
conf = ac.confident_control(m, rule)
z = np.load("runs/cache/pathcrf/real_tactical.npz"); ph = z["p_home"]
team = m.team
print("slots", m.n_slots, "team counts", [(int((team == k).sum())) for k in (0, 1)], "pitch", m.pitch)
for h, a, b in half_runs(m.half):
    pos = m.pos[a:b]
    x = pos[:, :, 0]
    print(f"H{h}: frames {b-a} vis slots/frame home {np.isfinite(x[:, team==0]).sum(1).mean():.1f} away {np.isfinite(x[:, team==1]).sum(1).mean():.1f}")
    mxh = np.nanmean(x[:, team == 0]); mxa = np.nanmean(x[:, team == 1])
    print(f"  mean x home {mxh:+.1f} away {mxa:+.1f} -> left team = {'home' if mxh < mxa else 'away'}; home attacks {'right(+x)' if mxh < mxa else 'left(-x)'}")
    # first 60 s median x per team (kick-off formation) and slot-stability: per-slot mean x split in halves of the half
    k0 = int(60 * m.fps)
    print(f"  first-60s median x home {np.nanmedian(x[:k0, team==0]):+.1f} away {np.nanmedian(x[:k0, team==1]):+.1f}; last-60s home {np.nanmedian(x[-k0:, team==0]):+.1f} away {np.nanmedian(x[-k0:, team==1]):+.1f}")
    # per 5-minute bin: team centroid x, pathcrf acc on confident frames, count
    step = int(300 * m.fps); rows = []
    for s in range(a, b, step):
        e = min(b, s + step); i = np.flatnonzero((conf[s:e] >= 0)) + s
        i = i[np.isfinite(ph[i])]
        acc = float(((ph[i] > 0.5) == (conf[i] == 0)).mean()) if len(i) > 20 else float("nan")
        cx = [float(np.nanmean(m.pos[s:e, team == k, 0])) for k in (0, 1)]
        rows.append(f"{(s-a)/m.fps/60:4.0f}m n={len(i):4d} acc={acc:.2f} cx_home={cx[0]:+.1f} cx_away={cx[1]:+.1f}")
    print("\n".join("   " + r for r in rows))
    # pathcrf accuracy conditional on which side the pseudo-label's team is
    i = np.flatnonzero((conf >= 0) & (m.half == h) & np.isfinite(ph))
    for lab, nm in ((0, "home"), (1, "away")):
        j = i[conf[i] == lab]; print(f"  confident {nm}: n={len(j)} pathcrf says home {float((ph[j] > 0.5).mean()):.2f}")
