"""Independent anchor: agreement of PathCRF, the pseudo-label and the a5c decode with the tag-implied team, per half
(tags are team-level phase labels, independent of ball proximity). Aggregates only."""
import json, os, sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "scripts"))
import numpy as np
import arb_common as ac
from mtah.evidence.tag_prior import load_tag_codes
from mtah.io.metrica_loader import load_tactical
from mtah.io.tags import load_tags
d = os.environ["MTAH_DATA_DIR"]; m = load_tactical(d); tags = load_tags(d); tab = load_tag_codes()
F = m.n_frames; imp = np.full(F, -1, np.int8)      # tag-implied team 0/1
for r in tags.itertuples(index=False):
    if r.code == "SET PIECES" or r.team not in ("H", "A") or r.code not in tab or int(tab[r.code]["sign"]) == 0:
        continue
    exp = r.team if int(tab[r.code]["sign"]) > 0 else ("A" if r.team == "H" else "H")
    imp[int(r.start):int(r.end) + 1] = 0 if exp == "H" else 1
conf = ac.confident_control(m, ac.RuleParams(**json.load(open("runs/arb/rule.json"))["rule"]))
ph = np.load("runs/cache/pathcrf/real_tactical.npz")["p_home"]; z = np.load("runs/arb/out/real_mnone.npz")
a5 = z["lab_a5c"]; a14 = z["lab_a14"]
for h in (1, 2):
    g = (m.half == h) & (imp >= 0)
    out = {"tag_frames": int(g.sum())}
    i = g & np.isfinite(ph); out["pathcrf_vs_tag"] = float(((ph[i] > .5) == (imp[i] == 0)).mean())
    i = g & (conf >= 0); out["pseudo_vs_tag"] = float(((conf[i] == imp[i])).mean()); out["pseudo_n"] = int(i.sum())
    i2 = i & np.isfinite(ph); out["pathcrf_vs_tag_on_pseudo_frames"] = float(((ph[i2] > .5) == (imp[i2] == 0)).mean())
    for nm, lab in (("a5c", a5), ("a14", a14)):
        j = g & (lab < 2); out[nm + "_vs_tag"] = float((lab[j] == imp[j]).mean())
    j = g & (a5 < 2) & np.isfinite(ph); out["pathcrf_vs_a5c"] = float(((ph[j] > .5) == (a5[j] == 0)).mean())
    # tag agreement with a ball-free alternative: nearest-ball team already in pseudo; report ball visibility
    out["ball_visible_in_tag_frames"] = float(m.ball_present[g].mean())
    print(h, json.dumps({k: round(v, 3) if isinstance(v, float) else v for k, v in out.items()}))
