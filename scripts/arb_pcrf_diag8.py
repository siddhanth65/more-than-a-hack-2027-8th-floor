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
    s = ""
    for c0 in range(0, b - a - 125, 250):
        i = np.flatnonzero((conf[a+c0:a+c0+250] >= 0) & np.isfinite(ph[a+c0:a+c0+250])) + a + c0
        if len(i) < 3: s += "."; continue
        acc = ((ph[i] > .5) == (conf[i] == 0)).mean(); s += "#" if acc > .8 else ("x" if acc < .2 else "~")
    print(f"H{h} (one char per 10 s chunk; # right, x wrong, . none):\n{s}")
