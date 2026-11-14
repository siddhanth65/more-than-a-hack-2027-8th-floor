"""Label-free comparison of candidate files on the real match (aggregates only, no ground truth) + public-window identity of s_ko vs S."""
import json, os, sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT))
import numpy as np, pandas as pd
from mtah import baselines
from mtah.eval.drift import blackout_mask_from_match, drift_report
from mtah.io.metrica_loader import load_tactical
from mtah.io.tags import load_tags
D = os.environ.get("MTAH_DATA_DIR", os.path.join(os.path.expanduser("~"), "mtah-data"))
m = load_tactical(D); tags = load_tags(D); mask = blackout_mask_from_match(m); n = m.n_frames; g = m.graded
RHO = {"s_ko": 0.8921081072691707, "s": 0.8921081072691707, "a17": 0.8883, "a5c_fx2": 0.8719237643982939}
BLOCK = (110808, 113741)
def lab(name):
    s = pd.read_csv(ROOT / "runs" / "candidates" / f"{name}.csv"); x = np.full(n, "", object); x[s.frame.to_numpy()] = s.label.to_numpy(); return x
def spells(x):
    out = []
    for h in (1, 2):
        sel = np.flatnonzero(m.half == h); a, b = sel[0], sel[-1] + 1; s = a
        for i in range(a + 1, b + 1):
            if i == b or x[i] != x[s]:
                if x[s] in ("H", "A"): out.append(s)
                s = i
    return np.array(out)
res = {}; X = {k: lab(k) for k in ("s_ko", "s", "a17", "a5c_fx2")}
X["smooth50"] = baselines.smooth_vote(m, 50)
for name, x in X.items():
    d = drift_report(x, m, tags, mask); r = {}
    r["tag_agr_overall"] = d["tag_agreement_implied"]
    for h in (1, 2):
        r[f"tag_agr_H{h}"] = drift_report(np.where(m.half == h, x, "D"), m, tags, mask)["tag_agreement_implied"]
    blk = np.zeros(n, bool); blk[BLOCK[0]:BLOCK[1] + 1] = True
    vis = g & ~mask
    r["vis_dead_literal"] = float((x[vis] == "D").mean()); r["vis_dead_block_excl"] = float((x[vis & ~blk] == "D").mean())
    sp = spells(x); r["spells"] = int(len(sp)); gmin = g.sum() / 25 / 60
    r["R_literal"] = len(sp) / gmin
    sp_out = sp[~((sp >= BLOCK[0]) & (sp <= BLOCK[1]))]; r["R_block_excl"] = len(sp_out) / ((g & ~blk).sum() / 25 / 60)
    rho = RHO.get(name)
    if rho: r["rho"] = rho; r["R_over_rho_literal"] = r["R_literal"] / rho; r["R_over_rho_block_excl"] = r["R_block_excl"] / rho
    r["pred_D_share"] = d["pred_D_share"]; res[name] = r
res["window_identity"] = {"frames_11185_18685_differing_s_ko_vs_s": int((X["s_ko"][11185:18686] != X["s"][11185:18686]).sum()),
                          "frames_differing_total_s_ko_vs_s": int((X["s_ko"][g] != X["s"][g]).sum()),
                          "differing_outside_block": int((X["s_ko"][g & ~np.isin(np.arange(n), range(BLOCK[0], BLOCK[1] + 1))] != X["s"][g & ~np.isin(np.arange(n), range(BLOCK[0], BLOCK[1] + 1))]).sum())}
Path(ROOT / "docs/battery").mkdir(parents=True, exist_ok=True)
(ROOT / "docs/battery/real_labelfree.json").write_text(json.dumps(res, indent=1)); print(json.dumps(res, indent=1))
