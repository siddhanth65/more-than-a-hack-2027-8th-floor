"""Amendment 5 real-match label-free checks of candidate files (aggregates only, no ground truth, no window score).
Tag agreement is computed against the ORIGINAL (unshifted) tags. Usage: lead_real_metrics.py ref_name cand_name [...]"""
import hashlib, json, os, sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT))
os.environ.pop("MTAH_SP_SHIFT_REAL", None)
import numpy as np, pandas as pd
from mtah.eval.drift import blackout_mask_from_match, drift_report
from mtah.io.metrica_loader import load_tactical
from mtah.io.tags import load_tags
D = os.environ["MTAH_DATA_DIR"]
m = load_tactical(D); tags = load_tags(D); mask = blackout_mask_from_match(m); n = m.n_frames; g = m.graded
RHO = 0.8921081072691707; BLOCK = (110808, 113741)
blk = np.zeros(n, bool); blk[BLOCK[0]:BLOCK[1] + 1] = True
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
res = {}; X = {k: lab(k) for k in sys.argv[1:]}
for name, x in X.items():
    d = drift_report(x, m, tags, mask); r = {"md5": hashlib.md5((ROOT / "runs" / "candidates" / f"{name}.csv").read_bytes()).hexdigest()}
    r["tag_agr_overall"] = d["tag_agreement_implied"]
    vis = g & ~mask
    r["vis_dead_literal"] = float((x[vis] == "D").mean()); r["vis_dead_block_excl"] = float((x[vis & ~blk] == "D").mean())
    sp = spells(x); sp_out = sp[~((sp >= BLOCK[0]) & (sp <= BLOCK[1]))]
    r["R_literal"] = len(sp) / (g.sum() / 25 / 60); r["R_block_excl"] = len(sp_out) / ((g & ~blk).sum() / 25 / 60)
    r["R_over_rho_block_excl"] = r["R_block_excl"] / RHO; r["R_over_rho_literal"] = r["R_literal"] / RHO
    r["pred_D_share"] = d["pred_D_share"]
    ref = X[sys.argv[1]]; r["frames_differing_vs_ref"] = int((x[g] != ref[g]).sum())
    r["frames_differing_vs_ref_outside_block"] = int((x[g & ~blk] != ref[g & ~blk]).sum())
    r["checks"] = {"vis_dead_in_band": bool(0.116 <= r["vis_dead_block_excl"] <= 0.181),
                   "R_over_rho_in_band": bool(2.28 <= r["R_over_rho_block_excl"] <= 4.10)}
    res[name] = r
a0 = res[sys.argv[1]]["tag_agr_overall"]
for name, r in res.items():
    r["checks"]["tag_agr_ok_vs_ref_minus_0.02"] = bool(r["tag_agr_overall"] >= a0 - 0.02)
print(json.dumps(res, indent=1))
