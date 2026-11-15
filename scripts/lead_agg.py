"""Amendment 5 aggregation (aggregates only). corrected (a') = s_ko at lead 5.0 on the stand-in (identity, see the amendment);
mismatch = s_ko models on the shifted stand-ins. Per set: mean spellF1 (per half = mean over seeds), diff, wins/ties/losses.
Usage: lead_agg.py NAME   (NAME = out/sets_NAME and out/lomo_NAME)"""
import glob, json, os, sys
import numpy as np, pandas as pd
LD = os.path.expanduser("~/projects/mtah/lead"); B = os.path.expanduser("~/projects/mtah/battery")
name = sys.argv[1]
def sets_half(files, cand="S"):
    d = pd.concat([pd.read_csv(f) for f in files]); d = d[d.cand == cand]
    return d.groupby(["key", "variant"]).spell_f1.mean().unstack()
def lomo_half(f):
    d = pd.read_csv(f); return d.groupby(["key", "variant"]).spell_f1.mean().unstack()
out = {}
def cmp(setname, ref, mis):
    for v in ("ctrl_Dfull", "raw_Dfull"):
        r, m = ref[v].dropna(), mis[v].reindex(ref[v].dropna().index)
        diff = r - m
        out.setdefault(setname, {})[v] = {"n_halves": int(len(r)), "ref_lead5.0": round(float(r.mean()), 4), "mismatch": round(float(m.mean()), 4),
                                          "corrected_minus_mismatch": round(float(diff.mean()), 4),
                                          "wins/ties/losses_corrected": [int((diff > 1e-12).sum()), int((diff.abs() <= 1e-12).sum()), int((diff < -1e-12).sum())]}
for s in ("sealed", "Knat", "Ksim", "M"):
    ref = sets_half(glob.glob(f"{B}/out/{s}/sdec_part*.csv")); mis = sets_half(glob.glob(f"{LD}/out/sets_{name}/{s}/sdec_part*.csv"))
    cmp(s, ref, mis)
cmp("LOMO", lomo_half(f"{LD}/out/ref/lomo_ref.csv"), lomo_half(f"{LD}/out/lomo_{name}/lomo_V3.csv"))
if os.path.exists(f"{LD}/out/lomo_null/lomo_V3.csv"):
    cmp("LOMO_null_chain_check", lomo_half(f"{LD}/out/ref/lomo_ref.csv"), lomo_half(f"{LD}/out/lomo_null/lomo_V3.csv"))
print(json.dumps(out, indent=1))
