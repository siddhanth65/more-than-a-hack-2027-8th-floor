"""Aggregate battery CSVs (set, cand, key, seed, variant, spell_f1, macro_f1): per-half = mean over seeds; mean over halves; wins vs ref."""
import sys, glob, json
import pandas as pd, numpy as np
tag = sys.argv[1]; ref = "S"   # S == s_ko on every set here (kick-off rule is a no-op)
d = pd.concat([pd.read_csv(f) for f in glob.glob(f"docs/battery/csv/{tag}/*_part*.csv")])
h = d.groupby(["cand", "key", "variant"])[["spell_f1", "macro_f1"]].mean().reset_index()
rows = []
for v in sorted(h.variant.unique()):
    hv = h[h.variant == v]; p = hv.pivot(index="key", columns="cand", values="spell_f1")
    mf = hv.pivot(index="key", columns="cand", values="macro_f1")
    for c in p.columns:
        rows.append({"set": tag, "variant": v, "cand": c, "n_halves": int(p[c].notna().sum()), "spellF1": p[c].mean(), "macroF1": mf[c].mean(),
                     "wins_vs_S": int((p[c] > p[ref]).sum()) if c != ref else None, "ties": int((p[c] == p[ref]).sum()) if c != ref else None,
                     "losses_vs_S": int((p[c] < p[ref]).sum()) if c != ref else None})
r = pd.DataFrame(rows); pd.set_option("display.width", 200); print(r.round(4).to_string(index=False))
r.to_csv(f"docs/battery/agg_{tag}.csv", index=False)
