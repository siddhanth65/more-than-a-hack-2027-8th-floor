"""Aggregate score_kh.py outputs: per candidate mean/sd (spellF1, macroF1) and paired differences with win shares.
  python scripts/aggregate_kh.py NAME=DIR [NAME=DIR ...] --pairs a5c:a14 a5c:H a14:H --out runs/kh/agg.json"""
import argparse, glob, json
from pathlib import Path
import numpy as np, pandas as pd
ap = argparse.ArgumentParser(); ap.add_argument("builds", nargs="+"); ap.add_argument("--pairs", nargs="*", default=["a5c:a14", "a5c:H", "a14:H"])
ap.add_argument("--out", required=True); ap.add_argument("--keys-file", default=None); a = ap.parse_args()
res = {}
for b in a.builds:
    name, d = b.split("=", 1)
    df = pd.concat([pd.read_csv(f) for f in glob.glob(str(Path(d) / "scores_part*.csv"))], ignore_index=True)
    if a.keys_file:
        keep = set(json.load(open(a.keys_file))); df = df[df.key.isin(keep)]
    r = {"n_halves": int(df.key.nunique()), "cand": {}, "pairs": {}}
    for var in ("ctrl_Dfull", "raw_Dfull"):
        d1 = df[df.variant == var]; piv = d1.pivot(index="key", columns="cand", values="spell_f1"); mac = d1.pivot(index="key", columns="cand", values="macro_f1")
        for c in piv.columns:
            r["cand"].setdefault(c, {})[var] = {"spellF1_mean": float(piv[c].mean()), "sd": float(piv[c].std()), "macroF1_mean": float(mac[c].mean())}
        for p in a.pairs:
            x, y = p.split(":")
            if x in piv and y in piv:
                dd = (piv[x] - piv[y]).dropna()   # x minus y; "x leads" = positive
                r["pairs"].setdefault(p, {})[var] = {"mean_diff_x_minus_y": float(dd.mean()), "x_wins": float((dd > 0).mean()), "y_wins": float((dd < 0).mean()), "n": int(len(dd))}
    res[name] = r
Path(a.out).write_text(json.dumps(res, indent=1))
for n, r in res.items():
    print(n, r["n_halves"])
    for c, v in r["cand"].items():
        print(f"  {c:9s} ctrl {v['ctrl_Dfull']['spellF1_mean']:.3f} (sd {v['ctrl_Dfull']['sd']:.3f}) raw {v['raw_Dfull']['spellF1_mean']:.3f} macro {v['ctrl_Dfull']['macroF1_mean']:.3f}")
    for p, v in r["pairs"].items():
        print(f"  {p} ctrl diff {v['ctrl_Dfull']['mean_diff_x_minus_y']:+.4f} win {v['ctrl_Dfull']['x_wins']:.2f}")
