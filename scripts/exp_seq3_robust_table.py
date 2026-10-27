"""Plan 8 T2: aggregate table for the exp-seq3 robustness variants (sim 4 scores, macroF1, n_spells, folds won vs a14 and a13r,
merge off = th1 and th0.7), from spell_<variant>_haz0.csv (exp_seq3_spell.py) and the a13r decode rows of exp_seq2/lomo.
  <mtah venv python> scripts/exp_seq3_robust_table.py   -> exp_seq3/robust_table.csv"""
import glob, os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import pandas as pd
import seq3_common as c3

L = c3.EXP3.parent / "exp_seq2" / "lomo"


def perfold(d, cell, var="ctrl_Dfull"):
    x = d[(d.cell == cell) & (d.variant == var)]
    return x.groupby("fold").spell_f1.mean()


def main():
    a13 = pd.concat([pd.read_csv(f) for f in glob.glob(str(L / "decode_f*.csv"))])
    print(a13[a13.decoder == "a13"]["merge"].unique())
    a13r = a13[(a13.decoder == "a13") & (a13["merge"] == "on")]
    ref13 = a13r[a13r.variant == "ctrl_Dfull"].groupby("fold").spell_f1.mean()
    rows = []
    srcs = {v: pd.read_csv(c3.EXP3 / f"spell_{v}_haz0.csv") for v in ("a14", "base", "v1", "touch", "v1touch")}
    ref14 = {th: perfold(srcs["a14"], f"a14|haz0|th{th}") for th in ("1", "0.7")}
    for v, d in srcs.items():
        for th in ("1", "0.7"):
            cell = f"{v}|haz0|th{th}"; pf = perfold(d, cell); c = d[d.cell == cell]
            r = {"variant": v, "merge": "off" if th == "1" else "th0.7"}
            for var in ("ctrl_Dfull", "raw_Dfull", "ctrl_Dnarrow", "raw_Dnarrow"):
                r[var] = c[c.variant == var].spell_f1.mean()
            cc = c[c.variant == "ctrl_Dfull"]
            r["macroF1"] = cc.macro_f1.mean(); r["n_spells"] = cc.n_pred.mean(); r["n_gt"] = cc.n_gt.mean()
            r["won_vs_a14_nomerge"] = int((pf > ref14["1"]).sum()); r["won_vs_a14_samemerge"] = int((pf > ref14[th]).sum())
            r["won_vs_a13r"] = int((pf > ref13).sum()); rows.append(r)
    t = pd.DataFrame(rows); t.to_csv(c3.EXP3 / "robust_table.csv", index=False)
    print("a13r ctrl_Dfull", ref13.mean(), "raw", a13r[a13r.variant == "raw_Dfull"].spell_f1.mean(),
          "ctrl_Dn", a13r[a13r.variant == "ctrl_Dnarrow"].spell_f1.mean(), "raw_Dn", a13r[a13r.variant == "raw_Dnarrow"].spell_f1.mean(),
          "macro", a13r[a13r.variant == "ctrl_Dfull"].macro_f1.mean(), "n", a13r[a13r.variant == "ctrl_Dfull"].n_pred.mean())
    pd.set_option("display.width", 250); print(t.round(4).to_string(index=False))


main()
