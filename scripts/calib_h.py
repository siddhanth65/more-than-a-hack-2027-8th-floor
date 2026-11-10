"""Task H: dead-offset calibration of H = a14 with learned-stream weight halved (seqB 12 -> 6), decode-only on the
6-fold LOMO cache of exp_seq2 (folds 9001-9006, seeds 100 101, same as a14's selection) + the real-match label-free drift.
Admissible under the NEW bands: visible-dead in select.REAL_D_SHARE_BAND and R / rho in select.REAL_SPELL_RATE_BAND,
rho = mean over halves of n_pred / n_gt on ctrl_Dfull of the same cell. Chosen = best mean rank among admissible cells.
  python scripts/calib_h.py   -> runs/kh/H/calib_B6.csv / calib_B6.json (+ the a14 (B12) cells re-judged, for reference)"""
import os, sys, json
from pathlib import Path
os.environ.setdefault("OMP_NUM_THREADS", "2")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, pandas as pd
import exp_seq_lomo as e
import mtah.select as sel

OUT = Path(__file__).resolve().parents[1] / "runs" / "kh" / "H"
LOMO = e.L   # cache dir of exp_seq2 (halves, predictions); calib_B6* files are written there, then copied
e.DECODERS["B6"] = {"B": 6.0}
e.CAL_DEAD = (-4.0, -3.5, -3.0, -2.5, -2.0, -1.5)
e.CAL_SEQD = (-3.0, -2.0, -1.0, -0.5, 0.0, 0.5, 1.0)


def judge(t, folds):
    f = folds[folds.variant == "ctrl_Dfull"].copy()
    f["r"] = f.n_pred / f.n_gt.clip(lower=1)
    rho = f.groupby(["dead_offset", "seq_d_offset"]).r.mean().rename("rho").reset_index()
    t = t.drop(columns=[c for c in ("rho", "implied_rate") if c in t.columns]).merge(rho, on=["dead_offset", "seq_d_offset"])
    t["implied_rate"] = t.real_spells_per_min / t.rho
    t["admissible"] = [bool(sel.real_band_checks(v, r, rho=h)["real_plausible"])
                       for v, r, h in zip(t.real_vis_dead, t.real_spells_per_min, t.rho)]
    return t.sort_values(["mean_rank", "ctrl_Dfull"], ascending=[True, False]).reset_index(drop=True)


if __name__ == "__main__":
    e.stage_calib("B6")
    import shutil
    for n in ("calib_B6.csv", "calib_B6_folds.csv"):
        shutil.copy2(LOMO / n, OUT / n)
    t = pd.read_csv(OUT / "calib_B6.csv"); folds = pd.read_csv(OUT / "calib_B6_folds.csv")
    t = judge(t, folds); t.to_csv(OUT / "calib_B6.csv", index=False)
    adm = t[t.admissible]
    (OUT / "calib_B6.json").write_text(json.dumps({"bands": {"vis_dead": list(sel.REAL_D_SHARE_BAND), "rate": list(sel.REAL_SPELL_RATE_BAND)},
        "chosen": adm.iloc[0].to_dict() if len(adm) else None, "table": t.to_dict(orient="records")}, indent=1, default=float))
    # a14 (B12) cells from the registered calibration files, re-judged under the new bands (reference only)
    P = (Path(__file__).resolve().parents[1] / "runs" / "select" / "exp_seq2" / "lomo")
    for name, tab, fo in (("B12", "calib_B12.csv", "calib_B12_folds.csv"), ("B12w", "calib_B12w.csv", "calib_B12w_folds.csv")):
        j = judge(pd.read_csv(P / tab), pd.read_csv(P / fo)); j.to_csv(OUT / f"rejudged_{name}.csv", index=False)
    print(adm.round(4).to_string(index=False) if len(adm) else "NO ADMISSIBLE CELL")
