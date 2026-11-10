"""a17 calibration: the a14f configuration with PathCRF smoothing 1.0 s on the REAL feed (simulator inputs unchanged), dead_offset x
seq_d_offset re-set by the pre-registered rule on the fixed real evidence.

Grid = the union of a14's two calibration grids (exp_seq2 lomo/calib_B12 and calib_B12w; B weight 12), with the stored 6-fold LOMO
scores (decode-only; the simulator side does not depend on the real feed). Real side: full registered pipeline per cell with the fixed
PathCRF cache (smooth_s 1.0); the BiGRU inputs stay at dead -2.0 (input_dead_offset). rho = LOMO mean n_pred / n_gt (ctrl_Dfull).
Chosen = best simulator mean rank (4 truth variants, as a14's selection) among cells with visible-dead in
[0.116, 0.181] and R / rho in [2.28, 4.10].
  python scripts/calib_a17.py [--base a14f-dir] [--smooth 1.0]   -> runs/kh/a17/calib_a17.csv/.json"""
import argparse, copy, json, os, sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "scripts"))
os.environ.setdefault("OMP_NUM_THREADS", "2")
import numpy as np, pandas as pd
import score_kh                       # noqa: F401  (memoises the BiGRU stream per input hash)
import mtah.select as sel
from make_candidate import candidate_labels, load_selection
from mtah.eval.drift import blackout_mask_from_match, drift_report
from mtah.io.metrica_loader import load_tactical
from mtah.io.tags import load_tags

MAIN = Path(__file__).resolve().parents[1]
LOMO = MAIN / "runs" / "select" / "exp_seq2" / "lomo"
VARS = ("ctrl_Dfull", "raw_Dfull", "ctrl_Dnarrow", "raw_Dnarrow")


def lomo_table():
    f = pd.concat([pd.read_csv(LOMO / n) for n in ("calib_B12_folds.csv", "calib_B12w_folds.csv")], ignore_index=True)
    f = f.drop_duplicates(["fold", "dead_offset", "seq_d_offset", "key", "seed", "variant"])
    sf = f.pivot_table(index=["dead_offset", "seq_d_offset"], columns="variant", values="spell_f1", aggfunc="mean")
    c = f[f.variant == "ctrl_Dfull"].copy(); c["r"] = c.n_pred / c.n_gt.clip(lower=1)
    t = sf.join(c.groupby(["dead_offset", "seq_d_offset"]).r.mean().rename("rho")).reset_index()
    return t


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default=str(MAIN / "runs" / "select" / "a14f")); ap.add_argument("--smooth", type=float, default=1.0)
    ap.add_argument("--data", default=os.environ.get("MTAH_DATA_DIR", os.path.join(os.path.expanduser("~"), "mtah-data")))
    ap.add_argument("--out", default=str(ROOT / "runs" / "kh" / "a17"))
    a = ap.parse_args()
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    cfg0, dead, lengths, team, merge, switch, seq = load_selection(a.base, with_seq=True)
    cfg0.pathcrf.smooth_s = a.smooth; cfg0.pathcrf.cache_dir = str(MAIN / "runs" / "cache" / "pathcrf_fx")
    m = load_tactical(a.data); tags = load_tags(a.data); mask = blackout_mask_from_match(m)
    t = lomo_table(); rows = []
    for r in t.itertuples():
        cfg = copy.deepcopy(cfg0); cfg.dead_offset = float(r.dead_offset)
        sp = copy.deepcopy(seq); sp["d_offset"] = float(r.seq_d_offset)
        lab = candidate_labels(m, tags, cfg, dead, team, merge, switch, seq=sp, sel_dir=Path(a.base))
        d = drift_report(lab, m, tags, mask, code_table=cfg.tag.code_table)
        rows.append({"dead_offset": r.dead_offset, "seq_d_offset": r.seq_d_offset, "real_vis_dead": d["non_blackout_D_share"],
                     "R": d["spells_per_graded_min"], "real_tag_implied": d["tag_agreement_implied"], "real_pred_D": d["pred_D_share"]})
        print(rows[-1], flush=True)
    t = t.merge(pd.DataFrame(rows), on=["dead_offset", "seq_d_offset"])
    t["R_over_rho"] = t.R / t.rho
    t["admissible"] = [bool(sel.real_band_checks(v, R, rho=h)["real_plausible"]) for v, R, h in zip(t.real_vis_dead, t.R, t.rho)]
    t = sel.mean_rank(t, list(VARS))
    t.to_csv(out / "calib_a17.csv", index=False)
    adm = t[t.admissible]
    (out / "calib_a17.json").write_text(json.dumps({"chosen": adm.iloc[0].to_dict() if len(adm) else None, "table": t.to_dict(orient="records")}, indent=1, default=float))
    pd.set_option("display.width", 250)
    print(t[["dead_offset", "seq_d_offset", "ctrl_Dfull", "raw_Dfull", "real_vis_dead", "R", "rho", "R_over_rho", "admissible", "mean_rank"]].round(4).to_string(index=False))


if __name__ == "__main__":
    main()
