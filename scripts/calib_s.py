"""S calibration: dead_offset x seq_d_offset by the same rule as a17 (calib_a17.py).

  sim   (main env, server)  decode the chosen variant's fold predictions (V3) at the union of a14's two calibration grids
                            (dead {-2,-2.5,-3} x seq_d {-1,-2,-3} and dead {-3,-3.5,-4} x seq_d {-1,-0.5,0,0.5}), four truth
                            variants -> <out>/calib_s_folds.csv
  real  (main env, with MTAH_DATA_DIR, MTAH_PATHCRF_PY)  full registered pipeline per cell on the contest match with S's
                            per-half B checkpoints in --base (a selection dir), PathCRF smoothing 1.0 s; label-free drift only.
                            Chosen = best simulator mean rank (4 truth variants) among cells with visible-dead in [0.116, 0.181]
                            and R / rho in [2.28, 4.10] -> <out>/calib_s.csv / calib_s.json
No score is computed on the real match.
"""
import argparse, copy, hashlib, json, os, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "scripts"))

import numpy as np
import pandas as pd

CELLS = sorted({(o, d) for o in (-2.0, -2.5, -3.0) for d in (-1.0, -2.0, -3.0)} | {(o, d) for o in (-3.0, -3.5, -4.0) for d in (-1.0, -0.5, 0.0, 0.5)})
VARS = ("ctrl_Dfull", "raw_Dfull", "ctrl_Dnarrow", "raw_Dnarrow")


def stage_sim(a):
    import s_selftrain as S
    import exp_seq_lomo as L
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    lomo = Path(a.lomo); rows = []
    for F in S.MATCHES:
        for k in S.keys_of(F):
            for s in S.SEEDS:
                h = S.HalfB(lomo, F, k, s, a.sim, Path(a.pred) / a.variant / f"f{F}" / f"{k}_s{s}.npz")
                for o, d in CELLS:
                    lab, _ = S.decode_cell(h, dead_offset=o, seq_d=d)
                    rows += L.score_rows(h, lab, {"fold": F, "dead_offset": o, "seq_d_offset": d})
        print("fold", F, "done", flush=True)
    pd.DataFrame(rows).to_csv(out / "calib_s_folds.csv", index=False)


def lomo_table(path):
    f = pd.read_csv(path).drop_duplicates(["fold", "dead_offset", "seq_d_offset", "key", "seed", "variant"])
    sf = f.pivot_table(index=["dead_offset", "seq_d_offset"], columns="variant", values="spell_f1", aggfunc="mean")
    c = f[f.variant == "ctrl_Dfull"].copy(); c["r"] = c.n_pred / c.n_gt.clip(lower=1)
    return sf.join(c.groupby(["dead_offset", "seq_d_offset"]).r.mean().rename("rho")).reset_index()


def stage_real(a):
    import mtah.evidence.seqstream as sq
    import mtah.select as sel
    from make_candidate import candidate_labels, load_selection
    from mtah.eval.drift import blackout_mask_from_match, drift_report
    from mtah.io.metrica_loader import load_tactical
    from mtah.io.tags import load_tags
    orig, memo = sq.compute_streams, {}

    def cs(z, spec, model_dir):
        mb = spec["modelB"]; files = list(mb.values()) if isinstance(mb, dict) else [mb]
        key = (tuple(hashlib.md5((Path(model_dir) / f).read_bytes()).hexdigest() for f in files),
               hashlib.md5(np.ascontiguousarray(z["ev_dead"]).tobytes()).hexdigest())
        if key not in memo:
            memo[key] = orig(z, spec, model_dir)
        return {k: v.copy() for k, v in memo[key].items()}

    sq.compute_streams = cs
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    cfg0, dead, lengths, team, merge, switch, seq = load_selection(a.base, with_seq=True)
    cfg0.pathcrf.smooth_s = a.smooth; cfg0.pathcrf.cache_dir = str(ROOT / "runs" / "cache" / "pathcrf_fx")
    m = load_tactical(a.data); tags = load_tags(a.data); mask = blackout_mask_from_match(m)
    t = lomo_table(out / "calib_s_folds.csv"); rows = []
    for r in t.itertuples():
        cfg = copy.deepcopy(cfg0); cfg.dead_offset = float(r.dead_offset)
        sp = copy.deepcopy(seq); sp["d_offset"] = float(r.seq_d_offset)
        lab = candidate_labels(m, tags, cfg, dead, team, merge, switch, seq=sp, sel_dir=Path(a.base))
        d = drift_report(lab, m, tags, mask, code_table=cfg.tag.code_table)
        rows.append({"dead_offset": r.dead_offset, "seq_d_offset": r.seq_d_offset, "real_vis_dead": d["non_blackout_D_share"],
                     "R": d["spells_per_graded_min"], "real_tag_implied": d["tag_agreement_implied"], "real_pred_D": d["pred_D_share"]})
        print({k: (round(v, 4) if isinstance(v, float) else v) for k, v in rows[-1].items()}, flush=True)
    t = t.merge(pd.DataFrame(rows), on=["dead_offset", "seq_d_offset"])
    t["R_over_rho"] = t.R / t.rho
    t["admissible"] = [bool(sel.real_band_checks(v, R, rho=h)["real_plausible"]) for v, R, h in zip(t.real_vis_dead, t.R, t.rho)]
    t = sel.mean_rank(t, list(VARS))
    t.to_csv(out / "calib_s.csv", index=False)
    adm = t[t.admissible]
    (out / "calib_s.json").write_text(json.dumps({"chosen": adm.iloc[0].to_dict() if len(adm) else None, "table": t.to_dict(orient="records")}, indent=1, default=float))
    pd.set_option("display.width", 250)
    print(t[["dead_offset", "seq_d_offset", "ctrl_Dfull", "raw_Dfull", "real_vis_dead", "R", "rho", "R_over_rho", "admissible", "mean_rank"]].round(4).to_string(index=False))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["sim", "real"])
    ap.add_argument("--out", default=str(ROOT / "runs" / "s" / "calib")); ap.add_argument("--variant", default="V3")
    ap.add_argument("--pred", default=str(ROOT / "runs" / "s" / "lomo" / "pred")); ap.add_argument("--lomo", default=str(ROOT / "runs" / "select" / "exp_seq2" / "lomo"))
    ap.add_argument("--sim", default=str(ROOT / "runs" / "sim")); ap.add_argument("--base", default=None)
    ap.add_argument("--smooth", type=float, default=1.0); ap.add_argument("--data", default=os.environ.get("MTAH_DATA_DIR"))
    a = ap.parse_args()
    {"sim": stage_sim, "real": stage_real}[a.stage](a)


if __name__ == "__main__":
    main()
