"""Plan 8 T3: decode-only recalibration of the per-half-normalised learned stream v1 (exp_seq3/v1) in the a13r base.
Grid dead_offset x seq_d_offset x merge {off, th0.7}. Sim: 6-fold LOMO held-out halves (fold-v1 models' predictions);
merge model per (fold, cell) fitted ONLY on that fold's inner-validation match V (inner-model predictions, decoded in
the same cell, labels ctrl_Dfull), applied to the fold's test halves. Real: v1 FINAL stream, label-free drift; merge model
per cell pooled over the six folds' V tables (all out-of-sample predictions). Aggregates only; no ground truth beyond
the sim folds, no public/sealed read.
  python scripts/calib_v1.py folds | real | table"""
import os, sys, json, time, pickle, copy
os.environ["OMP_NUM_THREADS"] = "4"
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import pandas as pd

import exp_seq_lomo as e
import exp_seq3_eval as ev3
import seq3_common as c3
import mtah.select as sel
from mtah.evidence.dead import dead_evidence
from mtah.spells.merge import MergeModel, apply_merge_model, match_spell_table, merge_labels

DEADS = (-3.0, -3.5, -4.0, -4.5, -5.0); SEQDS = (-1.0, -2.0, -3.0, -4.0); TH = 0.7; W = 12.0
OUT = c3.EXP3 / "calib_v1"; OUT.mkdir(exist_ok=True)
CELLS = [(o, d) for o in DEADS for d in SEQDS]


def cell_cfg(base, o):
    cfg = sel.apply_cell(base, {**sel.GRID8[0], "dead_offset": o})
    cfg.decoder.lengths = sel.estimate_lengths([], cfg, None); return cfg


def decode_cell(h, cfg, o, d, v1):
    ev = dict(h.ev); ev["dead"] = dead_evidence(h.p_dead, logit_offset=o)
    hh = copy.copy(h); v = v1.copy(); v[:, 2] += np.float32(d); hh.seq = {"B": v}
    return ev, e.decode_labels(hh, ev, {"B": W}, cfg)


def folds():
    base = e.base_cfg()
    for F in e.MATCHES:
        out = OUT / f"fold{F}.csv"
        if out.exists():
            print("fold", F, "done"); continue
        t0 = time.time(); train, V, _ = e.split(F); rows = []; vtabs = {}
        val = [e.Half(F, V, k, s) for k, s in e.halves_of(V)]
        test = [e.Half(F, F, k, s) for k, s in e.halves_of(F)]
        for hs in (val, test):
            for h in hs:
                h.v1 = ev3.load_variant("v1", F, h)[0]
        for o, d in CELLS:
            cfg = cell_cfg(base, o); tabs = []
            for h in val:
                ev, lab = decode_cell(h, cfg, o, d, h.v1)
                tab = match_spell_table(lab, h.s.match, ev, h.strong); tab["y"] = merge_labels(h.s.truth[sel.MERGE_TRUTH], lab, tab)
                tabs.append(tab[np.isfinite(tab["y"].to_numpy(float))])
            df = pd.concat(tabs, ignore_index=True); vtabs[(o, d)] = df
            mm = MergeModel().fit(df, df["y"].to_numpy(float))
            for h in test:
                ev, lab = decode_cell(h, cfg, o, d, h.v1)
                tg = {"fold": F, "dead_offset": o, "seq_d_offset": d}
                rows += e.score_rows(h, lab, {**tg, "merge": "off"})
                rows += e.score_rows(h, apply_merge_model(lab, h.s.match, ev, h.strong, mm, TH), {**tg, "merge": "th0.7"})
            print(f"fold {F} cell {o},{d} {time.time() - t0:.0f}s", flush=True)
        pickle.dump(vtabs, open(OUT / f"vtabs{F}.pkl", "wb"))
        pd.DataFrame(rows).to_csv(out, index=False)


def real():
    from make_candidate import load_selection
    from mtah.eval.drift import blackout_mask_from_match, drift_report
    from mtah.evidence import seq_feats as fx
    from mtah.evidence.dead import dead_table
    from mtah.io.match import half_runs
    from mtah.io.tags import load_tags
    from mtah.pipeline import compute_evidence, compute_strong, predict
    from touch_candidates import agreement
    from touch_common import load_real
    data = os.environ.get("MTAH_DATA_DIR", r"HOME/mtah-data")
    cfg0, dead, lengths, team, _, _, _ = load_selection(e.MAIN / "runs" / "select" / "a13r", with_seq=True)
    cfg0.pathcrf.cache_dir = str(e.MAIN / "runs" / "cache" / "pathcrf"); cfg0.post.merge_theta = 1.0
    m = load_real(); tags = load_tags(data)
    zc = np.load(e.MAIN / "runs" / "select" / "exp_robust" / "real_cache.npz", allow_pickle=False)
    ev0 = compute_evidence(m, tags, cfg0, dead, cache_key="real_tactical", team_model=team)
    p_dead = dead.predict_proba(dead_table(m, tags, dead.policy))
    strong = compute_strong(m, cfg0); mask = blackout_mask_from_match(m)
    st = []
    for h, s, en in half_runs(m.half):
        n = (en - s) // fx.STEP * fx.STEP
        if n > 0:
            st.append(s + np.arange(0, n, fx.STEP))
    z = np.load(c3.EXP3 / "v1" / "real_stream.npz"); base = fx.upsample(z["logp"], np.concatenate(st), m.n_frames, m.half)
    vt = [pickle.load(open(OUT / f"vtabs{F}.pkl", "rb")) for F in e.MATCHES]
    res = []
    for o, d in CELLS:
        df = pd.concat([t[(o, d)] for t in vt], ignore_index=True)
        mm = MergeModel().fit(df, df["y"].to_numpy(float))
        for mg in ("off", "th0.7"):
            cfg = copy.deepcopy(cfg0); cfg.dead_offset = o; cfg.post.merge_theta = TH if mg != "off" else 1.0
            ev = dict(ev0); ev["dead"] = dead_evidence(p_dead, logit_offset=o)
            v = base.copy(); v[:, 2] += np.float32(d); ev["seqB"] = v
            cfg.decoder.weights["seqB"] = W; cfg.decoder.lengths["seqB"] = 1.0
            lab = predict(m, tags, cfg, dead, evidence=ev, strong=strong, merge_model=mm if mg != "off" else None)
            r = drift_report(lab, m, tags, mask, code_table=cfg.tag.code_table); ag = agreement(lab, m, zc)
            res.append({"dead_offset": o, "seq_d_offset": d, "merge": mg, "real_vis_dead": r["non_blackout_D_share"],
                        "real_spells_per_min": r["spells_per_graded_min"], "real_tag_implied": r["tag_agreement_implied"],
                        "real_pathcrf_agree": ag["pathcrf_confident_HA_agree"], "real_replay_D": ag["D_share_when_replay_cue"],
                        "real_pred_D": r["pred_D_share"]})
            print(res[-1], flush=True)
    pd.DataFrame(res).to_csv(OUT / "real.csv", index=False)


def table():
    d = pd.concat([pd.read_csv(OUT / f"fold{F}.csv") for F in e.MATCHES], ignore_index=True)
    k = ["dead_offset", "seq_d_offset", "merge"]
    sf = d.pivot_table(index=k, columns="variant", values="spell_f1", aggfunc="mean")
    oth = d[d.variant == "ctrl_Dfull"].groupby(k)[["macro_f1", "n_pred"]].mean()
    t = sf.join(oth).reset_index().merge(pd.read_csv(OUT / "real.csv"), on=k)
    t["admissible"] = [bool(sel.real_band_checks(v, r)["real_plausible"]) for v, r in zip(t.real_vis_dead, t.real_spells_per_min)]
    t = sel.mean_rank(t, list(e.VARS)); t.to_csv(OUT / "calib_table.csv", index=False)
    pd.set_option("display.width", 300); pd.set_option("display.max_rows", 200)
    print(t.round(4).to_string(index=False))
    adm = t[t.admissible]
    print("\nADMISSIBLE", len(adm)); print(adm.round(4).head(10).to_string(index=False))
    json.dump({"chosen": adm.iloc[0].to_dict() if len(adm) else None}, open(OUT / "chosen.json", "w"), indent=1, default=float)


if __name__ == "__main__":
    {"folds": folds, "real": real, "table": table}[sys.argv[1]]()
