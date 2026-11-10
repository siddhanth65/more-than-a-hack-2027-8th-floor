"""exp-tau: short possessions. a14 (B12 on the a13 cell, dead -3.0 / seq_d -1.0) under-segments on the sim folds
(n_pred ~157 vs n_gt ~175 per half, false merges ~2x false splits; true spells <= 5 s are 48 % of all and matched at
IoU >= 0.5 only 17-42 %). Decode-only levers on the 6-fold LOMO caches (no real labels, no public/sealed read):
  tau_touch  H<->A switch cost within +-TOUCH_W frames of a detected touch (the a13 cell's touches, cached)
  tau_rest   H<->A switch cost elsewhere (a14: 30 everywhere)
  aba_s      postprocess A-B-A merge threshold in seconds (a14: 1.0)
    python scripts/exp_tau.py grid   -> lomo/tau_grid.csv / tau_grid.json
    python scripts/exp_tau.py real --cells 12|30|0.5 ...   -> lomo/tau_real.json (label-free real-match bands)
"""
import argparse, copy, json, os, sys, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import pandas as pd

import exp_dstart as x
import exp_seq_lomo as e
import mtah.select as sel
from mtah.decode.viterbi import decode
from mtah.evidence.dead import dead_evidence
from mtah.spells.postprocess import postprocess

TOUCH_W = 3
TAU_TOUCH = (30.0, 20.0, 12.0, 6.0)
TAU_REST = (30.0, 40.0, 60.0)
ABA = (1.0, 0.5, 0.0)


def tau_map(n, touches, tau_touch, tau_rest):
    t = np.full(n, tau_rest, np.float64)
    if len(touches) and tau_touch != tau_rest:
        fr = touches[:, 0].astype(int)
        for d in range(-TOUCH_W, TOUCH_W + 1):
            f = np.clip(fr + d, 0, n - 1); t[f] = np.minimum(t[f], tau_touch)
    return t


def evidence(h, wts, seq_d=x.SEQ_D):
    ev = dict(h.ev); ev["dead"] = dead_evidence(h.p_dead, logit_offset=x.DEAD_OFF)
    for model in wts:
        v = h.seq[model].copy(); v[:, 2] += np.float32(seq_d); ev[f"seq{model}"] = v
    return ev


def decode_tau(ev, wts, half, fps, strong, cfg, tau_touch, tau_rest, aba_s):
    dcfg = copy.deepcopy(cfg.decoder)
    for model, w in wts.items():
        dcfg.weights[f"seq{model}"] = float(w); dcfg.lengths[f"seq{model}"] = 1.0
    n = len(half)
    tt = None if tau_touch == tau_rest == dcfg.tau_HA else tau_map(n, strong[2], tau_touch, tau_rest)
    lab = decode(ev, half, dcfg, tau_HA_t=tt)
    post = copy.deepcopy(cfg.post); post.aba_s = float(aba_s)
    return postprocess(lab, half, fps, post, strong_H=strong[0], strong_A=strong[1], touches=strong[2])


def stage_grid():
    cfg = x.cfg_for(); rows = []
    assert cfg.decoder.tau_HA == 30.0 and cfg.post.aba_s == 1.0, (cfg.decoder.tau_HA, cfg.post.aba_s)
    for F, h in x.iter_halves():
        t0 = time.time(); m = h.s.match
        ev = evidence(h, x.WTS)
        for tt in TAU_TOUCH:
            for tr in TAU_REST:
                for ab in ABA:
                    lab = decode_tau(ev, x.WTS, m.half, m.fps, h.strong, cfg, tt, tr, ab)
                    rows += e.score_rows(h, lab, {"fold": F, "cell": f"{tt:g}|{tr:g}|{ab:g}"})
        e.log(f"tau grid {h.s.key} s{h.s.seed} {time.time() - t0:.0f}s")
        pd.DataFrame(rows).to_csv(e.L / "tau_grid_rows.csv", index=False)
    d = pd.DataFrame(rows)
    sf = d.pivot_table(index="cell", columns="variant", values="spell_f1", aggfunc="mean")
    mf = d[d.variant == "ctrl_Dfull"].groupby("cell")[["macro_f1", "n_pred", "n_gt"]].mean()
    pf = d[d.variant == "ctrl_Dfull"].pivot_table(index="cell", columns="fold", values="spell_f1", aggfunc="mean")
    ref = pf.loc["30|30|1"]
    t = sf.join(mf).reset_index()
    t["wins_vs_a14"] = [int((pf.loc[c] > ref).sum()) for c in t.cell]
    t["losses_vs_a14"] = [int((pf.loc[c] < ref).sum()) for c in t.cell]
    t["min_delta_vs_a14"] = [float((pf.loc[c] - ref).min()) for c in t.cell]
    t["mean_delta_vs_a14"] = [float((pf.loc[c] - ref).mean()) for c in t.cell]
    t = sel.mean_rank(t, list(e.VARS))
    t.to_csv(e.L / "tau_grid.csv", index=False)
    pd.set_option("display.width", 250)
    print(t.sort_values("mean_rank").round(4).to_string(index=False)); print(pf.round(4).to_string())
    (e.L / "tau_grid.json").write_text(json.dumps({"table": t.to_dict(orient="records"),
                                                    "per_fold_ctrl_Dfull": pf.round(5).to_dict()}, indent=1, default=float))


def stage_real(cells):
    """Label-free drift bands on the real match for the given cells, a14's final models; aggregates only.
    Cell 30|30|1 must reproduce runs/candidates/a14.csv exactly (checked)."""
    from make_candidate import load_selection
    from mtah.eval.drift import blackout_mask_from_match, drift_report
    from mtah.io.tags import load_tags
    from mtah.pipeline import compute_evidence, compute_strong
    from touch_common import load_real
    data = os.environ.get("MTAH_DATA_DIR", os.path.join(os.path.expanduser("~"), "mtah-data"))
    cfg, dead, lengths, team, merge, switch, seq = load_selection(e.MAIN / "runs" / "select" / "a14", with_seq=True)
    cfg.pathcrf.cache_dir = str(e.MAIN / "runs" / "cache" / "pathcrf")
    m = load_real(); tags = load_tags(data)
    ev = compute_evidence(m, tags, cfg, dead, cache_key="real_tactical", team_model=team)
    arr = np.load(e.FINAL / "real_stream.npy")
    wts = {}
    for i, (name, w) in enumerate(seq["weights"].items()):
        v = arr[i].copy(); v[:, 2] += np.float32(seq["d_offset"]); ev[name] = v; wts[name[len("seq"):]] = w
    strong = compute_strong(m, cfg); mask = blackout_mask_from_match(m)
    a14 = pd.read_csv(e.MAIN / "runs" / "candidates" / "a14.csv")["label"].to_numpy().astype("<U1")
    res = {}
    for c in ["30|30|1"] + [c for c in cells if c != "30|30|1"]:
        tt, tr, ab = (float(v) for v in c.split("|"))
        lab = decode_tau(ev, wts, m.half, m.fps, strong, cfg, tt, tr, ab)
        if c == "30|30|1" and not np.array_equal(lab, a14):
            raise SystemExit(f"30|30|1 does not reproduce a14.csv ({int((lab != a14).sum())} frames differ)")
        r = drift_report(lab, m, tags, mask, code_table=cfg.tag.code_table)
        b = sel.real_band_checks(r["non_blackout_D_share"], r["spells_per_graded_min"])
        res[c] = {"real_vis_dead": r["non_blackout_D_share"], "real_spells_per_min": r["spells_per_graded_min"],
                  "real_spells_per_HA_min": r.get("spells_per_HA_min"), "real_spell_len_median_s": r.get("spell_len_median_s"),
                  "admissible": bool(b["real_plausible"])}
        print(c, json.dumps(res[c], default=float), flush=True)
    (e.L / "tau_real.json").write_text(json.dumps(res, indent=1, default=float))


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("stage", choices=["grid", "real"])
    ap.add_argument("--cells", nargs="*", default=[])
    a = ap.parse_args()
    {"grid": stage_grid, "real": lambda: stage_real(a.cells)}[a.stage]()


if __name__ == "__main__":
    main()
