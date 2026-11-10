"""exp-dstart: dead-ball START placement of a14 (B12 on the a13 cell, dead -3.0 / seq_d -1.0) on the 6-fold LOMO caches.

Motivation (label-free real-match audit, runs/select/exp_seq2/boundary_audit_dead.json): a10/a14 start D runs further
from the visible ball-out frame than a5c/a8r. Decode-only levers, chosen on the sim folds only (no real labels, no
public/sealed read):
  vis_k   extra log-offset on the learned stream's D column where the ball is visible and inside the pitch (> 1 m in)
  snap    after postprocess: a predicted live->D edge with a visible ball-out frame within [-pre, +post] frames is moved
          to that frame (the D run is extended back or the live spell forward; never across another edge)
Stages:  diag  -> D-start offset vs truth for a13 / a14 (ball visible vs not)       lomo/dstart_diag.json
         grid  -> spellF1 per fold for every (vis_k, snap) cell + mean rank           lomo/dstart_grid.csv / .json
         real  -> label-free real-match drift bands for the top cells (final models)  lomo/dstart_real.json
"""
import argparse, copy, json, os, sys, time
from pathlib import Path
os.environ["OMP_NUM_THREADS"] = "4"
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import pandas as pd

import exp_seq_lomo as e
import mtah.select as sel
from mtah.evidence.dead import dead_evidence

DEAD_OFF, SEQ_D = -3.0, -1.0          # a14's registered calibration
WTS = e.DECODERS["B12"]
VIS_K = (0.0, -1.0, 1.0)
SNAPS = {"off": None, "s25_75": (25, 75), "s50_75": (50, 75)}
SHIFTS = (0, 6, 12, 20, 30)   # live->D edges moved this many frames earlier (sim diag: a14 D starts are late)
SIDE_M = 1.0


def pitch_dims(m):
    p = getattr(m, "pitch", None)
    try:
        return float(p[0]), float(p[1])
    except Exception:
        return 105.0, 68.0


def ball_geometry(m):
    """(visible, inside: visible and > SIDE_M m inside the lines, out_onsets: first visible-out frame after in)."""
    L, W = pitch_dims(m)
    b = m.ball.astype(float); vis = np.isfinite(b[:, 0])
    ax, ay = np.abs(np.nan_to_num(b[:, 0])), np.abs(np.nan_to_num(b[:, 1]))
    inside = vis & (ax < L / 2 - SIDE_M) & (ay < W / 2 - SIDE_M)
    out = vis & ((ax > L / 2) | (ay > W / 2))
    # onset: out now, and the last visible frame before was in-bounds (within 1 s)
    on = np.flatnonzero(out & ~np.r_[False, out[:-1]])
    keep = [o for o in on if (vis[max(o - 25, 0):o] & ~out[max(o - 25, 0):o]).any()]
    return vis, inside, np.array(keep, int)


def cfg_for():
    base = e.base_cfg()
    cfg = sel.apply_cell(base, {**sel.GRID8[0], "dead_offset": DEAD_OFF})
    cfg.decoder.lengths = sel.estimate_lengths([], cfg, None)
    return cfg


def decode(h, cfg, wts, vis_k, inside, seq_d=SEQ_D):
    ev = dict(h.ev); ev["dead"] = dead_evidence(h.p_dead, logit_offset=DEAD_OFF)
    hh = copy.copy(h); hh.seq = {}
    for model in wts:
        v = h.seq[model].copy(); v[:, 2] += np.float32(seq_d)
        if vis_k:
            v[inside, 2] += np.float32(vis_k)
        hh.seq[model] = v
    return e.decode_labels(hh, ev, wts, cfg)


def snap_dstart(lab, outs, pre, post, half):
    """Move each live->D edge to the nearest ball-out onset in [edge - pre, edge + post] (same half, no edge crossed)."""
    lab = lab.copy()
    isd = lab == "D"
    edges = np.flatnonzero(isd[1:] & ~isd[:-1]) + 1
    if not len(outs):
        return lab
    for c in edges:
        cand = outs[(outs >= c - pre) & (outs <= c + post) & (half[outs] == half[c])]
        if not len(cand):
            continue
        o = int(cand[np.abs(cand - c).argmin()])
        if o > c:      # D started too early: live continues until the ball is out, if the D run lasts that long
            run_end = c + int(np.argmax(lab[c:] != "D")) if (lab[c:] != "D").any() else len(lab)
            if o < run_end:
                lab[c:o] = lab[c - 1]
        elif o < c:    # D started too late: extend back within the preceding live spell
            seg = lab[o:c]
            if (seg == lab[c - 1]).all():
                lab[o:c] = "D"
    return lab


def shift_dstart(lab, s, half):
    """Move every live->D edge s frames earlier, within the preceding live spell and the same half."""
    if not s:
        return lab
    lab = lab.copy(); isd = lab == "D"
    for c in np.flatnonzero(isd[1:] & ~isd[:-1]) + 1:
        a = max(c - s, 0)
        seg = lab[a:c]
        if (seg == lab[c - 1]).all() and half[a] == half[c]:
            lab[a:c] = "D"
    return lab


def dstart_offsets(truth, lab, vis, w=250):
    """Signed offset (pred edge - true edge) of the nearest predicted live->D edge for each true live->D edge."""
    def edges(x):
        d = x == "D"; return np.flatnonzero(d[1:] & ~d[:-1]) + 1
    te, pe = edges(truth), edges(lab)
    rows = []
    for c in te:
        near = pe[np.abs(pe - c) <= w]
        rows.append({"vis": bool(vis[max(c - 12, 0):c + 12].mean() > 0.8),
                     "off": int((near - c)[np.abs(near - c).argmin()]) if len(near) else None})
    return rows


def iter_halves():
    for F in e.MATCHES:
        for k, s in e.halves_of(F):
            yield F, e.Half(F, F, k, s)


def summarise_offsets(rows):
    out = {}
    for name, sub in (("visible", [r for r in rows if r["vis"]]), ("not_visible", [r for r in rows if not r["vis"]])):
        o = np.array([r["off"] for r in sub if r["off"] is not None], float)
        out[name] = {"n_true": len(sub), "missed_10s": round(1 - len(o) / max(len(sub), 1), 3),
                     "abs_median": float(np.median(np.abs(o))) if len(o) else None,
                     "within_12": round(float((np.abs(o) <= 12).mean()), 3) if len(o) else None,
                     "early_share": round(float((o < -12).mean()), 3) if len(o) else None,
                     "late_share": round(float((o > 12).mean()), 3) if len(o) else None,
                     "q10_50_90": [float(x) for x in np.percentile(o, [10, 50, 90])] if len(o) else None}
    return out


def stage_diag():
    cfg = cfg_for(); acc = {"a13": [], "a14": [], "a14_snap": []}
    for F, h in iter_halves():
        vis, inside, outs = ball_geometry(h.s.match)
        t = h.s.truth["ctrl_Dfull"]
        lab13 = decode(h, cfg, {}, 0.0, inside)
        lab14 = decode(h, cfg, WTS, 0.0, inside)
        acc["a13"] += dstart_offsets(t, lab13, vis); acc["a14"] += dstart_offsets(t, lab14, vis)
        acc["a14_snap"] += dstart_offsets(t, snap_dstart(lab14, outs, 25, 75, h.s.match.half), vis)
        e.log(f"dstart diag {h.s.key} s{h.s.seed} done")
    res = {k: summarise_offsets(v) for k, v in acc.items()}
    (e.L / "dstart_diag.json").write_text(json.dumps(res, indent=1))
    print(json.dumps(res, indent=1))


def stage_grid():
    cfg = cfg_for(); rows = []
    out = e.L / "dstart_grid_rows.csv"
    for F, h in iter_halves():
        t0 = time.time()
        vis, inside, outs = ball_geometry(h.s.match)
        base13 = decode(h, cfg, {}, 0.0, inside)
        rows += e.score_rows(h, base13, {"fold": F, "vis_k": None, "snap": "a13"})
        for k in VIS_K:
            lab = decode(h, cfg, WTS, k, inside)
            for sn, pp in SNAPS.items():
                lab2 = lab if pp is None else snap_dstart(lab, outs, pp[0], pp[1], h.s.match.half)
                for sh in SHIFTS:
                    lab3 = shift_dstart(lab2, sh, h.s.match.half)
                    rows += e.score_rows(h, lab3, {"fold": F, "vis_k": k, "snap": f"{sn}|{sh}"})
        e.log(f"dstart grid {h.s.key} s{h.s.seed} {time.time() - t0:.0f}s")
        pd.DataFrame(rows).to_csv(out, index=False)
    d = pd.DataFrame(rows)
    d["cell"] = d["snap"].astype(str) + "|" + d["vis_k"].astype(str)
    sf = d.pivot_table(index="cell", columns="variant", values="spell_f1", aggfunc="mean")
    mf = d[d.variant == "ctrl_Dfull"].groupby("cell")[["macro_f1", "n_pred", "n_gt"]].mean()
    pf = d[d.variant == "ctrl_Dfull"].pivot_table(index="cell", columns="fold", values="spell_f1", aggfunc="mean")
    ref = pf.loc["off|0|0.0"]
    t = sf.join(mf).reset_index()
    t["wins_vs_a14"] = [int((pf.loc[c] > ref).sum()) for c in t.cell]
    t["losses_vs_a14"] = [int((pf.loc[c] < ref).sum()) for c in t.cell]
    t["min_delta_vs_a14"] = [float((pf.loc[c] - ref).min()) for c in t.cell]
    t = sel.mean_rank(t, list(e.VARS))
    t.to_csv(e.L / "dstart_grid.csv", index=False)
    pd.set_option("display.width", 250)
    print(t.round(4).to_string(index=False)); print(pf.round(4).to_string())
    (e.L / "dstart_grid.json").write_text(json.dumps({"table": t.to_dict(orient="records"),
                                                      "per_fold_ctrl_Dfull": pf.round(5).to_dict()}, indent=1, default=float))


def stage_real(cells):
    """Label-free drift on the real match for the given cells ('snap|vis_k'), final a14 models; aggregates only."""
    from make_candidate import load_selection
    from mtah.eval.drift import blackout_mask_from_match, drift_report
    from mtah.io.tags import load_tags
    from mtah.pipeline import compute_evidence, compute_strong, predict
    from touch_common import load_real
    data = os.environ.get("MTAH_DATA_DIR", os.path.join(os.path.expanduser("~"), "mtah-data"))
    sel_dir = e.MAIN / "runs" / "select" / "a14"
    cfg, dead, lengths, team, merge, switch, seq = load_selection(sel_dir, with_seq=True)
    cfg.pathcrf.cache_dir = str(e.MAIN / "runs" / "cache" / "pathcrf")
    m = load_real(); tags = load_tags(data)
    vis, inside, outs = ball_geometry(m)
    ev0 = compute_evidence(m, tags, cfg, dead, cache_key="real_tactical", team_model=team)
    arr = np.load(e.FINAL / "real_stream.npy")   # (n_streams, F, 3), d_offset NOT applied
    strong = compute_strong(m, cfg); mask = blackout_mask_from_match(m)
    res = {}
    for c in cells:
        sn, sh, k = c.split("|"); sh = int(sh); k = float(k)
        ev = dict(ev0); c2 = copy.deepcopy(cfg)
        for i, (name, w) in enumerate(seq["weights"].items()):
            v = arr[i].copy(); v[:, 2] += np.float32(seq["d_offset"])
            if k:
                v[inside, 2] += np.float32(k)
            ev[name] = v; c2.decoder.weights[name] = float(w); c2.decoder.lengths[name] = 1.0
        lab = predict(m, tags, c2, dead, evidence=ev, strong=strong)
        if SNAPS[sn] is not None:
            lab = snap_dstart(np.asarray(lab).astype("<U1"), outs, *SNAPS[sn], m.half)
        lab = shift_dstart(np.asarray(lab).astype("<U1"), sh, m.half)
        r = drift_report(lab, m, tags, mask, code_table=cfg.tag.code_table)
        bands = sel.real_band_checks(r["non_blackout_D_share"], r["spells_per_graded_min"])
        res[c] = {"real_vis_dead": r["non_blackout_D_share"], "real_spells_per_min": r["spells_per_graded_min"],
                  "real_blackout_D": r.get("blackout_D_share"), "real_spell_len_median_s": r.get("spell_len_median_s"),
                  "admissible": bool(bands["real_plausible"])}
        print(c, json.dumps(res[c]), flush=True)
    (e.L / "dstart_real.json").write_text(json.dumps(res, indent=1, default=float))


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("stage", choices=["diag", "grid", "real"])
    ap.add_argument("--cells", nargs="*", default=[])
    a = ap.parse_args()
    {"diag": stage_diag, "grid": stage_grid, "real": lambda: stage_real(a.cells)}[a.stage]()


if __name__ == "__main__":
    main()
