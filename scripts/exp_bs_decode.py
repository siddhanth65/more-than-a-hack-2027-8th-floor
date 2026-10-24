"""Decode the spell-weighted Model Bs (exp_seq_lomo_gru_sw.py) against a14's Model B on the 6-fold LOMO test halves.
a14 decoder (a13 cell, dead -3.0, merge off); cells: stream B or Bs at weight w, seq D offset d. Sim only.
    python scripts/exp_bs_decode.py [--folds 9001 ...]   -> lomo/bs_grid.csv / bs_grid.json
    --blend: B and Bs streams together (a14 under-segments, Bs over-segments)   -> lomo/bs_blend.csv / bs_blend.json"""
import argparse, json, os, sys, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import pandas as pd
import exp_dstart as x
import exp_seq_lomo as e
import exp_tau as xt
import mtah.select as sel

CELLS = [({"B": 12.0}, -1.0)] + [({"Bs": w}, d) for w in (8.0, 12.0, 16.0) for d in (-1.0, -2.0, 0.0)]
BLEND = [({"B": 12.0}, -1.0)] + [({"B": b, "Bs": w}, d) for b, w in ((12.0, 2.0), (12.0, 4.0), (12.0, 6.0),
                                                                     (10.0, 4.0), (8.0, 4.0)) for d in (-1.0, -0.5)]


def cell_name(wts, d):
    return "+".join(f"{k}{w:g}" for k, w in wts.items()) + f"|{d:g}"


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--folds", type=int, nargs="*", default=e.MATCHES)
    ap.add_argument("--blend", action="store_true")
    a = ap.parse_args()
    cells, out = (BLEND, "bs_blend") if a.blend else (CELLS, "bs_grid")
    cfg = x.cfg_for(); rows = []
    for F in a.folds:
        for k, s in e.halves_of(F):
            t0 = time.time(); h = e.Half(F, F, k, s); m = h.s.match
            p = np.load(e.L / "pred_Bs" / f"f{F}" / f"{k}_s{s}.npz")
            h.seq["Bs"] = e.fx.upsample(p["logp"], p["starts"], m.n_frames, m.half)
            for wts, d in cells:
                ev = xt.evidence(h, wts, seq_d=d)
                lab = xt.decode_tau(ev, wts, m.half, m.fps, h.strong, cfg, 30.0, 30.0, 1.0)
                rows += e.score_rows(h, lab, {"fold": F, "cell": cell_name(wts, d)})
            e.log(f"bs decode {k} s{s} {time.time() - t0:.0f}s")
    d = pd.DataFrame(rows)
    sf = d.pivot_table(index="cell", columns="variant", values="spell_f1", aggfunc="mean")
    mf = d[d.variant == "ctrl_Dfull"].groupby("cell")[["macro_f1", "n_pred", "n_gt"]].mean()
    pf = d[d.variant == "ctrl_Dfull"].pivot_table(index="cell", columns="fold", values="spell_f1", aggfunc="mean")
    ref = pf.loc[cell_name({"B": 12.0}, -1.0)]
    tb = sf.join(mf).reset_index()
    tb["mean_delta"] = [float((pf.loc[c] - ref).mean()) for c in tb.cell]
    tb["min_delta"] = [float((pf.loc[c] - ref).min()) for c in tb.cell]
    tb["wins"] = [int((pf.loc[c] > ref).sum()) for c in tb.cell]
    tb = sel.mean_rank(tb, list(e.VARS))
    tb.to_csv(e.L / f"{out}.csv", index=False)
    (e.L / f"{out}.json").write_text(json.dumps({"folds": a.folds, "table": tb.to_dict(orient="records"),
                                                  "per_fold": pf.round(5).to_dict()}, indent=1, default=float))
    pd.set_option("display.width", 250)
    print(tb.sort_values("mean_rank").round(4).to_string(index=False)); print(pf.round(4).to_string())


if __name__ == "__main__":
    main()
