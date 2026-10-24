"""Decode the spell-weighted Model Bs (exp_seq_lomo_gru_sw.py) against a14's Model B on the 6-fold LOMO test halves.
a14 decoder (a13 cell, dead -3.0, merge off); cells: stream B or Bs at weight w, seq D offset d. Sim only.
    python scripts/exp_bs_decode.py [--folds 9001 ...]   -> lomo/bs_grid.csv / bs_grid.json"""
import argparse, json, os, sys, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import pandas as pd
import exp_dstart as x
import exp_seq_lomo as e
import exp_tau as xt
import mtah.select as sel

CELLS = [("B", 12.0, -1.0)] + [("Bs", w, d) for w in (8.0, 12.0, 16.0) for d in (-1.0, -2.0, 0.0)]


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--folds", type=int, nargs="*", default=e.MATCHES)
    a = ap.parse_args()
    cfg = x.cfg_for(); rows = []
    for F in a.folds:
        for k, s in e.halves_of(F):
            t0 = time.time(); h = e.Half(F, F, k, s); m = h.s.match
            p = np.load(e.L / "pred_Bs" / f"f{F}" / f"{k}_s{s}.npz")
            h.seq["Bs"] = e.fx.upsample(p["logp"], p["starts"], m.n_frames, m.half)
            for st, w, d in CELLS:
                wts = {st: w}
                ev = xt.evidence(h, wts, seq_d=d)
                lab = xt.decode_tau(ev, wts, m.half, m.fps, h.strong, cfg, 30.0, 30.0, 1.0)
                rows += e.score_rows(h, lab, {"fold": F, "cell": f"{st}|{w:g}|{d:g}"})
            e.log(f"bs decode {k} s{s} {time.time() - t0:.0f}s")
    d = pd.DataFrame(rows)
    sf = d.pivot_table(index="cell", columns="variant", values="spell_f1", aggfunc="mean")
    mf = d[d.variant == "ctrl_Dfull"].groupby("cell")[["macro_f1", "n_pred", "n_gt"]].mean()
    pf = d[d.variant == "ctrl_Dfull"].pivot_table(index="cell", columns="fold", values="spell_f1", aggfunc="mean")
    ref = pf.loc["B|12|-1"]
    tb = sf.join(mf).reset_index()
    tb["mean_delta"] = [float((pf.loc[c] - ref).mean()) for c in tb.cell]
    tb["min_delta"] = [float((pf.loc[c] - ref).min()) for c in tb.cell]
    tb["wins"] = [int((pf.loc[c] > ref).sum()) for c in tb.cell]
    tb = sel.mean_rank(tb, list(e.VARS))
    tb.to_csv(e.L / "bs_grid.csv", index=False)
    (e.L / "bs_grid.json").write_text(json.dumps({"folds": a.folds, "table": tb.to_dict(orient="records"),
                                                  "per_fold": pf.round(5).to_dict()}, indent=1, default=float))
    pd.set_option("display.width", 250)
    print(tb.sort_values("mean_rank").round(4).to_string(index=False)); print(pf.round(4).to_string())


if __name__ == "__main__":
    main()
