"""exp-oracle: can the a14 decoder emit short possessions at all, given perfect evidence? (sim 6-fold LOMO caches only)
Injects a TRUTH-derived stream into the a14 decode (B12, dead -3.0 / seq_d -1.0), same hook as exp_ctrl:
  onset   team of each true H/A spell on frames [start - 2, start + 12] (graded spells only), NaN elsewhere
  short   as onset, only for true spells <= 5 s
  full    team on every frame of every true H/A spell (evidence ceiling)
If short-spell recovery barely moves even here, the limit is the decoder/segmentation, not evidence.
    python scripts/exp_oracle.py   -> lomo/oracle_grid.csv / oracle_grid.json"""
import json, os, sys, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import pandas as pd
import exp_dstart as x
import exp_seq_lomo as e
import exp_tau as xt
import mtah.select as sel
from touch_team_audit import spells

W_OR = (0.0, 2.0, 8.0, 32.0)
MODES = ("onset", "short", "full")


def oracle_ev(t, graded, fps, mode):
    v = np.full((len(t), 3), np.nan, np.float32)
    for a, b, k in spells(t):
        if not graded[a:b].any():
            continue
        if mode == "short" and (b - a) > 5 * fps:
            continue
        lo, hi = (a, b) if mode == "full" else (max(a - 2, 0), min(a + 13, len(t)))
        v[lo:hi, 0] = 1.0 if k == "H" else 0.0; v[lo:hi, 1] = 0.0 if k == "H" else 1.0
    return v


def recovery(t, lab, m):
    """Per true graded H/A spell: (short, visible, recovered at IoU >= 0.5 by a same-team predicted spell)."""
    ps = spells(lab); vis = np.isfinite(m.ball[:, 0]); out = []
    for a, b, k in spells(t):
        if not m.graded[a:b].any():
            continue
        best = 0.0
        for c, d, kk in ps:
            if kk == k and d > a and c < b:
                best = max(best, (min(b, d) - max(a, c)) / (max(b, d) - min(a, c)))
        out.append({"short": (b - a) <= 5 * m.fps, "visible": vis[a:b].mean() > 0.8, "recovered": best >= 0.5})
    return out


def main():
    cfg = x.cfg_for(); rows = []; rec = []
    for F, h in x.iter_halves():
        t0 = time.time(); m = h.s.match; t = h.s.truth["ctrl_Dfull"]
        ev0 = xt.evidence(h, x.WTS)
        for md in MODES:
            for w in W_OR:
                if w == 0 and md != "onset":
                    continue
                ev = dict(ev0); cf = cfg; cell = f"{md}|{w:g}" if w else "a14"
                if w:
                    ev["oracle"] = oracle_ev(t, m.graded, m.fps, md); cf = x.copy.deepcopy(cfg)
                    cf.decoder.weights["oracle"] = float(w); cf.decoder.lengths["oracle"] = 1.0
                lab = xt.decode_tau(ev, x.WTS, m.half, m.fps, h.strong, cf, 30.0, 30.0, 1.0)
                rows += e.score_rows(h, lab, {"fold": F, "cell": cell})
                rec += [{**r, "cell": cell, "fold": F} for r in recovery(t, lab, m)]
        e.log(f"oracle grid {h.s.key} s{h.s.seed} {time.time() - t0:.0f}s")
    d = pd.DataFrame(rows); rc = pd.DataFrame(rec)
    sf = d.pivot_table(index="cell", columns="variant", values="spell_f1", aggfunc="mean")
    mf = d[d.variant == "ctrl_Dfull"].groupby("cell")[["macro_f1", "n_pred", "n_gt"]].mean()
    pf = d[d.variant == "ctrl_Dfull"].pivot_table(index="cell", columns="fold", values="spell_f1", aggfunc="mean")
    rr = rc.groupby(["cell", "short", "visible"]).recovered.mean().unstack(["short", "visible"])
    rr.columns = [f"rec_{'short' if s else 'long'}_{'vis' if v else 'hid'}" for s, v in rr.columns]
    tb = sf.join(mf).join(rr).reset_index()
    tb["mean_delta"] = [float((pf.loc[c] - pf.loc["a14"]).mean()) for c in tb.cell]
    tb = sel.mean_rank(tb, list(e.VARS))
    tb.to_csv(e.L / "oracle_grid.csv", index=False)
    (e.L / "oracle_grid.json").write_text(json.dumps({"table": tb.to_dict(orient="records"), "per_fold": pf.round(5).to_dict()},
                                                     indent=1, default=float))
    pd.set_option("display.width", 250)
    print(tb.sort_values("mean_rank").round(4).to_string(index=False)); print(pf.round(4).to_string())


if __name__ == "__main__":
    main()
