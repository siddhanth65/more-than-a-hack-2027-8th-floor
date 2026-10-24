"""exp-ctrl: decode-only "ball control" stream for short possessions (sim 6-fold LOMO caches; no real labels/reads).
ctrl evidence: on visible-ball frames where a player of team T is within R m of the ball and closer than every opponent
(by a factor RATIO), log-score +1 for T's column (H/A), NaN elsewhere (no opinion). Optionally only at control
ONSETS (first ONSET_W frames of a control run). a14 decoder (B12, dead -3.0 / seq_d -1.0) + w_ctrl * ctrl.
Also: coverage of true short visible spells by a new-team control onset in [start-5, start+12].
    python scripts/exp_ctrl.py   -> lomo/ctrl_grid.csv / ctrl_grid.json"""
import json, os, sys, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import pandas as pd
import exp_dstart as x
import exp_seq_lomo as e
import exp_tau as xt
import mtah.select as sel

R, RATIO = 1.5, 1.3
W_CTRL = (0.0, 0.5, 1.0, 2.0, 4.0)
MODES = ("all", "onset")
ONSET_W = 12


def control(m):
    """(F,) team code with control (0/1) or -1."""
    b = m.ball.astype(float); p = m.pos
    d = np.hypot(p[..., 0] - b[:, None, 0], p[..., 1] - b[:, None, 1]); d = np.where(np.isfinite(d), d, np.inf)
    d0 = np.where(m.team[None, :] == 0, d, np.inf).min(1); d1 = np.where(m.team[None, :] == 1, d, np.inf).min(1)
    c = np.full(len(b), -1)
    c[(d0 <= R) & (d0 * RATIO < d1)] = 0; c[(d1 <= R) & (d1 * RATIO < d0)] = 1
    return c


def onsets(c):
    o = np.full(len(c), -1)
    st = np.flatnonzero((c >= 0) & (np.r_[-2, c[:-1]] != c))
    for s in st:
        e_ = s + ONSET_W; o[s:e_] = np.where(c[s:e_] == c[s], c[s], -1)
    return o


def ctrl_ev(c, hcode):
    v = np.full((len(c), 3), np.nan, np.float32)
    on = c >= 0
    v[on, 0] = np.where(c[on] == hcode, 1.0, 0.0); v[on, 1] = np.where(c[on] == hcode, 0.0, 1.0)
    return v


def main():
    cfg = x.cfg_for(); rows = []; cov = []
    for F, h in x.iter_halves():
        t0 = time.time(); m = h.s.match; t = h.s.truth["ctrl_Dfull"]
        c = control(m); o = onsets(c)
        ha = np.isin(t, ["H", "A"]) & (c >= 0)
        hcode = 0 if ((c[ha] == 0) == (t[ha] == "H")).mean() > 0.5 else 1
        code = {"H": hcode, "A": 1 - hcode}
        ch = np.flatnonzero(t[1:] != t[:-1]) + 1; st = np.r_[0, ch]; en = np.r_[ch, len(t)]
        for a, b_ in zip(st, en):
            if t[a] in ("H", "A") and m.graded[a:b_].any() and (b_ - a) <= 5 * m.fps and np.isfinite(m.ball[a:b_, 0]).mean() > 0.8:
                w = o[max(a - 5, 0):a + 13]
                cov.append({"right": bool((w == code[t[a]]).any()), "wrong_only": bool((w == 1 - code[t[a]]).any() and not (w == code[t[a]]).any())})
        ev0 = xt.evidence(h, x.WTS)
        evs = {"all": ctrl_ev(c, hcode), "onset": ctrl_ev(o, hcode)}
        for md in MODES:
            for w in W_CTRL:
                if w == 0 and md != "all":
                    continue
                ev = dict(ev0)
                cf = cfg
                if w:
                    ev["ctrl"] = evs[md]; cf = x.copy.deepcopy(cfg); cf.decoder.weights["ctrl"] = float(w); cf.decoder.lengths["ctrl"] = 1.0
                lab = xt.decode_tau(ev, x.WTS, m.half, m.fps, h.strong, cf, 30.0, 30.0, 1.0)
                rows += e.score_rows(h, lab, {"fold": F, "cell": f"{md}|{w:g}"})
        e.log(f"ctrl grid {h.s.key} s{h.s.seed} {time.time() - t0:.0f}s")
        pd.DataFrame(rows).to_csv(e.L / "ctrl_grid_rows.csv", index=False)
    d = pd.DataFrame(rows); cv = pd.DataFrame(cov)
    sf = d.pivot_table(index="cell", columns="variant", values="spell_f1", aggfunc="mean")
    mf = d[d.variant == "ctrl_Dfull"].groupby("cell")[["macro_f1", "n_pred", "n_gt"]].mean()
    pf = d[d.variant == "ctrl_Dfull"].pivot_table(index="cell", columns="fold", values="spell_f1", aggfunc="mean")
    ref = pf.loc["all|0"]
    tb = sf.join(mf).reset_index()
    tb["mean_delta"] = [float((pf.loc[c] - ref).mean()) for c in tb.cell]
    tb["min_delta"] = [float((pf.loc[c] - ref).min()) for c in tb.cell]
    tb["wins"] = [int((pf.loc[c] > ref).sum()) for c in tb.cell]
    tb = sel.mean_rank(tb, list(e.VARS))
    tb.to_csv(e.L / "ctrl_grid.csv", index=False)
    res = {"coverage_short_visible": {"n": int(len(cv)), "right_onset": float(cv.right.mean()), "wrong_only": float(cv.wrong_only.mean())},
           "table": tb.to_dict(orient="records"), "per_fold": pf.round(5).to_dict()}
    (e.L / "ctrl_grid.json").write_text(json.dumps(res, indent=1, default=float))
    pd.set_option("display.width", 250)
    print(json.dumps(res["coverage_short_visible"])); print(tb.sort_values("mean_rank").round(4).to_string(index=False)); print(pf.round(4).to_string())


if __name__ == "__main__":
    main()
