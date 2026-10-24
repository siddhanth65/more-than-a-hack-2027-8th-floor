"""Sim-only audit: are short true possessions backed by a correctly-teamed touch, and does a14 recover them?
For each true H/A spell (ctrl_Dfull truth, graded) on the 6-fold test halves (a13-cell touches, cached):
  has_touch      a touch by the spell's team in [start - 5, start + 12] frames
  touch_wrong    a touch in that window assigned to the OTHER team (and none to the right team)
  touch_contest  only contested touches (team -1) in the window
  recovered      a14 has a same-team spell with IoU >= 0.5
Also per-touch accuracy: touch team vs truth owner 12 frames after the touch (H/A frames only).
    -> runs/select/exp_seq2/lomo/touch_team_audit.json (aggregates only)"""
import json, os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import pandas as pd
import exp_dstart as x
import exp_seq_lomo as e


def spells(l):
    ch = np.flatnonzero(l[1:] != l[:-1]) + 1
    st = np.r_[0, ch]; en = np.r_[ch, len(l)]
    return [(a, b, l[a]) for a, b in zip(st, en) if l[a] in ("H", "A")]


def main():
    cfg = x.cfg_for(); rows = []; tacc = []
    for F, h in x.iter_halves():
        m = h.s.match; t = h.s.truth["ctrl_Dfull"]; vis = np.isfinite(m.ball[:, 0])
        tc = h.strong[2]; fr = tc[:, 0].astype(int); tt = tc[:, 2].astype(int)
        own = t[np.clip(fr + 12, 0, len(t) - 1)]
        ok = np.isin(own, ["H", "A"]) & (tt >= 0)
        # team code of H on this half: majority agreement
        hc = 0 if ((tt[ok] == 0) == (own[ok] == "H")).mean() > 0.5 else 1
        code = {"H": hc, "A": 1 - hc}
        for f, team, o in zip(fr, tt, own):
            if o in ("H", "A"):
                tacc.append({"contested": team < 0, "correct": team == code[o], "vis": True})
        lab = x.decode(h, cfg, x.WTS, 0.0, None)
        ps = spells(lab)
        for a, b, k in spells(t):
            if not m.graded[a:b].any():
                continue
            w = (fr >= a - 5) & (fr <= a + 12)
            right = (w & (tt == code[k])).any(); wrong = (w & (tt == 1 - code[k])).any(); cont = (w & (tt < 0)).any()
            best = 0.0
            for c, d, kk in ps:
                if kk == k and d > a and c < b:
                    best = max(best, (min(b, d) - max(a, c)) / (max(b, d) - min(a, c)))
            rows.append({"dur": (b - a) / m.fps, "vis": float(vis[a:b].mean()), "has_touch": bool(right),
                         "touch_wrong": bool(wrong and not right), "touch_contest_only": bool(cont and not right and not wrong),
                         "recovered": best >= 0.5})
        print(F, h.s.key, h.s.seed, flush=True)
    d = pd.DataFrame(rows); ta = pd.DataFrame(tacc)
    d["short"] = d.dur <= 5; d["visible"] = d.vis > 0.8
    g = d.groupby(["short", "visible"])
    tab = pd.DataFrame({"n": g.size(), "has_touch": g.has_touch.mean(), "touch_wrong": g.touch_wrong.mean(),
                        "contest_only": g.touch_contest_only.mean(), "recovered": g.recovered.mean()})
    sv = d[d.short & d.visible]
    by = sv.groupby("has_touch").recovered.agg(["size", "mean"])
    res = {"per_touch": {"n": int(len(ta)), "contested_share": float(ta.contested.mean()),
                         "correct_among_assigned": float(ta[~ta.contested].correct.mean())},
           "spells": tab.reset_index().to_dict(orient="records"),
           "short_visible_recovered_by_touch": by.reset_index().to_dict(orient="records")}
    (e.L / "touch_team_audit.json").write_text(json.dumps(res, indent=1, default=float))
    pd.set_option("display.width", 200)
    print(json.dumps(res["per_touch"])); print(tab.round(3).to_string()); print(by.round(3).to_string())


if __name__ == "__main__":
    main()
