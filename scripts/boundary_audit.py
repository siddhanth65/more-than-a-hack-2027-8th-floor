"""Label-free audit of where candidates place their H<->A hand-overs on the real match, relative to detected touches.

For every H<->A change in a candidate (graded frames, ball visible within +-1 s of the change) the signed offset to
the nearest touch by the NEW team within +-3 s (touch frame - change frame; >0 = change placed before the touch).
Also the share of changes with no such touch (the change is not backed by a visible touch). No ground truth is read.
    python scripts/boundary_audit.py a5c a10 a14     -> runs/select/exp_seq2/boundary_audit.json
"""
import json, os, sys
from pathlib import Path
os.environ["OMP_NUM_THREADS"] = "4"
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "scripts"))
import numpy as np
import pandas as pd

from make_candidate import load_selection
from mtah.evidence.touches import detect_touches
from touch_common import load_real

GRADED = [(11185, 90651), (110808, 190197)]
W_BALL, W_TOUCH = 25, 75


def labels(name):
    return pd.read_csv(ROOT / "runs" / "candidates" / f"{name}.csv")["label"].to_numpy().astype("<U1")


def main():
    names = sys.argv[1:] or ["a5c", "a10", "a14"]
    m = load_real()
    cfg = load_selection(ROOT / "runs" / "select" / "a14", with_seq=True)[0]
    t = detect_touches(m, cfg.touch)
    fr, tt = t[:, 0].astype(int), t[:, 2].astype(int)
    vis = np.isfinite(m.ball[:, 0])
    graded = np.zeros(len(vis), bool)
    for a, b in GRADED:
        graded[a:b + 1] = True
    ref = labels(names[0])
    # team code of H: the touch-team code that agrees most with H frames of the first candidate
    ok = tt >= 0
    h_code = int(pd.Series(tt[ok][ref[fr[ok]] == "H"]).mode()[0])
    code = {"H": h_code, "A": 1 - h_code}
    out = {"touch_team_code_H": h_code, "n_touches": int(len(t))}
    for n in names:
        lab = labels(n)
        ch = np.flatnonzero((lab[1:] != lab[:-1]) & np.isin(lab[1:], ["H", "A"]) & np.isin(lab[:-1], ["H", "A"])) + 1
        ch = ch[graded[ch]]
        seen = np.array([vis[max(c - W_BALL, 0):c + W_BALL].mean() > 0.8 for c in ch], bool)
        offs, unbacked = [], 0
        for c in ch[seen]:
            sel = (np.abs(fr - c) <= W_TOUCH) & (tt == code[lab[c]])
            if not sel.any():
                unbacked += 1; continue
            d = fr[sel] - c
            offs.append(int(d[np.abs(d).argmin()]))
        o = np.array(offs)
        out[n] = {"n_HA_changes": int(len(ch)), "n_ball_visible": int(seen.sum()),
                  "share_unbacked": round(unbacked / max(seen.sum(), 1), 3),
                  "offset_median_frames": float(np.median(o)) if len(o) else None,
                  "offset_abs_median_frames": float(np.median(np.abs(o))) if len(o) else None,
                  "share_within_5": round(float((np.abs(o) <= 5).mean()), 3) if len(o) else None,
                  "share_change_before_touch": round(float((o > 0).mean()), 3) if len(o) else None,
                  "share_change_after_touch": round(float((o < 0).mean()), 3) if len(o) else None,
                  "offset_quantiles_10_50_90": [float(q) for q in np.percentile(o, [10, 50, 90])] if len(o) else None}
    p = ROOT / "runs" / "select" / "exp_seq2" / "boundary_audit.json"
    p.write_text(json.dumps(out, indent=1))
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
