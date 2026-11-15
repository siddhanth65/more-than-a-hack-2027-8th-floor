"""Label-free measurement of the set-piece tag lead on the real match (restart instant from the ball itself).
Detector (fixed before looking at leads): a visible ball that stays within 1.0 m of its median position for >= 1.0 s
(>= 90 % of frames visible), then leaves that 1.0 m disk and is >= 4 m from it within 1.5 s (visible at both ends).
The restart instant is the last frame inside the disk. Search window: tag start +-10 s. If several qualify, the one
closest to start + 3.4 s. Tags whose [start-10 s, end+10 s] touches the public window 11185-18685 are excluded.
Aggregates only (no per-tag rows are printed)."""
import sys, json, collections
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from mtah.io.metrica_loader import load_tactical
from mtah.io.tags import load_tags

PUB = (11185, 18685)

def detect(pos, pres, a, b, fps, R=1.0, FAR=4.0, HOLD_S=1.0, OUT_S=1.5):
    """candidate restart frames in [a, b)"""
    hold = int(HOLD_S * fps); out = int(OUT_S * fps); c = []
    t = a + hold
    while t < b:
        w = slice(t - hold, t)
        pv = pres[w]
        if pv.mean() >= 0.9 and pres[t - 1]:
            med = np.nanmedian(pos[w][pv], axis=0)
            d = np.hypot(*(pos[w][pv] - med).T)
            if d.max() <= R:
                # run continues: find first frame leaving the disk
                k = t
                while k < b and (not pres[k] or np.hypot(*(pos[k] - med)) <= R):
                    k += 1
                if k < b and k - t < out * 4:
                    seg = np.arange(k, min(len(pres), k + out))
                    ok = pres[seg]
                    if ok.any() and np.hypot(*(pos[seg[ok]] - med).T).max() >= FAR and pres[min(len(pres) - 1, k)]:
                        c.append(k - 1)
                t = k + hold
                continue
        t += 1
    return c

def main(data):
    m = load_tactical(data); tags = load_tags(data); fps = m.fps
    pres = np.asarray(m.ball_present, bool); pos = np.asarray(m.ball, float)[:, :2]
    W = int(10 * fps); rows = []
    for r in tags.itertuples(index=False):
        if r.code != "SET PIECES": continue
        s, e = int(r.start), int(r.end)
        if s - W <= PUB[1] and e + W >= PUB[0]: continue
        typ = r.type or "NA"
        c = detect(pos, pres, max(0, s - W), min(m.n_frames, s + W), fps)
        lead = np.nan
        if c:
            lead = (min(c, key=lambda f: abs(f - (s + 3.4 * fps))) - s) / fps
        rows.append((typ, lead, (e - s) / fps))
    by = collections.defaultdict(list); ln = collections.defaultdict(list); tot = collections.Counter()
    for typ, lead, L in rows:
        tot[typ] += 1; ln[typ].append(L)
        if np.isfinite(lead): by[typ].append(lead)
    out = {}; allv = []
    for typ in sorted(tot):
        v = np.array(by[typ]); allv += list(v)
        out[typ] = {"tags": tot[typ], "used": len(v),
                    "p25_med_p75": np.percentile(v, [25, 50, 75]).round(2).tolist() if len(v) else None,
                    "tag_len_s_med": float(np.median(ln[typ]))}
    v = np.array(allv)
    rest = np.array([x for t in by if len(by[t]) < 8 for x in by[t]])
    out["REST_types_below_8"] = {"used": len(rest), "p25_med_p75": np.percentile(rest, [25, 50, 75]).round(2).tolist() if len(rest) else None}
    out["ALL"] = {"tags": sum(tot.values()), "used": len(v),
                  "p25_med_p75": np.percentile(v, [25, 50, 75]).round(2).tolist() if len(v) else None}
    # kick-off check against the FAQ frame 113742 (tag lead 3.28 s published), detector sees it?
    ko2 = 113742
    c = detect(pos, pres, ko2 - W, ko2 + W, fps)
    out["detector_on_FAQ_kickoff2"] = {"found": bool(c), "restart_minus_FAQ_s": [round((x - ko2) / fps, 2) for x in c][:3]}
    print(json.dumps(out))

if __name__ == "__main__":
    main(sys.argv[1])
