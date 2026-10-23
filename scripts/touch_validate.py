"""Step 3a (label-free, aggregates only): validate the feed-adaptive touch detector (mode rel) against the old one (abs) on
the REAL match. No ground truth is read; a5c / a10 labels and the PathCRF posterior come from the exp-robust real cache
(runs/select/exp_robust/real_cache.npz of the main tree, produced label-free).  -> runs/touch/validation.json

Metrics per variant (`abs`, `rel`, sensitivity variants):
  rate            touches per visible graded minute (real); sim reference = the same detector on sims 9001-9006 seed 100
  near            share of touches with a player within 1.5 / 2 m at the touch frame
  handover        tag-implied hand-overs (ATTACKING TRANSITION start -> that team gains; DEFENSIVE TRANSITION start -> the
                  other team gains; a PROGRESSION tag whose team differs from the previous PROGRESSION tag, <= 10 s
                  apart -> the new team gains): share with a touch of the gaining team within +-1 s (ball visible in
                  >= half of that window), against the chance level of a random visible frame
  a5c / a10       alignment of touches with the model's own H<->A boundaries: lag histogram (nearest touch - boundary,
                  0.5 s bins over +-2 s), share with a touch within +-0.5 s / +-1 s, share with a touch of the gaining team
                  within +-1 s, chance level
  pathcrf         PathCRF-confident hand-overs (p_home crosses 0.8 <-> 0.2 within 4 s): share with >= 1 touch / a touch of
                  the gaining team in [last old - 0.5 s, first new + 0.5 s], chance level for windows of the same length
Chance levels use the same visibility criterion at random visible graded frames (touch of a given team: both teams
averaged)."""
import json, os, sys, time
os.environ.setdefault("OMP_NUM_THREADS", "4")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from dataclasses import replace

import numpy as np

from touch_common import DATA, MAIN, OUT, load_real, sim_halves
from mtah.evidence.touches import TouchParams, detect_touches
from mtah.io.tags import load_tags

CACHE = MAIN / "runs" / "select" / "exp_robust" / "real_cache.npz"
FPS = 25
VARIANTS = {
    "abs": TouchParams(mode="abs"),
    "rel": TouchParams(mode="rel"),
    "rel_k5": TouchParams(mode="rel", k_rel=5, gate_frames=5),
    "rel_z2": TouchParams(mode="rel", z_thr=2.0, z_min=1.25),
    "rel_z3.5": TouchParams(mode="rel", z_thr=3.5),
    "rel_r2": TouchParams(mode="rel", r_rel_m=2.0),
}


def tag_handovers(tags):
    """[(frame, gaining team 0=H / 1=A)]"""
    ev = []; other = {"H": "A", "A": "H"}
    for r in tags.itertuples(index=False):
        if r.team not in ("H", "A"):
            continue
        if r.code == "ATTACKING TRANSITION":
            ev.append((int(r.start), r.team))
        elif r.code == "DEFENSIVE TRANSITION":
            ev.append((int(r.start), other[r.team]))
    prog = tags[tags.code == "PROGRESSION"].sort_values("start")
    prev = None
    for r in prog.itertuples(index=False):
        if prev is not None and r.team in ("H", "A") and prev.team in ("H", "A") and r.team != prev.team \
                and 0 <= int(r.start) - int(prev.end) <= 10 * FPS:
            ev.append((int(r.start), r.team))
        prev = r
    return sorted(ev)


def ha_boundaries(lab, half):
    """frames where the label switches directly H->A or A->H inside a half -> [(frame, gaining team 0/1)]"""
    out = []
    ch = np.flatnonzero((lab[1:] != lab[:-1]) & np.isin(lab[1:], ["H", "A"]) & np.isin(lab[:-1], ["H", "A"])
                        & (half[1:] == half[:-1]) & (half[1:] > 0)) + 1
    for f in ch:
        out.append((int(f), 0 if lab[f] == "H" else 1))
    return out


def pathcrf_handovers(p_home, half, max_gap=4 * FPS):
    st = np.zeros(len(p_home), np.int8)
    st[np.nan_to_num(p_home, nan=0.5) >= 0.8] = 1; st[np.nan_to_num(p_home, nan=0.5) <= 0.2] = -1
    idx = np.flatnonzero(st != 0)
    out = []
    for a, b in zip(idx[:-1], idx[1:]):
        if st[a] != st[b] and b - a <= max_gap and half[a] == half[b] and half[a] > 0:
            out.append((int(a), int(b), 0 if st[b] > 0 else 1))     # new state +1 = home
    return out


class Index:
    """Fast 'is there a touch (of team X) in [a, b]' queries."""
    def __init__(self, touches, F):
        fr = touches[:, 0].astype(int) if len(touches) else np.zeros(0, int)
        tm = touches[:, 2] if len(touches) else np.zeros(0)
        self.c = {t: np.r_[0, np.cumsum(np.bincount(fr[tm == t], minlength=F)[:F])] for t in (0, 1, -1)}
        self.c["any"] = self.c[0] + self.c[1] + self.c[-1]
        self.F = F

    def has(self, a, b, team="any"):
        a = np.clip(a, 0, self.F - 1); b = np.clip(b, 0, self.F - 1)
        return (self.c[team][b + 1] - self.c[team][a]) > 0


def vis_frac(vis_c, a, b):
    return (vis_c[b + 1] - vis_c[a]) / (b - a + 1)


def event_stats(events, idx, vis_c, F, w, rng, vis_graded):
    """events [(frame, team)] -> evaluable n, share touch of gaining team within +-w, share any touch, chance levels."""
    fr = np.array([e[0] for e in events]); tm = np.array([e[1] for e in events])
    a = np.clip(fr - w, 0, F - 1); b = np.clip(fr + w, 0, F - 1)
    ev = vis_frac(vis_c, a, b) >= 0.5
    if not ev.any():
        return {"n": 0}
    own = np.where(tm == 0, idx.has(a, b, 0), idx.has(a, b, 1))
    anyt = idx.has(a, b, "any")
    # chance: random visible graded frames with the same visibility criterion; both teams averaged
    cand = np.flatnonzero(vis_graded); cand = cand[rng.permutation(len(cand))[:40000]]
    ca = np.clip(cand - w, 0, F - 1); cb = np.clip(cand + w, 0, F - 1)
    ok = vis_frac(vis_c, ca, cb) >= 0.5
    ca, cb = ca[ok], cb[ok]
    ch_team = 0.5 * (idx.has(ca, cb, 0).mean() + idx.has(ca, cb, 1).mean())
    return {"n": int(ev.sum()), "n_total": len(fr), "share_gain_team_touch": float(own[ev].mean()),
            "share_any_touch": float(anyt[ev].mean()), "chance_gain_team": float(ch_team),
            "chance_any": float(idx.has(ca, cb, "any").mean())}


def boundary_stats(bds, touches, idx, vis_c, F, rng, vis_graded):
    out = event_stats(bds, idx, vis_c, F, FPS, rng, vis_graded)
    tf = np.sort(touches[:, 0].astype(int)) if len(touches) else np.zeros(0, int)
    fr = np.array([b[0] for b in bds])
    a = np.clip(fr - FPS, 0, F - 1); b = np.clip(fr + FPS, 0, F - 1)
    ev = vis_frac(vis_c, a, b) >= 0.5
    if len(tf) and ev.any():
        j = np.searchsorted(tf, fr[ev]); j0 = np.clip(j - 1, 0, len(tf) - 1); j1 = np.clip(j, 0, len(tf) - 1)
        near = np.where(np.abs(tf[j0] - fr[ev]) <= np.abs(tf[j1] - fr[ev]), tf[j0], tf[j1]) - fr[ev]
        lag = near / FPS
        edges = np.arange(-2.0, 2.01, 0.5)
        out["lag_hist_0p5s_bins_-2_to_2"] = np.histogram(lag, edges)[0].tolist()
        out["share_beyond_2s"] = float((np.abs(lag) > 2).mean())
        out["median_abs_lag_s"] = float(np.median(np.abs(lag)))
        out["share_touch_within_0.5s"] = float((np.abs(lag) <= 0.5).mean())
        cand = np.flatnonzero(vis_graded); cand = cand[rng.permutation(len(cand))[:40000]]
        ca = np.clip(cand - FPS, 0, F - 1); cb = np.clip(cand + FPS, 0, F - 1); okc = vis_frac(vis_c, ca, cb) >= 0.5
        cand = cand[okc]
        j = np.searchsorted(tf, cand); j0 = np.clip(j - 1, 0, len(tf) - 1); j1 = np.clip(j, 0, len(tf) - 1)
        nn = np.minimum(np.abs(tf[j0] - cand), np.abs(tf[j1] - cand)) / FPS
        out["chance_touch_within_0.5s"] = float((nn <= 0.5).mean())
        out["chance_median_abs_lag_s"] = float(np.median(nn))
    return out


def pathcrf_stats(hos, idx, vis_c, F, rng, vis_graded, p):
    a = np.array([h[0] for h in hos]) - 12; b = np.array([h[1] for h in hos]) + 12
    a = np.clip(a, 0, F - 1); b = np.clip(b, 0, F - 1); tm = np.array([h[2] for h in hos])
    ev = vis_frac(vis_c, a, b) >= 0.5
    own = np.where(tm == 0, idx.has(a, b, 0), idx.has(a, b, 1)); anyt = idx.has(a, b, "any")
    L = b - a
    cand = np.flatnonzero(vis_graded); cand = cand[rng.permutation(len(cand))[:30000]]
    ll = L[rng.integers(0, len(L), len(cand))]
    ca = np.clip(cand, 0, F - 1); cb = np.clip(cand + ll, 0, F - 1); ok = vis_frac(vis_c, ca, cb) >= 0.5
    ca, cb = ca[ok], cb[ok]
    return {"n_handovers": len(hos), "n_evaluable": int(ev.sum()), "share_any_touch": float(anyt[ev].mean()) if ev.any() else None,
            "share_gain_team_touch": float(own[ev].mean()) if ev.any() else None,
            "chance_any": float(idx.has(ca, cb, "any").mean()),
            "chance_gain_team": float(0.5 * (idx.has(ca, cb, 0).mean() + idx.has(ca, cb, 1).mean())),
            "touches_inside_per_handover": float(((idx.c["any"][b + 1] - idx.c["any"][a])[ev]).mean()) if ev.any() else None}


def main():
    T0 = time.time()
    m = load_real(); tags = load_tags(DATA); F = m.n_frames
    z = np.load(CACHE, allow_pickle=False)
    lab5, lab10, p_home = z["lab_a5c"], z["lab_a10"], z["p_home"]
    vis_graded = m.ball_present & m.graded
    vis_c = np.r_[0, np.cumsum(m.ball_present)]
    vmin = vis_graded.sum() / FPS / 60
    ho = tag_handovers(tags); b5 = ha_boundaries(lab5, m.half); b10 = ha_boundaries(lab10, m.half)
    pho = pathcrf_handovers(p_home, m.half)
    b = m.ball.astype(np.float64)
    res = {"graded_visible_min": float(vmin), "n_tag_handovers": len(ho), "n_a5c_boundaries": len(b5),
           "n_a10_boundaries": len(b10), "n_pathcrf_handovers": len(pho), "variants": {}, "sim": {}}
    sims = list(sim_halves())
    for name, p in VARIANTS.items():
        rng = np.random.default_rng(0)
        t = detect_touches(m, p); idx = Index(t, F)
        fr = t[:, 0].astype(int) if len(t) else np.zeros(0, int)
        d = np.hypot(m.pos[fr, :, 0] - b[fr, None, 0], m.pos[fr, :, 1] - b[fr, None, 1]) if len(fr) else np.zeros((0, m.n_slots))
        dn = np.nanmin(np.where(np.isfinite(d), d, np.inf), 1) if len(fr) else np.zeros(0)
        v = {"n_touches": int(len(t)), "per_visible_min": float(len(t) / vmin),
             "team_share_H_A_contested": [float((t[:, 2] == q).mean()) if len(t) else None for q in (0, 1, -1)],
             "near_le_1.5m": float((dn <= 1.5).mean()) if len(t) else None, "near_le_2m": float((dn <= 2.0).mean()) if len(t) else None,
             "tag_handover": event_stats(ho, idx, vis_c, F, FPS, rng, vis_graded),
             "a5c_boundaries": boundary_stats(b5, t, idx, vis_c, F, rng, vis_graded),
             "a10_boundaries": boundary_stats(b10, t, idx, vis_c, F, rng, vis_graded),
             "pathcrf_handovers": pathcrf_stats(pho, idx, vis_c, F, rng, vis_graded, p)}
        res["variants"][name] = v
        sr = []; sn = []
        for key, s in sims:
            ts = detect_touches(s.match, p); vm = (s.match.ball_present & s.match.graded).sum() / FPS / 60
            sr.append(len(ts) / vm)
        res["sim"][name] = {"per_visible_min_mean": float(np.mean(sr)), "per_visible_min_min": float(np.min(sr)),
                            "per_visible_min_max": float(np.max(sr))}
        print(name, v["n_touches"], round(v["per_visible_min"], 2), "sim", round(res["sim"][name]["per_visible_min_mean"], 2),
              "near2m", v["near_le_2m"], "tag", v["tag_handover"].get("share_gain_team_touch"), v["tag_handover"].get("chance_gain_team"),
              f"{time.time() - T0:.0f}s", flush=True)
    (OUT / "validation.json").write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
