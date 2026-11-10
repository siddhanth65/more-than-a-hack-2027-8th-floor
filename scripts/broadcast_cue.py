"""Broadcast replay / close-up cue (Plan 3r): thin CLI over mtah.evidence.replay (the cue definition, clock mapping
and thresholds are documented there; this script and the pipeline share that one implementation).
  --cue-only   compute and write runs/broadcast/cue.npz only (no validation; what reproduce.ps1 runs)
  default      also the label-free validation aggregates -> runs/broadcast/cue_validation.json (exploratory T1 run;
               section f reads the public window and section d the a5c candidate CSV)
Writes the per-frame cue to runs/broadcast/cue.npz (gitignored) and aggregates only to the json. Never prints
frame-level data."""
import argparse, json, os, time
from pathlib import Path

import numpy as np
import pandas as pd

from mtah.eval.drift import blackout_mask_from_match
from mtah.evidence.replay import FPS, compute_cue
from mtah.io.metrica_loader import load_broadcast, load_tactical
from mtah.io.tags import load_tags
from mtah.sim.blackout_sim import runs_of

D = Path(os.environ.get("MTAH_DATA_DIR", os.path.join(os.path.expanduser("~"), "mtah-data")))
OUT = Path(__file__).resolve().parents[1] / "runs" / "broadcast"
GT = Path(os.environ.get("MTAH_PUBLIC_GT", "BAR-ATH_possession_gt_public.csv"))
CAND = Path(__file__).resolve().parents[1] / "runs" / "candidates" / "a5c.csv"
PUB = (11185, 18685)
SP_TYPES = {"goal_kick": "Goal Kick", "corner": "Corner Kick", "throw_in": "Throw-in", "free_kick": "Free Kick",
            "kick_off": "Kick Off"}


def share(x, m):
    m = m & np.isfinite(x) if x.dtype.kind == "f" else m
    return {"share": float(x[m].mean()) if m.any() else None, "n": int(m.sum())}


def runlen_q(mask):
    L = np.array([(e - s) / FPS for s, e in runs_of(mask)])
    if not len(L):
        return {"n": 0}
    return {"n": int(len(L)), "median_s": float(np.median(L)), "p90_s": float(np.percentile(L, 90)),
            "p10_s": float(np.percentile(L, 10)), "max_s": float(L.max())}


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--no-pixel", action="store_true")
    ap.add_argument("--cue-only", action="store_true", help="write runs/broadcast/cue.npz only (no validation)")
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    T0 = time.time()
    if args.cue_only:
        compute_cue(D, cache=OUT / "cue.npz", pixel=not args.no_pixel)
        print(f"wrote {OUT / 'cue.npz'} ({time.time() - T0:.0f}s)")
        return
    b = load_broadcast(D); t = load_tactical(D); tags = load_tags(D); Ft = t.n_frames
    a, res = compute_cue(D, cache=OUT / "cue.npz", pixel=not args.no_pixel, b=b, t=t)
    has_b, hom_valid, in_sync, in_sync_strict = a["has_broadcast"], a["hom_valid"], a["in_sync"], a["in_sync_strict"]
    rl, rl_s, lowg = a["replay_like"], a["replay_like_s"], a["low_green"]
    green_b = lowg   # the low_green cue is validated iff the pixel proxy ran

    # ---- validation (aggregates only)
    g = t.graded; blk = blackout_mask_from_match(t); vis = t.ball_present
    U = g & has_b
    res["coverage"] = {"graded_with_broadcast_share": float(U.sum() / g.sum()), "n_frames_U": int(U.sum())}
    cues = {"raw": rl, "smooth": rl_s, "hom_invalid": has_b & ~hom_valid, "out_of_sync": has_b & ~in_sync,
            "out_of_sync_strict": has_b & ~in_sync_strict}
    if green_b is not None:
        cues["low_green"] = lowg
    res["overall_share_U"] = {c: float(x[U].mean()) for c, x in cues.items()}
    # (a)
    res["a_blackout"] = {c: {"in_blackout": float(x[U & blk].mean()), "outside_blackout": float(x[U & ~blk].mean()),
                             "visible_ball": float(x[U & vis].mean())} for c, x in cues.items()}
    # (b) 30 s before SET PIECES starts
    sp = tags[tags.code == "SET PIECES"]
    pre_all = np.zeros(Ft, bool); rb = {}
    for key, pref in SP_TYPES.items():
        m = np.zeros(Ft, bool); n = 0
        for s in sp[sp["type"].fillna("").str.startswith(pref)]["start"]:
            a0 = max(0, s - 30 * FPS); m[a0:max(a0, s)] = True; n += 1
        pre_all |= m
        rb[key] = {"n_tags": n, **{c: float(x[U & m].mean()) if (U & m).any() else None for c, x in cues.items()},
                   "n_frames": int((U & m).sum())}
    rb["all_setpieces"] = {c: float(x[U & pre_all].mean()) for c, x in cues.items()}
    rb["elsewhere"] = {c: float(x[U & ~pre_all].mean()) for c, x in cues.items()}
    rb["elsewhere_visible_ball"] = {c: float(x[U & ~pre_all & vis].mean()) for c, x in cues.items()}
    res["b_pre_setpiece_30s"] = rb
    # (c) BUILD UP / PROGRESSION mid-tag (central 50 % of each tag)
    rc = {}
    for code in ("BUILD UP", "PROGRESSION"):
        m = np.zeros(Ft, bool)
        for r in tags[tags.code == code].itertuples(index=False):
            q = (r.end - r.start) // 4; m[r.start + q:r.end - q] = True
        rc[code] = {"n_frames": int((U & m).sum()), **{c: float(x[U & m].mean()) for c, x in cues.items()}}
    res["c_live_midtag"] = rc
    # (d) a5c D vs H/A on visible-ball frames
    a5 = pd.read_csv(CAND)["label"].to_numpy().astype("<U1")
    Vd = U & vis
    res["d_a5c_visible"] = {c: {"P_rl_given_D": float(x[Vd & (a5 == "D")].mean()),
                                "P_rl_given_HA": float(x[Vd & (a5 != "D")].mean()),
                                "n_D": int((Vd & (a5 == "D")).sum()), "n_HA": int((Vd & (a5 != "D")).sum())}
                            for c, x in cues.items()}
    res["d_a5c_visible_lift_raw_smooth"] = {c: res["d_a5c_visible"][c]["P_rl_given_D"] / max(1e-9, res["d_a5c_visible"][c]["P_rl_given_HA"]) for c in ("raw", "smooth")}
    # (e) run lengths
    res["e_runlengths"] = {"raw": runlen_q(rl & g), "smooth": runlen_q(rl_s & g)}
    # (f) public window, aggregates only, primary cue = smooth
    gt = pd.read_csv(GT)["label"].to_numpy().astype("<U1")
    W = np.zeros(Ft, bool); W[PUB[0]:PUB[1]] = True; W &= has_b
    res["f_public_smooth"] = {"truthD_visible_share_rl": float(rl_s[W & (gt == "D") & vis].mean()),
                              "truthHA_share_rl": float(rl_s[W & np.isin(gt, ["H", "A"])].mean())}
    res["runtime_s"] = time.time() - T0
    (OUT / "cue_validation.json").write_text(json.dumps(res, indent=1))
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
