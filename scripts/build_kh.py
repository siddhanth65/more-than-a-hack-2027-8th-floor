"""Task K: SkillCorner natural-visibility hold-out (40 halves). Blackouts are SkillCorner's own: the ball is missing
(NaN) wherever the source frame had ball is_detected False; NO DFL blackout simulator, no player corruption.
Players keep SkillCorner's (partly extrapolated) positions; frames with no detected player at all (replays, close-ups)
become X (ungraded) in every truth variant. Truth = the usual variants (ctrl_Dfull is the ctrl convention).
Tags are simulated from the source events with the sim tag model (SkillCorner has no tagging feed).

  python scripts/build_kh.py [--src runs/extra] [--out runs/kh/sim] [--seed 100]
Writes game_91NN_Hk_s<seed>.* (the dataset.save_half format) and runs/kh/holdout_aggregates.json (aggregates only).
"""
import argparse, json, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np

from mtah.eval.drift import blackout_mask_from_match
from mtah.io.dfl_events import truth_variants
from mtah.io.dfl_proxy import load_events, load_half
from mtah.io.extra_npz import spells
from mtah.io.match import Match, speeds_from_positions
from mtah.sim import half_rng
from mtah.sim.blackout_sim import runs_of
from mtah.sim.dataset import SimHalf, half_index, save_half
from mtah.sim.tag_sim import TagSimParams, simulate_tags

LONG_S = 10.0


def build(npz, census, seed, tp):
    key = Path(npz).stem
    m, labels = load_half(npz, census)
    events = load_events(npz, census)
    with np.load(npz) as z:
        vis = z["ball_visible"].astype(bool)
        raw = z["raw"]; det = z["raw_detected"].astype(bool); n0 = int(z["ball_n"][0])
    F = m.n_frames
    ndet = np.bincount((raw[:, 0].astype(np.int64) - n0)[det], minlength=F)[:F]
    no_players = ndet == 0
    ball = m.ball.copy(); ball[~vis] = np.nan
    sim = Match(fps=m.fps, pos=m.pos, speed=m.speed, team=m.team, ball=ball,
                ball_speed=speeds_from_positions(ball, m.fps), half=m.half, pitch=m.pitch)
    truth = truth_variants(labels, events, m.ball, m.ball_speed, m.fps, m.pitch)   # clean ball: dead-run narrowing only
    for k in truth:
        t = truth[k].copy(); t[no_players] = "X"; truth[k] = t
    mask = blackout_mask_from_match(sim, LONG_S)
    rng = half_rng(seed, half_index(key))
    tags = simulate_tags(sim, labels, events, mask, tp, rng, raw_ball=m.ball)
    h = SimHalf(key=key, seed=seed, match=sim, tags=tags, blackout=mask, truth=truth, events=events)
    extra = {"ndet_mean": float(ndet.mean()), "ndet_frac_players": float((ndet / 22.0).mean())}
    return h, no_players, extra


def half_stats(h, no_players, extra, fps=25.0):
    t = h.truth["ctrl_Dfull"]; g = t != "X"
    miss = np.isnan(h.match.ball[:, 0])
    mask = h.blackout
    gaps = np.array([e - s for s, e in runs_of(miss)], float) / fps
    lg = gaps[gaps > LONG_S]
    gm = g.sum() / fps / 60
    D = t == "D"
    sp = spells(t)
    return {"key": h.key, "frames": int(len(t)), "graded_min": round(float(gm), 3), "x_share": round(float(no_players.mean()), 4),
            "ball_missing_share_graded": round(float(miss[g].mean()), 4),
            "blackout_share_graded": round(float(mask[g].mean()), 4),
            "n_gaps_gt10s": int(len(lg)), "gap_gt10s_median_s": float(np.median(lg)) if len(lg) else None,
            "gap_gt10s_p90_s": float(np.percentile(lg, 90)) if len(lg) else None,
            "n_gaps_all": int(len(gaps)),
            "dead_share_in_gaps": float(D[g & mask].mean()) if (g & mask).any() else None,
            "dead_share_out_gaps": float(D[g & ~mask].mean()) if (g & ~mask).any() else None,
            "dead_share_all": float(D[g].mean()), "spells": len(sp), "spells_per_graded_min": len(sp) / gm if gm else None,
            "players_detected_share": round(extra["ndet_frac_players"], 4)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default=str(ROOT / "runs" / "extra")); ap.add_argument("--out", default=str(ROOT / "runs" / "kh" / "sim"))
    ap.add_argument("--seed", type=int, default=100)
    a = ap.parse_args()
    tp = TagSimParams.from_dict(json.loads((ROOT / "configs" / "tag_sim.json").read_text()))
    src = Path(a.src); census = src / "fp1d_census.json"
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    rows = []
    for p in sorted(src.glob("game_91*_H*.npz")):
        h, nop, extra = build(p, census, a.seed, tp)
        save_half(h, out)
        rows.append(half_stats(h, nop, extra))
        print(rows[-1]["key"], "ok", flush=True)
    Path(a.out).parent.joinpath("holdout_halves.json").write_text(json.dumps(rows, indent=1))
    print(len(rows), "halves")


if __name__ == "__main__":
    main()
