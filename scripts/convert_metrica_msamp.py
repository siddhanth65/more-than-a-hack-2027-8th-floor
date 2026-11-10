"""Metrica public sample games 1, 2 (CSV) and 3 (EPTS + JSON) -> dfl_proxy npz halves in Metrica's own possession
convention (truth rule: mtah/io/metrica_truth.py), for the other-game validation set.

Untrusted-download hygiene: run with -I from a directory other than the download folders, paths as arguments:
  OMP_NUM_THREADS=2 python -I scripts/convert_metrica_msamp.py DOWNLOADS_DIR OUT_DIR [--games 1 2 3] [--blank-min-s 2.0]
DOWNLOADS_DIR holds metrica_sample_game{1,2,3}. Output halves game_9211/9212/9213_H{1,2} (games 1/2/3) and
fp1d_census.json in OUT_DIR. Prints/returns aggregates only.

Differences to scripts/convert_metrica_sample.py (kept for the training-side 9201 conversion): one shared truth module
for the three games; BALL LOST / BALL OUT / aerial duels / fouls do not move the owner (opponent RECOVERY / action
does); tackle / ground / dribble wins do; goal kicks count as restarts; dead additionally = ball blank for
>= blank_min_s (the sample data blanks the ball in stoppages).
Coordinates: games 1 and 2 are normalised 0..1 with no pitch size in the files -> 105 x 68 m assumed (identity
rescale to the simulator pitch); game 3 uses its metadata field size, rescaled to 105 x 68.
"""
import site
import sys
from pathlib import Path

site.addsitedir(site.getusersitepackages())
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "scripts"))

import argparse  # noqa: E402
import json  # noqa: E402
import re  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from convert_metrica_sample import read_meta, read_tracking  # noqa: E402
from mtah.io.extra_npz import (BLOCK, CROSS, DRIVE, HEADER, PASS, SHOT, TACKLE, THROW_IN, aggregate,  # noqa: E402
                               hda_labels, ltr_from_gk, merge_census, player_rows, to_dfl_metres, write_half)
from mtah.io.metrica_truth import (PITCH, derive_labels, events_from_csv_rows, norm_to_metres, parse_subs,  # noqa: E402
                                   spell_stats)

SYN = {1: 9211, 2: 9212, 3: 9213}


def sim_event_class(typ, subs):
    """(cls, poss_only) or None: the event classes the tag simulation reads (as convert_metrica_sample.event_class)."""
    subs = set(subs)
    if typ == "SET PIECE":
        return (THROW_IN, 0) if "THROW IN" in subs else (PASS, 1)
    if typ == "PASS":
        return (CROSS if "CROSS" in subs else HEADER if "HEAD" in subs else PASS), 0
    if typ == "SHOT":
        return SHOT, 0
    if typ == "CHALLENGE" and "WON" in subs and "FAULT" not in subs:
        return (HEADER, 0) if "AERIAL" in subs else (TACKLE, 0) if "TACKLE" in subs else (DRIVE, 1)
    if typ == "RECOVERY":
        return (BLOCK, 0) if ("INTERCEPTION" in subs or "BLOCKED" in subs) else (BLOCK, 1)
    if typ in ("BALL LOST", "BALL OUT"):
        return None if (subs & {"THEFT", "OFFSIDE"}) else (PASS, 0)
    if typ == "CARRY":
        return DRIVE, 1
    return None


def shirt_of(name):
    m = re.search(r"(\d+)", str(name))
    return int(m.group(1)) if m else 0


def load_csv_game(d, g):
    """-> frames (F,), period (F,), players list of dicts {team 'H'/'A', name, xy_norm (F,2)}, ball_norm (F,2), events."""
    pre = Path(d) / f"Sample_Game_{g}_"
    out = {}
    for side, team in (("Home", "H"), ("Away", "A")):
        df = pd.read_csv(f"{pre}RawTrackingData_{side}_Team.csv", header=2)
        out[team] = df
    h, a = out["H"], out["A"]
    if not (np.array_equal(h.Frame.to_numpy(), a.Frame.to_numpy()) and np.array_equal(h.Period.to_numpy(), a.Period.to_numpy())):
        raise ValueError("home / away tracking frames differ")
    frames = h.Frame.to_numpy().astype(np.int64)
    if not np.array_equal(frames, np.arange(frames[0], frames[0] + len(frames))):
        raise ValueError("tracking frames not contiguous")
    players = []
    for team, df in out.items():
        cols = list(df.columns)
        names = [c for c in cols[3:] if not c.startswith("Unnamed") and c.strip().lower() != "ball"]
        for nme in names:
            i = cols.index(nme)
            players.append({"team": team, "name": nme, "xy": df.iloc[:, i:i + 2].to_numpy(np.float64)})
    ib = list(h.columns).index("Ball")
    ball = h.iloc[:, ib:ib + 2].to_numpy(np.float64)
    if not np.array_equal(np.isnan(ball[:, 0]), np.isnan(a["Ball"].to_numpy(np.float64))):
        raise ValueError("home / away ball columns disagree")
    ev = pd.read_csv(f"{pre}RawEventsData.csv")
    return frames, h.Period.to_numpy().astype(int), players, ball, events_from_csv_rows(ev.to_dict("records"))


def load_epts_game(d):
    """Game 3 in the same shape as load_csv_game (normalised coordinates 0..1 per axis; pitch from metadata)."""
    d = Path(d)
    halves, pitch, teams, players_meta, specs = read_meta(d / "Sample_Game_3_metadata.xml")
    home_t = teams[0]
    tab = read_tracking(d / "Sample_Game_3_tracking.txt")
    frames = tab[:, 0].astype(np.int64)
    if not np.array_equal(frames, np.arange(frames[0], frames[0] + len(frames))):
        raise ValueError("tracking frames not contiguous")
    n_slots = (tab.shape[1] - 3) // 2
    period = np.zeros(len(frames), int)
    for i, (h0, h1) in enumerate(halves, start=1):
        period[(frames >= h0) & (frames <= h1)] = i
    pl, shirts = {}, {}
    for s0, s1, order in specs:
        if len(order) != n_slots:
            raise ValueError("spec / column count mismatch")
        rows = (frames >= s0) & (frames <= s1)
        for j, pid in enumerate(order):
            tm = "H" if players_meta[pid][0] == home_t else "A"
            # virtual slot = the player's index among his team's columns of this spec: a substitute takes the
            # leaver's column (as in the challenge export), so a team never exceeds 16 slots
            slot = f"{tm}{[q for q in order if players_meta[q][0] == players_meta[pid][0]].index(pid)}"
            a = pl.setdefault(slot, np.full((len(frames), 2), np.nan))
            a[rows] = tab[rows][:, 1 + 2 * j: 3 + 2 * j]
            shirts.setdefault(slot, players_meta[pid][1])
    players = [{"team": slot[0], "name": slot, "xy": xy, "shirt": shirts[slot]} for slot, xy in pl.items()]
    ball = tab[:, -2:]
    raw = json.loads((d / "Sample_Game_3_events.json").read_text(encoding="utf-8"))["data"]
    evs = []
    for e in raw:
        t = (e.get("team") or {}).get("id")
        if t not in teams:
            raise ValueError("event team not in metadata")
        subs = e.get("subtypes")
        subs = [] if subs is None else subs if isinstance(subs, list) else [subs]
        toks = tuple(tok for s in subs for tok in parse_subs(s["name"]))
        evs.append({"type": e["type"]["name"].upper(), "subs": toks, "team": "H" if t == home_t else "A",
                    "start": int(e["start"]["frame"]), "end": int(e["end"]["frame"]), "period": int(e["period"]),
                    "player": (e.get("from") or {}).get("id")})
    evs.sort(key=lambda e: (e["start"], e["end"]))
    return frames, period, players, ball, evs, halves, pitch


def convert_game(g, dl_dir, out_dir, blank_min_s=2.0):
    d = Path(dl_dir) / f"metrica_sample_game{g}"
    if g == 3:
        frames, period, players, ball_n, evs, halves, src_pitch = load_epts_game(d)
        halves_rng = halves
    else:
        frames, period, players, ball_n, evs = load_csv_game(d, g)
        halves_rng = [(int(frames[period == p].min()), int(frames[period == p].max())) for p in (1, 2)]
        src_pitch = PITCH
    ball_m_all = norm_to_metres(ball_n, src_pitch)
    per_half, sens, comp, census_halves = [], {}, [], {}
    for hi, (h0, h1) in enumerate(halves_rng, start=1):
        sel = (frames >= h0) & (frames <= h1)
        fr = frames[sel]
        ball = ball_m_all[sel]
        vis = ~np.isnan(ball[:, 0])
        hev = [e for e in evs if e["period"] == hi]
        res = derive_labels(hev, h0, h1, vis, 25.0, blank_min_s)
        # sensitivity of the truth statistics to the blank rule (aggregates only)
        for tag, kw in (("events_only", dict(blank_min_s=None)), ("blank1s", dict(blank_min_s=1.0)),
                        ("blank2s", dict(blank_min_s=2.0)), ("blank4s", dict(blank_min_s=4.0))):
            sens.setdefault(tag, []).append(derive_labels(hev, h0, h1, vis, 25.0, **kw)["labels"])
        # players of this half (>= 10 % of the half's frames tracked), slots = column order
        pm = [p for p in players if np.isfinite(p["xy"][sel, 0]).mean() >= 0.1]
        xm = {p["name"]: float(np.nanmedian(norm_to_metres(p["xy"][sel], src_pitch)[:, 0])) for p in pm}
        team_x = {t: float(np.mean([v for p in pm if p["team"] == t for v in [xm[p["name"]]]])) for t in "HA"}
        left = min(team_x, key=team_x.get)
        ltr = {left: 0, ("A" if left == "H" else "H"): 1}
        gk = {}
        for t in "HA":
            c = [p for p in pm if p["team"] == t]
            gk[t] = (min if ltr[t] == 0 else max)(c, key=lambda p: xm[p["name"]])["name"]
        raw_parts, det_parts = [], []
        for t in "HA":
            for j, p in enumerate([q for q in pm if q["team"] == t]):
                xy = norm_to_metres(p["xy"][sel], src_pitch)
                r, dd = player_rows(fr, xy, np.ones(len(fr), bool), (0 if t == "H" else 100) + j, ltr[t],
                                    p.get("shirt", shirt_of(p["name"])), 1 if p["name"] == gk[t] else 8, src_pitch)
                raw_parts.append(r); det_parts.append(dd)
        labels = res["labels"]
        status = (labels != "D").astype(np.int8)
        poss_ltr = np.array([ltr[c] for c in res["team"]], np.int8)      # owner also held through dead frames
        # tag-sim events: first scored class per frame, player slot unknown -> actor -1 (as the loaders accept)
        rows_by_fr = {}
        for e in hev:
            c = sim_event_class(e["type"], e["subs"])
            if c is None or not (h0 <= e["start"] <= h1):
                continue
            rows_by_fr.setdefault(e["start"], []).append(((e["start"], ltr[e["team"]], 0, c[0], 1, 0), c[1]))
        truth, poss_only, actor = [], [], []
        for f in sorted(rows_by_fr):
            grp = rows_by_fr[f]; sc = [x for x in grp if x[1] == 0]; x = sc[0] if sc else grp[0]
            truth.append(x[0]); poss_only.append(x[1]); actor.append(-1)
        key = f"game_{SYN[g]}_H{hi}"
        write_half(out_dir, key, raw_parts=raw_parts, det_parts=det_parts, ball_n=fr,
                   ball_xy=to_dfl_metres(ball, src_pitch), ball_z=np.full(len(fr), np.nan), status=status,
                   poss_ltr=poss_ltr, truth=truth, poss_only=poss_only, actor_pid=actor, ball_visible=vis,
                   interp=np.zeros(len(fr), bool))
        per_half.append({"labels": labels, "ball_visible": vis})
        comp.append({"frames": len(fr), "event_dead": float(res["event_dead"].mean()), "blank_dead": float(res["blank_dead"].mean()),
                     "dead": float(res["dead"].mean()), "ball_blank": float((~vis).mean()),
                     "blank_not_dead": float(((~vis) & ~res["dead"]).mean()), "dead_ball_visible": float((res["dead"] & vis).mean()),
                     "left_team": left, "gk": gk})
        census_halves[key] = {"section": f"half{hi}", "frames": int(len(fr)), "fps_from_T": 25.0,
                              "ltr": {"home": ltr["H"], "guest": ltr["A"]},
                              "ball_visible_share": round(float(vis.mean()), 4), "pitch_source": list(src_pitch)}
    merge_census(out_dir, f"metrica_sample_game{g}", {"match": f"metrica_sample_game{g}", "source": "metrica_sample",
                                                       "halves": census_halves})
    F = sum(c["frames"] for c in comp)
    wavg = lambda k: round(sum(c[k] * c["frames"] for c in comp) / F, 3)  # noqa: E731
    return {"game": g, "syn_id": SYN[g], "blank_min_s": blank_min_s,
            "truth": spell_stats([h["labels"] for h in per_half]),
            "dead_composition": {k: wavg(k) for k in ("event_dead", "blank_dead", "dead", "ball_blank", "blank_not_dead", "dead_ball_visible")},
            "visible_dead_share_of_visible": aggregate(per_half)["visible_dead_share"],
            "per_half_orientation": [{"left_team": c["left_team"], "gk": c["gk"]} for c in comp],
            "sensitivity": {k: spell_stats(v) for k, v in sens.items()}}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("downloads_dir"); ap.add_argument("out_dir")
    ap.add_argument("--games", type=int, nargs="+", default=[1, 2, 3])
    ap.add_argument("--blank-min-s", type=float, default=2.0)
    ap.add_argument("--json", default=None, help="write the aggregates here")
    args = ap.parse_args()
    out = Path(args.out_dir); out.mkdir(parents=True, exist_ok=True)
    res = [convert_game(g, args.downloads_dir, out, args.blank_min_s) for g in args.games]
    s = json.dumps(res, indent=1)
    if args.json:
        Path(args.json).write_text(s)
    print(s)


if __name__ == "__main__":
    main()
