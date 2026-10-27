"""Metrica Sample Game 3 (EPTS 25 fps + metadata xml + events json) -> dfl_proxy npz halves game_9201_H{1,2}.

Untrusted-download hygiene: run with -I from a directory other than the download folder, paths as arguments:
  OMP_NUM_THREADS=2 python -I scripts/convert_metrica_sample.py GAME3_DIR OUT_DIR
Prints aggregates only.

Label rule (per tracking frame, from the events; Metrica has no per-frame flag):
  team   = team of the latest possession event started at or before the frame. Possession events: PASS, CARRY,
           RECOVERY, SHOT, SET PIECE, BALL LOST, BALL OUT (CHALLENGE and FAULT RECEIVED do not move possession).
  D      = [trigger, start of the next SET PIECE) where trigger is
             BALL OUT (its end frame), SHOT with subtype OUT or GOAL (end frame),
             a foul (CHALLENGE with FAULT, FAULT RECEIVED) or BALL LOST OFFSIDE / HAND BALL / REFEREE HIT (start frame),
           kept only when the first possession event starting after the trigger frame is that SET PIECE
           (otherwise play went on: advantage, or an out the tagger did not close with a restart);
         plus each half's head up to its first SET PIECE (kick-off) and its tail after the last event's end frame.
Events table: one row per frame (scored classes preferred), classes as in parse_dfl_match: THROW_IN for SET PIECE
THROW IN, PASS (CROSS / HEAD subtypes -> CROSS / HEADER), SHOT, CHALLENGE AERIAL WON -> HEADER, TACKLE WON ->
TACKLE, RECOVERY INTERCEPTION/BLOCKED -> BLOCK (other RECOVERY -> poss-only BLOCK), BALL LOST / BALL OUT -> PASS
(a failed or out-of-play kick), CARRY / CHALLENGE GROUND WON -> poss-only DRIVE.
"""
import site
import sys
from pathlib import Path

site.addsitedir(site.getusersitepackages())
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import argparse  # noqa: E402
import io  # noqa: E402
import json  # noqa: E402
import xml.etree.ElementTree as ET  # noqa: E402,N817

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from mtah.io.extra_npz import (BLOCK, CROSS, DRIVE, HEADER, PASS, SHOT, TACKLE, THROW_IN, aggregate,  # noqa: E402
                               hda_labels, ltr_from_gk, merge_census, player_rows, to_dfl_metres, write_half)

SYN = 9201
POSS_TYPES = {"PASS", "CARRY", "RECOVERY", "SHOT", "SET PIECE", "BALL LOST", "BALL OUT"}


def read_meta(xml_path):
    root = ET.parse(xml_path).getroot()
    gp = {p.findtext("Name"): p.findtext("Value") for p in root.iter("ProviderParameter")}
    halves = [(int(gp["first_half_start"]), int(gp["first_half_end"])),
              (int(gp["second_half_start"]), int(gp["second_half_end"]))]
    fs = root.find(".//FieldSize")
    pitch = (float(fs.findtext("Width")), float(fs.findtext("Height")))
    teams = [t.get("id") for t in root.iter("Team")]
    players = {}
    for p in root.iter("Player"):
        prm = {q.findtext("Name"): q.findtext("Value") for q in p.iter("ProviderParameter")}
        players[p.get("id")] = (p.get("teamId"), int(p.findtext("ShirtNumber") or 0),
                                1 if prm.get("position_type") == "Goalkeeper" else 8)
    chan = {c.get("id"): c.get("playerId") for c in root.iter("PlayerChannel")}
    specs = []
    for s in root.iter("DataFormatSpecification"):
        order = []
        for r in s.iter("PlayerChannelRef"):
            pid = chan[r.get("playerChannelId")]
            if not order or order[-1] != pid:
                order.append(pid)
        specs.append((int(s.get("startFrame")), int(s.get("endFrame")), order))
    return halves, pitch, teams, players, specs


def read_tracking(txt_path):
    text = Path(txt_path).read_text().replace(":", ",").replace(";", ",")
    return pd.read_csv(io.StringIO(text), header=None, na_values=["NaN"], dtype=np.float64).to_numpy()


def _subs(e):
    s = e.get("subtypes")
    if s is None:
        return ()
    return tuple(x["name"] for x in s) if isinstance(s, list) else (s["name"],)


def event_class(typ, subs, won):
    """(cls, poss_only) or None."""
    if typ == "SET PIECE":
        return (THROW_IN, 0) if "THROW IN" in subs else (PASS, 1)
    if typ == "PASS":
        return (CROSS if "CROSS" in subs else HEADER if "HEAD" in subs else PASS), 0
    if typ == "SHOT":
        return SHOT, 0
    if typ == "CHALLENGE" and won:
        if "AERIAL" in subs:
            return HEADER, 0
        if "TACKLE" in subs:
            return TACKLE, 0
        return DRIVE, 1
    if typ == "RECOVERY":
        return (BLOCK, 0) if ("INTERCEPTION" in subs or "BLOCKED" in subs) else (BLOCK, 1)
    if typ in ("BALL LOST", "BALL OUT"):
        return (None if "THEFT" in subs or "OFFSIDE" in subs else (PASS, 0))
    if typ == "CARRY":
        return DRIVE, 1
    return None


def dead_mask(evs, h0, h1):
    """(F,) bool over frames h0..h1 (inclusive) per the module docstring rule."""
    F = h1 - h0 + 1
    dead = np.zeros(F, bool)
    poss = [e for e in evs if e["type"] in POSS_TYPES]
    sp = [e for e in poss if e["type"] == "SET PIECE"]
    if sp:
        dead[:max(sp[0]["start"] - h0, 0)] = True
    last_end = max(e["end"] for e in evs)
    dead[max(last_end - h0 + 1, 0):] = True
    for e in evs:
        typ, subs = e["type"], e["subs"]
        if typ == "BALL OUT" or (typ == "SHOT" and ("OUT" in subs or "GOAL" in subs)):
            trig, ref = e["end"], e["start"]
        elif (typ == "CHALLENGE" and "FAULT" in subs) or typ == "FAULT RECEIVED" or \
                (typ == "BALL LOST" and set(subs) & {"OFFSIDE", "HAND BALL", "REFEREE HIT"}):
            trig, ref = e["start"], e["start"]
        else:
            continue
        nxt = next((q for q in poss if q["start"] > ref and q is not e), None)
        if nxt is None or nxt["type"] != "SET PIECE" or nxt["start"] <= trig:
            continue
        dead[max(trig - h0, 0):max(nxt["start"] - h0, 0)] = True
    return dead


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("game3_dir")
    ap.add_argument("out_dir")
    args = ap.parse_args()
    d = Path(args.game3_dir); out = Path(args.out_dir); out.mkdir(parents=True, exist_ok=True)
    halves, pitch, teams, players, specs = read_meta(d / "Sample_Game_3_metadata.xml")
    home_t = teams[0]
    tab = read_tracking(d / "Sample_Game_3_tracking.txt")
    frames = tab[:, 0].astype(np.int64)
    if not np.array_equal(frames, np.arange(frames[0], frames[0] + len(frames))):
        raise ValueError("tracking frames not contiguous")
    n_slots = (tab.shape[1] - 3) // 2
    L, W = pitch
    # per-player full-length arrays in centred metres
    pxy = {pid: np.full((len(frames), 2), np.nan) for pid in players}
    for s0, s1, order in specs:
        if len(order) != n_slots:
            raise ValueError("spec / column count mismatch")
        rows = (frames >= s0) & (frames <= s1)
        for j, pid in enumerate(order):
            pxy[pid][rows, 0] = (tab[rows, 1 + 2 * j] - 0.5) * L
            pxy[pid][rows, 1] = (tab[rows, 2 + 2 * j] - 0.5) * W
    ball_all = np.c_[(tab[:, -2] - 0.5) * L, (tab[:, -1] - 0.5) * W]
    raw_ev = json.loads((d / "Sample_Game_3_events.json").read_text(encoding="utf-8"))["data"]
    evs = []
    for e in raw_ev:
        team = (e.get("team") or {}).get("id")
        if team not in teams:
            raise ValueError("event team not in metadata")
        subs = _subs(e)
        evs.append({"type": e["type"]["name"], "subs": subs, "team": team, "start": int(e["start"]["frame"]),
                    "end": int(e["end"]["frame"]), "period": int(e["period"]),
                    "player": (e.get("from") or {}).get("id"), "won": "WON" in subs})
    evs.sort(key=lambda e: (e["start"], e["end"]))
    per_half = []
    vs = {}
    for s0, _s1, order in specs:
        for pid in order:
            tm = players[pid][0]
            vs[(s0, pid)] = (100 if tm == home_t else 200) + [q for q in order if players[q][0] == tm].index(pid)

    def vslot(s0, pid):
        return vs[(s0, pid)]
    census = {"match": "metrica_sample_game3", "source": "metrica_sample", "halves": {}}
    for hi, (h0, h1) in enumerate(halves, start=1):
        sel = (frames >= h0) & (frames <= h1)
        fr = frames[sel]
        hev = [e for e in evs if e["period"] == hi]
        dead = dead_mask(hev, h0, h1)
        # team: latest possession event started at or before the frame
        team = np.full(len(fr), None, object)
        pe = [e for e in hev if e["type"] in POSS_TYPES]
        for e, nxt in zip(pe, pe[1:] + [None]):
            a = max(e["start"] - h0, 0); b = (nxt["start"] - h0) if nxt else len(fr)
            team[a:b] = e["team"]
        first = next(t for t in team if t is not None)
        team = np.array([t if t is not None else first for t in team], object)
        gk = {}
        for pid, (tm, _, role) in players.items():
            if role == 1 and not np.isnan(pxy[pid][sel, 0]).all():
                gk.setdefault(tm, []).append(np.nanmean(pxy[pid][sel, 0]))
        gkm = {t: float(np.mean(v)) for t, v in gk.items()}
        ltr = ltr_from_gk(gkm)
        # slots follow the EPTS columns (one player per column at a time, substitutes take the leaver's
        # column), as in the challenge export; pid = virtual slot id so a team never exceeds 16 slots
        raw_parts, det_parts = [], []
        for s0, s1, order in specs:
            rows = sel & (frames >= s0) & (frames <= s1)
            if not rows.any():
                continue
            for pid in order:
                tm, shirt, role = players[pid]
                r, dd = player_rows(frames[rows], pxy[pid][rows], np.ones(int(rows.sum()), bool),
                                    vslot(s0, pid), ltr[tm], shirt, role, pitch)
                raw_parts.append(r); det_parts.append(dd)
        ball = ball_all[sel]
        vis = ~np.isnan(ball[:, 0])
        status = (~dead).astype(np.int8)
        poss_ltr = np.array([ltr[t] for t in team], np.int8)
        rows_by_fr = {}
        for e in hev:
            c = event_class(e["type"], e["subs"], e["won"])
            if c is None or e["player"] not in players or players[e["player"]][0] != e["team"]:
                continue
            if not (h0 <= e["start"] <= h1):
                continue
            pid = e["player"]
            s0 = next((a for a, b, o in specs if a <= e["start"] <= b and pid in o), None)
            if s0 is None:
                continue
            rows_by_fr.setdefault(e["start"], []).append(
                ((e["start"], ltr[e["team"]], players[pid][1], c[0], 1, 0), c[1], vslot(s0, pid)))
        truth, poss_only, actor = [], [], []
        for f in sorted(rows_by_fr):
            grp = rows_by_fr[f]
            sc = [x for x in grp if x[1] == 0]
            x = sc[0] if sc else grp[0]
            truth.append(x[0]); poss_only.append(x[1]); actor.append(x[2])
        key = f"game_{SYN}_H{hi}"
        write_half(out, key, raw_parts=raw_parts, det_parts=det_parts, ball_n=fr,
                   ball_xy=to_dfl_metres(ball, pitch), ball_z=np.full(len(fr), np.nan), status=status,
                   poss_ltr=poss_ltr, truth=truth, poss_only=poss_only, actor_pid=actor, ball_visible=vis,
                   interp=np.zeros(len(fr), bool))
        labels = hda_labels(status, poss_ltr, ltr[home_t])
        per_half.append({"labels": labels, "ball_visible": vis})
        census["halves"][key] = {
            "section": f"half{hi}", "frames": int(len(fr)), "fps_from_T": 25.0,
            "n_players_tracked": len({p for a, b, o in specs for p in o if a <= h1 and b >= h0}),
            "gk_mean_xn": {("home" if t == home_t else "guest"): round((v + L / 2) / L, 4) for t, v in gkm.items()},
            "ltr": {("home" if t == home_t else "guest"): v for t, v in ltr.items()},
            "ball_alive_share": round(float(status.mean()), 4), "events_kept": len(truth),
            "ball_visible_share": round(float(vis.mean()), 4), "pitch_source": list(pitch),
        }
    merge_census(out, census["match"], census)
    print(json.dumps({"metrica_sample_game3": aggregate(per_half)}, indent=1))


if __name__ == "__main__":
    main()
