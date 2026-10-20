"""Standalone DFL/Sportec match parser -> mtah dfl_proxy npz layout (raw, ball_*, plus the event arrays truth, poss_only, actor_pid).

Usage: python scripts/parse_dfl_match.py DFL-MAT-J03WR9 RAW_DIR OUT_DIR [syn_id]
Logic copied from football-synthesizer/tools/tactical/fp1d_dfl.py (build_match), tracking part only.
"""
from __future__ import annotations

import json
import re
import sys
import xml.etree.ElementTree as ET  # noqa: N817
from datetime import datetime
from pathlib import Path

import numpy as np

SECTIONS = ("firstHalf", "secondHalf")
ROLE_MAP = {"TW": 1, "LV": 2, "IVL": 3, "IVZ": 4, "IVR": 5, "RV": 13,
            "LM": 6, "DLM": 6, "HL": 6, "RM": 7, "DRM": 7, "HR": 7,
            "DML": 8, "DMR": 8, "DMZ": 8, "ZO": 9,
            "OLM": 10, "LA": 10, "ORM": 11, "RA": 11, "STL": 12, "STR": 12, "STZ": 12}
_FS = re.compile(rb'<FrameSet GameSection="(\w+)" MatchId="([^"]+)" TeamId="([^"]+)" '
                 rb'PersonId="([^"]+)"')
_FR = re.compile(rb'<Frame N="(\d+)" T="([^"]+)" X="([-\d.]+)" Y="([-\d.]+)"')
_BALL = re.compile(rb'Z="([-\d.]+)".*BallPossession="(\d)" BallStatus="(\d)"')


def paths(raw: Path, mid: str) -> dict:
    hits = {}
    for k, pat in (("info", "DFL_02_01_matchinformation_*"), ("events", "DFL_03_02_events_raw_*"),
                   ("pos", "DFL_04_03_positions_raw_observed_*")):
        f = list(Path(raw).glob(f"{pat}{mid}.xml"))
        assert len(f) == 1, (k, f)
        hits[k] = f[0]
    return hits


def _ms(ts):
    return int(round(datetime.fromisoformat(ts).timestamp() * 1000))


def load_info(p):
    r = ET.parse(p["info"]).getroot()
    teams, people = {}, {}
    for t in r.iter("Team"):
        if t.get("TeamId") and t.get("Role") in ("home", "guest"):
            teams[t.get("TeamId")] = t.get("Role")
            for pl in t.iter("Player"):
                people[pl.get("PersonId")] = (t.get("TeamId"), int(pl.get("ShirtNumber")),
                                              pl.get("PlayingPosition"))
    assert sorted(teams.values()) == ["guest", "home"]
    return teams, people


def parse_positions(p, mid):
    out = {s: {"players": {}, "ball": None} for s in SECTIONS}
    cur, buf = None, []
    with open(p["pos"], "rb") as f:
        for line in f:
            if b"<Frame " in line:
                if cur is None:
                    continue
                m = _FR.search(line)
                if cur[2]:
                    b = _BALL.search(line)
                    buf.append((int(m.group(1)), _ms(m.group(2).decode()), float(m.group(3)),
                                float(m.group(4)), float(b.group(1)), int(b.group(2)),
                                int(b.group(3))))
                else:
                    buf.append((int(m.group(1)), float(m.group(3)), float(m.group(4))))
                continue
            m = _FS.search(line)
            if m:
                assert m.group(2).decode() == mid
                sec, team, pid = m.group(1).decode(), m.group(3).decode(), m.group(4).decode()
                cur = (sec, team, team == "BALL", pid) if sec in SECTIONS else None
                buf = []
                continue
            if b"</FrameSet>" in line and cur is not None:
                sec, team, is_ball, pid = cur
                arr = np.array(buf, dtype=np.float64)
                if is_ball:
                    out[sec]["ball"] = arr
                elif team != "referee":
                    out[sec]["players"][pid] = (team, arr)
                cur, buf = None, []
    return out

DRIVE, PASS, CROSS, THROW_IN, SHOT, HEADER, TACKLE, BLOCK = range(1, 9)
WRAPPERS = ("KickOff", "FreeKick", "CornerKick", "GoalKick", "ThrowIn", "Penalty")


def map_event(ev):
    """(kind, class, actor PersonId, actor TeamId, rule) or None; copied from fp1d_dfl.map_event."""
    kids = list(ev)
    if len(kids) != 1:
        return None
    top = kids[0]
    wrapper = top.tag if top.tag in WRAPPERS else None
    inner = top
    if wrapper:
        sub = [c for c in top if c.tag in ("Play", "ShotAtGoal")]
        if not sub:
            return None
        inner = sub[0]
    if inner.tag == "Play":
        kid = {c.tag for c in inner}
        if wrapper == "ThrowIn":
            cls, rule = THROW_IN, "ThrowIn/Play -> throw-in"
        elif "Cross" in kid:
            cls, rule = CROSS, f"{wrapper or ''}/Play/Cross -> cross"
        else:
            cls, rule = PASS, f"{wrapper or ''}/Play/{'Pass' if 'Pass' in kid else 'other'} -> pass"
        return "SCORED", cls, inner.get("Player"), inner.get("Team"), rule
    if inner.tag == "ShotAtGoal":
        return "SCORED", SHOT, inner.get("Player"), inner.get("Team"), "ShotAtGoal -> shot"
    if inner.tag == "TacklingGame":
        res, role, typ = inner.get("WinnerResult"), inner.get("WinnerRole"), inner.get("Type")
        who, tm = inner.get("Winner"), inner.get("WinnerTeam")
        if res == "fouled":
            return None
        if typ == "air":
            return "SCORED", HEADER, who, tm, "TacklingGame air (winner) -> header"
        if role == "withoutBallControl":
            return "SCORED", TACKLE, who, tm, "TacklingGame ground won w/o ball -> tackle"
        return "POSS", DRIVE, who, tm, "TacklingGame ground won with ball -> poss-only"
    if inner.tag == "BallClaiming":
        typ = inner.get("Type")
        if typ in ("InterceptedBall", "Block"):
            return "SCORED", BLOCK, inner.get("Player"), inner.get("Team"), f"BallClaiming {typ} -> block"
        return "POSS", BLOCK, inner.get("Player"), inner.get("Team"), f"BallClaiming {typ} -> poss-only opener"
    if inner.tag == "OtherBallAction":
        return "POSS", DRIVE, inner.get("Player"), inner.get("Team"), "OtherBallAction -> poss-only"
    return None


def parse_events(p, mid):
    r = ET.parse(p["events"]).getroot()
    out = []
    for ev in r.iter("Event"):
        assert ev.get("MatchId") == mid
        m = map_event(ev)
        if m is None:
            continue
        kind, cls, actor, team, rule = m
        out.append({"id": int(ev.get("EventId")), "t": _ms(ev.get("EventTime")), "kind": kind,
                    "cls": cls, "actor": actor, "team": team})
    out.sort(key=lambda e: (e["t"], e["id"]))
    return out


def build(mid, raw_dir, out_dir, syn_id=9007):
    p = paths(raw_dir, mid)
    teams, people = load_info(p)
    pos = parse_positions(p, mid)
    evs = parse_events(p, mid)
    r = ET.parse(p["events"]).getroot()
    sub_pos, sub_out = {}, {}
    for s in r.iter("Substitution"):
        sub_pos[s.get("PlayerIn")] = s.get("PlayingPosition")
        sub_out[s.get("PlayerIn")] = s.get("PlayerOut")
    club_code = {t: (1 if role == "home" else 2) for t, role in teams.items()}
    census = {"match": mid, "halves": {}}
    pid_index = {}
    for hi, sec in enumerate(SECTIONS, start=1):
        ball = pos[sec]["ball"]
        bn = ball[:, 0].astype(np.int64)
        assert np.all(np.diff(bn) == 1)
        fps = 1000.0 / np.median(np.diff(ball[:, 1]))
        rows, gk_x = [], {t: [] for t in teams}
        for pd_, (team, arr) in pos[sec]["players"].items():
            team_, shirt, posn = people[pd_]
            assert team_ == team
            posn = posn or sub_pos.get(pd_)
            if posn is None and pd_ in sub_out and sub_out[pd_] in people:
                posn = people[sub_out[pd_]][2]
            role = ROLE_MAP.get(posn or "", 8)
            if pd_ not in pid_index:
                pid_index[pd_] = club_code[team] * 100 + len(
                    [q for q in pid_index if people[q][0] == team])
            xn = (arr[:, 1] + 52.5) / 105.0
            yn = (arr[:, 2] + 34.0) / 68.0
            if role == 1:
                gk_x[team].append(xn)
            rows.append((arr[:, 0], pid_index[pd_], team, shirt, role, xn, yn))
        gkm = {t: float(np.concatenate(v).mean()) for t, v in gk_x.items() if v}
        assert len(gkm) == 2
        lo = min(gkm, key=gkm.get)
        ltr = {t: (0 if t == lo else 1) for t in teams}
        assert abs(gkm[lo] - max(gkm.values())) > 0.3
        n_rows = sum(len(x[0]) for x in rows)
        rawa = np.full((n_rows, 14), np.nan, np.float32)
        i = 0
        for fr, pidx, team, shirt, role, xn, yn in rows:
            n = len(fr)
            rawa[i:i + n, 0] = fr; rawa[i:i + n, 1] = pidx; rawa[i:i + n, 2] = ltr[team]
            rawa[i:i + n, 3] = shirt; rawa[i:i + n, 4] = role; rawa[i:i + n, 5] = xn
            rawa[i:i + n, 6] = yn; rawa[i:i + n, 9] = 1.0
            i += n
        home_t = [t for t, rl in teams.items() if rl == "home"][0]
        guest_t = [t for t, rl in teams.items() if rl == "guest"][0]
        ltr_poss = np.where(ball[:, 5] == 1, ltr[home_t], np.where(ball[:, 5] == 2, ltr[guest_t], -1))
        t0, t1 = ball[0, 1], ball[-1, 1]
        sel = [e for e in evs if t0 - 2000 <= e["t"] <= t1 + 2000]
        et = np.array([e["t"] for e in sel], np.float64)
        fidx = np.clip(np.searchsorted(ball[:, 1], et), 1, len(ball) - 1)
        prev_closer = np.abs(ball[fidx - 1, 1] - et) <= np.abs(ball[fidx, 1] - et)
        fidx = np.where(prev_closer, fidx - 1, fidx)
        present = {pd_: set(arr[:, 0].astype(np.int64).tolist())
                   for pd_, (_t, arr) in pos[sec]["players"].items()}
        cand = []
        for e, fi in zip(sel, fidx):
            fr = int(bn[fi]); pd_ = e["actor"]
            if pd_ not in present or fr not in present[pd_] or people[pd_][0] != e["team"]:
                continue
            cand.append((fr, ltr[e["team"]], people[pd_][1], e["cls"], 1, 0,
                         0 if e["kind"] == "SCORED" else 1, pid_index[pd_]))
        by_fr = {}
        for x in cand:
            by_fr.setdefault(x[0], []).append(x)
        ev_rows = []
        for fr in sorted(by_fr):
            grp = by_fr[fr]
            sc = [x for x in grp if x[6] == 0]
            ev_rows.append(sc[0] if sc else grp[0])
        truth = np.array([x[:6] for x in ev_rows], np.int64).reshape(-1, 6)
        poss_only = np.array([x[6] for x in ev_rows], np.int64)
        actor_pid = np.array([x[7] for x in ev_rows], np.int64)
        key = f"game_{syn_id}_H{hi}"
        np.savez(Path(out_dir) / f"{key}.npz", raw=rawa, ball_n=bn, ball_xy=ball[:, 2:4],
                 ball_z=ball[:, 4], ball_status=ball[:, 6].astype(np.int8),
                 ball_poss_ltr=ltr_poss.astype(np.int8), truth=truth, poss_only=poss_only,
                 actor_pid=actor_pid)
        census["halves"][key] = {"section": sec, "frames": int(len(bn)),
                                 "fps_from_T": round(float(fps), 4),
                                 "n_players_tracked": len(rows),
                                 "gk_mean_xn": {teams[t]: round(v, 4) for t, v in gkm.items()},
                                 "ltr": {teams[t]: v for t, v in ltr.items()},
                                 "ball_alive_share": round(float((ball[:, 6] == 1).mean()), 4),
                                 "events_kept": len(ev_rows)}
    (Path(out_dir) / "fp1d_census.json").write_text(json.dumps({mid: census}, indent=1))
    return census


if __name__ == "__main__":
    mid, raw_dir, out_dir = sys.argv[1:4]
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    c = build(mid, raw_dir, out_dir, int(sys.argv[4]) if len(sys.argv) > 4 else 9007)
    print(json.dumps(c, indent=1))
