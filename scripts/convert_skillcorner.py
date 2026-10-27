"""SkillCorner open data (A-League 2024/25, 10 fps broadcast tracking) -> dfl_proxy npz halves game_91NN_H{1,2}.

Usage (light, single process):
  OMP_NUM_THREADS=2 python -I scripts/convert_skillcorner.py SKILLCORNER_MATCHES_DIR OUT_DIR [--only N ...]
Match order = sorted SkillCorner match ids -> syn ids 9101..9120. Prints aggregates only.

Label rule (per 10 fps source frame, then resampled):
  alive  = frame inside a phases_of_play interval (in-play by SkillCorner's definition) OR possession.group set.
  team   = possession.group where set; inside an alive run, frames with no group carry the last group forward
           (the team keeps possession while the ball travels / is loose), the run's head falls back to the
           phase's team_in_possession, then to the run's first group.
  D      = not alive (outside every phase and no player in possession): stoppages, out of play, replays.
Resample 10 -> 25 fps on the period clock: positions linear (NaN if either neighbour is NaN), labels / flags
nearest source frame; `interp` marks output frames that do not coincide with a source frame (4 of every 5).
Positions are SkillCorner's *extrapolated* ones (all 22 players every frame); `raw_detected` / `ball_visible`
carry the real is_detected flags.
Events (for truth_variants' ctrl filter / tag simulation): one row per dynamic_events player_possession at
frame_start (the reception / gain), class from end_type (pass/clearance -> PASS, shot -> SHOT, else DRIVE
poss-only); a possession that starts with a throw-in reception (game_interruption_before throw_in_*) is a
THROW_IN row.
"""
import site
import sys
from pathlib import Path

site.addsitedir(site.getusersitepackages())                 # -I drops the user site; numpy/pandas live there
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # this checkout's mtah, not an installed copy

import argparse  # noqa: E402
import json  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from mtah.io.extra_npz import (DRIVE, PASS, SHOT, THROW_IN, aggregate, hda_labels, ltr_from_gk,  # noqa: E402
                               merge_census, player_rows, to_dfl_metres, write_half)

SRC_FPS = 10.0
OUT_FPS = 25.0
SYN0 = 9101


def read_tracking(path):
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            b = r.get("ball_data") or {}
            pos = r.get("possession") or {}
            rows.append((r["frame"], r.get("period"), b.get("x"), b.get("y"), b.get("z"),
                         bool(b.get("is_detected")), pos.get("group"),
                         [(p["player_id"], p["x"], p["y"], bool(p.get("is_detected"))) for p in r.get("player_data") or []]))
    return rows


def _fill_team(alive, group, phase_team):
    """group: (N,) object 'H'/'A'/None. Returns (N,) 'H'/'A' (D frames get the previous team, irrelevant)."""
    N = len(alive)
    team = np.empty(N, object)
    last = None
    i = 0
    while i < N:
        if not alive[i]:
            team[i] = last; i += 1; continue
        j = i
        while j < N and alive[j]:
            j += 1
        g = group[i:j]
        known = [x for x in g if x is not None]
        cur = phase_team[i] or (known[0] if known else None) or last
        for k in range(i, j):
            if group[k] is not None:
                cur = group[k]
            team[k] = cur
        last = cur
        i = j
    # leading frames before any team: first known
    first = next((t for t in team if t is not None), "H")
    return np.array([t if t is not None else first for t in team], object)


def convert_match(mdir, syn, out_dir):
    mid = mdir.name
    meta = json.loads((mdir / f"{mid}_match.json").read_text(encoding="utf-8"))
    pitch = (float(meta["pitch_length"]), float(meta["pitch_width"]))
    home_id = meta["home_team"]["id"]
    people = {p["id"]: ("home" if p["team_id"] == home_id else "away", int(p.get("number") or 0),
                        1 if (p.get("player_role") or {}).get("acronym") == "GK" else 8)
              for p in meta["players"]}
    rows = read_tracking(mdir / f"{mid}_tracking_extrapolated.jsonl")
    fr_all = np.array([r[0] for r in rows], np.int64)
    idx_of = {f: i for i, f in enumerate(fr_all.tolist())}
    phases = pd.read_csv(mdir / f"{mid}_phases_of_play.csv")
    dyn = pd.read_csv(mdir / f"{mid}_dynamic_events.csv", low_memory=False,
                      usecols=["event_type", "frame_start", "period", "team_id", "player_id", "end_type",
                               "start_type", "game_interruption_before"])
    pp = dyn[dyn["event_type"] == "player_possession"]
    pid_index, census, per_half = {}, {"match": f"skillcorner_{mid}", "source": "skillcorner", "halves": {}}, []
    for per in meta["match_periods"]:
        p = int(per["period"])
        f0, f1 = int(per["start_frame"]), int(per["end_frame"])
        src = np.arange(f0, f1 + 1)
        have = np.array([f in idx_of for f in src.tolist()])
        N = len(src)
        ball = np.full((N, 3), np.nan); bdet = np.zeros(N, bool); group = np.full(N, None, object)
        pl = {}
        for k, f in enumerate(src.tolist()):
            if f not in idx_of:
                continue
            r = rows[idx_of[f]]
            if r[2] is not None:
                ball[k] = (r[2], r[3], r[4] if r[4] is not None else np.nan)
            bdet[k] = r[5]
            group[k] = {"home team": "H", "away team": "A"}.get(r[6])
            for pid, x, y, d in r[7]:
                if pid not in people:
                    continue
                a = pl.setdefault(pid, [np.full((N, 2), np.nan), np.zeros(N, bool)])
                a[0][k] = (x, y); a[1][k] = d
        in_phase = np.zeros(N, bool); phase_team = np.full(N, None, object)
        for _, ph in phases[phases["period"] == p].iterrows():
            a, b = int(ph["frame_start"]) - f0, int(ph["frame_end"]) - f0 + 1
            a, b = max(a, 0), min(b, N)
            if b <= a:
                continue
            in_phase[a:b] = True
            phase_team[a:b] = "H" if ph["team_in_possession_id"] == home_id else "A"
        alive = in_phase | np.array([g is not None for g in group])
        team10 = _fill_team(alive, group, phase_team)
        # 10 -> 25 fps on the period clock
        K = int(np.floor((N - 1) * OUT_FPS / SRC_FPS)) + 1
        s = np.arange(K) * (SRC_FPS / OUT_FPS)
        lo = np.floor(s).astype(int); hi = np.minimum(lo + 1, N - 1); w = (s - lo)[:, None]
        near = np.rint(s).astype(int)
        interp = (s - lo) > 1e-9

        def lin(a):
            return a[lo] * (1 - w) + a[hi] * w

        gk_x = {"home": [], "away": []}
        for pid, (xy, d) in pl.items():
            if people[pid][2] == 1:
                gk_x[people[pid][0]].append(np.nanmean(xy[:, 0]))
        gkm = {t: float(np.nanmean(v)) for t, v in gk_x.items() if v}
        side_flag = meta["home_team_side"][p - 1] == "left_to_right"
        ltr = ltr_from_gk(gkm) if len(gkm) == 2 else {"home": 0 if side_flag else 1, "away": 1 if side_flag else 0}
        side_agrees = (ltr["home"] == 0) == side_flag
        raw_parts, det_parts = [], []
        frames_out = np.arange(K, dtype=np.int64)
        for pid, (xy, d) in sorted(pl.items()):
            tm, shirt, role = people[pid]
            if pid not in pid_index:
                pid_index[pid] = (100 if tm == "home" else 200) + sum(1 for q in pid_index if people[q][0] == tm)
            r, dd = player_rows(frames_out, lin(xy), d[near], pid_index[pid], ltr[tm], shirt, role, pitch)
            raw_parts.append(r); det_parts.append(dd)
        ball25 = lin(ball)
        bxy = to_dfl_metres(ball25[:, :2], pitch)
        lab10 = np.where(alive, team10, "D")
        status = (lab10[near] != "D").astype(np.int8)
        team25 = team10[near]
        poss_ltr = np.where(team25 == "H", ltr["home"], ltr["away"]).astype(np.int8)
        # events
        ev = pp[pp["period"] == p]
        truth, poss_only, actor = [], [], []
        seen = set()
        for _, e in ev.sort_values("frame_start", kind="stable").iterrows():
            pid = int(e["player_id"]) if pd.notna(e["player_id"]) else None
            if pid not in people:
                continue
            k = int(round((int(e["frame_start"]) - f0) * OUT_FPS / SRC_FPS))
            if not (0 <= k < K) or k in seen:
                continue
            tm, shirt, _ = people[pid]
            et = e["end_type"] if isinstance(e["end_type"], str) else ""
            gib = e["game_interruption_before"] if isinstance(e["game_interruption_before"], str) else ""
            if gib.startswith("throw_in") and str(e["start_type"]).startswith("throw_in"):
                cls, po = THROW_IN, 0
            elif et in ("pass", "clearance"):
                cls, po = PASS, 0
            elif et == "shot":
                cls, po = SHOT, 0
            else:
                cls, po = DRIVE, 1
            if pid not in pid_index:
                pid_index[pid] = (100 if tm == "home" else 200) + sum(1 for q in pid_index if people[q][0] == tm)
            seen.add(k)
            truth.append((k, ltr[tm], shirt, cls, 1, 0)); poss_only.append(po); actor.append(pid_index[pid])
        key = f"game_{syn}_H{p}"
        write_half(out_dir, key, raw_parts=raw_parts, det_parts=det_parts, ball_n=frames_out, ball_xy=bxy,
                   ball_z=ball25[:, 2], status=status, poss_ltr=poss_ltr, truth=truth, poss_only=poss_only,
                   actor_pid=actor, ball_visible=bdet[near], interp=interp)
        labels = hda_labels(status, poss_ltr, ltr["home"])
        per_half.append({"labels": labels, "ball_visible": bdet[near]})
        census["halves"][key] = {
            "section": f"period{p}", "frames": int(K), "fps_from_T": OUT_FPS, "source_fps": SRC_FPS,
            "source_frames": int(N), "source_frames_missing": int((~have).sum()),
            "n_players_tracked": len(pl), "gk_mean_xn": {t: round((v + pitch[0] / 2) / pitch[0], 4) for t, v in gkm.items()},
            "ltr": ltr, "ltr_agrees_home_team_side": bool(side_agrees),
            "ball_alive_share": round(float(status.mean()), 4), "events_kept": len(truth),
            "interp_share": round(float(interp.mean()), 4), "ball_visible_share": round(float(bdet[near].mean()), 4),
            "phase_share": round(float(in_phase.mean()), 4),
            "group_share": round(float(np.mean([g is not None for g in group])), 4),
            "pitch_source": list(pitch),
        }
    merge_census(out_dir, census["match"], census)
    return per_half


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("matches_dir")
    ap.add_argument("out_dir")
    ap.add_argument("--only", type=int, nargs="+", default=None, help="syn ids (9101..) to convert")
    args = ap.parse_args()
    out = Path(args.out_dir); out.mkdir(parents=True, exist_ok=True)
    mdirs = sorted([d for d in Path(args.matches_dir).iterdir() if d.is_dir()], key=lambda d: int(d.name))
    halves = []
    for i, d in enumerate(mdirs):
        syn = SYN0 + i
        if args.only and syn not in args.only:
            continue
        halves += convert_match(d, syn, out)
        print(f"game_{syn} done", flush=True)
    print(json.dumps({"skillcorner": aggregate(halves)}, indent=1))


if __name__ == "__main__":
    main()
