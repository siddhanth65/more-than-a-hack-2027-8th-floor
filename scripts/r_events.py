"""Task R: typed SkillCorner restart events per K half (25 fps half-relative frame, type, taking team H/A), plus the
SkillCorner dead time before each restart (s, 10 fps events, as diag_prerestart.skillcorner). SkillCorner data only.
  python scripts/r_events.py  -> runs/kh/R/restarts.csv  (key, frame, type, team, dead_s)"""
import os
import json, sys
from pathlib import Path
import numpy as np, pandas as pd
ROOT = Path(__file__).resolve().parents[1]
SC = Path(os.environ.get("MTAH_SKILLCORNER_DIR", os.path.join(os.path.expanduser("~"), "skillcorner-opendata", "data", "matches")))
TYPES = ("throw_in", "free_kick", "goal_kick", "corner", "goal")
rows = []
for i, md in enumerate(sorted((p for p in SC.iterdir() if p.is_dir()), key=lambda d: int(d.name))):
    syn = 9101 + i; mid = md.name
    meta = json.loads((md / f"{mid}_match.json").read_text(encoding="utf-8")); home = meta["home_team"]["id"]
    ev = pd.read_csv(md / f"{mid}_dynamic_events.csv", low_memory=False,
                     usecols=["event_type", "frame_start", "frame_end", "period", "team_id", "game_interruption_before"])
    pp = ev[ev.event_type == "player_possession"].sort_values(["period", "frame_start"]).reset_index(drop=True)
    prev_end = pp.groupby("period").frame_end.shift(1)
    gib = pp.game_interruption_before.fillna(""); typ = gib.str.replace(r"_(for|against)$", "", regex=True)
    for per in meta["match_periods"]:
        p = int(per["period"]); f0 = int(per["start_frame"])
        m = (pp.period == p) & typ.isin(TYPES) & prev_end.notna()
        for fs, ty, pe, tid in zip(pp.frame_start[m], typ[m], prev_end[m], pp.team_id[m]):
            rows.append((f"game_{syn}_H{p}", int(round((fs - f0) * 2.5)), ty, "H" if tid == home else "A", (fs - pe) / 10.0))
df = pd.DataFrame(rows, columns=["key", "frame", "type", "team", "dead_s"])
(ROOT / "runs" / "kh" / "R").mkdir(parents=True, exist_ok=True); df.to_csv(ROOT / "runs" / "kh" / "R" / "restarts.csv", index=False)
print(len(df), df.groupby("type").dead_s.describe().round(1).to_string())
