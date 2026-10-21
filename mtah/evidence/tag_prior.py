"""Team prior from Smart Tagging phases and team-tagged set pieces."""
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np

TAG_CODES_PATH = Path(__file__).resolve().parents[2] / "configs" / "tag_codes.json"


def load_tag_codes(path=None):
    """code -> {"sign": +1|-1|0, "p": float}; {} if the file is missing."""
    path = Path(path) if path is not None else TAG_CODES_PATH
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


@dataclass
class TagPriorParams:
    p_tag: float = 0.85
    p_sp_team: float = 0.8
    sp_team_after_s: float = 10.0   # reserved for Plan 2b (taker-team prior after the restart)
    code_table: Optional[dict] = None   # code -> {sign, p}; unknown codes: sign +1, p_tag
    # set-piece dead prior (Plan 2c): D log-score + sp_dead_bonus on [start - before, start + after] of every
    # SET PIECES row (0 = off; configs saved before it existed load as 0)
    sp_dead_bonus: float = 0.0
    sp_dead_before_s: float = 2.0
    sp_dead_after_s: float = 3.0


def tag_prior_evidence(tags, n_frames, params, fps=25.0):
    """(n_frames, 3) float32 log-scores (H, A, D); NaN = no opinion (the decoder centres finite entries per frame).

    Team-tagged rows give H/A scores; with sp_dead_bonus > 0 every SET PIECES row (any team) also gives the D
    column 0 + bonus on its window (overlapping windows count once), and its H/A entries that are still NaN
    become 0, so a frame without a team opinion reads [0, 0, bonus]: a pure D advantage of `bonus`."""
    ev = np.full((n_frames, 3), np.nan, np.float32)
    for r in tags.itertuples(index=False):
        if r.team not in ("H", "A"):
            continue
        entry = (params.code_table or {}).get(r.code)
        if entry is not None:
            sign, p = int(entry["sign"]), float(entry["p"])
        else:
            sign, p = 1, (params.p_sp_team if r.code == "SET PIECES" else params.p_tag)
        if sign == 0:
            continue
        a = max(0, int(r.start)); b = min(n_frames - 1, int(r.end))
        if b < a:
            continue
        col = 0 if r.team == "H" else 1
        if sign < 0:
            col = 1 - col   # tagged team is the one WITHOUT the ball
        ev[a:b + 1, col] = np.fmax(ev[a:b + 1, col], np.log(p))
        ev[a:b + 1, 1 - col] = np.fmax(ev[a:b + 1, 1 - col], np.log(1 - p))
    if params.sp_dead_bonus:
        win = np.zeros(n_frames, bool)
        before, after = int(round(params.sp_dead_before_s * fps)), int(round(params.sp_dead_after_s * fps))
        for r in tags.itertuples(index=False):
            if r.code == "SET PIECES":
                win[max(0, int(r.start) - before):max(0, min(n_frames, int(r.start) + after + 1))] = True
        hd = ev[win]
        hd[:, :2] = np.where(np.isnan(hd[:, :2]), 0.0, hd[:, :2])
        hd[:, 2] = np.where(np.isnan(hd[:, 2]), 0.0, hd[:, 2]) + params.sp_dead_bonus
        ev[win] = hd
    return ev
