"""Team prior from Smart Tagging phases and team-tagged set pieces."""
import json
from dataclasses import dataclass, field
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
    sp_dead_before_s: float = 2.0    # pre-start window of a set-piece type missing from sp_dead_before_by_type
    sp_dead_after_s: float = 3.0
    # Plan 2f: type-specific pre-restart window (seconds before the tag start; key = lower-case type prefix as in
    # mtah.evidence.dead._sp_code). The tag marks the kick itself; the dead time before a goal kick / corner runs
    # 15-30 s (ball visible and moving), so the window must start well before the tag.
    sp_dead_before_by_type: dict = field(default_factory=lambda: {"goal": 25.0, "corner": 20.0, "free": 10.0,
                                                                  "kick": 15.0, "throw": 6.0})
    # ramp: full bonus on [start - SP_RAMP_FULL_S, start + after], rising linearly from 0 at start - before_type to the
    # full bonus at start - SP_RAMP_FULL_S (weak far from the kick, strong near it); False = flat bonus on the window
    sp_dead_ramp: bool = True


SP_RAMP_FULL_S = 3.0


def _sp_before_s(sp_type, params):
    t = sp_type.lower() if isinstance(sp_type, str) else ""
    for key, before in (params.sp_dead_before_by_type or {}).items():
        if t.startswith(key):
            return float(before)
    return float(params.sp_dead_before_s)


def sp_dead_weights(tags, n_frames, params, fps=25.0):
    """(n_frames,) float weights in [0, 1] of the set-piece dead bonus (max over overlapping windows; 0 = outside)."""
    w = np.zeros(n_frames, np.float64)
    after = int(round(params.sp_dead_after_s * fps)); full = int(round(SP_RAMP_FULL_S * fps))
    t = np.arange(n_frames)
    for r in tags.itertuples(index=False):
        if r.code != "SET PIECES":
            continue
        s = int(r.start); before = int(round(_sp_before_s(getattr(r, "type", None), params) * fps))
        a, b = max(0, s - before), max(0, min(n_frames, s + after + 1))
        if b <= a:
            continue
        seg = np.ones(b - a)
        if params.sp_dead_ramp and before > full:
            seg = np.clip((t[a:b] - (s - before)) / (before - full), 0.0, 1.0)
        w[a:b] = np.maximum(w[a:b], seg)
    return w


def tag_prior_evidence(tags, n_frames, params, fps=25.0):
    """(n_frames, 3) float32 log-scores (H, A, D); NaN = no opinion (the decoder centres finite entries per frame).

    Team-tagged rows give H/A scores; with sp_dead_bonus > 0 every SET PIECES row (any team) also gives the D
    column 0 + bonus on its window (overlapping windows count once), and its H/A entries that are still NaN
    become 0, so a frame without a team opinion reads [0, 0, bonus]: a pure D advantage of `bonus`. The window is
    [start - before_type, start + sp_dead_after_s] (sp_dead_weights): ramped by default, frames at ramp weight 0 are
    left untouched."""
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
        w = sp_dead_weights(tags, n_frames, params, fps)
        win = w > 0
        hd = ev[win]
        hd[:, :2] = np.where(np.isnan(hd[:, :2]), 0.0, hd[:, :2])
        hd[:, 2] = np.where(np.isnan(hd[:, 2]), 0.0, hd[:, 2]) + params.sp_dead_bonus * w[win]
        ev[win] = hd
    return ev
