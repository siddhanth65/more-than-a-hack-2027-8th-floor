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


def tag_prior_evidence(tags, n_frames, params):
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
    return ev
