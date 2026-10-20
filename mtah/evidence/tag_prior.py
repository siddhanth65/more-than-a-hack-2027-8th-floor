"""Team prior from Smart Tagging phases and team-tagged set pieces."""
from dataclasses import dataclass

import numpy as np


@dataclass
class TagPriorParams:
    p_tag: float = 0.85
    p_sp_team: float = 0.8
    sp_team_after_s: float = 10.0   # reserved for Plan 2b (taker-team prior after the restart)


def tag_prior_evidence(tags, n_frames, params):
    ev = np.full((n_frames, 3), np.nan, np.float32)
    for r in tags.itertuples(index=False):
        if r.team not in ("H", "A"):
            continue
        p = params.p_sp_team if r.code == "SET PIECES" else params.p_tag
        a = max(0, int(r.start)); b = min(n_frames - 1, int(r.end))
        if b < a:
            continue
        col = 0 if r.team == "H" else 1
        ev[a:b + 1, col] = np.fmax(ev[a:b + 1, col], np.log(p))
        ev[a:b + 1, 1 - col] = np.fmax(ev[a:b + 1, 1 - col], np.log(1 - p))
    return ev
