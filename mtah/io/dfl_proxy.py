"""DFL open data (pre-parsed by the football-synthesizer project) as a Metrica-shaped proxy."""
import json
from pathlib import Path

import numpy as np

from mtah.io.match import Match, speeds_from_positions

DFL_PITCH = (105.0, 68.0)
SLOTS_PER_TEAM = 16
FPS = 25.0


def _ltr_home(census_path, key):
    census = json.loads(Path(census_path).read_text())
    for match in census.values():
        halves = match.get("halves", {})
        if key in halves:
            return int(halves[key]["ltr"]["home"])
    raise KeyError(f"{key} not in {census_path}")


def load_half(npz_path, census_path):
    npz_path = Path(npz_path)
    key = npz_path.stem
    z = np.load(npz_path)
    raw = z["raw"]; ball_n = z["ball_n"].astype(np.int64)
    n0 = int(ball_n[0]); F = len(ball_n)
    if not np.array_equal(ball_n, np.arange(n0, n0 + F)):
        raise ValueError(f"{npz_path}: ball frames not contiguous")
    ltr_home = _ltr_home(census_path, key)
    S = 2 * SLOTS_PER_TEAM
    pos = np.full((F, S, 2), np.nan, np.float32)
    team = np.repeat(np.array([0, 1], np.int8), SLOTS_PER_TEAM)
    frame = raw[:, 0].astype(np.int64) - n0
    pid = raw[:, 1].astype(np.int64)
    is_home = raw[:, 2].astype(np.int64) == ltr_home
    slot = np.empty(len(raw), np.int64)
    for home_flag, base in ((True, 0), (False, SLOTS_PER_TEAM)):
        sel = is_home == home_flag
        order = {p: i for i, p in enumerate(sorted(set(pid[sel].tolist())))}
        if len(order) > SLOTS_PER_TEAM:
            raise ValueError(f"{npz_path}: {len(order)} players on one side > {SLOTS_PER_TEAM} slots")
        slot[sel] = base + np.array([order[p] for p in pid[sel]], np.int64)
    ok = (frame >= 0) & (frame < F)
    L, W = DFL_PITCH
    pos[frame[ok], slot[ok], 0] = raw[ok, 5] * L - L / 2
    pos[frame[ok], slot[ok], 1] = raw[ok, 6] * W - W / 2
    ball = z["ball_xy"].astype(np.float32)
    status = z["ball_status"].astype(np.int8); poss = z["ball_poss_ltr"].astype(np.int8)
    labels = np.where(status == 0, "D", np.where(poss == ltr_home, "H", "A")).astype("<U1")
    m = Match(fps=FPS, pos=pos, speed=speeds_from_positions(pos, FPS), team=team, ball=ball,
              ball_speed=speeds_from_positions(ball, FPS), half=np.ones(F, np.int8), pitch=DFL_PITCH)
    return m, labels


def list_halves(dfl_dir):
    return [(p.stem, p) for p in sorted(Path(dfl_dir).glob("game_*_H*.npz"))]
