"""Automatic-looking tags from the corrupted stream: phases by nearest team, set pieces at restarts."""
from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd

from mtah.baselines import nearest_team
from mtah.sim.blackout_sim import runs_of

COLS = ["start", "end", "code", "team", "type", "half"]


@dataclass
class TagSimParams:
    hysteresis_s: float = 1.0
    min_phase_s: float = 3.0
    sp_before_s: float = 5.0
    sp_after_s: float = 30.0
    keep_outside: float = 0.88
    keep_inside: float = 0.75
    jitter_s: float = 1.0
    carry_max_s: float = 60.0

    def to_dict(self):
        return asdict(self)

    @classmethod
    def from_dict(cls, d):
        return cls(**d)


def _phase_runs(codes, fps, hyst_s, min_s, carry_max_s=60.0):
    """Runs of team codes with hysteresis. A -1 (no ball) frame does not end a run: the team is carried
    through missing-ball frames (as the real tags do through blackouts). A run ends on a confirmed team
    change, or at the last ball frame once the gap exceeds carry_max_s."""
    F = len(codes); hyst = int(hyst_s * fps); min_len = int(min_s * fps); carry = int(carry_max_s * fps)
    out = []
    cur = -1; start = 0; pending = None; last_ball = -1
    for t in range(F):
        c = int(codes[t])
        if c == -1:
            if cur != -1 and t - last_ball > carry:
                if last_ball - start + 1 >= min_len:
                    out.append((start, last_ball, cur))
                cur, pending = -1, None
            continue
        last_ball = t
        if cur == -1:
            cur, start, pending = c, t, None
            continue
        if c != cur:
            if pending is None:
                pending = t
            elif t - pending >= hyst:
                if pending - start >= min_len:
                    out.append((start, pending - 1, cur))
                cur, start, pending = c, pending, None
        else:
            pending = None
    if cur != -1 and F - start >= min_len:
        out.append((start, F - 1, cur))
    return out


def _sp_type(xy, pitch):
    L, W = pitch
    x, y = float(xy[0]), float(xy[1])
    if abs(y) > W / 2 - 1.0:
        return "Throw-in"
    if abs(x) > L / 2 - 1.5 and abs(y) > 20:
        return "Corner Kick"
    if abs(x) > L / 2 - 16.5:
        return "Goal Kick"
    return "Free Kick"


def simulate_tags(match, labels, events, blackout_mask, params, rng):
    fps = match.fps; F = match.n_frames
    rows = []
    codes = nearest_team(match)
    for s, e, team in _phase_runs(codes, fps, params.hysteresis_s, params.min_phase_s, params.carry_max_s):
        mid = (s + e) // 2
        keep = params.keep_inside if blackout_mask[mid] else params.keep_outside
        if rng.random() >= keep:
            continue
        rows.append((s, e, "PROGRESSION", "H" if team == 0 else "A", None, 1))
    ev_frames = events["frame"].to_numpy() if len(events) else np.zeros(0, int)
    ev_team = events["team"].to_numpy() if len(events) else np.zeros(0, object)
    look = int(2.0 * fps)
    for s, e in runs_of(np.asarray(labels) == "D"):
        cand = np.flatnonzero((ev_frames >= e - look) & (ev_frames <= e + look))
        if not cand.size:
            continue
        r = int(ev_frames[cand[0]]); team = str(ev_team[cand[0]])
        k = s - 1
        while k >= 0 and np.isnan(match.ball[k, 0]):
            k -= 1
        typ = _sp_type(match.ball[k], match.pitch) if k >= 0 else "Free Kick"
        if typ == "Throw-in":
            team = None
        rows.append((r - int(params.sp_before_s * fps), r + int(params.sp_after_s * fps), "SET PIECES", team, typ, 1))
    out = []
    jit = int(params.jitter_s * fps)
    for s, e, code, team, typ, half in rows:
        if jit:
            s += int(rng.integers(-jit, jit + 1)); e += int(rng.integers(-jit, jit + 1))
        s = max(0, s); e = min(F - 1, e)
        if e > s:
            out.append((s, e, code, team, typ, half))
    data = {c: [r[i] for r in out] for i, c in enumerate(COLS)}
    return pd.DataFrame({c: pd.Series(data[c], dtype=(object if c in ("team", "type", "half") else None))
                         for c in COLS}, columns=COLS).sort_values("start", kind="stable").reset_index(drop=True)
