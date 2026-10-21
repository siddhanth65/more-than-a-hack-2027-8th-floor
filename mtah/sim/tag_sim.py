"""Automatic-looking tags from the corrupted stream: phases by nearest team, set pieces at restarts."""
from dataclasses import asdict, dataclass, field

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
    # per set-piece type keep probability (case-insensitive type names; types not listed are kept):
    # per-type calibration knobs, starting values: re-calibrated after the next rebuild now that restarts are typed
    # from THROW_IN events and the clean DFL ball (real: ~5 free kicks and ~0.245 throw-ins/min)
    keep_sp_by_type: dict = field(default_factory=lambda: {"Free kick": 0.15, "Throw-in": 0.55})

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


THROW_IN_CLS = 4   # scripts/parse_dfl_match.py THROW_IN; same scheme for every half via load_events


def _sp_type(cls, restart_ball, pitch):
    """Restart type from the restart event class and the clean ball position at the restart.
    A THROW_IN event is a throw-in whatever the geometry; otherwise corner flag / goal area / centre spot / free kick."""
    if cls is not None and int(cls) == THROW_IN_CLS:
        return "Throw-in"
    L, W = pitch
    rx, ry = float(restart_ball[0]), float(restart_ball[1])
    if np.isnan(rx) or np.isnan(ry):
        return "Free kick"
    if abs(rx) > L / 2 - 3 and abs(ry) > W / 2 - 3:
        return "Corner"
    if abs(rx) > L / 2 - 7 and abs(ry) < 12:
        return "Goal kick"
    if np.hypot(rx, ry) < 2.0:
        return "Kick-off"
    return "Free kick"


def _ball_at(ball, r):
    """Ball at frame r, else the last non-NaN ball before it, else NaN."""
    k = min(int(r), len(ball) - 1)
    while k >= 0 and np.isnan(ball[k, 0]):
        k -= 1
    return ball[k] if k >= 0 else np.array([np.nan, np.nan])


def simulate_tags(match, labels, events, blackout_mask, params, rng, raw_ball=None):
    """raw_ball: clean DFL ball (F,2) used to type restarts; falls back to match.ball."""
    rball = match.ball if raw_ball is None else np.asarray(raw_ball)
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
    ev_cls = events["cls"].to_numpy() if len(events) else np.zeros(0, int)
    look = int(2.0 * fps)
    keep_sp = {str(k).lower(): float(v) for k, v in (params.keep_sp_by_type or {}).items()}
    for s, e in runs_of(np.asarray(labels) == "D"):
        cand = np.flatnonzero((ev_frames >= e - look) & (ev_frames <= e + look))
        if cand.size:
            r = int(ev_frames[cand[0]]); team = str(ev_team[cand[0]]); cls = int(ev_cls[cand[0]])
            typ = _sp_type(cls, _ball_at(rball, r), match.pitch)
        elif s == 0:
            r = int(e); team = None; typ = "Kick-off"
        else:
            continue
        if keep_sp.get(typ.lower(), 1.0) < 1.0 and rng.random() >= keep_sp[typ.lower()]:
            continue
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
