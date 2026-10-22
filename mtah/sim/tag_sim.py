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
    keep_sp_by_type: dict = field(default_factory=lambda: {"Free kick": 0.16, "Goal kick": 0.78, "Throw-in": 0.72})

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


GOAL_LINE_TOL = 0.5     # m inside the goal line / touchline that counts as "crossed" (ball radius + tracking noise)
CORNER_R = 4.0          # restart within this distance of a corner flag
KICKOFF_R = 3.0         # restart within this distance of the centre spot
KICKOFF_MIN_DEAD_S = 15.0   # a mid-half kick-off follows a goal: a long dead run (celebration)
EXCURSION_MARGIN_S = 2.0    # the raw-ball excursion window extends the dead run by this on both sides


def _sp_type(cls, restart_ball, pitch, max_ax=np.nan, max_ay=np.nan, dead_s=0.0, at_start=False):
    """Restart type from the restart event class, the raw-ball restart position and the raw-ball excursion
    (max |x|, max |y|) during the dead run (+ margin).

    THROW_IN event -> Throw-in (whatever the geometry). Goal-line crossing (max |x| > L/2 - 0.5) with the restart within
    4 m of a corner flag -> Corner; with the restart in the goal area (|x| > L/2 - 7, |y| < 12) -> Goal kick. Restart
    within 3 m of the centre spot and (dead run >= 15 s or the run starts the half) -> Kick-off. Touchline crossing
    (max |y| > W/2 - 0.5) without a goal-line crossing -> Throw-in. Else Free kick."""
    if cls is not None and int(cls) == THROW_IN_CLS:
        return "Throw-in"
    L, W = pitch
    rx, ry = float(restart_ball[0]), float(restart_ball[1])
    goal_line = bool(np.isfinite(max_ax) and max_ax > L / 2 - GOAL_LINE_TOL)
    touch_line = bool(np.isfinite(max_ay) and max_ay > W / 2 - GOAL_LINE_TOL)
    if np.isfinite(rx) and np.isfinite(ry):
        if goal_line and np.hypot(abs(rx) - L / 2, abs(ry) - W / 2) < CORNER_R:
            return "Corner"
        if goal_line and abs(rx) > L / 2 - 7 and abs(ry) < 12:
            return "Goal kick"
        if np.hypot(rx, ry) < KICKOFF_R and (dead_s >= KICKOFF_MIN_DEAD_S or at_start):
            return "Kick-off"
    if touch_line and not goal_line:
        return "Throw-in"
    return "Free kick"


def _excursion(ball, s, e, margin):
    """(max |x|, max |y|) of the ball over [s - margin, e + margin) (NaN frames ignored; NaN if none)."""
    w = np.abs(np.asarray(ball[max(0, s - margin):min(len(ball), e + margin)], np.float64))
    ok = np.isfinite(w).all(axis=1)
    if not ok.any():
        return np.nan, np.nan
    return float(w[ok, 0].max()), float(w[ok, 1].max())


SP_TYPES = ("Throw-in", "Corner", "Goal kick", "Free kick", "Kick-off")
# real set-piece tag rates per graded minute (public real match; reference for the sim calibration)
REAL_SP_RATES = {"Throw-in": 0.245, "Corner": 0.113, "Goal kick": 0.094, "Free kick": 0.047, "Kick-off": 0.057}
REAL_SP_TOTAL = 0.557


def sp_rate_by_type(tags_df, graded_minutes):
    """Set-piece tag rate per graded minute by type: dict over SP_TYPES (+ any other type seen) and "total"."""
    sp = tags_df[tags_df["code"] == "SET PIECES"] if len(tags_df) else tags_df
    counts = sp["type"].fillna("None").astype(str).value_counts().to_dict() if len(sp) else {}
    gm = float(graded_minutes)
    out = {t: (counts.get(t, 0) / gm if gm > 0 else float("nan")) for t in SP_TYPES}
    for t, n in counts.items():
        if t not in out:
            out[t] = n / gm if gm > 0 else float("nan")
    out["total"] = (sum(counts.values()) / gm) if gm > 0 else float("nan")
    return out


def _ball_at(ball, r):
    """Ball at frame r, else the last non-NaN ball before it, else NaN."""
    k = min(int(r), len(ball) - 1)
    while k >= 0 and np.isnan(ball[k, 0]):
        k -= 1
    return ball[k] if k >= 0 else np.array([np.nan, np.nan])


def simulate_tags(match, labels, events, blackout_mask, params, rng, raw_ball=None):
    """raw_ball: clean DFL ball (F,2) used to type restarts (restart position + the excursion during the dead run, see
    _sp_type); falls back to match.ball. A dead run starting at frame 0 with no restart event is a Kick-off."""
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
    margin = int(EXCURSION_MARGIN_S * fps)
    for s, e in runs_of(np.asarray(labels) == "D"):
        cand = np.flatnonzero((ev_frames >= e - look) & (ev_frames <= e + look))
        if cand.size:
            r = int(ev_frames[cand[0]]); team = str(ev_team[cand[0]]); cls = int(ev_cls[cand[0]])
            max_ax, max_ay = _excursion(rball, s, e, margin)
            typ = _sp_type(cls, _ball_at(rball, r), match.pitch, max_ax, max_ay, dead_s=(e - s) / fps,
                           at_start=(s == 0))
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
