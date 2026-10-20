"""Truth variants on the DFL proxy: raw vs event-confirmed control, full vs narrowed dead runs."""
import numpy as np

from mtah.io.match import distance_to_line
from mtah.sim.blackout_sim import runs_of

CTRL_WINDOW_S = 1.5
STATIC_MPS = 0.5
OUT_MARGIN_M = 0.5
RESTART_LOOK_S = 2.0


def _segments(labels):
    """[(start, stop, label)] maximal constant-label runs."""
    out = []
    F = len(labels)
    s = 0
    for i in range(1, F + 1):
        if i == F or labels[i] != labels[s]:
            out.append((s, i, labels[s])); s = i
    return out


def ctrl_filter(labels, events, fps):
    """Relabel team runs whose start is not confirmed by an event of that team within +-1.5 s."""
    out = np.asarray(labels).copy()
    win = int(round(CTRL_WINDOW_S * fps))
    ev_frames = {"H": np.sort(events.loc[events["team"] == "H", "frame"].to_numpy()),
                 "A": np.sort(events.loc[events["team"] == "A", "frame"].to_numpy())}
    prev_team = None
    for s, e, lab in _segments(out):
        if lab == "D":
            continue
        if prev_team is not None and lab != prev_team:
            f = ev_frames[lab]
            supported = f.size and np.any((f >= s - win) & (f <= s + win))
            if not supported:
                out[s:e] = prev_team
                lab = prev_team
        prev_team = lab
    return out


def narrow_dead(labels, events, ball, ball_speed, fps, pitch):
    """Trim each D run to [first static/out frame, first restart event]; keep if inverted."""
    out = np.asarray(labels).copy()
    look = int(round(RESTART_LOOK_S * fps))
    ev_all = np.sort(events["frame"].to_numpy())
    segs = _segments(out)
    for idx, (s, e, lab) in enumerate(segs):
        if lab != "D":
            continue
        before = segs[idx - 1][2] if idx > 0 else None
        after = segs[idx + 1][2] if idx + 1 < len(segs) else None
        static = ball_speed[s:e] < STATIC_MPS
        outp = distance_to_line(ball[s:e], pitch) < -OUT_MARGIN_M
        hit = np.flatnonzero(static | outp)
        s2 = s + int(hit[0]) if hit.size else s
        cand = ev_all[(ev_all >= s) & (ev_all <= e + look)]
        e2 = int(cand[0]) if cand.size else e
        e2 = min(e2, e)
        if e2 <= s2:
            continue
        fill_before = before if before is not None else after
        fill_after = after if after is not None else before
        if fill_before is not None:
            out[s:s2] = fill_before
        if fill_after is not None:
            out[e2:e] = fill_after
    return out


def truth_variants(labels, events, ball, ball_speed, fps, pitch):
    raw = np.asarray(labels).astype("<U1")
    ctrl = ctrl_filter(raw, events, fps)
    return {
        "raw_Dfull": raw,
        "ctrl_Dfull": ctrl,
        "raw_Dnarrow": narrow_dead(raw, events, ball, ball_speed, fps, pitch),
        "ctrl_Dnarrow": narrow_dead(ctrl, events, ball, ball_speed, fps, pitch),
    }
