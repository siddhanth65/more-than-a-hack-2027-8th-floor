"""Per-frame H/A/D truth for the public Metrica sample games, derived from the event stream and the tracking.

Convention aimed at: "ball owner and dead flag of the corrected tracking" (the contest ground truth), not a
tagger's possession-phase notion. Everything here is algorithmic (no hand labelling) and pure (numpy only).

Events are dicts {type, subs (tuple of upper-case tokens), team ('H' / 'A'), start, end, period, player}; frames are
the source's absolute frame numbers; a half is the inclusive frame range h0..h1.

Dead  (`event_dead`): the head of a half up to its first SET PIECE (kick-off), the tail after the last event's end
      frame, and [trigger, start of the next restart (SET PIECE or goal kick)) where trigger is BALL OUT (end frame), SHOT with token OUT /
      GOAL (end frame), a foul (CHALLENGE with token FAULT, FAULT RECEIVED) or BALL LOST OFFSIDE / HAND BALL /
      REFEREE HIT (start frame) -- kept only when the first possession event starting after the trigger is that
      restart (otherwise play went on: advantage, or an out the tagger did not close with a restart).
      `blank_dead` adds runs of >= `min_s` seconds in which the tracking has no ball (the sample data blanks the ball
      in stoppages): the owner/dead flag of a tracking without a ball is "dead" by the contest's own convention.
Owner (`team_track`): the team of the latest *owner-setting* event started at or before the frame, held through
      flights and loose balls. Owner-setting: PASS, CARRY, SHOT, RECOVERY, SET PIECE and the winner of a CHALLENGE
      TACKLE / GROUND / DRIBBLE won. BALL LOST / BALL OUT / aerial duels / fouls never move the owner (the
      opponent's RECOVERY or next action does), i.e. possession changes only when the opponent gains control.
"""
import numpy as np

OWNER_TYPES = {"PASS", "CARRY", "SHOT", "RECOVERY", "SET PIECE"}
POSS_TYPES = OWNER_TYPES | {"BALL LOST", "BALL OUT"}       # events that count as "play went on" for the restart check
FOUL_BALL_LOST = {"OFFSIDE", "HAND BALL", "REFEREE HIT"}
CHALLENGE_OWNER = {"TACKLE", "GROUND", "DRIBBLE"}
PITCH = (105.0, 68.0)                                       # Metrica sample games 1 and 2 carry no pitch size


def parse_subs(s):
    """'HEAD-OFF TARGET-OUT' -> ('HEAD', 'OFF TARGET', 'OUT'); NaN / None / '' -> ()."""
    if s is None or (isinstance(s, float) and np.isnan(s)):
        return ()
    return tuple(t.strip().upper() for t in str(s).split("-") if t.strip())


def norm_to_metres(xy_norm, pitch=PITCH):
    """Metrica normalised (0..1, origin at a corner) -> centred metres. NaN propagates."""
    a = np.asarray(xy_norm, np.float64)
    return (a - 0.5) * np.asarray(pitch, np.float64)


def events_from_csv_rows(rows):
    """rows: iterable of dicts with the Metrica CSV columns (Team, Type, Subtype, Period, Start Frame, End Frame, From)."""
    evs = []
    for r in rows:
        team = {"HOME": "H", "AWAY": "A"}.get(str(r["Team"]).strip().upper())
        if team is None:
            continue
        s = int(r["Start Frame"]); e = r["End Frame"]
        e = s if (e is None or (isinstance(e, float) and np.isnan(e)) or int(e) < s) else int(e)
        evs.append({"type": str(r["Type"]).strip().upper(), "subs": parse_subs(r.get("Subtype")), "team": team,
                    "start": s, "end": e, "period": int(r["Period"]), "player": r.get("From")})
    evs.sort(key=lambda e: (e["start"], e["end"]))
    return evs


def is_restart(q, poss):
    """A restart event: SET PIECE, or a goal kick (the CSV files tag goal kicks as PASS / BALL LOST with token GOAL KICK,
    often preceded by the keeper's RECOVERY in the same frame)."""
    if q["type"] == "SET PIECE" or "GOAL KICK" in q["subs"]:
        return True
    return q["type"] == "RECOVERY" and any("GOAL KICK" in o["subs"] and 0 <= o["start"] - q["start"] <= 1 for o in poss)


def event_dead(evs, h0, h1):
    """(F,) bool over frames h0..h1 inclusive: the event-derived dead flag (module docstring)."""
    F = h1 - h0 + 1
    dead = np.zeros(F, bool)
    if not evs:
        dead[:] = True
        return dead
    poss = [e for e in evs if e["type"] in POSS_TYPES]
    sp = [e for e in poss if e["type"] == "SET PIECE"]
    if sp:
        dead[:max(sp[0]["start"] - h0, 0)] = True
    last_end = max(e["end"] for e in evs)
    dead[max(last_end - h0 + 1, 0):] = True
    for e in evs:
        typ, subs = e["type"], e["subs"]
        if typ == "BALL OUT" or (typ == "SHOT" and ("OUT" in subs or "GOAL" in subs)):
            trig, ref = e["end"], e["start"]
        elif (typ == "CHALLENGE" and "FAULT" in subs) or typ == "FAULT RECEIVED" or \
                (typ == "BALL LOST" and set(subs) & FOUL_BALL_LOST):
            trig, ref = e["start"], e["start"]
        else:
            continue
        nxt = next((q for q in poss if q["start"] > ref and q is not e), None)
        if nxt is None or not is_restart(nxt, poss) or nxt["start"] <= trig:
            continue
        dead[max(trig - h0, 0):max(nxt["start"] - h0, 0)] = True
    return dead


def blank_runs(visible, min_frames):
    """(F,) bool: frames inside maximal runs of ~visible that are >= min_frames long."""
    v = np.asarray(visible, bool)
    out = np.zeros(len(v), bool)
    edge = np.diff(np.r_[0, (~v).astype(np.int8), 0])
    for a, b in zip(np.flatnonzero(edge == 1), np.flatnonzero(edge == -1)):
        if b - a >= min_frames:
            out[a:b] = True
    return out


def is_owner_event(e):
    t, subs = e["type"], e["subs"]
    if t in OWNER_TYPES:
        return True
    return t == "CHALLENGE" and "WON" in subs and "FAULT" not in subs and "ADVANTAGE" not in subs and bool(CHALLENGE_OWNER & set(subs))


def team_track(evs, h0, h1):
    """(F,) object 'H'/'A': latest owner-setting event at or before each frame; the head before the first one takes
    the first owner's team (that stretch is dead in the usual case)."""
    F = h1 - h0 + 1
    own = [e for e in evs if is_owner_event(e)]
    team = np.full(F, None, object)
    for e, nxt in zip(own, own[1:] + [None]):
        a = min(max(e["start"] - h0, 0), F)
        b = F if nxt is None else min(max(nxt["start"] - h0, 0), F)
        if b > a:
            team[a:b] = e["team"]
    first = next((t for t in team if t is not None), "H")
    return np.array([t if t is not None else first for t in team], object)


def derive_labels(evs, h0, h1, ball_visible=None, fps=25.0, blank_min_s=None):
    """-> dict(labels (F,) '<U1' H/A/D, team (F,) owner held through dead frames, dead (F,), event_dead (F,), blank_dead (F,)).
    blank_min_s: None = ignore the tracking's ball visibility; else blank runs >= that many seconds count as dead."""
    ed = event_dead(evs, h0, h1)
    bd = np.zeros_like(ed)
    if ball_visible is not None and blank_min_s is not None:
        bd = blank_runs(ball_visible, int(round(blank_min_s * fps)))
    dead = ed | bd
    team = team_track(evs, h0, h1)
    labels = np.where(dead, "D", team).astype("<U1")
    return {"labels": labels, "team": team, "dead": dead, "event_dead": ed, "blank_dead": bd}


def spell_stats(halves_labels, fps=25.0):
    """Aggregates over a list of (F,) label arrays: dead share, spells per minute of all (graded) time, length
    quantiles / shares, direct-handover share = spells whose successor run is a spell of the other team (no dead
    gap) over all spells that have a successor run inside the half."""
    lens, direct, followed, F, dead = [], 0, 0, 0, 0
    for lab in halves_labels:
        lab = np.asarray(lab); F += len(lab); dead += int((lab == "D").sum())
        cut = np.flatnonzero(lab[1:] != lab[:-1]) + 1
        st = np.r_[0, cut]; en = np.r_[cut, len(lab)]
        runs = [(lab[a], b - a) for a, b in zip(st, en)]
        for i, (c, n) in enumerate(runs):
            if c not in ("H", "A"):
                continue
            lens.append(n)
            if i + 1 < len(runs):
                followed += 1
                direct += runs[i + 1][0] in ("H", "A")
    s = np.asarray(lens, float) / fps
    minutes = F / fps / 60
    return {
        "minutes": round(minutes, 1), "dead_share": round(dead / F, 3),
        "n_spells": int(s.size), "spells_per_min": round(s.size / minutes, 2),
        "lt1s": round(float((s < 1).mean()), 3), "lt3s": round(float((s < 3).mean()), 3),
        "lt5s": round(float((s < 5).mean()), 3), "median_s": round(float(np.median(s)), 2),
        "direct_handover_share": round(direct / max(followed, 1), 3),
    }
