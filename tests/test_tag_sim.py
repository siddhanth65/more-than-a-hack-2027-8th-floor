import numpy as np
import pandas as pd

from mtah.sim.tag_sim import TagSimParams, _phase_runs, simulate_tags
from tests.test_baselines import two_player_match


def test_phases_and_setpiece():
    m = two_player_match(F=25 * 120)
    m.ball[:] = 0.0
    m.ball[:25 * 40, 0] = -8.0            # near home for 40 s
    m.ball[25 * 40:25 * 70, 0] = 8.0      # near away for 30 s
    m.ball[25 * 70:25 * 80, :] = np.nan   # blackout 10 s
    m.ball[25 * 80:, 0] = -8.0
    labels = np.full(m.n_frames, "H", dtype="<U1")
    labels[25 * 40:25 * 70] = "A"
    labels[25 * 70:25 * 80] = "D"
    m.ball[25 * 69:25 * 70, 1] = 34.0     # last ball frames before the D run (incl. 1749): on the touchline -> throw-in
    events = pd.DataFrame([(25 * 80, "H", 8, False)], columns=["frame", "team", "cls", "poss_only"])
    mask = np.zeros(m.n_frames, bool); mask[25 * 70:25 * 80] = True
    p = TagSimParams(keep_outside=1.0, keep_inside=1.0, jitter_s=0.0)
    df = simulate_tags(m, labels, events, mask, p, np.random.default_rng(0))
    phases = df[df["code"] == "PROGRESSION"]
    assert list(phases["team"]) == ["H", "A", "H"]
    assert int(phases.iloc[1]["end"]) == 25 * 80 - 1   # the A phase now persists through the 10 s blackout
    sp = df[df["code"] == "SET PIECES"]
    assert len(sp) == 1 and sp.iloc[0]["type"] == "Throw-in" and sp.iloc[0]["team"] is None
    assert sp.iloc[0]["start"] == 25 * 80 - 125 and sp.iloc[0]["end"] == 25 * 80 + 750


def test_deletion_rate():
    m = two_player_match(F=25 * 600); m.ball[:] = 0.0; m.ball[:, 0] = -8.0
    labels = np.full(m.n_frames, "H", dtype="<U1")
    events = pd.DataFrame(columns=["frame", "team", "cls", "poss_only"])
    p = TagSimParams(keep_outside=0.0, keep_inside=0.0)
    df = simulate_tags(m, labels, events, np.zeros(m.n_frames, bool), p, np.random.default_rng(0))
    assert len(df) == 0


def test_phase_runs_carry_through_gaps():
    fps = 25.0
    def codes(gap_s):
        c = np.full(25 * (40 + 40) + int(gap_s * 25), -1, np.int8)
        c[:25 * 40] = 1
        c[25 * 40 + int(gap_s * 25):] = 1
        return c
    r = _phase_runs(codes(20), fps, 1.0, 3.0, 60.0)
    assert r == [(0, len(codes(20)) - 1, 1)]          # one A phase, spanning the 20 s gap
    r = _phase_runs(codes(90), fps, 1.0, 3.0, 60.0)
    assert r[0] == (0, 25 * 40 - 1, 1)                # 90 s gap ends the run at the last ball frame
    assert r[1][0] == 25 * 40 + 90 * 25
