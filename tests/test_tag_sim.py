import numpy as np
import pandas as pd

from mtah.sim.tag_sim import TagSimParams, simulate_tags
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
    m.ball[25 * 69, 1] = 34.0             # last ball before the D run: on the touchline -> throw-in
    events = pd.DataFrame([(25 * 80, "H", 8, False)], columns=["frame", "team", "cls", "poss_only"])
    mask = np.zeros(m.n_frames, bool); mask[25 * 70:25 * 80] = True
    p = TagSimParams(keep_outside=1.0, keep_inside=1.0, jitter_s=0.0)
    df = simulate_tags(m, labels, events, mask, p, np.random.default_rng(0))
    phases = df[df["code"] == "PROGRESSION"]
    assert list(phases["team"]) == ["H", "A", "H"]
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
