import os

import numpy as np
import pandas as pd
import pytest

from mtah.baselines import b0_nearest_hold, b4_blackout_dead, hold_last, nearest_team, smooth_vote
from mtah.io.match import Match, speeds_from_positions


def two_player_match(F=300):
    pos = np.zeros((F, 2, 2), np.float32)
    pos[:, 0, 0] = -10.0   # home at x=-10
    pos[:, 1, 0] = 10.0    # away at x=+10
    ball = np.zeros((F, 2), np.float32)
    ball[:100, 0] = -8.0   # near home
    ball[100:200, 0] = 8.0 # near away
    ball[200:, :] = np.nan # absent for 100 frames (4 s)
    half = np.ones(F, np.int8); half[:10] = 0
    return Match(fps=25.0, pos=pos, speed=speeds_from_positions(pos, 25.0), team=np.array([0, 1], np.int8),
                 ball=ball, ball_speed=speeds_from_positions(ball, 25.0), half=half, pitch=(105.3, 68.0))


def test_nearest_and_hold():
    m = two_player_match()
    nt = nearest_team(m)
    assert nt[50] == 0 and nt[150] == 1 and nt[250] == -1
    held = hold_last(nt, m.half)
    assert held[250] == 1
    assert held[5] == nt[5]        # frames outside every half are left as they are (not filled)
    assert hold_last(np.array([-1, -1, 0, -1], np.int8), np.array([0, 0, 1, 1], np.int8)).tolist() == [-1, -1, 0, 0]


def test_b0_and_b4():
    m = two_player_match()
    b0 = b0_nearest_hold(m)
    assert b0[5] == "D" and b0[50] == "H" and b0[150] == "A" and b0[250] == "A"
    b4 = b4_blackout_dead(m, min_gap_s=2.0)
    assert b4[250] == "D" and b4[150] == "A"
    assert b4_blackout_dead(m, min_gap_s=10.0)[250] == "A"
    assert b4_blackout_dead(m, min_gap_s=4.0)[250] == "A"   # a gap of exactly 4 s is not longer than 4 s


def test_smooth_vote_window():
    m = two_player_match()
    s = smooth_vote(m, 10)
    assert s[50] == "H" and s[150] == "A" and s[260] == "D"
    assert s[205] == "A"          # window still sees votes just after the ball vanishes


@pytest.mark.real
def test_smooth50_parity(data_dir):
    from mtah.io.metrica_loader import load_tactical
    m = load_tactical(data_dir)
    ours = smooth_vote(m, 50)
    ref_path = os.environ.get("MTAH_CONTEST_REPO", "")
    ref_path = os.path.join(ref_path, "evaluation", "results", "baseline_smooth50.csv")
    if not os.path.exists(ref_path):
        pytest.skip("organisers' baseline_smooth50.csv not available (set MTAH_CONTEST_REPO to the organisers' repository checkout)")
    ref = pd.read_csv(ref_path)["label"].to_numpy()
    agree = float((ours == ref).mean())
    assert agree == 1.0


def test_hold_last_does_not_cross_half_boundary():
    held = hold_last(np.array([0, -1, -1, -1, -1, -1], np.int8), np.array([1, 1, 1, 2, 2, 2], np.int8))
    assert held.tolist() == [0, 0, 0, -1, -1, -1]
