"""Tests of the arbitration helpers (scripts/arb_common.py): pseudo-label rule and pseudo-blackout sampler."""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import arb_common as ac   # noqa: E402
from mtah.io.match import Match, speeds_from_positions   # noqa: E402

FPS = 25.0


def make_match(F=400, S=4):
    pos = np.full((F, S, 2), 30.0, np.float32)
    pos[:, 0] = [0.0, 0.0]; pos[:, 1] = [20.0, 0.0]          # home 0, home 1
    pos[:, 2] = [10.0, 10.0]; pos[:, 3] = [-15.0, 5.0]       # away 2, away 3
    team = np.array([0, 0, 1, 1], np.int8)
    ball = np.tile(np.array([0.5, 0.0], np.float32), (F, 1))  # next to home player 0, others far
    half = np.ones(F, np.int8)
    return Match(fps=FPS, pos=pos, speed=speeds_from_positions(pos, FPS)[..., 0] if False else np.zeros((F, S), np.float32),
                 team=team, ball=ball, ball_speed=np.zeros(F, np.float32), half=half)


def test_rule_labels_middle_of_long_run():
    m = make_match()
    lab = ac.confident_control(m, ac.RuleParams(r_near=1.5, r_opp=3.0, v_rel=2.0, min_run_s=1.0, trim_s=0.5))
    assert (lab[20:380] == 0).all()                  # home player 0 closest, nearest opponent > 3 m
    assert (lab[:12] == -1).all() and (lab[-12:] == -1).all()   # trimmed ends (incl. edge frames with NaN velocity)


def test_rule_needs_opponent_far_and_run_length():
    m = make_match()
    m.pos[100:200, 2] = [3.0, 0.0]                   # an away player 2.5 m from the ball (< r_opp)
    lab = ac.confident_control(m)
    assert (lab[100:200] == -1).all()
    m2 = make_match(); m2.pos[:, 2] = [50.0, 50.0]
    m2.pos[50:70, 0] = [40.0, 0.0]                   # the closest player leaves for 20 frames: runs 1.2 s / break
    lab2 = ac.confident_control(m2)
    assert (lab2[50:70] == -1).all()


def test_rule_ball_missing_and_moving():
    m = make_match(); m.ball[100:140] = np.nan
    lab = ac.confident_control(m)
    assert (lab[100:140] == -1).all()
    m = make_match()
    m.ball[:, 0] = 0.5 + np.arange(400) * 0.4        # ball moves 10 m/s away from a static player
    assert (ac.confident_control(m) == -1).all()


def test_rule_away_team():
    m = make_match(); m.pos[:, 2] = [0.5, 0.0]; m.pos[:, 0] = [8.0, 8.0]
    lab = ac.confident_control(m)
    assert (lab[20:380] == 1).all()


def test_mask_only_in_visible_and_fraction():
    m = make_match(F=20000); m.ball[5000:9000] = np.nan
    pool = np.array([6.0, 12.0, 20.0, 40.0])
    mk = ac.sample_mask(m, seed=3, pool_s=pool, unit_id=1, target=0.3)
    assert not (mk & ~m.ball_present).any()
    frac = mk.sum() / m.ball_present.sum()
    assert 0.28 <= frac <= 0.36
    assert all((e - s) / FPS >= 5.0 for s, e in ac.runs_of(mk))   # no window under 5 s (merges only lengthen)
    mk2 = ac.sample_mask(m, seed=3, pool_s=pool, unit_id=1, target=0.3)
    assert (mk == mk2).all()                                              # deterministic
    assert (ac.sample_mask(m, seed=4, pool_s=pool, unit_id=1, target=0.3) != mk).any()


def test_apply_mask_hides_ball_and_recomputes_speed():
    m = make_match(); mk = np.zeros(400, bool); mk[100:200] = True
    m2 = ac.apply_mask(m, mk)
    assert np.isnan(m2.ball[100:200]).all() and m2.ball_present[:100].all()
    assert np.isnan(m2.ball_speed[100:200]).all()


def test_entry_matched_mask_and_centered():
    m = make_match(F=40000); m.ball[:, 0] = np.linspace(-50, 50, 40000)   # sweeps across the pitch: entries near/far lines
    pool = np.array([10.0, 20.0, 40.0])
    mk = ac.sample_mask(m, 1, pool, 0, entry=True)
    assert not (mk & ~m.ball_present).any() and 0.25 < mk.mean() < 0.4
    conf = np.full(40000, -1, np.int8); conf[::50] = 0
    cm, ctr = ac.centered_mask(m, conf, 0)
    assert len(ctr) > 3 and cm[ctr].all() and np.diff(ctr).min() >= 45 * 25
