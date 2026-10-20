import numpy as np
import pytest

from mtah.io.match import speeds_from_positions
from mtah.sim.blackout_sim import (BlackoutParams, apply_blackouts, blackout_mask, blackout_stats, entry_geometry,
                                   ks_statistic, pooled_blackout_stats, runs_of)
from tests.test_match import tiny_match


def synthetic(F=25 * 600, fps=25):
    """10 minutes: alternating 40 s alive / 20 s dead; ball near a line when dead."""
    labels = np.full(F, "H", dtype="<U1")
    ball = np.zeros((F, 2), np.float32)
    t = np.arange(F)
    dead = (t % (60 * fps)) >= 40 * fps
    labels[dead] = "D"
    ball[dead, 0] = 52.0            # 0.5 m from the goal line while dead
    return labels, ball


def test_runs_of():
    assert runs_of(np.array([0, 1, 1, 0, 1], bool)) == [(1, 3), (4, 5)]


def test_dead_anchored_only():
    labels, ball = synthetic()
    p = BlackoutParams(p_mask_near=1.0, p_mask_far=0.0, c_lo=1.0, c_hi=1.0,
                       pre_lo_s=0, pre_hi_s=0, post_lo_s=0, post_hi_s=0, live_rate_per_min=0.0)
    mask = blackout_mask(labels, ball, (105.0, 68.0), 25, p, np.random.default_rng(0))
    assert mask[labels == "D"].all() and not mask[labels == "H"].any()
    st = blackout_stats(mask, labels, 25)
    assert st["n_blackouts"] == 10 and st["dead_share_inside"] == pytest.approx(1.0)
    assert st["median_s"] == pytest.approx(20.0)


def test_live_component_and_seed_reproducible():
    labels, ball = synthetic()
    p = BlackoutParams(p_mask_near=0.0, p_mask_far=0.0, live_rate_per_min=2.0)
    m1 = blackout_mask(labels, ball, (105.0, 68.0), 25, p, np.random.default_rng(1))
    m2 = blackout_mask(labels, ball, (105.0, 68.0), 25, p, np.random.default_rng(1))
    assert np.array_equal(m1, m2) and m1.any()
    assert blackout_stats(m1, labels, 25)["masked_share"] < 0.5


def test_apply_blackouts():
    m = tiny_match()
    mask = np.zeros(m.n_frames, bool); mask[0] = True
    m2 = apply_blackouts(m, mask)
    assert np.isnan(m2.ball[0]).all() and not np.isnan(m2.ball[1]).any()
    assert not np.isnan(m.ball[0]).any()   # original untouched


def test_near_far_branch():
    labels, ball = synthetic()
    far = BlackoutParams(p_mask_near=0.0, p_mask_far=1.0, c_lo=1.0, c_hi=1.0,
                         pre_lo_s=0, pre_hi_s=0, post_lo_s=0, post_hi_s=0, live_rate_per_min=0.0)
    # every dead run starts 0.5 m from the goal line -> all are "near" -> p_far never applies
    assert not blackout_mask(labels, ball, (105.0, 68.0), 25, far, np.random.default_rng(0)).any()


def test_nan_first_frame_is_skipped():
    labels, ball = synthetic()
    ball = ball.copy()
    for s, e in runs_of(labels == "D"):
        ball[s:s + 3, :] = np.nan          # first 3 dead frames have no ball
    p = BlackoutParams(p_mask_near=1.0, p_mask_far=0.0, c_lo=1.0, c_hi=1.0,
                       pre_lo_s=0, pre_hi_s=0, post_lo_s=0, post_hi_s=0, live_rate_per_min=0.0)
    mask = blackout_mask(labels, ball, (105.0, 68.0), 25, p, np.random.default_rng(0))
    assert mask[labels == "D"].all()       # distance taken from the first non-NaN frame


def test_merge_short_gaps():
    labels = np.full(25 * 60, "H", dtype="<U1"); ball = np.zeros((len(labels), 2), np.float32)
    labels[0:250] = "D"; labels[260:510] = "D"      # two dead runs 10 frames (0.4 s) apart
    ball[labels == "D", 0] = 52.0
    p = BlackoutParams(p_mask_near=1.0, p_mask_far=0.0, c_lo=1.0, c_hi=1.0,
                       pre_lo_s=0, pre_hi_s=0, post_lo_s=0, post_hi_s=0, live_rate_per_min=0.0, merge_s=1.0)
    mask = blackout_mask(labels, ball, (105.0, 68.0), 25, p, np.random.default_rng(0))
    assert runs_of(mask) == [(0, 510)]


def test_min_duration_drops_short_runs():
    mask = np.zeros(25 * 60, bool); mask[100:150] = True; mask[500:700] = True   # 2 s and 8 s
    from mtah.sim.blackout_sim import drop_short_runs
    out = drop_short_runs(mask, 25, 5.0)
    assert runs_of(out) == [(500, 700)]


def test_short_component_lengths():
    labels = np.full(25 * 600, "H", dtype="<U1"); ball = np.zeros((len(labels), 2), np.float32)
    p = BlackoutParams(p_mask_near=0.0, p_mask_far=0.0, live_rate_per_min=0.0, short_rate_per_min=1.0, merge_s=0.0)
    mask = blackout_mask(labels, ball, (105.0, 68.0), 25, p, np.random.default_rng(3))
    lens = np.array([e - s for s, e in runs_of(mask)]) / 25.0
    assert len(lens) >= 5 and lens.min() >= 5.0 and lens.max() <= 10.0 + 10.0   # overlaps may chain two


def test_stats_per_graded_minute():
    labels = np.full(25 * 120, "H", dtype="<U1")
    mask = np.zeros(len(labels), bool); mask[500:1000] = True            # one 20 s run
    graded = np.ones(len(labels), bool); graded[:1500] = False           # 60 s ungraded, 60 s graded
    st = blackout_stats(mask, labels, 25, graded=graded)
    assert st["n_blackouts"] == 0 and st["per_graded_min"] == 0.0        # the run lies in the ungraded part
    mask[2000:2500] = True
    st = blackout_stats(mask, labels, 25, graded=graded)
    assert st["n_blackouts"] == 1 and st["per_graded_min"] == pytest.approx(1.0)
    assert st["lengths_s"] == [20.0] and st["masked_share"] == pytest.approx(500 / 1500)
    assert blackout_stats(mask, labels, 25)["per_graded_min"] == pytest.approx(2 / 2.0)


def entry_fixture():
    F = 25 * 120
    ball = np.zeros((F, 2), np.float32)
    ball[1000 - 6:1000, 0] = np.linspace(40, 52, 6)     # heads to the goal line, then vanishes
    mask = np.zeros(F, bool); mask[1000:1500] = True    # 20 s
    return mask, ball


def test_entry_geometry():
    mask, ball = entry_fixture()
    assert entry_geometry(mask, ball, (105.0, 68.0)) == (1.0, 1.0)
    assert all(np.isnan(v) for v in entry_geometry(np.zeros(len(mask), bool), ball, (105.0, 68.0)))


def test_pooled_stats_two_halves():
    labels = np.full(25 * 120, "H", dtype="<U1")
    mask, ball = entry_fixture()
    m2 = np.zeros_like(mask); m2[100:1100] = True                      # 40 s
    graded = np.ones(len(mask), bool)
    st = pooled_blackout_stats([(mask, labels, 25, graded, ball, (105.0, 68.0)),
                                (m2, labels, 25, graded, ball, (105.0, 68.0))])
    assert st["n_blackouts"] == 2 and st["per_graded_min"] == pytest.approx(2 / 4.0)
    assert st["median_s"] == pytest.approx(30.0) and st["max_s"] == pytest.approx(40.0)
    assert st["masked_share"] == pytest.approx(1500 / 6000)
    assert sorted(st["lengths_s"]) == [20.0, 40.0]
    assert st["entry_near_line_share"] == pytest.approx(0.5)           # only the first entry is near the line
    empty = pooled_blackout_stats([(np.zeros(len(mask), bool), labels, 25, graded, ball, (105.0, 68.0))])
    assert np.isnan(empty["median_s"]) and np.isnan(empty["p90_s"]) and np.isnan(empty["max_s"])


def test_ks_statistic():
    a = [1.0, 2.0, 3.0]
    assert ks_statistic(a, a) == 0.0
    assert ks_statistic([1.0, 2.0], [10.0, 11.0]) == 1.0
    assert ks_statistic([1.0, 2.0, 3.0, 4.0], [3.0, 4.0, 5.0, 6.0]) == pytest.approx(0.5)


def test_half_rng_reproducible_and_distinct():
    from mtah.sim import EVAL_SEEDS, FIT_SEEDS, half_rng
    assert half_rng(3, 0).random() == half_rng(3, 0).random()
    assert half_rng(3, 0).random() != half_rng(3, 1).random()
    assert not set(FIT_SEEDS) & set(EVAL_SEEDS)


def test_live_durations_resampled_above_min():
    labels = np.full(25 * 600, "H", dtype="<U1"); ball = np.zeros((len(labels), 2), np.float32)
    p = BlackoutParams(p_mask_near=0.0, p_mask_far=0.0, live_rate_per_min=2.0, live_median_s=1.0,
                       live_sigma_log=0.3, merge_s=0.0, min_s=5.0)
    mask = blackout_mask(labels, ball, (105.0, 68.0), 25, p, np.random.default_rng(0))
    # a median of 1 s would drop nearly every dropout; resampling / clamping to min_s keeps them
    assert len(runs_of(mask)) >= 10


def test_apply_blackouts_recomputes_ball_speed():
    m = tiny_match()
    m.ball[:] = 0.0; m.ball[:, 0] = np.arange(m.n_frames)
    m.ball_speed[:] = speeds_from_positions(m.ball, 25.0)
    mask = np.zeros(m.n_frames, bool); mask[5:8] = True
    m2 = apply_blackouts(m, mask)
    assert np.isnan(m2.ball_speed[4]) and np.isnan(m2.ball_speed[8])    # neighbour of a mask edge is NaN
    assert m2.ball_speed[3] == pytest.approx(25.0)
