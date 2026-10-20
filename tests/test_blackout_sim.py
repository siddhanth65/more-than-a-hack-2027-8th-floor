import numpy as np
import pytest

from mtah.sim.blackout_sim import BlackoutParams, apply_blackouts, blackout_mask, blackout_stats, runs_of
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
    labels = np.full(25 * 60, "H", dtype="<U1"); ball = np.zeros((len(labels), 2), np.float32)
    p = BlackoutParams(p_mask_near=0.0, p_mask_far=0.0, live_rate_per_min=0.0, min_s=5.0)
    mask = np.zeros(len(labels), bool); mask[100:150] = True; mask[500:700] = True   # 2 s and 8 s
    from mtah.sim.blackout_sim import drop_short_runs
    out = drop_short_runs(mask, 25, 5.0)
    assert runs_of(out) == [(500, 700)]
