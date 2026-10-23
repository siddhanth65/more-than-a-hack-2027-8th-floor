import numpy as np

from mtah.evidence.touches import TouchParams, detect_touches, strong_touch_frames, velocity_change
from mtah.io.match import Match, speeds_from_positions


def kick_match(F=100):
    pos = np.zeros((F, 2, 2), np.float32); pos[:, 0, 0] = 0.0; pos[:, 1, 0] = 30.0
    ball = np.zeros((F, 2), np.float32)
    ball[50:, 0] = np.arange(50) * 0.8     # ball at rest next to slot 0, then kicked at 20 m/s
    team = np.array([0, 1], np.int8)
    return Match(fps=25.0, pos=pos, speed=speeds_from_positions(pos, 25.0), team=team, ball=ball,
                 ball_speed=speeds_from_positions(ball, 25.0), half=np.ones(F, np.int8), pitch=(105.0, 68.0))


def test_velocity_change_peak():
    m = kick_match()
    dv = velocity_change(m.ball, 3, 25.0)
    assert np.nanargmax(dv) in (49, 50, 51, 52) and np.nanmax(dv) > 15


def test_detect_touch_by_home():
    m = kick_match()
    ABS = TouchParams(mode="abs")
    t = detect_touches(m, ABS)
    assert len(t) == 1 and t[0, 1] == 0 and t[0, 2] == 0 and t[0, 3] > 15
    strong = strong_touch_frames(t, 0, ABS, m.n_frames)
    assert strong.sum() == 1 and not strong_touch_frames(t, 1, ABS, m.n_frames).any()


def test_continuity_gate_drops_teleport():
    m = kick_match()
    m.ball[52, 0] += 10.0                   # a 10 m jump next to the touch
    assert len(detect_touches(m, TouchParams(mode="abs"))) == 0


# ---- mode "rel" (feed-adaptive) -------------------------------------------------------------------------------------
from dataclasses import replace  # noqa: E402


def pass_match(jitter=0.0, seed=0, F=220, pitch=(105.0, 68.0)):
    """Ball at rest at player 0 (x=0), kicked at 15 m/s along +x at frame 50, received at player 1 (x=30) at frame 100
    where it stops. A third player stands far away. One kink at the kick, one at the reception."""
    rng = np.random.default_rng(seed)
    pos = np.zeros((F, 3, 2), np.float32); pos[:, 0, 0] = 0.0; pos[:, 1, 0] = 30.0; pos[:, 2, :] = (-40.0, 20.0)
    x = np.zeros(F)
    t = np.arange(F)
    x = np.where(t < 50, 0.0, np.where(t < 100, (t - 50) * 15.0 / 25.0, 30.0))
    ball = np.c_[x, np.zeros(F)] + rng.normal(0, jitter, (F, 2))
    ball = ball.astype(np.float32)
    return Match(fps=25.0, pos=pos, speed=speeds_from_positions(pos, 25.0), team=np.array([0, 1, 1], np.int8), ball=ball,
                 ball_speed=speeds_from_positions(ball, 25.0), half=np.ones(F, np.int8), pitch=pitch)


REL = TouchParams(mode="rel")


def test_rel_pass_has_kick_and_reception():
    t = detect_touches(pass_match(0.02), REL)
    assert len(t) == 2
    assert abs(t[0, 0] - 50) <= 4 and t[0, 1] == 0 and t[0, 2] == 0
    assert abs(t[1, 0] - 100) <= 4 and t[1, 1] == 1 and t[1, 2] == 1


def test_rel_noise_only_path_has_no_touch():
    m = pass_match(0.05, seed=3)
    m.ball[:, 0] = m.ball[0, 0] + np.random.default_rng(1).normal(0, 0.05, len(m.ball))   # ball never leaves the player
    m.ball[:, 1] = np.random.default_rng(2).normal(0, 0.05, len(m.ball))
    assert len(detect_touches(m, REL)) == 0


def test_rel_same_touches_at_different_jitter_levels():
    lo = detect_touches(pass_match(0.02, seed=1), REL)[:, 0]
    hi = detect_touches(pass_match(0.08, seed=2), REL)[:, 0]
    assert len(lo) == len(hi) == 2 and np.all(np.abs(lo - hi) <= 3)


def test_rel_threshold_is_relative_to_the_half_noise_level():
    """A kink of fixed size is a touch in a quiet half and (with the same sigma-multiple) not in a very noisy one;
    the abs rule at theta 5 cannot tell the two feeds apart (it fires on the jitter peaks of the noisy one)."""
    quiet = pass_match(0.01, seed=4); noisy = pass_match(0.15, seed=5)
    # put the player on the ball for the whole noisy path so the proximity gate cannot hide jitter peaks
    noisy.pos[:, 0, 0] = noisy.ball[:, 0]; noisy.pos[:, 0, 1] = 0.0
    n_abs = len(detect_touches(noisy, TouchParams(k=3, mode="abs")))
    n_rel = len(detect_touches(noisy, replace(REL, max_rate_per_min=1e9)))
    assert n_abs > 10 and n_rel < n_abs
    assert len(detect_touches(quiet, REL)) == 2


def test_default_is_rel_and_abs_available():
    assert TouchParams().mode == "rel"
    m = kick_match()
    assert np.array_equal(detect_touches(m, TouchParams()), detect_touches(m, TouchParams(mode="rel")))
    assert len(detect_touches(m, TouchParams(mode="abs"))) == 1


def test_old_config_without_mode_loads_as_abs():
    from mtah.pipeline import PipelineConfig
    d = PipelineConfig().to_dict(); assert d["touch"]["mode"] == "rel"
    old = {k: v for k, v in d["touch"].items() if k in ("k", "theta_dv", "r_touch_m", "contest_ratio", "theta_hi")}
    d["touch"] = old
    assert PipelineConfig.from_dict(d).touch.mode == "abs"
    d["touch"] = dict(old, mode="rel")
    assert PipelineConfig.from_dict(d).touch.mode == "rel"


def test_rel_rate_cap_and_unknown_mode():
    import pytest
    with pytest.raises(ValueError):
        detect_touches(kick_match(), TouchParams(mode="nope"))
    m = pass_match(0.15, seed=6); m.pos[:, 0, 0] = m.ball[:, 0]; m.pos[:, 0, 1] = 0.0
    n = len(detect_touches(m, replace(REL, max_rate_per_min=1e9)))
    cap = len(detect_touches(m, replace(REL, max_rate_per_min=6.0, min_rate_per_min=0.0)))
    assert cap <= int(6.0 * 220 / 25 / 60) and cap < n


def test_rel_teleport_and_gap_guard():
    m = pass_match(0.02); m.ball[52, 0] += 10.0
    assert all(abs(f - 50) > 8 for f in detect_touches(m, REL)[:, 0])
    m2 = pass_match(0.02); m2.ball[45:48] = np.nan
    assert len(detect_touches(m2, REL)) == 1        # the kick sits next to a ball gap: only the reception survives
