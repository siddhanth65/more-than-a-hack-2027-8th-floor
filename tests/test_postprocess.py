import numpy as np

from mtah.spells.postprocess import PostParams, postprocess, snap_team_boundaries


def lab(s):
    return np.array(list(s), dtype="<U1")


def test_island_and_fill_and_aba():
    half = np.ones(60, np.int8)
    x = lab("D" * 10 + "H" * 5 + "D" * 10 + "H" * 10 + "D" * 3 + "H" * 10 + "A" * 4 + "H" * 8)
    p = PostParams(island_s=0.4, d_fill_s=0.2, aba_s=0.2)      # 10 / 5 / 5 frames at 25 fps
    out = postprocess(x, half, 25.0, p)
    assert "".join(out[10:15]) == "D" * 5            # 5-frame island dropped
    assert "".join(out[35:38]) == "H" * 3            # 3-frame D hole filled
    assert "".join(out[48:52]) == "H" * 4            # 4-frame A blip merged


def test_strong_touch_protects_blip():
    half = np.ones(40, np.int8)
    x = lab("H" * 20 + "A" * 4 + "H" * 16)
    strong_A = np.zeros(40, bool); strong_A[22] = True
    out = postprocess(x, half, 25.0, PostParams(aba_s=0.2), strong_A=strong_A)
    assert "".join(out[20:24]) == "A" * 4


def test_chained_interactions_fill_first():
    half = np.ones(33, np.int8)
    x = lab("D" * 10 + "H" * 5 + "D" * 3 + "H" * 5 + "D" * 10)
    out = postprocess(x, half, 25.0, PostParams(island_s=0.4, d_fill_s=0.2, aba_s=0.0))
    assert "".join(out[10:23]) == "H" * 13


def test_many_blips_is_fast():
    import time
    n = 60000
    x = np.array(["H"] * n, dtype="<U1")
    for k in range(2000):
        s = 100 + k * 28
        x[s:s + 3] = "A"
    half = np.ones(n, np.int8)
    t0 = time.perf_counter()
    out = postprocess(x, half, 25.0, PostParams(island_s=0.4, d_fill_s=0.2, aba_s=0.5))
    assert time.perf_counter() - t0 < 2.0
    assert (out == "H").all()


def test_float_thresholds_not_rounded():
    half = np.ones(40, np.int8)
    x = lab("D" * 10 + "H" * 11 + "D" * 10 + "A" * 9)      # island_s 0.45 s = 11.25 frames at 25 fps
    out = postprocess(x, half, 25.0, PostParams(island_s=0.45, d_fill_s=0.0, aba_s=0.0))
    assert "".join(out[10:21]) == "D" * 11                   # an 11-frame run is shorter than 11.25


def _touches(*rows):
    return np.array(rows, dtype=float).reshape(-1, 4)


def _snap_case(touch_rows, snap_w=50):
    half = np.ones(200, np.int8)
    x = lab("H" * 100 + "A" * 100)
    return snap_team_boundaries(x, half, _touches(*touch_rows), snap_w)


def test_snap_moves_boundary_to_new_team_touch():
    out = _snap_case([(93, 0, 1, 1.0)])
    assert "".join(out[:93]) == "H" * 93 and "".join(out[93:]) == "A" * 107


def test_snap_moves_boundary_later():
    out = _snap_case([(107, 0, 1, 1.0)])
    assert "".join(out[:107]) == "H" * 107 and "".join(out[107:]) == "A" * 93


def test_snap_ignores_other_team_touch():
    out = _snap_case([(93, 0, 0, 1.0)])
    assert "".join(out) == "H" * 100 + "A" * 100


def test_snap_ignores_touch_outside_window():
    out = _snap_case([(49, 0, 1, 1.0), (151, 0, 1, 1.0)])
    assert "".join(out) == "H" * 100 + "A" * 100
    out = _snap_case([(50, 0, 1, 1.0)])           # exactly at f - snap_w is inside
    assert "".join(out[:50]) == "H" * 50 and out[50] == "A"


def test_snap_nearest_and_tie_goes_earlier():
    out = _snap_case([(95, 0, 1, 1.0), (90, 0, 1, 1.0)])
    assert out[94] == "H" and out[95] == "A"
    out = _snap_case([(95, 0, 1, 1.0), (105, 0, 1, 1.0)])
    assert out[94] == "H" and out[95] == "A"


def test_snap_cannot_swallow_one_frame_run():
    half = np.ones(40, np.int8)
    x = lab("H" * 19 + "A" + "H" * 20)          # A at 19, then H run 20..39
    # transition A->H at 20; touch for H at 10 would cross the A->? boundary at 19
    out = snap_team_boundaries(x, half, _touches((10, 0, 0, 1.0)), 50)
    assert "".join(out[:19]) == "H" * 19 and out[19] == "A" and out[20] == "H"
    # H->A at 19 can move back to 10 only if previous run allows: prev run is 0..18, start 0
    out = snap_team_boundaries(x, half, _touches((10, 0, 1, 1.0)), 50)
    assert "".join(out[:10]) == "H" * 10 and "".join(out[10:20]) == "A" * 10 and out[20] == "H"


def test_snap_clips_to_keep_next_run():
    half = np.ones(40, np.int8)
    x = lab("H" * 10 + "A" * 2 + "H" * 28)
    # touch for A at 30 would push the H->A boundary past the A run; clipped to 11 (A keeps 1 frame)
    out = snap_team_boundaries(x, half, _touches((30, 0, 1, 1.0)), 50)
    assert "".join(out[:11]) == "H" * 11 and out[11] == "A" and out[12] == "H"


def test_snap_d_boundaries_untouched():
    half = np.ones(60, np.int8)
    x = lab("H" * 20 + "D" * 10 + "A" * 30)
    out = snap_team_boundaries(x, half, _touches((15, 0, 0, 1.0), (25, 0, 1, 1.0), (35, 0, 1, 1.0)), 50)
    assert "".join(out) == "".join(x)


def test_snap_stays_inside_half():
    half = np.concatenate([np.ones(100, np.int8), 2 * np.ones(100, np.int8)])
    x = lab("H" * 100 + "A" * 100)             # H->A at the half boundary is not a transition
    out = snap_team_boundaries(x, half, _touches((95, 0, 1, 1.0)), 50)
    assert "".join(out) == "".join(x)
    x = lab("H" * 70 + "A" * 130)             # transition at 70 in half 1; touch at 120 is in half 2
    out = snap_team_boundaries(x, half, _touches((120, 0, 1, 1.0)), 50)
    assert "".join(out) == "".join(x)


def test_snap_left_to_right_uses_updated_labels():
    half = np.ones(60, np.int8)
    x = lab("H" * 20 + "A" * 20 + "H" * 20)
    out = snap_team_boundaries(x, half, _touches((15, 0, 1, 1.0), (35, 0, 0, 1.0)), 50)
    assert "".join(out[:15]) == "H" * 15 and "".join(out[15:35]) == "A" * 20 and "".join(out[35:]) == "H" * 25


def test_postprocess_snap_wiring_and_disable():
    half = np.ones(200, np.int8)
    x = lab("H" * 100 + "A" * 100)
    t = _touches((93, 0, 1, 1.0))
    out = postprocess(x, half, 25.0, PostParams(), touches=t)
    assert out[93] == "A" and out[92] == "H"
    assert "".join(postprocess(x, half, 25.0, PostParams(snap_w=0), touches=t)) == "".join(x)
    assert "".join(postprocess(x, half, 25.0, PostParams())) == "".join(x)
