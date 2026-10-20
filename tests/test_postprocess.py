import numpy as np

from mtah.spells.postprocess import PostParams, postprocess


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
