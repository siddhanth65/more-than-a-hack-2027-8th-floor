import numpy as np

from mtah.decode.scale import correlation_length, evidence_margin
from mtah.decode.viterbi import DecoderParams, decode


def test_correlation_length():
    rng = np.random.default_rng(0)
    white = rng.standard_normal(5000)
    assert 0.8 < correlation_length(white) < 1.3
    smooth = np.convolve(white, np.ones(25) / 25, mode="same")
    assert correlation_length(smooth) > 10


def test_decode_sticky_vs_flicker():
    F = 300
    ev = np.full((F, 3), np.nan, np.float32)
    ev[:, 0] = np.log(0.7); ev[:, 1] = np.log(0.3)          # weak H everywhere
    ev[100:103, 0] = np.log(0.1); ev[100:103, 1] = np.log(0.9)  # 3-frame A blip
    ev[200:, 0] = np.log(0.1); ev[200:, 1] = np.log(0.9)        # real switch to A
    half = np.ones(F, np.int8); half[:10] = 0
    p = DecoderParams(weights={"x": 1.0}, lengths={"x": 1.0}, tau_HA=10.0, tau_HD=10.0, tau_DH=10.0)
    lab = decode({"x": ev}, half, p)
    assert "".join(lab[:10]) == "D" * 10
    assert "".join(lab[10:200]) == "H" * 190 and "".join(lab[200:]) == "A" * 100
    p0 = DecoderParams(weights={"x": 1.0}, lengths={"x": 1.0}, tau_HA=0.0, tau_HD=0.0, tau_DH=0.0)
    assert "".join(decode({"x": ev}, half, p0)[100:103]) == "AAA"


def test_decode_dead_stream():
    F = 100
    team = np.full((F, 3), np.nan, np.float32); team[:, 0] = np.log(0.8); team[:, 1] = np.log(0.2)
    dead = np.full((F, 3), np.nan, np.float32); dead[:, 2] = np.log(0.01); dead[:, 0] = dead[:, 1] = np.log(0.99)
    dead[40:70, 2] = np.log(0.95); dead[40:70, 0] = dead[40:70, 1] = np.log(0.05)
    p = DecoderParams(weights={"t": 1.0, "d": 1.0}, lengths={"t": 1.0, "d": 1.0}, tau_HA=5, tau_HD=3, tau_DH=3)
    lab = decode({"t": team, "d": dead}, np.ones(F, np.int8), p)
    assert "".join(lab[:40]) == "H" * 40 and "".join(lab[40:70]) == "D" * 30 and "".join(lab[70:]) == "H" * 30
    assert evidence_margin(team)[0] > 0


def test_residual_margin_removes_signal():
    from mtah.decode.scale import residual_margin
    rng = np.random.default_rng(0)
    truth = np.array((["H"] * 200 + ["A"] * 200) * 5, dtype="<U1")
    margin = np.where(truth == "H", 4.0, -4.0) + rng.normal(size=len(truth))
    res = residual_margin(margin, truth, ("H", "A"))
    assert correlation_length(res) < 2 and correlation_length(margin) > 20
    margin[3] = np.nan; truth[5] = "D"
    res = residual_margin(margin, truth, ("H", "A"))
    assert np.isnan(res[3]) and np.isnan(res[5]) and np.isfinite(res[6])
