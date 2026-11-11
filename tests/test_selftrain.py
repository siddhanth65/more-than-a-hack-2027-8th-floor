import numpy as np
import pytest

from mtah import selftrain as st


def test_cross_half_source_is_other_half():
    assert st.cross_half_source(1, [1, 2]) == 2
    assert st.cross_half_source(2, [2, 1]) == 1
    with pytest.raises(ValueError):
        st.cross_half_source(3, [1, 2])
    with pytest.raises(ValueError):
        st.cross_half_source(1, [1, 2, 3])


def test_leakage_check_refuses_own_half():
    st.leakage_check(1, 2, [3, 4])
    with pytest.raises(AssertionError):
        st.leakage_check(1, 1, [3])
    with pytest.raises(AssertionError):
        st.leakage_check(1, 2, [1, 3])


def _seqs(rng, n=3000, nf=4):
    X = rng.normal(size=(n, nf)).astype(np.float32); y = rng.integers(0, 3, n)
    return X, y


def test_epoch_never_contains_target_half_pseudo():
    """Pseudo windows come only from the pseudo sequence passed in (the other half): tag every feature row of the
    other half with a marker and check that windows holding it are exactly the pseudo ones."""
    rng = np.random.default_rng(0)
    Xs, ys = _seqs(rng)                                   # simulator
    Xo = np.full((3000, 4), 7.0, np.float32)              # the OTHER half's rows carry the marker 7
    yo = np.full(3000, st.IGNORE); yo[100:900] = 0
    mir = lambda x: x; sw = lambda y: y
    ep = st.build_epoch([(Xs, ys)], (Xo, yo), 1.0, rng, 600, 300, mir, sw)
    pw = [e for e in ep if e["pseudo"]]; sw_ = [e for e in ep if not e["pseudo"]]
    assert pw and all((e["x"] == 7.0).all() for e in pw)
    assert sw_ and all(not (e["x"] == 7.0).any() for e in sw_)
    assert len(sw_) <= st.SIM_PER_PSEUDO * len(pw)


def test_window_exclusion():
    fl = np.zeros(30000, np.int8)
    out = st.exclude_frames(fl, [st.PUBLIC_WINDOW])
    assert (out[11185:18686] == -1).all() and (out[:11185] == 0).all() and (out[18686:] == 0).all()
    assert fl[12000] == 0                                      # input untouched
    # an excluded frame never becomes a step label
    starts = np.arange(0, 30000, 5)
    y = st.to_step_labels(out, starts)
    inside = (starts + 2 >= 11185) & (starts + 2 <= 18685)
    assert (y[inside] == st.IGNORE).all() and (y[~inside] == 0).all()
    # the V3 labels fill gaps but the excluded frames are excluded after combination
    comb = st.exclude_frames(st.combine_frame_labels(np.full(30000, -1, np.int8), np.full(30000, 2, np.int8)), [st.PUBLIC_WINDOW])
    assert (comb[11185:18686] == -1).all() and (comb[:11185] == 2).all()


def test_combine_primary_wins():
    p = np.array([0, -1, 1, -1], np.int8); s = np.array([1, 2, 2, -1], np.int8)
    assert st.combine_frame_labels(p, s).tolist() == [0, 2, 1, -1]
    assert st.combine_frame_labels(p).tolist() == [0, -1, 1, -1]


def test_merged_posterior_threshold():
    cost = np.array([[0.0, 5.0, 5.0], [0.0, 0.1, 5.0], [5.0, 0.0, 5.0]])
    dec = np.array(list("HHA"))
    out = st.merged_posterior_labels(cost, dec, 0.9)
    assert out.tolist() == [0, -1, 1]                          # frame 1: posterior of H is about 0.5 -> dropped
    assert st.merged_posterior_labels(cost, np.array(list("ADD")), 0.9).tolist() == [-1, -1, -1]


def test_step_weights_and_weighted_ce():
    y = np.array([0, 1, 3, 2])
    assert st.step_weights(y, 0.5, True).tolist() == [0.5, 0.5, 0.0, 0.5]
    assert st.step_weights(y, 0.5, False).tolist() == [1.0, 1.0, 0.0, 1.0]
    lp = np.log(np.full((4, 3), 1 / 3)); lp[0, 0] = np.log(0.5); lp[1, 1] = np.log(0.25)
    w = np.array([1.0, 0.5, 0.0, 0.5])
    exp = (1.0 * -np.log(0.5) + 0.5 * -np.log(0.25) + 0.5 * np.log(3)) / 2.0
    assert st.weighted_ce(lp, y, w) == pytest.approx(exp)
    assert st.weighted_ce(lp, y, np.zeros(4)) == 0.0


def test_pseudo_weight_scales_pseudo_windows_only():
    rng = np.random.default_rng(1)
    Xs, ys = _seqs(rng); Xo, yo = _seqs(rng); yo[:] = st.IGNORE; yo[0:1500] = 1
    for w in (1.0, 0.5, 0.0):
        ep = st.build_epoch([(Xs, ys)], (Xo, yo), w, np.random.default_rng(2), 600, 300, lambda x: x, lambda y: y)
        for e in ep:
            lab = e["y"] != st.IGNORE
            assert (e["w"][~lab] == 0).all()
            assert (e["w"][lab] == (w if e["pseudo"] else 1.0)).all()


def test_pseudo_windows_need_enough_labels():
    y = np.full(3000, st.IGNORE); y[0:20] = 0; y[1200:1900] = 1
    wins = st.pseudo_windows(y, 600, 300, 0, 30)
    assert 0 not in wins and 900 in wins and 1200 in wins
    assert all((y[a:a + 600] != st.IGNORE).sum() >= 30 for a in wins)


def test_swap_augmentation_flips_labels_consistently():
    rng = np.random.default_rng(3)
    Xs, ys = _seqs(rng); Xo, yo = _seqs(rng)
    sw = lambda y: np.where(y == 0, 1, np.where(y == 1, 0, y))
    ep = st.build_epoch([(Xs, ys)], (Xo, yo), 1.0, rng, 600, 300, lambda x: -x, sw)
    for e in ep:
        src = Xo if e["pseudo"] else Xs
        # each window is either the original or its team-swapped version
        found = False
        for a in range(0, len(src) - 600 + 1):
            if np.array_equal(e["x"], src[a:a + 600]) or np.array_equal(e["x"], -src[a:a + 600]):
                found = True; break
        assert found


def test_per_half_model_dispatch(monkeypatch, tmp_path):
    """seqstream.compute_streams with a per-half modelB dict runs each half's checkpoint on that half's steps only."""
    import mtah.evidence.seqstream as sq
    import mtah.evidence.seq_feats as fx
    seen = {}

    def fake_b(path, Xs, seg, names, python=None, threads=4):
        seen[path.name] = sorted(set(seg.tolist()))
        lp = np.zeros((len(Xs), 3), np.float32); lp[:, 0] = {"h1.pt": -1.0, "h2.pt": -2.0}[path.name]
        return lp

    monkeypatch.setattr(sq, "model_b_logp", fake_b)
    F = 400
    z = {"fps": np.float32(25.0), "half": np.r_[np.full(200, 1), np.full(200, 2)].astype(np.int8)}
    monkeypatch.setattr(sq.fx, "frame_features", lambda z: (np.zeros((F, 3), np.float32), ["a", "b", "c"]))
    spec = {"weights": {"seqB": 1.0}, "modelB": {"1": "h1.pt", "2": "h2.pt"}}
    out = sq.compute_streams(z, spec, tmp_path)["seqB"]
    assert seen == {"h1.pt": [1], "h2.pt": [2]}
    assert (out[:200, 0] == -1.0).all() and (out[200:, 0] == -2.0).all()
