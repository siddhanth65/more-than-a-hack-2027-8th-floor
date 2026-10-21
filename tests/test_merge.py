"""Per-spell merge model (mtah.spells.merge): spell table, exact spellF1, labels, apply_merges, predict wiring."""
import numpy as np
import pandas as pd
import pytest

from mtah.eval.scorer import score
from mtah.spells.merge import (FEATURES, NB_DEAD, NB_NONE, NB_OTHER, MergeModel, SpellF1, apply_merges, merge_labels,
                               spell_table)
from mtah.eval.official import possession_metrics as pm


def L(s):
    """'H5A1H2' -> array of labels."""
    out, i = [], 0
    while i < len(s):
        lab = s[i]; j = i + 1
        while j < len(s) and s[j].isdigit():
            j += 1
        out += [lab] * int(s[i + 1:j]); i = j
    return np.array(out, "<U1")


def _random_labels(rng, n, alphabet, mean_run):
    out = []
    while len(out) < n:
        out += [alphabet[rng.integers(len(alphabet))]] * int(rng.integers(1, 2 * mean_run))
    return np.array(out[:n], "<U1")


@pytest.mark.parametrize("seed", range(8))
def test_spellf1_equals_official_scorer(seed):
    rng = np.random.default_rng(seed)
    n = 3000
    gt = _random_labels(rng, n, "HHAAD", 40)
    gt[rng.integers(0, n // 2):][: rng.integers(0, 300)] = "X"        # an ungraded hole splits the valid runs
    pred = _random_labels(rng, n, "HHAD", 25 + seed)
    if seed % 2:
        pred[rng.integers(0, n, 40)] = "X"                             # abstentions are forward-filled by the scorer
    sc = SpellF1(pm.label_codes(gt))
    assert sc.f1(pm.label_codes(pred)) == pytest.approx(score(gt, pred)["spell_f1"], abs=1e-12)
    assert sc.f1(pm.label_codes(gt)) == pytest.approx(score(gt, gt)["spell_f1"], abs=1e-12)


def _ev(margin_h):
    """(F, 3) evidence with H - A = margin_h (H = m/2, A = -m/2), D NaN."""
    m = np.asarray(margin_h, np.float32)
    ev = np.full((len(m), 3), np.nan, np.float32); ev[:, 0] = m / 2; ev[:, 1] = -m / 2
    return ev


def test_spell_table_features():
    lab = L("D2H5A1H2D3A4")                      # fps 1 for readable lengths
    F = len(lab)
    ball = np.full(F, 1.0); ball[7] = -3.0       # frame 7 is the A spell: evidence favours A there by 3
    ball[8] = np.nan                             # frame 8 (H, first of H2): no opinion
    touches = np.array([[3, 0, 0, 1.0], [4, 0, 1, 1.0], [7, 0, 1, 1.0], [14, 0, 1, 1.0]])
    blackout = np.zeros(F, bool); blackout[13:15] = True
    bp = np.ones(F, bool); bp[8] = False
    sH = np.zeros(F, bool); sH[3] = True; sA = np.zeros(F, bool)
    tab = spell_table(lab, {"ball": _ev(ball)}, touches, 1.0, blackout=blackout, ball_present=bp, strong=(sH, sA))
    assert list(tab.columns[:5]) == ["start", "stop", "team", "target", "mergeable"] and set(FEATURES) <= set(tab.columns)
    assert tab[["start", "stop"]].values.tolist() == [[2, 7], [7, 8], [8, 10], [13, 17]]
    assert tab["team"].tolist() == ["H", "A", "H", "A"]
    h5, a1, h2, a4 = (tab.iloc[i] for i in range(4))
    assert h5.length_s == 5 and h5.prev_kind == NB_DEAD and h5.next_kind == NB_OTHER and h5.dead_prev == 1
    assert h5.mergeable and h5.target == "A" and h5.target_len_s == 1 and h5.aba == 0
    assert a1.aba == 1 and a1.target == "H" and a1.target_len_s == 7 and a1.prev_len_s == 5 and a1.next_len_s == 2
    assert a1.ball_mean == pytest.approx(3.0) and a1.ball_min == pytest.approx(3.0)      # signed towards A
    assert h5.ball_mean == pytest.approx(1.0) and h2.ball_mean == pytest.approx(1.0)     # NaN frame skipped
    assert h2.target == "A" and h2.next_kind == NB_DEAD and h2.target_len_s == 1 and h2.ball_absent_frac == 0.5
    assert a4.prev_kind == NB_DEAD and a4.next_kind == NB_NONE and not a4.mergeable and a4.target == ""
    assert a4.blackout_frac == 0.5 and np.isnan(a4.tag_mean) and np.isnan(a4.teamctrl_min)
    assert h5.n_touch_own == 1 and h5.n_touch_other == 1 and h5.n_strong_own == 1 and h5.touch_own_rate == pytest.approx(0.2)
    assert a1.n_touch_own == 1 and a4.n_touch_own == 1 and h2.n_touch_own == 0


def test_spell_table_respects_halves():
    lab = L("H3A3")
    half = np.array([1, 1, 1, 2, 2, 2])
    tab = spell_table(lab, {}, None, 1.0, half=half)
    assert len(tab) == 2 and not tab["mergeable"].any()       # the neighbour across the half boundary does not count
    tab2 = spell_table(lab, {}, None, 1.0)
    assert tab2["mergeable"].all()


def test_apply_merges_off_and_shortest_first():
    lab = L("H5A1H2A5")
    tab = spell_table(lab, {}, None, 1.0)
    p = np.array([0.0, 0.9, 0.95, 0.0])
    assert (apply_merges(lab, tab, p, 1.0) == lab).all()                 # theta 1.0: off
    assert (apply_merges(lab, tab, np.ones(4), 1.0) == lab).all()        # p = 1 is not > 1
    # shortest first: A1 merges into H (H8), then H2 is part of a grown run and is skipped
    assert "".join(apply_merges(lab, tab, p, 0.7)) == "".join(L("H8A5"))
    # only H2 above theta: it merges into the surrounding A
    assert "".join(apply_merges(lab, tab, np.array([0.0, 0.5, 0.95, 0.0]), 0.7)) == "".join(L("H5A8"))
    # D on both sides: never mergeable whatever p
    lab2 = L("D3H1D3A2")
    tab2 = spell_table(lab2, {}, None, 1.0)
    assert (apply_merges(lab2, tab2, np.ones(len(tab2)), 0.5) == lab2).all()
    # D on one side: extends the H/A neighbour, D untouched
    lab3 = L("D3A1H4")
    tab3 = spell_table(lab3, {}, None, 1.0)
    assert "".join(apply_merges(lab3, tab3, np.array([0.9, 0.0]), 0.5)) == "".join(L("D3H5"))


def test_apply_merges_rederives_neighbours():
    # A1 (shortest) merges into H first; H2 then has neighbours H... (grown run) -> skipped; the long spells only
    # merge when their run is unchanged: H6 is grown by A1's merge and is skipped too, A5 merges into the H run
    lab = L("H6A1H2A5D2")
    tab = spell_table(lab, {}, None, 1.0)
    out = apply_merges(lab, tab, np.array([0.9, 0.9, 0.9, 0.9]), 0.5)
    assert "".join(out) == "".join(L("H14D2"))


def test_merge_label_toy_one_second_spell_inside_h():
    fps = 25
    truth = L("D25H100D25")
    pred = L("D25H40A25H35D25")                   # a 1 s A spell inside a true H spell
    tab = spell_table(pred, {}, None, fps)
    y = merge_labels(truth, pred, tab)
    a = int(np.flatnonzero(tab["team"].to_numpy() == "A")[0])
    assert y[a] == 1.0
    merged = pred.copy(); merged[65:90] = "H"
    assert score(truth, merged)["spell_f1"] > score(truth, pred)["spell_f1"]   # the label agrees with the official scorer
    # the same spell is real in the truth: merging it hurts -> 0
    y2 = merge_labels(pred, pred, tab)
    assert y2[a] == 0.0
    assert np.isnan(merge_labels(truth, L("D25H100D25"), spell_table(L("D25H100D25"), {}, None, fps))).all()   # D-bounded


def test_merge_labels_cap():
    rng = np.random.default_rng(0)
    pred = _random_labels(rng, 2000, "HA", 10)
    truth = _random_labels(rng, 2000, "HA", 30)
    tab = spell_table(pred, {}, None, 25)
    y = merge_labels(truth, pred, tab, max_rows=5)
    assert np.isfinite(y).sum() == 5


def test_merge_model_fit_predict_roundtrip(tmp_path):
    rng = np.random.default_rng(0)
    pred = _random_labels(rng, 5000, "HAD", 12)
    tab = spell_table(pred, {"ball": _ev(rng.normal(size=len(pred)))}, None, 25)
    y = (tab["length_s"].to_numpy() < 0.5).astype(float)
    m = MergeModel().fit(tab, y)
    p = m.predict_proba(tab)
    assert p.shape == (len(tab),) and ((p >= 0) & (p <= 1)).all()
    assert p[y == 1].mean() > p[y == 0].mean()
    m.save(tmp_path / "m.joblib")
    m2 = MergeModel.load(tmp_path / "m.joblib")
    assert np.array_equal(m2.predict_proba(tab), p) and m2.info["n_rows"] == len(tab)
    c = MergeModel().fit(tab, np.zeros(len(tab)))                  # single class: constant
    assert (c.predict_proba(tab) == 0.0).all()
    c.save(tmp_path / "c.joblib"); assert (MergeModel.load(tmp_path / "c.joblib").predict_proba(tab) == 0.0).all()


def test_predict_merge_theta_off_is_identical_and_on_needs_model():
    from mtah.pipeline import PipelineConfig, compute_evidence, predict
    from tests.test_select_smoke import _fake_half
    h = _fake_half(9001, 1, 100, np.random.default_rng(0))
    cfg = PipelineConfig()
    ev = compute_evidence(h.match, h.tags, cfg, None)
    base = predict(h.match, h.tags, cfg, None, evidence=ev)
    always = MergeModel(); always.const = 1.0
    assert cfg.post.merge_theta == 1.0
    assert (predict(h.match, h.tags, cfg, None, evidence=ev, merge_model=always) == base).all()   # off: model ignored
    cfg.post.merge_theta = 0.7
    with pytest.raises(ValueError, match="merge_model"):
        predict(h.match, h.tags, cfg, None, evidence=ev)
    never = MergeModel(); never.const = 0.0
    assert (predict(h.match, h.tags, cfg, None, evidence=ev, merge_model=never) == base).all()
    merged = predict(h.match, h.tags, cfg, None, evidence=ev, merge_model=always)
    n_sp = lambda x: int(((x[1:] != x[:-1]) & np.isin(x[1:], ("H", "A"))).sum())
    assert n_sp(merged) < n_sp(base)                                 # every mergeable spell merged


def test_postparams_roundtrip_keeps_merge_theta():
    from mtah.pipeline import PipelineConfig
    cfg = PipelineConfig(); cfg.post.merge_theta = 0.7
    assert PipelineConfig.from_dict(cfg.to_dict()).post.merge_theta == 0.7
    d = cfg.to_dict(); d["post"].pop("merge_theta")
    assert PipelineConfig.from_dict(d).post.merge_theta == 1.0       # configs saved before T3: off


def test_merge_model_save_is_byte_reproducible(tmp_path):
    """Two fits on the same table with different wall-clock timings in info save to identical bytes (timings are not
    pickled); the in-memory info keeps them for the fold json / registration."""
    rng = np.random.default_rng(1)
    pred = _random_labels(rng, 5000, "HAD", 12)
    tab = spell_table(pred, {"ball": _ev(rng.normal(size=len(pred)))}, None, 25)
    y = (tab["length_s"].to_numpy() < 0.5).astype(float)
    a = MergeModel().fit(tab, y); a.info.update(held_match=9003, label_seconds=1.23, total_seconds=45.6)
    b = MergeModel().fit(tab, y); b.info.update(held_match=9003, label_seconds=9.87, total_seconds=65.4)
    a.save(tmp_path / "a.joblib"); a.save(tmp_path / "a2.joblib"); b.save(tmp_path / "b.joblib")
    ba = (tmp_path / "a.joblib").read_bytes()
    assert ba == (tmp_path / "a2.joblib").read_bytes() == (tmp_path / "b.joblib").read_bytes()
    info = MergeModel.load(tmp_path / "a.joblib").info
    assert info["held_match"] == 9003 and not any(k.endswith("_seconds") for k in info)
    assert a.info["label_seconds"] == 1.23   # in memory: unchanged
