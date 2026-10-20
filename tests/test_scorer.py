import numpy as np
import pytest

from mtah.eval.scorer import OFFICIAL_MD5, decompose, official_md5, score, window_labels


def lab(s):
    return np.array(list(s), dtype="<U1")


def test_official_file_pinned():
    assert official_md5() == OFFICIAL_MD5


def test_perfect():
    gt = lab("HHHHAAAADDDDHHHH")
    r = score(gt, gt)
    assert r["spell_f1"] == pytest.approx(1.0) and r["macro_f1"] == pytest.approx(1.0)
    assert r["n_gt_spells"] == 3 and r["n_pred_spells"] == 3


def test_split_in_half():
    gt = lab("H" * 20)
    pred = lab("H" * 10 + "D" + "H" * 9)
    r = score(gt, pred)
    # one true spell; two predicted pieces of IoU 0.5 and 0.45: at IoU<=0.5 one matches
    # (P 0.5, R 1 -> F1 0.667), at 0.6/0.7 none -> mean = 0.667*3/5 = 0.4
    assert r["spell_f1"] == pytest.approx(0.4, abs=1e-6)
    d = r["decomposition"]
    assert d["false_splits"] == 1 and d["false_merges"] == 0


def test_merge():
    gt = lab("HHHHHDDHHHHH")
    pred = lab("H" * 12)
    r = score(gt, pred)
    assert r["n_gt_spells"] == 2 and r["n_pred_spells"] == 1
    assert r["decomposition"]["false_merges"] == 1


def test_x_forward_fill_and_ungraded():
    gt = lab("XXHHHHAAAAXX")
    pred = lab("XXHHXXAAAAXX")
    r = score(gt, pred)
    assert r["spell_f1"] == pytest.approx(1.0)   # X inside a possession is filled with H
    assert r["macro_f1"] < 1.0                    # but charged per frame


def test_window_labels():
    w = window_labels(lab("HHHHHH"), 2, 4)
    assert "".join(w) == "XXHHXX"
