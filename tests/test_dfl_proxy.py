import json

import numpy as np
import pytest

from mtah.io.dfl_proxy import list_halves, load_events, load_half


def make_npz(tmp_path):
    F = 6; n0 = 100
    rows = []
    for f in range(F):
        rows.append([n0 + f, 101, 1, 7, 2, (10.0 + f + 52.5) / 105, (0.0 + 34) / 68, np.nan, np.nan, 1, np.nan, np.nan, np.nan, np.nan])  # home pid
        rows.append([n0 + f, 201, 0, 9, 2, (-10.0 + 52.5) / 105, (5.0 + 34) / 68, np.nan, np.nan, 1, np.nan, np.nan, np.nan, np.nan])     # guest pid
    raw = np.array(rows, np.float32)
    p = tmp_path / "game_9001_H1.npz"
    np.savez(p, raw=raw, ball_n=np.arange(n0, n0 + F), ball_xy=np.zeros((F, 2)),
             ball_status=np.array([1, 1, 0, 0, 1, 1], np.int8), ball_poss_ltr=np.array([1, 1, 0, 0, 0, 0], np.int8))
    census = {"DFL-MAT-X": {"halves": {"game_9001_H1": {"ltr": {"home": 1, "guest": 0}}}}}
    c = tmp_path / "fp1d_census.json"; c.write_text(json.dumps(census))
    return p, c


def test_load_half_tiny(tmp_path):
    p, c = make_npz(tmp_path)
    m, labels = load_half(p, c)
    assert m.n_frames == 6 and m.n_slots == 32 and m.pitch == (105.0, 68.0)
    assert m.team[:16].tolist() == [0] * 16 and m.team[16:].tolist() == [1] * 16
    assert m.pos[0, 0].tolist() == pytest.approx([10.0, 0.0], abs=1e-4)   # home player in slot 0
    assert m.pos[0, 16].tolist() == pytest.approx([-10.0, 5.0], abs=1e-4)  # guest player in slot 16
    assert np.isnan(m.pos[0, 1]).all()
    assert "".join(labels) == "HHDDAA"
    assert m.half.tolist() == [1] * 6


def test_list_halves(tmp_path):
    make_npz(tmp_path)
    assert [k for k, _ in list_halves(tmp_path)] == ["game_9001_H1"]


@pytest.mark.dfl
def test_real_half(dfl_dir):
    m, labels = load_half(dfl_dir / "game_9001_H1.npz", dfl_dir / "fp1d_census.json")
    n_frames = m.n_frames
    labels_ok = bool(np.isin(labels, ["H", "A", "D"]).all())
    dead_share = float((labels == "D").mean())
    mean_present = float((~np.isnan(m.pos[..., 0])).sum(axis=1).mean())
    assert n_frames == 70708
    assert labels_ok
    assert dead_share == pytest.approx(23483 / 70708, abs=1e-6)
    assert 20 <= mean_present <= 22


def test_load_half_rejects_non_binary(tmp_path):
    p, c = make_npz(tmp_path)
    z = dict(np.load(p))
    z["ball_poss_ltr"] = np.array([1, 1, 2, 0, 0, 0], np.int8)
    np.savez(p, **z)
    with pytest.raises(ValueError):
        load_half(p, c)


def _events_npz(tmp_path, team_col):
    p, c = make_npz(tmp_path)
    z = dict(np.load(p))
    z["truth"] = np.array([[101, team_col, 0, 8], [103, 1, 0, 8]], np.float32)
    z["poss_only"] = np.array([0, 1], np.int8)
    np.savez(p, **z)
    return p, c


def test_load_events_ok(tmp_path):
    p, c = _events_npz(tmp_path, 0)
    ev = load_events(p, c)
    assert list(ev["team"]) == ["A", "H"]


def test_load_events_rejects_bad_team_column(tmp_path):
    p, c = _events_npz(tmp_path, 2)
    with pytest.raises(ValueError):
        load_events(p, c)
