import numpy as np
import pandas as pd
import pytest

from mtah.io.dfl_events import truth_variants


def lab(s):
    return np.array(list(s), dtype="<U1")


def ev(rows):
    return pd.DataFrame(rows, columns=["frame", "team", "cls", "poss_only"])


def test_ctrl_rejects_unsupported_flip():
    labels = lab("H" * 50 + "A" * 50 + "H" * 50)      # A run at 50..99 with no A event
    events = ev([(10, "H", 1, False), (120, "H", 1, False)])
    ball = np.zeros((150, 2), np.float32); speed = np.ones(150, np.float32)
    v = truth_variants(labels, events, ball, speed, 25, (105.0, 68.0))
    assert "".join(v["raw_Dfull"]) == "".join(labels)
    assert "".join(v["ctrl_Dfull"]) == "H" * 150


def test_ctrl_accepts_supported_flip():
    labels = lab("H" * 50 + "A" * 50)
    events = ev([(60, "A", 1, False)])                  # within 1.5 s (37 frames) of frame 50
    ball = np.zeros((100, 2), np.float32); speed = np.ones(100, np.float32)
    v = truth_variants(labels, events, ball, speed, 25, (105.0, 68.0))
    assert "".join(v["ctrl_Dfull"]) == "".join(labels)


def test_dnarrow_trims_dead_run():
    labels = lab("H" * 100 + "D" * 100 + "A" * 100)
    ball = np.zeros((300, 2), np.float32); speed = np.ones(300, np.float32)
    speed[130:200] = 0.0                                # static from frame 130
    events = ev([(180, "A", 8, False)])                 # restart at 180
    v = truth_variants(labels, events, ball, speed, 25, (105.0, 68.0))
    d = v["raw_Dnarrow"]
    assert "".join(d[100:130]) == "H" * 30 and "".join(d[130:180]) == "D" * 50 and "".join(d[180:200]) == "A" * 20


def test_dnarrow_unchanged_when_inverted():
    labels = lab("H" * 100 + "D" * 100 + "A" * 100)
    ball = np.zeros((300, 2), np.float32); speed = np.ones(300, np.float32)
    speed[190:200] = 0.0                                # static only at the very end
    events = ev([(120, "A", 8, False)])                 # "restart" before static start -> inverted
    v = truth_variants(labels, events, ball, speed, 25, (105.0, 68.0))
    assert "".join(v["raw_Dnarrow"]) == "".join(labels)


@pytest.mark.dfl
def test_load_events_real(dfl_dir):
    from mtah.io.dfl_proxy import load_events
    df = load_events(dfl_dir / "game_9001_H1.npz", dfl_dir / "fp1d_census.json")
    assert len(df) > 500 and set(df["team"]) == {"H", "A"} and df["frame"].is_monotonic_increasing
