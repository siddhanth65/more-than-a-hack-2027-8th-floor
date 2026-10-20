import numpy as np
import pandas as pd

from mtah.io.match import Match, speeds_from_positions
from mtah.pipeline import PipelineConfig, predict


def test_two_halves_with_ungraded_gap():
    F = 2200
    half = np.array([1] * 1000 + [0] * 200 + [2] * 1000, np.int8)
    pos = np.zeros((F, 4, 2), np.float32)
    pos[:, 0, 0] = np.linspace(-20, 20, F); pos[:, 1, 0] = -30
    pos[:, 2, 0] = np.linspace(20, -20, F); pos[:, 3, 0] = 30
    ball = np.zeros((F, 2), np.float32)
    ball[:1500, 0] = pos[:1500, 0, 0] + 0.3
    ball[1500:, 0] = pos[1500:, 2, 0] - 0.3
    m = Match(fps=25.0, pos=pos, speed=speeds_from_positions(pos, 25.0), team=np.array([0, 0, 1, 1], np.int8),
              ball=ball, ball_speed=speeds_from_positions(ball, 25.0), half=half, pitch=(105.0, 68.0))
    tags = pd.DataFrame(columns=["start", "end", "code", "team", "type", "half"])
    lab = predict(m, tags, PipelineConfig(), None)
    assert len(lab) == F
    tol = 25
    assert (lab[1000:1200] == "D").all()
    h1 = lab[:1000]; h2 = lab[1200:]
    assert (h1[20:] == "H").mean() > 0.97 and (h1[:1000] == "H").sum() >= 1000 - 25
    # switch at frame 1500 -> index 300 within half 2
    assert (h2[:300 - tol] == "H").all() or (h2[:300 - tol] == "H").mean() > 0.97
    assert (h2[300 + tol:] == "A").all()
