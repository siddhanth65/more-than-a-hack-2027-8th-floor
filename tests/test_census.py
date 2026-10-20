import numpy as np
import pandas as pd

from mtah.census import real_match_census
from tests.test_baselines import two_player_match


def test_census_keys_and_values():
    m = two_player_match(F=25 * 120)
    m.ball[:] = 0.0                       # ball present everywhere first
    m.ball[25 * 60:, :] = np.nan          # one 60 s blackout
    m.ball[25 * 60 - 6:25 * 60, 0] = np.linspace(40, 52, 6)   # heading to the goal line
    tags = pd.DataFrame([{"start": 25 * 61, "end": 25 * 70, "code": "SET PIECES", "team": None, "type": "Throw-in", "half": 1}])
    c = real_match_census(m, tags)
    assert c["n_blackouts"] == 1 and c["blackout_median_s"] == 60.0
    assert c["entry_near_line_share"] == 1.0 and c["entry_outward_share"] == 1.0
    assert c["setpiece_in_blackout_share"] == 1.0 and c["throwins_with_team"] == 0
    assert 0 < c["tag_coverage_inside"] < 1
