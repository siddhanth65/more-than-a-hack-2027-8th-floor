import numpy as np
import pandas as pd

from mtah.evidence.tag_prior import TagPriorParams, tag_prior_evidence


def test_tag_prior_columns():
    tags = pd.DataFrame([{"start": 10, "end": 20, "code": "BUILD UP", "team": "H", "type": None, "half": 1},
                         {"start": 30, "end": 40, "code": "SET PIECES", "team": None, "type": "Throw-in", "half": 1},
                         {"start": 50, "end": 60, "code": "SET PIECES", "team": "A", "type": "Corner Kick", "half": 1}])
    ev = tag_prior_evidence(tags, 100, TagPriorParams(p_tag=0.85, p_sp_team=0.8))
    assert ev.shape == (100, 3) and np.isnan(ev[:, 2]).all()
    assert np.isclose(ev[15, 0], np.log(0.85)) and np.isclose(ev[15, 1], np.log(0.15))
    assert np.isnan(ev[35]).all()                      # throw-in without team: no opinion
    assert np.isclose(ev[55, 1], np.log(0.8)) and np.isnan(ev[5]).all()
