import numpy as np
import pytest

from mtah.sim.dataset import SEALED_MATCH, lomo_folds, match_id


def test_match_id_and_folds():
    assert match_id("game_9003_H2") == 9003 and SEALED_MATCH == 9006
    keys = [f"game_{g}_H{h}" for g in range(9001, 9007) for h in (1, 2)]
    folds = lomo_folds(keys)
    assert [f[0] for f in folds] == [9001, 9002, 9003, 9004, 9005]
    for held, train, test in folds:
        assert all(match_id(k) == held for k in test) and len(test) == 2
        assert all(match_id(k) not in (held, SEALED_MATCH) for k in train) and len(train) == 8


@pytest.mark.dfl
def test_build_and_roundtrip(dfl_dir, tmp_path):
    from mtah.sim.blackout_sim import BlackoutParams
    from mtah.sim.corrupt import CorruptParams
    from mtah.sim.dataset import build_half, load_half_sim, save_half
    from mtah.sim.tag_sim import TagSimParams
    sim = build_half("game_9001_H1", dfl_dir / "game_9001_H1.npz", dfl_dir / "fp1d_census.json", 100,
                     BlackoutParams(), CorruptParams(), TagSimParams())
    assert sim.match.n_frames == 70708 and set(sim.truth) == {"raw_Dfull", "ctrl_Dfull", "raw_Dnarrow", "ctrl_Dnarrow"}
    assert sim.blackout.any() and np.isnan(sim.match.ball[sim.blackout, 0]).all()
    assert len(sim.tags) > 20
    save_half(sim, tmp_path)
    back = load_half_sim(tmp_path, "game_9001_H1", 100)
    assert np.array_equal(back.truth["raw_Dfull"], sim.truth["raw_Dfull"]) and len(back.tags) == len(sim.tags)
