import numpy as np

from mtah.io.periods import HALF_BLOCKS, KICKOFF_FRAMES, force_pre_kickoff_dead

N = 199319


def test_block_forced_to_dead():
    lab = np.array(["H"] * N)
    out = force_pre_kickoff_dead(lab)
    a, ko = HALF_BLOCKS[2][0], KICKOFF_FRAMES[2]
    assert (out[a:ko] == "D").all() and ko - a == 2934
    assert out[ko] == "H" and out[a - 1] == "H"


def test_half1_noop_and_outside_untouched():
    rng = np.random.default_rng(0)
    lab = rng.choice(np.array(["H", "A", "D"]), N)
    out = force_pre_kickoff_dead(lab)
    a, ko = HALF_BLOCKS[2][0], KICKOFF_FRAMES[2]
    assert KICKOFF_FRAMES[1] == HALF_BLOCKS[1][0]
    assert (out[:a] == lab[:a]).all() and (out[ko:] == lab[ko:]).all()
    assert (out != lab).sum() == (lab[a:ko] != "D").sum()
    assert out is not lab and (lab[a:ko] != "D").any()   # input not modified
