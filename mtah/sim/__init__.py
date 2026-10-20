"""Blackout simulator and its seeding convention."""
import numpy as np

FIT_SEEDS = (0, 1)                       # seeds the parameter search sees
EVAL_SEEDS = (100, 101, 102, 103, 104)   # held out from the search; used for gates and for scoring


def half_rng(seed, half_index):
    """Independent, reproducible generator for one half of one seed."""
    return np.random.default_rng([int(seed), int(half_index)])
