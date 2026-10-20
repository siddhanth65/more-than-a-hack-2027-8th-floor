"""Graded frame blocks of the BAR-ATH tactical video (DATA.md / games.py in the challenge repo)."""
import numpy as np

HALF_BLOCKS = {1: (11185, 90651), 2: (110808, 190197)}   # inclusive video frames
KICKOFF_FRAMES = {1: 11185, 2: 113742}


def half_array(n_frames):
    half = np.zeros(n_frames, dtype=np.int8)
    for h, (a, b) in HALF_BLOCKS.items():
        half[a:b + 1] = h
    return half
