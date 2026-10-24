"""a14 recalibration on a wider grid: a14's calibration (exp_seq_lomo.stage_calib('B12')) only tried seq_d {-1,-2,-3},
and its pick -1 was the grid edge; the sim folds favour seq_d 0 (B12|0 0.5663 vs a14 0.5568, exp_bs_decode check).
Same stage, same real-match label-free bands, grid dead {-3,-3.5,-4} x seq_d {-1,-0.5,0,+0.5}; cell -3/-1 is a14.
    python scripts/calib_wide.py   -> lomo/calib_B12w.csv / calib_B12w.json / calib_B12w_folds.csv"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import exp_seq_lomo as e

e.CAL_DEAD = (-3.0, -3.5, -4.0)
e.CAL_SEQD = (-1.0, -0.5, 0.0, 0.5)
e.DECODERS["B12w"] = e.DECODERS["B12"]

if __name__ == "__main__":
    e.stage_calib("B12w")
