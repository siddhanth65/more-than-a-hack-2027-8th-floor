"""Final correctness check of the submission file (plan9 pre-registration, step 6). NOT run automatically.

  python scripts/final_check.py runs/candidates/<name>.csv [PUBLIC_GT.csv]

The public ground-truth file is the organisers' evaluation/gt/BAR-ATH_possession_gt_public.csv; pass its path as the second
argument or set MTAH_PUBLIC_GT. Nothing in this repository stores it.

1. Official validator + row count (199,319): on failure prints the reason and exits 1, before any ground truth is touched.
2. ONE check against the public window (frames 11185-18685): prints exactly two booleans and nothing else:
     best_offset_is_zero  - among frame offsets {-50, -45, ..., 50} the window spellF1 is maximal at offset 0 (ties with 0 count)
     swap_HA_is_worse     - swapping H and A (at offset 0) gives a strictly lower window spellF1
   No score, no per-offset number, no log, nothing written to disk. Only a demonstrable alignment or orientation bug (a False
   in either) may change the final file.
"""
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np

from mtah import N_FRAMES
from mtah.eval.official import possession_metrics as pm
from mtah.eval.scorer import score, window_labels
from mtah.submission import validate

WINDOW = (11185, 18685)
OFFSETS = tuple(range(-50, 51, 5))


def shifted(pred, k):
    """pred moved k frames later (k > 0) or earlier (k < 0); the uncovered edge is D."""
    out = np.full(len(pred), "D", dtype="<U1")
    if k > 0:
        out[k:] = pred[:-k]
    elif k < 0:
        out[:k] = pred[-k:]
    else:
        out[:] = pred
    return out


def swapped(pred):
    return np.where(pred == "H", "A", np.where(pred == "A", "H", pred))


def main():
    if len(sys.argv) not in (2, 3):
        raise SystemExit("usage: final_check.py SUBMISSION.csv [PUBLIC_GT.csv]  (or set MTAH_PUBLIC_GT)")
    path = sys.argv[1]
    GT = Path(sys.argv[2] if len(sys.argv) == 3 else os.environ.get("MTAH_PUBLIC_GT", ""))
    reason = validate(path)
    if reason is not None:
        print(f"INVALID: {reason}")
        raise SystemExit(1)
    pred = pm.read_label_csv(path)["label"].to_numpy().astype("<U1")
    if len(pred) != N_FRAMES:
        print(f"INVALID: {len(pred)} rows, expected {N_FRAMES}")
        raise SystemExit(1)
    print(f"valid, rows {len(pred)}")
    if not GT.is_file():
        raise SystemExit("public ground-truth file not given or not found (second argument or MTAH_PUBLIC_GT)")
    gt = pm.read_label_csv(GT)["label"].to_numpy().astype("<U1")

    def f1(p):
        return score(gt, window_labels(p, *WINDOW))["spell_f1"]

    s = {k: f1(shifted(pred, k)) for k in OFFSETS}
    best_zero = bool(s[0] >= max(s.values()))
    worse = bool(f1(swapped(pred)) < s[0])
    print(best_zero)
    print(worse)


if __name__ == "__main__":
    main()
