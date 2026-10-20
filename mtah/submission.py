"""Write and validate a frame,label submission; CLI runs a baseline end to end."""
import argparse
import os
from pathlib import Path

import numpy as np

from mtah import N_FRAMES
from mtah.eval.official import possession_metrics as pm


def write_submission(labels, path):
    labels = np.asarray(labels)
    if len(labels) != N_FRAMES:
        raise ValueError(f"{len(labels)} labels, expected {N_FRAMES}")
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="\n") as f:
        f.write("frame,label\n")
        f.write("\n".join(f"{i},{l}" for i, l in enumerate(labels.tolist())))
        f.write("\n")
    return path


def validate(path):
    try:
        frame = pm.read_label_csv(path)
    except Exception as e:  # unparseable
        return str(e)
    return pm.validate_submission(frame, N_FRAMES)


def main():
    from mtah import baselines
    from mtah.io.metrica_loader import load_tactical
    ap = argparse.ArgumentParser()
    ap.add_argument("--baseline", choices=["b0", "b4", "smooth50"], default="b4")
    ap.add_argument("--out", default="runs/submission.csv")
    ap.add_argument("--data", default=os.environ.get("MTAH_DATA_DIR", r"HOME/mtah-data"))
    args = ap.parse_args()
    m = load_tactical(args.data)
    labels = {"b0": baselines.b0_nearest_hold, "b4": baselines.b4_blackout_dead,
              "smooth50": lambda mm: baselines.smooth_vote(mm, 50)}[args.baseline](m)
    p = write_submission(labels, args.out)
    reason = validate(p)
    counts = {k: int((labels == k).sum()) for k in "HADX"}
    print(f"wrote {p}  counts {counts}  validation: {'OK' if reason is None else reason}")


if __name__ == "__main__":
    main()
