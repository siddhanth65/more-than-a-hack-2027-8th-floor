"""Counted read of the public 5-minute window. Appends to docs/reads/public.jsonl (tracked)."""
import argparse, json, subprocess, time
from pathlib import Path

import numpy as np

from mtah.eval.official import possession_metrics as pm
from mtah.eval.scorer import score, window_labels
from mtah.io import readlog
from mtah.submission import validate

ROOT = Path(__file__).resolve().parents[1]
GT = Path(r"WORKDIR/repo\evaluation\gt\BAR-ATH_possession_gt_public.csv")


def check_budget():
    n_prev = len(readlog.read_log("public"))
    if n_prev >= 3:   # budget raised from 3 to 4 total reads by the user (2026-10-22), report-only
        raise SystemExit(f"look budget: {n_prev} reads already logged (+1 done by hand on 2026-10-20; budget 4 total); refusing")


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--candidate", required=True); ap.add_argument("--note", required=True)
    args = ap.parse_args()
    reason = validate(args.candidate)
    if reason is not None:
        raise SystemExit(f"invalid candidate: {reason}")
    check_budget()
    gt = pm.read_label_csv(GT)["label"].to_numpy().astype("<U1")
    pred = pm.read_label_csv(args.candidate)["label"].to_numpy().astype("<U1")
    r = score(gt, window_labels(pred, 11185, 18685))
    rec = {"time": time.strftime("%Y-%m-%d %H:%M"), "git": subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT).decode().strip(),
           "candidate": args.candidate, "note": args.note, "spell_f1": r["spell_f1"], "macro_f1": r["macro_f1"],
           "n_pred": r["n_pred_spells"], "n_gt": r["n_gt_spells"], "decomposition": r["decomposition"]}
    readlog.append_read("public", rec)
    print(json.dumps(rec, indent=1))


if __name__ == "__main__":
    main()
