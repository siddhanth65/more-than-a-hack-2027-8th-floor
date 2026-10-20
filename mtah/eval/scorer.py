"""Thin wrapper over the organisers' metric file (vendored verbatim, md5-pinned)."""
import hashlib
from pathlib import Path

import numpy as np

from mtah.eval.official import possession_metrics as pm

# md5 of the organisers' evaluation/possession_metrics.py with LF line endings (git blob)
OFFICIAL_MD5 = "7718cb8a237452b48b35713984d31c75"
OFFICIAL_PATH = Path(__file__).with_name("official") / "possession_metrics.py"


def official_md5():
    return hashlib.md5(OFFICIAL_PATH.read_bytes().replace(b"\r\n", b"\n")).hexdigest()


def codes(labels):
    return pm.label_codes(np.asarray(labels))


def window_labels(labels, start, stop):
    out = np.full(len(labels), "X", dtype="<U1")
    out[start:stop] = np.asarray(labels)[start:stop]
    return out


def decompose(gt, pred):
    g = codes(gt); p = codes(pred)
    runs = pm.valid_runs(g)
    filled = pm.fill_abstentions(p, runs)
    gs = pm.possession_spells(g, runs)
    ps = pm.possession_spells(filled, runs)
    ov = pm.spell_overlaps(gs, ps)
    per_gt = np.zeros(len(gs), int); per_pred = np.zeros(len(ps), int)
    for _, i, j in ov:
        per_gt[i] += 1; per_pred[j] += 1
    m03 = pm.matched_gt_flags(ov, len(gs), 0.3)
    m07 = pm.matched_gt_flags(ov, len(gs), 0.7)
    return {
        "false_splits": int(np.maximum(per_gt - 1, 0).sum()),
        "false_merges": int(np.maximum(per_pred - 1, 0).sum()),
        "boundary_only": int((m03 & ~m07).sum()),
        "matched_07": int(m07.sum()),
        "unmatched_gt_03": int((~m03).sum()),
    }


def score(gt, pred):
    g = codes(gt); p = codes(pred)
    r = pm.evaluate(g, p, bootstrap_n=0)
    return {
        "spell_f1": float(r["spell_f1_mean"]),
        "macro_f1": float(r["macro_f1"]),
        "n_gt_spells": int(r["n_gt_spells"]),
        "n_pred_spells": int(r["n_pred_spells"]),
        "per_iou": r["spell_f1"],
        "mean_spell_s": float(r["mean_spell_duration_s"]),
        "gt_mean_spell_s": float(r["gt_mean_spell_duration_s"]),
        "decomposition": decompose(gt, pred),
    }
