"""Leave-one-match-out selection on the simulated dataset."""
import copy
import itertools

import numpy as np
import pandas as pd

from mtah.decode.scale import correlation_length, evidence_margin, residual_margin
from mtah.eval.scorer import score
from mtah.evidence.dead import DeadModel, dead_features
from mtah.pipeline import compute_evidence, predict
from mtah.sim.dataset import VARIANTS, match_id

GRID = [{"tau_HA": a, "tau_D": d, "w_dead": w, "aba_s": m}
        for a, d, w, m in itertools.product((3.0, 10.0, 30.0, 100.0), (3.0, 10.0, 30.0, 100.0), (0.5, 1.0, 2.0), (0.0, 0.5, 1.0))]


def apply_cell(cfg, cell):
    cfg = copy.deepcopy(cfg)
    cfg.decoder.tau_HA = cell["tau_HA"]; cfg.decoder.tau_HD = cell["tau_D"]; cfg.decoder.tau_DH = cell["tau_D"]
    cfg.decoder.weights["dead"] = cell["w_dead"]; cfg.post.aba_s = cell["aba_s"]
    return cfg


def train_dead(sims, variant="ctrl_Dfull"):
    X = np.concatenate([dead_features(s.match, s.tags) for s in sims])
    y = np.concatenate([(s.truth[variant] == "D").astype(int) for s in sims])
    return DeadModel().fit(X, y)


LENGTHS_MODE = "unit"   # "unit": L_k = 1 for every stream (the tau grid absorbs the scale);
                        # "residual": integrated autocorrelation of the errors (pilot 2026-10-20: 164/606/380 -> too sticky)
EVIDENCE_STREAMS = ("ball", "tag", "dead")


def estimate_lengths(train_sims, cfg, train_fn, mode=None):
    mode = mode or LENGTHS_MODE
    if mode == "unit":
        return {k: 1.0 for k in EVIDENCE_STREAMS}
    """Correlation lengths L_k: integrated autocorrelation of each stream's errors (spec 4.4), not of its signal.

    Errors are the evidence margin minus its per-truth-state mean (truth = ctrl_Dfull): ball and tag over
    H/A frames, dead (ev[:,2]-ev[:,0]) as D vs not-D. L_k is the median over held-out halves.

    The dead model used here is trained on all train matches except the last one (by match_id);
    lengths are estimated on that held-out train match's halves (all seeds), so the dead evidence
    is out-of-sample as it will be on the test match. The fold's final dead model is trained
    separately on all train halves. With < 2 train matches, falls back to in-sample (warns).
    """
    mids = sorted({match_id(s.key) for s in train_sims})
    if len(mids) < 2:
        print("WARNING: estimate_lengths: <2 train matches, falling back to in-sample dead model")
        inner, held = list(train_sims), list(train_sims)
    else:
        inner = [s for s in train_sims if match_id(s.key) != mids[-1]]
        held = [s for s in train_sims if match_id(s.key) == mids[-1]]
    dead_model = train_fn(inner)
    acc = {}
    for s in held:
        ev = compute_evidence(s.match, s.tags, cfg, dead_model)
        truth = s.truth["ctrl_Dfull"]
        for k, e in ev.items():
            if k == "dead":
                res = residual_margin(e[:, 2] - e[:, 0], np.where(truth == "D", "D", "N"), ("D", "N"))
            else:
                res = residual_margin(evidence_margin(e), truth, ("H", "A"))
            acc.setdefault(k, []).append(correlation_length(res))
    return {k: float(np.median(v)) for k, v in acc.items()}


def score_cell(sims, evidences, cfg, variants, strongs=None):
    """Predict once per (sim, evidence); score the same labels against every variant.

    Returns ({variant: {spell_f1, macro_f1, ratio}}, long-format rows DataFrame)."""
    rows = []
    for i, (s, ev) in enumerate(zip(sims, evidences)):
        lab = predict(s.match, s.tags, cfg, None, evidence=ev, strong=None if strongs is None else strongs[i])
        for v in variants:
            r = score(s.truth[v], lab)
            rows.append({"key": s.key, "seed": s.seed, "variant": v, "spell_f1": r["spell_f1"],
                         "macro_f1": r["macro_f1"], "n_gt": r["n_gt_spells"], "n_pred": r["n_pred_spells"]})
    df = pd.DataFrame(rows)
    out = {}
    for v in variants:
        d = df[df["variant"] == v]
        out[v] = {"spell_f1": float(d["spell_f1"].mean()), "macro_f1": float(d["macro_f1"].mean()),
                  "ratio": float((d["n_pred"] / d["n_gt"].clip(lower=1)).mean())}
    return out, df


def mean_rank(table, variants):
    t = table.copy()
    ranks = np.column_stack([t[v].rank(ascending=False, method="average") for v in variants])
    t["mean_rank"] = ranks.mean(1)
    return t.sort_values(["mean_rank", "ctrl_Dfull"], ascending=[True, False]).reset_index(drop=True)
