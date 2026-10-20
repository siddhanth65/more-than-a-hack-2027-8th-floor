"""Leave-one-match-out selection on the simulated dataset."""
import copy
import itertools

import numpy as np
import pandas as pd

from mtah.decode.scale import correlation_length, evidence_margin
from mtah.eval.scorer import score
from mtah.evidence.dead import DeadModel, dead_features
from mtah.pipeline import compute_evidence, predict
from mtah.sim.dataset import VARIANTS

GRID = [{"tau_HA": a, "tau_D": d, "w_dead": w, "aba_s": m}
        for a, d, w, m in itertools.product((3.0, 10.0, 30.0), (3.0, 10.0, 30.0), (0.5, 1.0, 2.0), (0.0, 0.5, 1.0))]


def apply_cell(cfg, cell):
    cfg = copy.deepcopy(cfg)
    cfg.decoder.tau_HA = cell["tau_HA"]; cfg.decoder.tau_HD = cell["tau_D"]; cfg.decoder.tau_DH = cell["tau_D"]
    cfg.decoder.weights["dead"] = cell["w_dead"]; cfg.post.aba_s = cell["aba_s"]
    return cfg


def train_dead(sims, variant="ctrl_Dfull"):
    X = np.concatenate([dead_features(s.match, s.tags) for s in sims])
    y = np.concatenate([(s.truth[variant] == "D").astype(int) for s in sims])
    return DeadModel().fit(X, y)


def estimate_lengths(sims, cfg, dead_model):
    acc = {}
    for s in sims:
        ev = compute_evidence(s.match, s.tags, cfg, dead_model)
        for k, e in ev.items():
            m = evidence_margin(e) if k != "dead" else (e[:, 2] - e[:, 0])
            acc.setdefault(k, []).append(correlation_length(m))
    return {k: float(np.median(v)) for k, v in acc.items()}


def score_cell(sims, evidences, cfg, variant):
    rows = []
    for s, ev in zip(sims, evidences):
        lab = predict(s.match, s.tags, cfg, None, evidence=ev)
        r = score(s.truth[variant], lab)
        rows.append({"key": s.key, "seed": s.seed, "spell_f1": r["spell_f1"], "macro_f1": r["macro_f1"],
                     "ratio": r["n_pred_spells"] / max(1, r["n_gt_spells"])})
    df = pd.DataFrame(rows)
    return {"spell_f1": df["spell_f1"].mean(), "macro_f1": df["macro_f1"].mean(), "ratio": df["ratio"].mean(), "rows": df}


def mean_rank(table, variants):
    t = table.copy()
    ranks = np.column_stack([t[v].rank(ascending=False, method="average") for v in variants])
    t["mean_rank"] = ranks.mean(1)
    return t.sort_values(["mean_rank", "ctrl_Dfull"], ascending=[True, False]).reset_index(drop=True)
