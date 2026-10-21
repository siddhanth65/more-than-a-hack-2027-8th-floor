"""Leave-one-match-out selection on the simulated dataset."""
import copy
import itertools

import numpy as np
import pandas as pd

from mtah.decode.scale import correlation_length, evidence_margin, residual_margin
from mtah.eval.scorer import score
from mtah.evidence.dead import DeadModel, dead_features
from mtah.evidence.team_ctrl import TeamCtrlModel, team_features
from mtah.pipeline import compute_evidence, predict
from mtah.sim.dataset import VARIANTS, match_id

GRID = [{"tau_HA": a, "tau_D": d, "w_dead": w, "aba_s": m}
        for a, d, w, m in itertools.product((3.0, 10.0, 30.0, 100.0), (3.0, 10.0, 30.0, 100.0), (0.5, 1.0, 2.0), (0.0, 0.5, 1.0))]

# grid 2: PathCRF stream weight (0 = off) and the dead-model feature policy join the search; aba_s fixed at 1.0
GRID2 = [{"tau_HA": a, "tau_D": d, "w_dead": w, "w_pathcrf": c, "dead_policy": pol, "aba_s": 1.0}
         for a, d, w, c, pol in itertools.product((10.0, 30.0), (30.0, 100.0), (1.0, 2.0), (0.0, 0.5, 1.0, 2.0),
                                                  ("full", "no_ball_at_all"))]
# grid 3: dead weight/policy search around grid 2's winner, plus the set-piece dead prior (tag-stream bonus)
GRID3 = [{"tau_HA": 30.0, "tau_D": d, "w_dead": w, "w_pathcrf": c, "dead_policy": pol, "sp_dead_bonus": b, "aba_s": 1.0}
         for d, w, c, pol, b in itertools.product((30.0, 100.0), (2.0, 3.0, 4.0, 6.0), (0.5, 1.0),
                                                  ("full", "no_ball_at_all"), (0.0, 3.0))]
GRIDS = {"grid1": GRID, "grid2": GRID2, "grid3": GRID3}

# label-free real-match plausibility: non-blackout D share of the prediction must lie in this band
REAL_D_SHARE_BAND = (0.10, 0.18)


def apply_cell(cfg, cell):
    cfg = copy.deepcopy(cfg)
    cfg.decoder.tau_HA = cell["tau_HA"]; cfg.decoder.tau_HD = cell["tau_D"]; cfg.decoder.tau_DH = cell["tau_D"]
    cfg.decoder.weights["dead"] = cell["w_dead"]; cfg.post.aba_s = cell["aba_s"]
    cfg.dead_policy = cell.get("dead_policy", "full")
    if "w_pathcrf" in cell:
        cfg.decoder.weights["pathcrf"] = cell["w_pathcrf"]   # 0 disables the stream in cost_matrix; use_pathcrf is the caller's (base config)
    if "sp_dead_bonus" in cell:
        cfg.tag.sp_dead_bonus = float(cell["sp_dead_bonus"])
    return cfg


def train_dead(sims, variant="ctrl_Dfull", policy="full"):
    X = np.concatenate([dead_features(s.match, s.tags) for s in sims])
    y = np.concatenate([(s.truth[variant] == "D").astype(int) for s in sims])
    return DeadModel(policy=policy).fit(X, y)


def train_team_ctrl(sims, variant="ctrl_Dfull", policy="full", stride=5, touch_params=None):
    """TeamCtrlModel on frames whose truth is H or A AND whose ball is visible (every `stride`-th such frame per
    half; consecutive frames are near duplicates). y = 1 for H. The model adds the team-swap mirror itself."""
    Xs, ys = [], []
    for s in sims:
        t = s.truth[variant]
        idx = np.flatnonzero(np.isin(t, ("H", "A")) & s.match.ball_present)[::stride]
        Xs.append(team_features(s.match, touch_params)[idx]); ys.append((t[idx] == "H").astype(int))
    return TeamCtrlModel(policy=policy).fit(np.concatenate(Xs), np.concatenate(ys))


LENGTHS_MODE = "unit"   # "unit": L_k = 1 for every stream (the tau grid absorbs the scale);
                        # "residual": integrated autocorrelation of the errors (pilot 2026-10-20: 164/606/380 -> too sticky)
EVIDENCE_STREAMS = ("ball", "tag", "dead", "pathcrf")


def estimate_lengths(train_sims, cfg, train_fn, mode=None):
    """Correlation lengths L_k per evidence stream.

    mode "unit" (LENGTHS_MODE, the default): L_k = 1.0 for every stream in EVIDENCE_STREAMS and for every other
    stream the decoder knows (cfg.decoder.weights / cfg.decoder.lengths keys); nothing is computed or trained.

    mode "residual": integrated autocorrelation of each stream's errors (spec 4.4), not of its signal.
    Errors are the evidence margin minus its per-truth-state mean (truth = ctrl_Dfull): ball and tag over
    H/A frames, dead (ev[:,2]-ev[:,0]) as D vs not-D. L_k is the median over held-out halves.
    The dead model used here is trained on all train matches except the last one (by match_id);
    lengths are estimated on that held-out train match's halves (all seeds), so the dead evidence
    is out-of-sample as it will be on the test match. The fold's final dead model is trained
    separately on all train halves. With < 2 train matches, falls back to in-sample (warns).
    """
    mode = mode or LENGTHS_MODE
    if mode == "unit":
        keys = dict.fromkeys(EVIDENCE_STREAMS)
        if cfg is not None:
            keys.update(dict.fromkeys(cfg.decoder.weights)); keys.update(dict.fromkeys(cfg.decoder.lengths))
        return {k: 1.0 for k in keys}
    if mode != "residual":
        raise ValueError(f"unknown lengths mode {mode!r}; expected 'unit' or 'residual'")
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
        ev = compute_evidence(s.match, s.tags, cfg, dead_model, cache_key=f"{s.key}_s{s.seed}")
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


def is_real_plausible(non_blackout_D_share, band=REAL_D_SHARE_BAND):
    """True when the real-match non-blackout D share lies in the band (NaN is never plausible)."""
    v = float(non_blackout_D_share)
    return bool(np.isfinite(v) and band[0] <= v <= band[1])


def choose_plausible(ranked, plaus):
    """Cell index of the highest-ranked cell flagged real_plausible in `plaus` ({cell: {"real_plausible": bool}}).

    `ranked` is the mean-rank-sorted summary (column "cell", best first). Cells absent from `plaus` were not
    checked on the real match and are never chosen over the fallback, which is rank 1 (with a WARNING)."""
    order = [int(c) for c in ranked["cell"]]
    for c in order:
        if c in plaus and plaus[c].get("real_plausible"):
            return c
    print(f"WARNING: no real-plausible cell among the checked ones {sorted(plaus)}; falling back to rank 1 (cell {order[0]})")
    return order[0]
