"""Leave-one-match-out selection on the simulated dataset."""
import copy
import itertools
import json

import numpy as np
import pandas as pd

from mtah.decode.scale import correlation_length, evidence_margin, residual_margin
from mtah.decode.viterbi import AUX_STREAMS
from mtah.eval.scorer import score
from mtah.evidence import switch as switch_mod
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
# grid 4 (Plan 2d T1 + T2b + T3): learned team-control stream weight/policy around a3b's cell 52 (tau_D 100,
# w_pathcrf 1, full dead policy, no set-piece bonus); stickier tau_HA 60, the per-spell merge model threshold
# merge_theta (1.0 = off, 0.7 = merge spells with P(merge) > 0.7; mtah.spells.merge) and the switch hazard
# (mtah.evidence.switch; T2b) join the search.
# FINAL axes (96 cells): tau_HA {30, 60} x w_teamctrl {0, 1, 2} x w_dead {3, 4} x team_policy {full, no_touch}
# x w_switch {0, 2} x merge_theta {1.0, 0.7}; fixed: tau_D 100, w_pathcrf 1, dead_policy full, sp_dead_bonus 0,
# w_ball 1.0 (0.5 dropped in T3), aba_s 1.0 (the aba_s {1, 2} axis was replaced by w_switch in T2b).
# snap_mode is tied to w_switch: "hazard" (boundary snap to the hazard argmax) when w_switch > 0, else "touch".
# w_teamctrl 0 cells repeat across team_policy (identical labels when merge is off): kept so the grid is a full
# product.
GRID4 = [{"tau_HA": a, "tau_D": 100.0, "w_dead": w, "w_pathcrf": 1.0, "dead_policy": "full", "sp_dead_bonus": 0.0,
          "w_teamctrl": t, "w_ball": 1.0, "team_policy": tp, "aba_s": 1.0, "w_switch": s,
          "snap_mode": "hazard" if s > 0 else "touch", "merge_theta": q}
         for a, t, w, tp, s, q in itertools.product((30.0, 60.0), (0.0, 1.0, 2.0), (3.0, 4.0), ("full", "no_touch"),
                                                    (0.0, 2.0), (1.0, 0.7))]
GRIDS = {"grid1": GRID, "grid2": GRID2, "grid3": GRID3, "grid4": GRID4}

# label-free real-match plausibility: non-blackout D share of the prediction must lie in this band
REAL_D_SHARE_BAND = (0.10, 0.18)
# ... AND the predicted spell rate (drift_report spells_per_HA_min: H/A spells per minute of predicted H/A time) in
# this band: 3.4/min (public-window truth, 17 spells in 5 min) x [0.8, 1.25], the band the sim count-ratio bar uses
REAL_SPELL_RATE_BAND = (2.7, 4.3)


def apply_cell(cfg, cell):
    cfg = copy.deepcopy(cfg)
    cfg.decoder.tau_HA = cell["tau_HA"]; cfg.decoder.tau_HD = cell["tau_D"]; cfg.decoder.tau_DH = cell["tau_D"]
    cfg.decoder.weights["dead"] = cell["w_dead"]; cfg.post.aba_s = cell["aba_s"]
    cfg.dead_policy = cell.get("dead_policy", "full")
    if "w_pathcrf" in cell:
        cfg.decoder.weights["pathcrf"] = cell["w_pathcrf"]   # 0 disables the stream in cost_matrix; use_pathcrf is the caller's (base config)
    if "sp_dead_bonus" in cell:
        cfg.tag.sp_dead_bonus = float(cell["sp_dead_bonus"])
    if "w_teamctrl" in cell:
        cfg.decoder.weights["teamctrl"] = cell["w_teamctrl"]   # weight only; use_teamctrl is the caller's (base config)
    if "w_ball" in cell:
        cfg.decoder.weights["ball"] = cell["w_ball"]
    cfg.team_policy = cell.get("team_policy", cfg.team_policy)
    if "merge_theta" in cell:
        cfg.post.merge_theta = float(cell["merge_theta"])
    if "w_switch" in cell:
        cfg.decoder.w_switch = float(cell["w_switch"])   # weight only; use_switch is the caller's (base config)
    if "snap_mode" in cell:
        cfg.post.snap_mode = cell["snap_mode"]
    return cfg


def uses_switch(cell):
    """The cell reads the switch hazard: per-frame tau_HA (w_switch > 0) or hazard boundary snapping."""
    return float(cell.get("w_switch", 0.0)) > 0 or cell.get("snap_mode", "touch") == "hazard"


def evidence_key(cell):
    """Evidence depends on a cell only through (dead_policy, sp_dead_bonus, team_policy). The switch hazard (when
    the base config has use_switch) is the same for every key: the selection's switch model is trained without the
    teamctrl margin features (SWITCH_POLICY), so it does not depend on the dead or team policy."""
    return (cell.get("dead_policy", "full"), float(cell.get("sp_dead_bonus", 0.0)), cell.get("team_policy", "full"))


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


# Switch-hazard feature policy used by the selection. "no_team": trained WITHOUT the teamctrl margin features, so no
# out-of-fold teamctrl streams (an inner split like the merge model's) are needed and the hazard does not depend on
# the cell's team_policy. A richer variant ("full" with out-of-fold team_streams) can be added later under its own
# policy name; PipelineConfig.switch_policy / the registration's switch_policy record which one was used.
SWITCH_POLICY = "no_team"


def train_switch(sims, variant="ctrl_Dfull", policy=SWITCH_POLICY, touch_params=None, team_streams=None):
    """SwitchModel (mtah.evidence.switch.train_switch) on the labelled frames of `sims`. `team_streams` (one (F, 3)
    teamctrl evidence per sim) must be out-of-sample when given; the selection passes None (SWITCH_POLICY)."""
    return switch_mod.train_switch(sims, variant=variant, policy=policy, touch_params=touch_params,
                                   team_streams=team_streams)


def label_duplicate_cells(grid):
    """Indices of cells whose labels equal those of another grid cell: w_teamctrl 0 (stream unweighted) with
    team_policy != "full" when the same cell with team_policy "full" is in the grid. Not for merge cells (the merge
    features read the cell's teamctrl stream) nor for switch cells when the selection's switch policy keeps the
    teamctrl features. real_plausibility drops them before taking the top-k (they would waste real-match slots)."""
    hazard_reads_team = not set(switch_mod.TEAMCTRL_COLS) <= set(switch_mod.FEATURE_POLICIES[SWITCH_POLICY])
    keys = {json.dumps(c, sort_keys=True) for c in grid}
    out = set()
    for i, c in enumerate(grid):
        if (float(c.get("w_teamctrl", 0.0)) == 0.0 and c.get("team_policy", "full") != "full"
                and float(c.get("merge_theta", 1.0)) >= 1.0 and not (hazard_reads_team and uses_switch(c))
                and json.dumps({**c, "team_policy": "full"}, sort_keys=True) in keys):
            out.add(i)
    return out


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
    cfg = copy.deepcopy(cfg); cfg.use_teamctrl = False; cfg.use_switch = False   # no team/switch models here
    acc = {}
    for s in held:
        ev = compute_evidence(s.match, s.tags, cfg, dead_model, cache_key=f"{s.key}_s{s.seed}")
        truth = s.truth["ctrl_Dfull"]
        for k, e in ev.items():
            if k in AUX_STREAMS:
                continue   # the switch hazard is not an evidence stream
            if k == "dead":
                res = residual_margin(e[:, 2] - e[:, 0], np.where(truth == "D", "D", "N"), ("D", "N"))
            else:
                res = residual_margin(evidence_margin(e), truth, ("H", "A"))
            acc.setdefault(k, []).append(correlation_length(res))
    return {k: float(np.median(v)) for k, v in acc.items()}


def score_cell(sims, evidences, cfg, variants, strongs=None, merge_model=None):
    """Predict once per (sim, evidence); score the same labels against every variant. `merge_model` is applied when
    cfg.post.merge_theta < 1 (required then; see pipeline.predict).

    Returns ({variant: {spell_f1, macro_f1, ratio}}, long-format rows DataFrame)."""
    rows = []
    for i, (s, ev) in enumerate(zip(sims, evidences)):
        lab = predict(s.match, s.tags, cfg, None, evidence=ev, strong=None if strongs is None else strongs[i],
                      merge_model=merge_model if cfg.post.merge_theta < 1.0 else None)
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


def real_band_checks(non_blackout_D_share, spells_per_HA_min, band=REAL_D_SHARE_BAND, rate_band=REAL_SPELL_RATE_BAND):
    """{"d_share_ok", "spell_rate_ok", "real_plausible"} for the two label-free real-match bands (NaN never passes)."""
    v, r = float(non_blackout_D_share), float(spells_per_HA_min)
    d_ok = bool(np.isfinite(v) and band[0] <= v <= band[1])
    r_ok = bool(np.isfinite(r) and rate_band[0] <= r <= rate_band[1])
    return {"d_share_ok": d_ok, "spell_rate_ok": r_ok, "real_plausible": d_ok and r_ok}


def is_real_plausible(non_blackout_D_share, spells_per_HA_min, band=REAL_D_SHARE_BAND, rate_band=REAL_SPELL_RATE_BAND):
    """True when the real-match non-blackout D share lies in `band` AND the predicted spell rate (spells per minute of
    predicted H/A time, drift_report spells_per_HA_min) lies in `rate_band` (NaN is never plausible)."""
    return real_band_checks(non_blackout_D_share, spells_per_HA_min, band, rate_band)["real_plausible"]


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


# ---- per-spell merge model (Plan 2d T3): honest training set from an inner split of the fold's train matches ----
MERGE_TRUTH = "ctrl_Dfull"      # truth variant the merge label is scored against
MERGE_REF_CELLS = 4             # pre-merge labels for the training set come from up to this many merge cells
MERGE_MAX_ROWS_PER_HALF = None  # cap on label evaluations per half and reference cell (None = every mergeable spell)


def merge_cells(grid):
    return [c for c in grid if float(c.get("merge_theta", 1.0)) < 1.0]


def merge_ref_cells(grid, k=MERGE_REF_CELLS):
    """Up to k distinct merge cells (merge_theta reset to 1.0: training labels are scored on pre-merge labels), evenly
    spaced over the grid order so the decoder settings (tau_HA, teamctrl weight/policy, w_dead, aba_s) vary."""
    seen, uniq = set(), []
    for c in merge_cells(grid):
        c0 = {**c, "merge_theta": 1.0}
        key = tuple(sorted(c0.items()))
        if key not in seen:
            seen.add(key); uniq.append(c0)
    if not uniq:
        return []
    idx = np.unique(np.linspace(0, len(uniq) - 1, min(k, len(uniq))).round().astype(int))
    return [uniq[i] for i in idx]


def merge_training_table(train_sims, base, ref_cells, lengths, train_dead_fn=None, train_team_fn=None,
                         max_rows=MERGE_MAX_ROWS_PER_HALF, train_switch_fn=None):
    """Spell tables + merge labels on honest (out-of-sample) predictions.

    Inner split as in estimate_lengths: the stream models (dead per dead_policy; teamctrl per team_policy when
    base.use_teamctrl; the switch hazard model when base.use_switch) are trained on all train matches except the
    last one (by match_id), and the reference cells
    are decoded (pre-merge) on that held-out match's halves (all seeds). Only the fold's TRAIN matches are touched.
    Returns (DataFrame of mergeable spells with label column "y", info dict incl. label/total seconds)."""
    import time
    from mtah.pipeline import compute_strong
    from mtah.spells.merge import match_spell_table, merge_labels
    train_dead_fn = train_dead_fn or train_dead
    train_team_fn = train_team_fn or train_team_ctrl
    train_switch_fn = train_switch_fn or train_switch
    t0 = time.time()
    mids = sorted({match_id(s.key) for s in train_sims})
    if len(mids) < 2:
        print("WARNING: merge_training_table: <2 train matches, falling back to in-sample stream models")
        inner, held = list(train_sims), list(train_sims)
    else:
        inner = [s for s in train_sims if match_id(s.key) != mids[-1]]
        held = [s for s in train_sims if match_id(s.key) == mids[-1]]
    keys = sorted({evidence_key(c) for c in ref_cells})
    dead = {p: train_dead_fn(inner, policy=p) for p in sorted({k[0] for k in keys})}
    team = ({p: train_team_fn(inner, policy=p, touch_params=base.touch) for p in sorted({k[2] for k in keys})}
            if base.use_teamctrl else {})
    switch = (train_switch_fn(inner, policy=base.switch_policy, touch_params=base.touch)
              if base.use_switch else None)
    cfgs = {}
    for c in ref_cells:
        k = evidence_key(c)
        if k not in cfgs:
            cfgs[k] = apply_cell(base, c); cfgs[k].decoder.lengths = lengths
    tabs, t_label = [], 0.0
    for s in held:
        strong = compute_strong(s.match, next(iter(cfgs.values())))
        evs = {k: compute_evidence(s.match, s.tags, cfg, dead[k[0]], cache_key=f"{s.key}_s{s.seed}",
                                   team_model=team.get(k[2]), switch_model=switch) for k, cfg in cfgs.items()}
        for ci, c in enumerate(ref_cells):
            cfg = apply_cell(base, {**c, "merge_theta": 1.0}); cfg.decoder.lengths = lengths
            ev = evs[evidence_key(c)]
            lab = predict(s.match, s.tags, cfg, None, evidence=ev, strong=strong)
            tab = match_spell_table(lab, s.match, ev, strong)
            t1 = time.time()
            tab["y"] = merge_labels(s.truth[MERGE_TRUTH], lab, tab, max_rows=max_rows)
            t_label += time.time() - t1
            tab["key"] = s.key; tab["seed"] = s.seed; tab["ref_cell"] = ci
            tabs.append(tab[np.isfinite(tab["y"].to_numpy(float))])
    df = pd.concat(tabs, ignore_index=True) if tabs else pd.DataFrame()
    info = {"held_match": int(mids[-1]) if mids else None, "n_halves": len(held), "n_ref_cells": len(ref_cells),
            "n_rows": int(len(df)), "pos_rate": float(df["y"].mean()) if len(df) else float("nan"),
            "label_seconds": round(t_label, 2), "total_seconds": round(time.time() - t0, 2)}
    return df, info


def train_merge_model(train_sims, base, grid, lengths, train_dead_fn=None, train_team_fn=None, train_switch_fn=None):
    """MergeModel for one fold (None when no grid cell has merge_theta < 1); model.info carries the training stats."""
    from mtah.spells.merge import MergeModel
    refs = merge_ref_cells(grid)
    if not refs:
        return None
    df, info = merge_training_table(train_sims, base, refs, lengths, train_dead_fn, train_team_fn,
                                    train_switch_fn=train_switch_fn)
    m = MergeModel().fit(df, df["y"].to_numpy(float) if len(df) else np.zeros(0))
    m.info.update(info)
    print(f"merge model: {json.dumps(m.info)}", flush=True)
    return m
