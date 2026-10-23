"""LOMO selection over a grid (grid1 | grid2) on the simulated dataset; writes fold tables, summary and registration.

With a real-data dir, the top-k cells by mean rank are also decoded on the real match (label-free drift metrics);
the chosen cell is the highest-ranked one that is real-plausible: non-blackout D share in select.REAL_D_SHARE_BAND AND
predicted spells per GRADED minute in select.REAL_SPELL_RATE_BAND (tag_agreement_implied is reported, not gated).
Cells with merge_theta < 1 use a per-fold MergeModel (mtah.spells.merge) trained on an inner split of the fold's train
matches (select.train_merge_model); the registered one is written as merge_model.joblib.
Cells with w_switch > 0 or snap_mode "hazard" read the switch hazard (mtah.evidence.switch): one SwitchModel per fold
on the fold's train halves (select.train_switch, policy select.SWITCH_POLICY = "no_team": no teamctrl margin features,
so no out-of-fold teamctrl streams are needed); the registered one is written as switch_model.joblib.
This real-match plausibility step runs for EVERY grid (grid1 included) unless --no-real is given.
PathCRF threads are pinned to PathCRFParams().threads (4): the posterior cache is keyed on it, no CLI knob.

Training-only extra halves (--train-extra DIR [--extra-weight W]; mtah.sim.dataset.load_extra_halves, built by
scripts/build_sim.py --extra-dir ... --out DIR): added to EVERY fold's training set at the cell's `extra_weight` (W for
cells without one; 0 = DFL only) for the dead, team-control and switch models (sample weights; the merge model's
stream models are trained on them too, through its inner split), and to the training of the plausibility step and the
registered models. They are never a test fold, never decoded for the merge table (the merge-held match is the last
DFL match), and never part of the drift reference or the plausibility test sets. Cells are grouped by extra_weight:
each group trains its own models. The registration records the extra set."""
import argparse, hashlib, itertools, json, os, subprocess, sys, time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd

from mtah import baselines
from mtah.eval.drift import blackout_mask_from_match, drift_report
from mtah.eval.scorer import OFFICIAL_PATH, score
from mtah.evidence.dead import DeadModel
from mtah.evidence.pathcrf import PathCRFParams
from mtah.evidence.team_ctrl import TeamCtrlModel
from mtah.io import readlog
from mtah.io.metrica_loader import load_tactical
from mtah.io.tags import load_tags
from mtah.pipeline import PipelineConfig, compute_evidence, compute_strong, predict
import mtah.select as sel
from mtah.select import (LENGTHS_MODE, REAL_D_SHARE_BAND, REAL_SPELL_RATE_BAND, SWITCH_POLICY, apply_cell,
                         choose_plausible, estimate_lengths, is_real_plausible, mean_rank, real_band_checks, score_cell,
                         train_dead, train_switch, train_team_ctrl, uses_switch)
from mtah.evidence.switch import SwitchModel
from mtah.spells.merge import MergeModel
from mtah.select import ExtraSet, cell_extra_weight, extra_weight_groups
from mtah.sim.dataset import (SEALED_MATCH, VARIANTS, list_sim_keys, load_extra_halves, load_half_sim, lomo_folds,
                              match_id)

REAL_KEYS = ("non_blackout_D_share", "tag_agreement_implied", "pred_D_share", "spells_per_graded_min",
             "spells_per_HA_min")   # gated: D share + spells_per_graded_min; the rest reported only
REAL_TOP_K = 5


def git_sha():
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT).decode().strip()
    except Exception:
        return "unknown"


def load_sims(sim_dir, seeds, keys):
    return [load_half_sim(sim_dir, k, s) for k in keys for s in seeds]


def data_dir_or_exit(path):
    """--data, defaulting to $MTAH_DATA_DIR; the real-match data location is machine specific."""
    if not path:
        raise SystemExit("real-match data dir needed: pass --data or set MTAH_DATA_DIR (or use --no-real)")
    return path


def base_for_grid(grid, threads=None):
    """Base config for a grid: the PathCRF stream is computed iff some cell gives it a positive weight
    (w_pathcrf = 0 cells then ignore it through the zero weight); likewise the learned team-control stream
    (use_teamctrl iff some cell has w_teamctrl > 0; team models are then trained per needed team_policy) and the
    switch hazard (use_switch iff some cell has w_switch > 0 or snap_mode "hazard"; switch_policy SWITCH_POLICY).
    Threads stay at the pinned PathCRFParams default
    (4; the posterior and its cache key depend on it); `threads` exists for tests only, no CLI sets it."""
    base = PipelineConfig()
    base.use_pathcrf = any(c.get("w_pathcrf", 0.0) > 0 for c in grid)
    base.use_teamctrl = any(c.get("w_teamctrl", 0.0) > 0 for c in grid)
    base.use_switch = any(uses_switch(c) for c in grid)
    if base.use_switch:
        base.switch_policy = SWITCH_POLICY
    if threads is not None:
        base.pathcrf.threads = int(threads)
    return base


def ev_key(cell):
    return sel.evidence_key(cell)


def uses_merge(cell):
    return float(cell.get("merge_theta", 1.0)) < 1.0


def train_merge(train, base, grid, lengths):
    """Fold merge model (None unless some grid cell has merge_theta < 1); stream models via this module's
    train_dead / train_team_ctrl / train_switch so they stay patchable."""
    return sel.train_merge_model(train, base, grid, lengths, train_dead_fn=train_dead, train_team_fn=train_team_ctrl,
                                 train_switch_fn=train_switch)


def train_switch_model(train, base):
    """SwitchModel on `train` (policy base.switch_policy, touch params base.touch) when base.use_switch, else None."""
    return train_switch(train, policy=base.switch_policy, touch_params=base.touch) if base.use_switch else None


def policy_cfgs(base, cells, lengths):
    """{(dead_policy, sp_dead_bonus, team_policy, dead_offset): cfg} with cfg = apply_cell(base, first cell of that key). Evidence
    depends on the cell only through dead_policy (dead stream), dead_offset (its log-odds shift),
    sp_dead_bonus (tag stream) and team_policy (teamctrl stream); weights/taus act in the decoder, so one evidence set per key serves every cell. The PathCRF posterior is
    disk-cached per match, so extra keys are cheap."""
    out = {}
    for c in cells:
        p = ev_key(c)
        if p not in out:
            cfg = apply_cell(base, c); cfg.decoder.lengths = lengths; out[p] = cfg
    return dict(sorted(out.items()))


def train_team_models(train, base, policies):
    """{team_policy: TeamCtrlModel} trained on `train` when base.use_teamctrl, else {} (no stream, no training)."""
    return ({p: train_team_ctrl(train, policy=p, touch_params=base.touch) for p in sorted(set(policies))}
            if base.use_teamctrl else {})


BASELINES = {"smooth50": lambda m: baselines.smooth_vote(m, 50), "b4": baselines.b4_blackout_dead}


def _drift_summary(rows):
    ref = {}
    for k in sorted({k for r in rows for k in r}):
        v = np.array([r[k] for r in rows if k in r], float)
        if v.size and not np.all(np.isnan(v)):
            ref[k] = {"min": float(np.nanmin(v)), "median": float(np.nanmedian(v)), "max": float(np.nanmax(v))}
    return ref


def _group_cells(grid, ew, extra):
    """The grid's cells in extra_weight group `ew`."""
    d = extra.default_weight if extra is not None else 0.0
    return [c for c in grid if cell_extra_weight(c, d) == ew]


def out_of_sample_drift_reference(sim_dir, seeds, keys, base, best_cell, lengths_by_fold, fold_models, grid=None,
                                  extra=None):
    """Re-decode the best cell on every fold's TEST halves (never the extras) and summarise the drift metrics (fold
    dead/team/merge/switch models of the best cell's extra_weight group, `fold_models[held][ew]`; a resumed fold retrains
    its models from `grid` -- deterministic HGB -- on the fold's train halves plus the extras at that weight)."""
    rows = []
    policy = best_cell.get("dead_policy", "full"); tpol = best_cell.get("team_policy", "full")
    ew = cell_extra_weight(best_cell, extra.default_weight if extra is not None else 0.0)
    for held, train_keys, test_keys in lomo_folds(keys):
        base_train = load_sims(sim_dir, seeds, train_keys); test = load_sims(sim_dir, seeds, test_keys)
        train = base_train + (extra.at(ew) if extra is not None else [])
        lengths = lengths_by_fold.get(held) or estimate_lengths(base_train, base, train_dead)
        m = fold_models.get(held, {}).get(ew, {})
        dead = m.get("dead", {}).get(policy)
        if dead is None:
            dead = train_dead(train, policy=policy)
        cfg = apply_cell(base, best_cell); cfg.decoder.lengths = lengths
        team = m.get("team", {}).get(tpol)
        if team is None and cfg.use_teamctrl:   # (also at weight 0: merge features read the stream, as in the folds)
            team = train_team_ctrl(train, policy=tpol, touch_params=base.touch)   # resumed fold: retrain (deterministic HGB)
        merge = m.get("merge")
        if merge is None and uses_merge(best_cell):
            merge = train_merge(train, base, _group_cells(grid, ew, extra) if grid is not None else [best_cell], lengths)
        switch = m.get("switch")
        if switch is None and base.use_switch and uses_switch(best_cell):
            switch = train_switch_model(train, base)   # resumed fold: retrain (deterministic HGB)
        for s in test:
            ev = compute_evidence(s.match, s.tags, cfg, dead, cache_key=f"{s.key}_s{s.seed}", team_model=team,
                                  switch_model=switch)
            lab = predict(s.match, s.tags, cfg, dead, evidence=ev, merge_model=merge if uses_merge(best_cell) else None)
            rows.append(drift_report(lab, s.match, s.tags, blackout_mask_from_match(s.match), code_table=cfg.tag.code_table))
    return _drift_summary(rows)


RATIO_BAND = (0.8, 1.25)
RATIO_VARIANTS = ("ctrl_Dfull", "ctrl_Dnarrow")   # raw truth counts uncontrolled touches: raw ratios sit ~0.6 by design


def compute_bar(allf, halves, best, best_i):
    """Spec bar: margins over the strongest baseline (>= max(0.05, 2x seed SD), all four variants), folds won on
    ctrl_Dfull (>= n_folds - 1), count ratios n_pred/n_gt in RATIO_BAND for the ctrl_* variants only. All four
    ratios are recorded; the raw ones are informative only (raw truth splits spells at uncontrolled touches)."""
    base_rows = allf[allf["cell"].isin(list(BASELINES))].groupby("cell")[list(VARIANTS)].mean()
    baseline_used = {v: str(base_rows[v].idxmax()) for v in VARIANTS}
    margins = {v: float(best[v] - base_rows[v].max()) for v in VARIANTS}
    h = halves[halves["cell"] == best_i]
    seed_sd = {}
    for v in VARIANTS:
        per_seed = h[h["variant"] == v].groupby("seed")["spell_f1"].mean()
        seed_sd[v] = float(per_seed.std(ddof=1)) if len(per_seed) > 1 else 0.0
    cellf = allf[allf["cell"] == best_i].set_index("held")["ctrl_Dfull"]
    basef = allf[allf["cell"].isin(list(BASELINES))].groupby("held")["ctrl_Dfull"].max()
    folds_won = int(((cellf - basef.reindex(cellf.index)) >= 0.05).sum())
    n_folds = int(len(cellf)); folds_needed = max(1, n_folds - 1)
    ratios = {v: float(best[f"{v}_ratio"]) for v in VARIANTS}
    ratios_ok = {v: bool(RATIO_BAND[0] <= ratios[v] <= RATIO_BAND[1]) for v in RATIO_VARIANTS}
    need = {v: max(0.05, 2.0 * seed_sd[v]) for v in VARIANTS}
    bar_pass = bool(all(margins[v] >= need[v] for v in VARIANTS) and folds_won >= folds_needed
                    and all(ratios_ok.values()))
    bar = {"margins_vs_best_baseline": margins, "required_margin": need, "seed_sd": seed_sd,
           "baseline_used": baseline_used, "folds_won_ctrl_Dfull": folds_won, "n_folds": n_folds,
           "folds_won_needed": folds_needed, "count_ratios": ratios, "ratio_band": list(RATIO_BAND),
           "ratio_rule_variants": list(RATIO_VARIANTS), "count_ratios_ok": ratios_ok, "bar_pass": bar_pass}
    return bar, base_rows


def real_plausibility(sim_dir, seeds, keys, base, grid, ranked, data_dir, out, top_k=REAL_TOP_K, shared=None, extra=None):
    """Decode the top-k cells (by mean rank; label-identical duplicates, select.label_duplicate_cells, skipped) on the
    real match; label-free drift metrics per cell.

    Dead models (one per needed policy) and lengths come from ALL non-sealed sim halves, as in the sealed/candidate
    path (team-control models likewise, one per needed team_policy, when base.use_teamctrl; the switch model when
    base.use_switch; the merge model, when a top-k cell has merge_theta < 1, from an inner split of all non-sealed
    matches with the full grid's reference cells). With an ExtraSet the models of each extra_weight group of the top
    cells also train on the extras at that weight (one model set per group; the real match is the only thing decoded).
    Plausible = D-share band AND spell-rate band (both recorded); tag_agreement_implied is reported, not gated.
    If `shared` is a dict it receives {"lengths", "by_ew": {extra_weight: {"dead", "team", "merge", "switch"}}} so
    registration reuses them. Writes summary_real.csv; returns {cell index: {metric: value, "d_share_ok",
    "spell_rate_ok", "real_plausible", ...}}."""
    dup = sel.label_duplicate_cells(grid)   # label-identical repeats (w_teamctrl 0 across team_policy): one real slot each
    top = [int(c) for c in ranked["cell"] if int(c) not in dup][:top_k]
    dw = extra.default_weight if extra is not None else 0.0
    base_train = load_sims(sim_dir, seeds, keys)
    lengths = estimate_lengths(base_train, base, train_dead)
    m = load_tactical(data_dir); tags = load_tags(data_dir); mask = blackout_mask_from_match(m)
    by_ew, ev_of, strong = {}, {}, None
    for ew in sorted({cell_extra_weight(grid[i], dw) for i in top}):
        tops = [i for i in top if cell_extra_weight(grid[i], dw) == ew]
        train = base_train + (extra.at(ew) if extra is not None else [])
        cfgs = policy_cfgs(base, [grid[i] for i in tops], lengths)
        dead = {p: train_dead(train, policy=p) for p in sorted({k[0] for k in cfgs})}
        team = train_team_models(train, base, [k[2] for k in cfgs])
        merge = (train_merge(train, base, _group_cells(grid, ew, extra), lengths)
                 if any(uses_merge(grid[i]) for i in tops) else None)
        switch = train_switch_model(train, base) if any(uses_switch(grid[i]) for i in tops) else None
        del train
        by_ew[ew] = dict(dead=dead, team=team, merge=merge, switch=switch)
        ev = {k: compute_evidence(m, tags, cfg, dead[k[0]], cache_key="real_tactical", team_model=team.get(k[2]),
                                  switch_model=switch)
              for k, cfg in cfgs.items()}
        if strong is None:
            strong = compute_strong(m, next(iter(cfgs.values())))
        for i in tops:
            ev_of[i] = ev[ev_key(grid[i])]
    del base_train
    if shared is not None:
        shared.update(lengths=lengths, by_ew=by_ew)
    mr = dict(zip(ranked["cell"].astype(int), ranked["mean_rank"]))
    rows, plaus = [], {}
    for rank, i in enumerate(top, 1):
        cfg = apply_cell(base, grid[i]); cfg.decoder.lengths = lengths
        merge = by_ew[cell_extra_weight(grid[i], dw)]["merge"]
        lab = predict(m, tags, cfg, None, evidence=ev_of[i], strong=strong,
                      merge_model=merge if uses_merge(grid[i]) else None)
        d = drift_report(lab, m, tags, mask, code_table=cfg.tag.code_table)
        rec = {"rank": rank, "mean_rank": float(mr[i]), **{k: float(d[k]) for k in REAL_KEYS}}
        rec.update(real_band_checks(rec["non_blackout_D_share"], rec["spells_per_graded_min"]))
        plaus[i] = rec
        rows.append({"cell": i, **grid[i], **rec})
    tab = pd.DataFrame(rows); tab.to_csv(out / "summary_real.csv", index=False)
    print(f"REAL-MATCH PLAUSIBILITY (top {len(top)}; plausible = non_blackout_D_share in {list(REAL_D_SHARE_BAND)} "
          f"AND spells_per_graded_min in {list(REAL_SPELL_RATE_BAND)}; tag_agreement_implied and spells_per_HA_min "
          "reported only)")
    print(tab.to_string(index=False), flush=True)
    return plaus


TEAM_MODEL_FILE = "team_model.joblib"
MERGE_MODEL_FILE = "merge_model.joblib"
SWITCH_MODEL_FILE = "switch_model.joblib"


def write_candidate_artifacts(out, base, best_cell, lengths, dead, team=None, merge=None, switch=None):
    """config.json (selected PipelineConfig + lengths), dead_model.joblib (chosen policy, all non-sealed keys) and,
    when the chosen config uses the teamctrl stream, team_model.joblib, when it has merge_theta < 1,
    merge_model.joblib, and when it uses the switch hazard (use_switch), switch_model.joblib next to it: the
    artefacts used by make_candidate.py AND the sealed read. An unused model file left in `out` is deleted."""
    cfg = apply_cell(base, best_cell); cfg.decoder.lengths = lengths
    if cfg.post.merge_theta < 1.0 and merge is None:
        raise ValueError("chosen cell has merge_theta < 1 but no merge model was given")
    if cfg.use_switch and switch is None:
        raise ValueError("chosen config uses the switch hazard but no switch model was given")
    dead.save(out / "dead_model.joblib")
    written = {TEAM_MODEL_FILE: team, MERGE_MODEL_FILE: merge if cfg.post.merge_theta < 1.0 else None,
               SWITCH_MODEL_FILE: switch if cfg.use_switch else None}
    for name, model in written.items():
        if model is not None:
            model.save(out / name)
        elif (out / name).exists():
            (out / name).unlink()   # stale artefact of an earlier registration in this dir: never left behind
    (out / "config.json").write_text(json.dumps({**cfg.to_dict(), "lengths": lengths}, indent=1))


def load_candidate_artifacts(sel_dir):
    """(cfg, dead, lengths) from a selection dir's config.json + dead_model.joblib."""
    cp, dp = Path(sel_dir) / "config.json", Path(sel_dir) / "dead_model.joblib"
    if not cp.exists() or not dp.exists():
        raise SystemExit(f"{sel_dir} lacks config.json/dead_model.joblib (written at registration); rerun the selection")
    raw = json.loads(cp.read_text()); lengths = raw.pop("lengths")
    cfg = PipelineConfig.from_dict(raw); cfg.decoder.lengths = lengths
    return cfg, DeadModel.load(dp), lengths


def load_team_model(sel_dir, cfg):
    """TeamCtrlModel from a selection dir's team_model.joblib, or None when cfg.use_teamctrl is off. Loaded whenever
    the stream is on, also at teamctrl weight 0 (a merge cell registered with the grid's teamctrl stream: its spell
    features read it). A config with use_teamctrl without the file is refused."""
    tp = Path(sel_dir) / TEAM_MODEL_FILE
    if not cfg.use_teamctrl:
        return None
    if not tp.exists():
        raise SystemExit(f"{sel_dir}: config uses the teamctrl stream but {TEAM_MODEL_FILE} is missing; rerun the selection")
    return TeamCtrlModel.load(tp)


def load_merge_model(sel_dir, cfg):
    """MergeModel from a selection dir's merge_model.joblib, or None when cfg.post.merge_theta >= 1 (off).
    A config with merge_theta < 1 without the file is refused."""
    mp = Path(sel_dir) / MERGE_MODEL_FILE
    if not cfg.post.merge_theta < 1.0:
        return None
    if not mp.exists():
        raise SystemExit(f"{sel_dir}: config has merge_theta {cfg.post.merge_theta} < 1 but {MERGE_MODEL_FILE} is missing; "
                         "rerun the selection")
    return MergeModel.load(mp)


def load_switch_model(sel_dir, cfg):
    """SwitchModel from a selection dir's switch_model.joblib, or None when cfg.use_switch is off. A config with
    use_switch without the file is refused."""
    sp = Path(sel_dir) / SWITCH_MODEL_FILE
    if not cfg.use_switch:
        return None
    if not sp.exists():
        raise SystemExit(f"{sel_dir}: config uses the switch hazard but {SWITCH_MODEL_FILE} is missing; rerun the selection")
    return SwitchModel.load(sp)


GRID_PARAM_COLS = ("tau_HA", "tau_D", "w_dead", "aba_s", "w_pathcrf", "dead_policy", "sp_dead_bonus", "w_teamctrl",
                   "w_ball", "team_policy", "merge_theta", "w_switch", "snap_mode", "dead_offset", "extra_weight")


def grid_hash(grid):
    return hashlib.md5(json.dumps(grid, sort_keys=True).encode()).hexdigest()


def run_fingerprint(seeds, grid, grid_name=None, extra=None):
    """What a fold table was produced with: seeds, grid and the extra training set must match to resume it; git only warns."""
    fx = None if extra is None else {k: v for k, v in extra.info().items() if k != "dir"}
    return {"git": git_sha(), "seeds": [int(s) for s in seeds], "grid": grid_name, "grid_hash": grid_hash(grid), "extra": fx}


def _fold_csv_matches_grid(df, grid):
    """Parameter columns of a fold CSV equal grid[i] for every cell i (and no cell is missing or extra)."""
    rows = df[~df["cell"].isin(list(BASELINES))].copy()
    if len(rows) != len(grid):
        return False
    rows["cell"] = rows["cell"].astype(int)
    rows = rows.set_index("cell")
    for i, cell in enumerate(grid):
        if i not in rows.index:
            return False
        for c in GRID_PARAM_COLS:
            if c in rows.columns and c in cell:
                v = rows.at[i, c]
                same = (v == cell[c]) if isinstance(cell[c], str) else bool(np.isclose(float(v), float(cell[c])))
                if not same:
                    return False
    return True


def check_resume(meta, df, grid, fp, out, held=None):
    """The one resume check. The fold's meta fingerprint must exist and hard-match the seeds and the grid hash, and the
    fold CSV's parameter columns must equal the grid. A different git sha only warns (an unrelated commit must not
    force a multi-hour rerun); the registration records the current sha."""
    where = f"fold {held} in {out}" if held is not None else str(out)
    if meta is None:
        raise SystemExit(f"refusing to resume {where}: no fold meta fingerprint (seeds/grid unverifiable); use a fresh --out")
    if [int(s) for s in meta.get("seeds") or []] != fp["seeds"]:
        raise SystemExit(f"refusing to resume {where}: seeds {meta.get('seeds')} != current {fp['seeds']}; use a fresh --out")
    if meta.get("grid_hash") != fp["grid_hash"] or not _fold_csv_matches_grid(df, grid):
        raise SystemExit(f"refusing to resume {where}: fold CSVs were produced with a different grid; use a fresh --out")
    if meta.get("extra") != fp.get("extra"):
        raise SystemExit(f"refusing to resume {where}: extra training set {meta.get('extra')} != current {fp.get('extra')}; "
                         "use a fresh --out")
    if meta.get("git") != fp["git"]:
        print(f"WARNING: resuming {where} produced at git {meta.get('git')} (now {fp['git']})", flush=True)


def run_grid(sim_dir, seeds, keys, base, out, grid, real_data_dir=None, grid_name=None, top_k=REAL_TOP_K, extra=None):
    fold_tables, lengths_by_fold, fold_models = [], {}, {}
    groups = extra_weight_groups(grid, extra)   # {extra_weight: [cell index]}: one model set per group
    resumed_folds = []
    fp = run_fingerprint(seeds, grid, grid_name, extra)
    for held, train_keys, test_keys in lomo_folds(keys):
        fold_csv = out / f"fold_{held}.csv"; halves_csv = out / f"fold_{held}_halves.csv"
        meta_path = out / f"fold_{held}.meta.json"
        if fold_csv.exists() and halves_csv.exists():
            meta = json.loads(meta_path.read_text()) if meta_path.exists() else None
            df = pd.read_csv(fold_csv); check_resume(meta, df, grid, fp, out, held)
            fold_tables.append(df); resumed_folds.append(held)
            lengths_by_fold[held] = json.loads(df["lengths"].iloc[0])
            print(f"fold {held} resumed", flush=True)
            continue
        base_train = load_sims(sim_dir, seeds, train_keys); test = load_sims(sim_dir, seeds, test_keys)
        lengths = estimate_lengths(base_train, base, train_dead)  # DFL halves only (the extras never decide L_k)
        lengths_by_fold[held] = lengths; fold_models[held] = {}
        rows_by_cell, halves, test_strong = {}, [], None
        for ew, idxs in groups.items():
            sub = [grid[i] for i in idxs]
            train = base_train + (extra.at(ew) if extra is not None else [])   # the extras: training only, at weight ew
            policies = sorted({c.get("dead_policy", "full") for c in sub})
            team_policies = sorted({c.get("team_policy", "full") for c in sub})
            dead = {p: train_dead(train, policy=p) for p in policies}  # final fold dead models: all train halves, one per policy
            team = train_team_models(train, base, team_policies)   # fold team-control models (none unless base.use_teamctrl)
            merge = train_merge(train, base, sub, lengths)         # fold merge model, inner split of train (None: no merge cell)
            switch = train_switch_model(train, base)               # fold switch model, all train halves (None unless use_switch)
            fold_models[held][ew] = dict(dead=dead, team=team, merge=merge, switch=switch)
            del train
            cfgs = policy_cfgs(base, sub, lengths)
            test_ev = {k: [compute_evidence(s.match, s.tags, cfg, dead[k[0]], cache_key=f"{s.key}_s{s.seed}",
                                            team_model=team.get(k[2]), switch_model=switch) for s in test]
                       for k, cfg in cfgs.items()}  # one evidence set per (dead policy, sp_dead_bonus, team policy); the
                                                    # switch hazard (when present) is the same in every set
            if test_strong is None:
                test_strong = [compute_strong(s.match, next(iter(cfgs.values()))) for s in test]  # touches do not depend on the grid
            for i in idxs:
                cell = grid[i]
                cfg = apply_cell(base, cell); cfg.decoder.lengths = lengths
                res, hdf = score_cell(test, test_ev[ev_key(cell)], cfg, VARIANTS, strongs=test_strong,
                                      merge_model=merge if uses_merge(cell) else None)
                hdf.insert(0, "cell", i); halves.append(hdf)
                row = {"cell": i, **cell}
                for v in VARIANTS:
                    row[v] = res[v]["spell_f1"]; row[f"{v}_ratio"] = res[v]["ratio"]
                rows_by_cell[i] = row
        rows = [rows_by_cell[i] for i in range(len(grid))]
        del base_train
        meta_path.write_text(json.dumps(fp))   # before any fold CSV: a crash then recomputes instead of refusing
        pd.concat(halves).to_csv(halves_csv, index=False)
        brows = []
        for name, fn in BASELINES.items():
            row = {"cell": name}
            for v in VARIANTS:
                vals = [score(s.truth[v], fn(s.match))["spell_f1"] for s in test]
                brows += [{"key": s.key, "seed": s.seed, "variant": v, "baseline": name, "spell_f1": f}
                          for s, f in zip(test, vals)]
                row[v] = float(np.mean(vals))
            rows.append(row)
        pd.DataFrame(brows).to_csv(out / f"fold_{held}_baselines.csv", index=False)
        df = pd.DataFrame(rows); df["held"] = held; df["lengths"] = json.dumps(lengths)
        minfo = {ew: m["merge"].info for ew, m in fold_models[held].items() if m["merge"] is not None}
        if minfo:   # one merge model per extra_weight group; a single group keeps the plain info layout
            (out / f"fold_{held}_merge.json").write_text(json.dumps(next(iter(minfo.values())) if len(groups) == 1
                                                                    else {str(k): v for k, v in minfo.items()}, indent=1))
        df.to_csv(fold_csv, index=False); fold_tables.append(df)
        grid_rows = df[~df["cell"].isin(list(BASELINES))]
        print(f"fold {held} done; best ctrl_Dfull {grid_rows['ctrl_Dfull'].max():.3f}", flush=True)
    allf = pd.concat(fold_tables)
    is_b = allf["cell"].isin(list(BASELINES))
    gcells = allf[~is_b].copy(); gcells["cell"] = gcells["cell"].astype(int)
    cells = gcells.groupby("cell")[list(VARIANTS) + [f"{v}_ratio" for v in VARIANTS]].mean().reset_index()
    ranked = mean_rank(cells, list(VARIANTS)); ranked.to_csv(out / "summary.csv", index=False)
    print(ranked.head(top_k).to_string())
    plaus = None; shared = {}
    if real_data_dir is not None:
        plaus = real_plausibility(sim_dir, seeds, keys, base, grid, ranked, real_data_dir, out, top_k=top_k, shared=shared,
                                  extra=extra)
        best_i = choose_plausible(ranked, plaus)
    else:
        best_i = int(ranked.iloc[0]["cell"])
    is_best = (ranked["cell"].astype(int) == best_i).to_numpy()
    best = ranked[is_best].iloc[0].to_dict(); best_cell = grid[best_i]
    chosen_rank = int(np.flatnonzero(is_best)[0]) + 1
    print(f"chosen cell {best_i} (mean-rank position {chosen_rank}): {best_cell}")
    allf = allf.copy(); allf["cell"] = allf["cell"].astype(str)
    halves = pd.concat([pd.read_csv(out / f"fold_{h}_halves.csv") for h in sorted(allf["held"].unique())])
    halves["cell"] = halves["cell"].astype(str)
    bar, base_rows = compute_bar(allf, halves, best, str(best_i))
    print(base_rows.to_string())
    print("SPEC BAR", json.dumps(bar, indent=1))
    ref = out_of_sample_drift_reference(sim_dir, seeds, keys, base, best_cell, lengths_by_fold, fold_models, grid, extra)
    (out / "drift_reference.json").write_text(json.dumps(ref, indent=1))
    cbase = base_for_grid([best_cell])   # PathCRF / teamctrl / switch only if the chosen cell uses them ...
    need_merge = uses_merge(best_cell)
    if need_merge:
        # ... except for a merge cell: its spell features (mtah.spells.merge.MERGE_STREAMS) read every stream the
        # grid computed, also at decoder weight 0 (e.g. w_teamctrl 0). Register those streams (weights stay as in the
        # cell) so the registered pipeline reproduces the selection-time labels exactly.
        cbase.use_teamctrl = bool(base.use_teamctrl); cbase.use_pathcrf = bool(base.use_pathcrf)
    policy = best_cell.get("dead_policy", "full"); tpol = best_cell.get("team_policy", "full")
    best_ew = cell_extra_weight(best_cell, extra.default_weight if extra is not None else 0.0)
    need_team = bool(cbase.use_teamctrl); need_switch = bool(cbase.use_switch)
    sh = (shared or {}).get("by_ew", {}).get(best_ew)   # models the plausibility step trained for the chosen cell's group
    if (sh and policy in sh["dead"] and (not need_team or tpol in sh.get("team", {}))
            and (not need_merge or sh.get("merge") is not None)
            and (not need_switch or sh.get("switch") is not None)):
        c_lengths, c_dead = shared["lengths"], sh["dead"][policy]   # already trained by the plausibility step
        c_team = sh["team"][tpol] if need_team else None
        c_merge = sh.get("merge") if need_merge else None
        c_switch = sh.get("switch") if need_switch else None
    else:
        c_base = load_sims(sim_dir, seeds, keys)
        c_train = c_base + (extra.at(best_ew) if extra is not None else [])
        c_lengths = estimate_lengths(c_base, cbase, train_dead); c_dead = train_dead(c_train, policy=policy)
        c_team = train_team_ctrl(c_train, policy=tpol, touch_params=cbase.touch) if need_team else None
        c_merge = train_merge(c_train, base, _group_cells(grid, best_ew, extra), c_lengths) if need_merge else None   # base: the grid's streams
        c_switch = train_switch_model(c_train, cbase) if need_switch else None
        del c_train, c_base
    write_candidate_artifacts(out, cbase, best_cell, c_lengths, c_dead, c_team, c_merge, c_switch)
    selection = "mean rank over 4 variants, tie ctrl_Dfull"
    if plaus is not None:
        selection += (f"; highest-ranked real-plausible cell among the top {top_k} "
                      f"(real non-blackout D share in {list(REAL_D_SHARE_BAND)} and spells per graded minute in "
                      f"{list(REAL_SPELL_RATE_BAND)}; fallback rank 1)")
    real_bands = {"non_blackout_D_share": list(REAL_D_SHARE_BAND), "spells_per_graded_min": list(REAL_SPELL_RATE_BAND)}
    merge_info = c_merge.info if c_merge is not None else None
    real_json = None if plaus is None else {str(i): v for i, v in plaus.items()}
    chosen_real_plausible = None if plaus is None else bool(plaus.get(best_i, {}).get("real_plausible", False))
    reg = {"git": git_sha(), "seeds": seeds, "grid": grid_name, "grid_size": len(grid), "selection": selection,
           "scorer_md5": hashlib.md5(OFFICIAL_PATH.read_bytes().replace(b"\r\n", b"\n")).hexdigest(),
           "best_cell": best_cell, "best_cell_index": best_i, "chosen_rank": chosen_rank,
           "best": {v: best[v] for v in VARIANTS}, "baselines": base_rows.to_dict(),
           "seed_sd": bar["seed_sd"], "baseline_used": bar["baseline_used"], "spec_bar": bar,
           "use_pathcrf": bool(base.use_pathcrf), "pathcrf_threads": int(base.pathcrf.threads),
           "use_teamctrl": bool(cbase.use_teamctrl), "team_policy": tpol if cbase.use_teamctrl else None,
           "real_plausibility": real_json, "chosen_real_plausible": chosen_real_plausible, "real_bands": real_bands,
           "use_merge": need_merge, "merge_model": merge_info,
           "use_switch": need_switch, "switch_policy": cbase.switch_policy if need_switch else None,
           "pathcrf_train_matches": [9001, 9002, 9003, 9004, 9005], "pathcrf_val": 9006, "clean": 9007,
           "sealed_match": SEALED_MATCH, "lengths_mode": LENGTHS_MODE,
           "train_extra": None if extra is None else extra.info(), "extra_weight": best_ew if extra is not None else 0.0}
    (out / "registration.json").write_text(json.dumps(reg, indent=1))
    (out / "best.json").write_text(json.dumps({"cell": best_cell, "cell_index": best_i, "chosen_rank": chosen_rank,
                                               "scores": {v: best[v] for v in VARIANTS},
                                               "ratios": bar["count_ratios"], "seed_sd": bar["seed_sd"],
                                               "baseline_used": bar["baseline_used"], "spec_bar": bar,
                                               "real_plausibility": real_json, "chosen_real_plausible": chosen_real_plausible,
                                               "real_bands": real_bands,
                                               "git": git_sha(), "seeds": list(seeds), "grid": grid_name}, indent=1))
    return {"best_cell": best_cell, "resumed_folds": resumed_folds, "bar_pass": bar["bar_pass"],
            "chosen_real_plausible": chosen_real_plausible}


def run_selection(sim_dir, seeds, out_dir, grid=None, base=None, real_data_dir=None, grid_name=None, extra=None):
    """Importable LOMO selection loop: fold tables, summary, best/registration, out-of-sample drift reference.

    real_data_dir=None skips the real-match plausibility step (the chosen cell is then mean-rank 1).
    extra: an mtah.select.ExtraSet of training-only halves (see the module docstring)."""
    grid = grid if grid is not None else sel.GRID
    grid_name = grid_name or next((n for n, g in sel.GRIDS.items() if g is grid), "custom")
    out = Path(out_dir); out.mkdir(parents=True, exist_ok=True)
    keys = [k for k in list_sim_keys(sim_dir) if match_id(k) != SEALED_MATCH]
    base = base if base is not None else base_for_grid(grid)
    return run_grid(sim_dir, list(seeds), keys, base, out, grid, real_data_dir=real_data_dir, grid_name=grid_name,
                    extra=extra)


def run_pilot(args, keys, base, grid):
    """First fold only: print L_k and the eight tau/w_dead corner cells around grid[0]. Writes nothing."""
    held, train_keys, test_keys = lomo_folds(keys)[0]
    train = load_sims(args.sim, args.seeds, train_keys); test = load_sims(args.sim, args.seeds, test_keys)
    lengths = estimate_lengths(train, base, train_dead)
    print(f"PILOT fold {held} L_k:", json.dumps(lengths))
    cfg0 = apply_cell(base, grid[0]); cfg0.decoder.lengths = lengths
    dead = train_dead(train, policy=cfg0.dead_policy)
    team = train_team_models(train, base, [cfg0.team_policy]).get(cfg0.team_policy)
    ev = [compute_evidence(s.match, s.tags, cfg0, dead, cache_key=f"{s.key}_s{s.seed}", team_model=team) for s in test]
    strong = [compute_strong(s.match, cfg0) for s in test]
    for a, d, w in itertools.product((3.0, 100.0), (3.0, 100.0), (0.5, 2.0)):
        cell = {**grid[0], "tau_HA": a, "tau_D": d, "w_dead": w, "aba_s": 0.5 if grid is sel.GRID else grid[0]["aba_s"]}
        if "merge_theta" in cell:
            cell["merge_theta"] = 1.0   # the pilot never trains a merge model
        if "w_switch" in cell or "snap_mode" in cell:
            cell["w_switch"] = 0.0; cell["snap_mode"] = "touch"   # ... nor a switch model
        cfg = apply_cell(base, cell); cfg.decoder.lengths = lengths
        res, _ = score_cell(test, ev, cfg, ("ctrl_Dfull",), strongs=strong)
        r = res["ctrl_Dfull"]
        print(f"  tau_HA={a:>4} tau_D={d:>4} w_dead={w:>3}  n_pred/n_gt={r['ratio']:.3f}  spell_f1={r['spell_f1']:.3f}", flush=True)


def _repo_rel(path):
    """Path relative to the repo root when inside it (tracked log hygiene), else the basename."""
    p = Path(path).resolve()
    try:
        return p.relative_to(ROOT).as_posix()
    except ValueError:
        return p.name


def registration_id(reg):
    """Identity of a registration: grid name, seeds, and sha256 of registration.json's best_cell/grid_size/seeds."""
    core = {"best_cell": reg.get("best_cell"), "grid_size": reg.get("grid_size"), "seeds": reg.get("seeds")}
    return {"grid": reg.get("grid"), "seeds": reg.get("seeds"),
            "sha256": hashlib.sha256(json.dumps(core, sort_keys=True).encode()).hexdigest()}


def check_sealed_guard(reg_id, force_reason=None, match=SEALED_MATCH):
    """Refuse a second read of the sealed match for the same registration (docs/reads/sealed.jsonl, tracked) unless
    --force-sealed REASON is given. Returns the earlier reads of this registration (empty when it is the first)."""
    prev = [e for e in readlog.read_log("sealed")   # legacy entries without a `match` field are skipped (a1 was a 9006 read)
            if e.get("registration") == reg_id and "match" in e and int(e["match"]) == int(match)]
    if prev and not force_reason:
        raise SystemExit(f"refusing sealed read: registration {reg_id['grid']} seeds {reg_id['seeds']} "
                         f"sha256 {reg_id['sha256'][:12]} already read {len(prev)}x ({prev[-1].get('time')}); "
                         "pass --force-sealed REASON to override")
    if prev:
        print(f"!!! WARNING: --force-sealed ({force_reason}): re-reading the sealed match for an already-read registration !!!")
    return prev


def check_bar_gate(best, force_reason=None):
    """Sealed reads need spec_bar.bar_pass AND chosen_real_plausible not False in best.json (null = real step
    skipped with --no-real), or an explicit --force-sealed REASON (logged). Returns the list of failed gates."""
    bar = best.get("spec_bar") or {}
    failed = []
    if not bar.get("bar_pass"):
        failed.append("bar")
    if best.get("chosen_real_plausible") is False:
        failed.append("real_plausible")
    if not failed:
        return []
    print("SPEC BAR", json.dumps(bar, indent=1))
    print("chosen_real_plausible:", best.get("chosen_real_plausible"))
    if not force_reason:
        raise SystemExit(f"refusing sealed read: gate(s) failed {failed}; pass --force-sealed REASON to override")
    print(f"!!! WARNING: --force-sealed ({force_reason}): reading the sealed match although {failed} FAILED !!!")
    return failed


def load_best(best_dir, seeds):
    best = json.loads((Path(best_dir) / "best.json").read_text())
    missing = [k for k in ("git", "seeds") if k not in best]
    if missing:
        raise SystemExit(f"refusing --best-from {best_dir}: best.json lacks {missing}")
    if list(best["seeds"]) != list(seeds):
        raise SystemExit(f"refusing --best-from: recorded seeds {best['seeds']} != --seeds {list(seeds)}")
    if best["git"] != git_sha():
        print(f"WARNING: best.json was made at git {best['git']}, current is {git_sha()}")
    return best


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sim", default=str(ROOT / "runs" / "sim"))
    ap.add_argument("--seeds", type=int, nargs="+", default=[100, 101])
    ap.add_argument("--out", default=None,
                    help="selection dir (default runs/select/<timestamp>; with --sealed --best-from X, default X)")
    ap.add_argument("--grid", choices=sorted(sel.GRIDS), default="grid1",
                    help="grid1 (default) | grid2 | grid3 | grid4; the real-match plausibility step runs for every grid unless --no-real")
    ap.add_argument("--data", default=os.environ.get("MTAH_DATA_DIR"),
                    help="real-match data dir for the plausibility step (default $MTAH_DATA_DIR)")
    ap.add_argument("--no-real", action="store_true", help="skip the real-match plausibility step (choose mean-rank 1)")
    ap.add_argument("--train-extra", default=None, metavar="DIR",
                    help="build_sim output dir of training-only extra halves (e.g. runs/sim_extra: SkillCorner); added to every "
                         "fold's training set at the cell's extra_weight, never to test folds / drift reference / plausibility")
    ap.add_argument("--extra-weight", type=float, default=1.0, metavar="W",
                    help="training weight of the extra halves for cells without an extra_weight (default 1.0)")
    ap.add_argument("--best-from", default=None, help="selection dir whose best.json is reused (skips the grid loop)")
    ap.add_argument("--pilot", action="store_true", help="first fold only: print L_k and the 8 corner cells; writes nothing")
    ap.add_argument("--sealed", action="store_true")
    ap.add_argument("--force-sealed", default=None, metavar="REASON",
                    help="read the sealed match although a gate failed; the reason is logged in docs/reads/sealed.jsonl")
    args = ap.parse_args()
    grid = sel.GRIDS[args.grid]
    keys = [k for k in list_sim_keys(args.sim) if match_id(k) != SEALED_MATCH]
    if args.pilot:
        if args.train_extra:
            print("WARNING: --pilot ignores --train-extra")
        run_pilot(args, keys, base_for_grid(grid), grid)
        return
    extra = None
    if args.train_extra:
        extra = ExtraSet(load_extra_halves(args.train_extra, args.seeds), args.extra_weight, args.train_extra)
        print("training-only extra set:", json.dumps(extra.info()), flush=True)
    if args.out is None:
        args.out = args.best_from if (args.sealed and args.best_from) else str(ROOT / "runs" / "select" / time.strftime("%Y%m%d-%H%M"))
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    if args.best_from:
        best = load_best(args.best_from, args.seeds)
        best_cell = best["cell"]
        print("reusing best cell from", args.best_from, best_cell)
    else:
        best_cell = run_selection(args.sim, args.seeds, out, grid=grid, base=base_for_grid(grid),
                                  real_data_dir=None if args.no_real else data_dir_or_exit(args.data),
                                  grid_name=args.grid, extra=extra)["best_cell"]
    if args.sealed:
        sel_dir = Path(args.best_from or out)
        if not args.best_from:
            best = load_best(sel_dir, args.seeds)   # the best.json run_selection just wrote (--best-from loaded it above)
        reg_path = sel_dir / "registration.json"
        if not reg_path.exists():
            raise SystemExit(f"{reg_path} missing: cannot verify the registered seeds; refusing the sealed read")
        reg = json.loads(reg_path.read_text()); reg_seeds = reg.get("seeds")
        if reg_seeds != list(args.seeds):
            raise SystemExit(f"--seeds {list(args.seeds)} differ from the registered seeds {reg_seeds} in {reg_path}; "
                             "refusing the sealed read")
        if reg.get("best_cell") != best_cell:
            raise SystemExit(f"best.json cell {best_cell} != registered best_cell {reg.get('best_cell')} in {reg_path}")
        reg_id = registration_id(reg)
        failed = check_bar_gate(best, args.force_sealed)
        prev = check_sealed_guard(reg_id, args.force_sealed)
        cfg, dead, lengths = load_candidate_artifacts(sel_dir)   # byte-identical to what make_candidate.py uses
        team = load_team_model(sel_dir, cfg)
        merge = load_merge_model(sel_dir, cfg)
        switch = load_switch_model(sel_dir, cfg)
        names = (("config.json", "dead_model.joblib") + ((TEAM_MODEL_FILE,) if team is not None else ())
                 + ((MERGE_MODEL_FILE,) if merge is not None else ()) + ((SWITCH_MODEL_FILE,) if switch is not None else ()))
        for name in names:
            if not (out / name).exists() or (out / name).read_bytes() != (sel_dir / name).read_bytes():
                (out / name).write_bytes((sel_dir / name).read_bytes())
        ref_path = out / "drift_reference.json"
        if not ref_path.exists() and args.best_from and (Path(args.best_from) / "drift_reference.json").exists():
            ref_path.write_text((Path(args.best_from) / "drift_reference.json").read_text())
        if not ref_path.exists():
            print("WARNING: drift_reference.json absent; writing an IN-SAMPLE reference (train halves)")
            train = load_sims(args.sim, args.seeds, keys)
            rows = [drift_report(predict(s.match, s.tags, cfg, dead, cache_key=f"{s.key}_s{s.seed}", team_model=team,
                                         merge_model=merge, switch_model=switch),
                                 s.match, s.tags,
                                 blackout_mask_from_match(s.match), code_table=cfg.tag.code_table)
                    for s in train]
            ref_path.write_text(json.dumps(_drift_summary(rows), indent=1))
        sealed = load_sims(args.sim, args.seeds, [k for k in list_sim_keys(args.sim) if match_id(k) == SEALED_MATCH])
        ev = [compute_evidence(s.match, s.tags, cfg, dead, cache_key=f"{s.key}_s{s.seed}", team_model=team,
                               switch_model=switch) for s in sealed]
        res = {v: m["spell_f1"] for v, m in score_cell(sealed, ev, cfg, VARIANTS, merge_model=merge)[0].items()}
        forced_over = failed + (["repeat_read"] if prev else [])
        rec = {"time": time.strftime("%Y-%m-%d %H:%M"), "git": git_sha(), "match": SEALED_MATCH, "registration": reg_id,
               "select_dir": _repo_rel(sel_dir), "cell": best_cell, "seeds": list(args.seeds), "sealed": res,
               "bar_pass": bool(best.get("spec_bar", {}).get("bar_pass", False)),
               "chosen_real_plausible": best.get("chosen_real_plausible"),
               "forced": bool(forced_over), "forced_over": forced_over,
               "force_reason": args.force_sealed if forced_over else None}
        readlog.append_read("sealed", rec)
        print("SEALED", json.dumps(res))


if __name__ == "__main__":
    main()
