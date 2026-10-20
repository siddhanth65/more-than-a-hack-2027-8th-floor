"""LOMO selection over a grid (grid1 | grid2) on the simulated dataset; writes fold tables, summary and registration.

With a real-data dir, the top-k cells by mean rank are also decoded on the real match (label-free drift metrics);
the chosen cell is the highest-ranked one whose real non-blackout D share is plausible (select.REAL_D_SHARE_BAND).
This real-match plausibility step runs for EVERY grid (grid1 included) unless --no-real is given.
PathCRF threads are pinned to PathCRFParams().threads (4): the posterior cache is keyed on it, no CLI knob."""
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
from mtah.evidence.pathcrf import PathCRFParams
from mtah.io.metrica_loader import load_tactical
from mtah.io.tags import load_tags
from mtah.pipeline import PipelineConfig, compute_evidence, compute_strong, predict
import mtah.select as sel
from mtah.select import (REAL_D_SHARE_BAND, apply_cell, choose_plausible, estimate_lengths, is_real_plausible, mean_rank,
                         score_cell, train_dead)
from mtah.sim.dataset import SEALED_MATCH, VARIANTS, list_sim_keys, load_half_sim, lomo_folds, match_id

REAL_KEYS = ("non_blackout_D_share", "tag_agreement_implied", "pred_D_share", "spells_per_HA_min")
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
    (w_pathcrf = 0 cells then ignore it through the zero weight). Threads stay at the pinned PathCRFParams default
    (4; the posterior and its cache key depend on it); `threads` exists for tests only, no CLI sets it."""
    base = PipelineConfig()
    base.use_pathcrf = any(c.get("w_pathcrf", 0.0) > 0 for c in grid)
    if threads is not None:
        base.pathcrf.threads = int(threads)
    return base


def policy_cfgs(base, cells, lengths):
    """{dead_policy: cfg} with cfg = apply_cell(base, first cell of that policy). Evidence depends on the cell only
    through dead_policy (weights/taus act in the decoder), so one evidence set per policy serves every cell."""
    out = {}
    for c in cells:
        p = c.get("dead_policy", "full")
        if p not in out:
            cfg = apply_cell(base, c); cfg.decoder.lengths = lengths; out[p] = cfg
    return dict(sorted(out.items()))


BASELINES = {"smooth50": lambda m: baselines.smooth_vote(m, 50), "b4": baselines.b4_blackout_dead}


def _drift_summary(rows):
    ref = {}
    for k in sorted({k for r in rows for k in r}):
        v = np.array([r[k] for r in rows if k in r], float)
        if v.size and not np.all(np.isnan(v)):
            ref[k] = {"min": float(np.nanmin(v)), "median": float(np.nanmedian(v)), "max": float(np.nanmax(v))}
    return ref


def out_of_sample_drift_reference(sim_dir, seeds, keys, base, best_cell, lengths_by_fold, dead_by_fold):
    """Re-decode the best cell on every fold's TEST halves and summarise the drift metrics."""
    rows = []
    policy = best_cell.get("dead_policy", "full")
    for held, train_keys, test_keys in lomo_folds(keys):
        train = load_sims(sim_dir, seeds, train_keys); test = load_sims(sim_dir, seeds, test_keys)
        lengths = lengths_by_fold.get(held) or estimate_lengths(train, base, train_dead)
        dead = dead_by_fold.get(held, {}).get(policy)
        if dead is None:
            dead = train_dead(train, policy=policy)
        cfg = apply_cell(base, best_cell); cfg.decoder.lengths = lengths
        for s in test:
            ev = compute_evidence(s.match, s.tags, cfg, dead, cache_key=f"{s.key}_s{s.seed}")
            lab = predict(s.match, s.tags, cfg, dead, evidence=ev)
            rows.append(drift_report(lab, s.match, s.tags, blackout_mask_from_match(s.match)))
    return _drift_summary(rows)


def compute_bar(allf, halves, best, best_i):
    """Spec bar: margins over the strongest baseline, 2x seed SD, folds won, count ratios."""
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
    ratios = {v: float(best[f"{v}_ratio"]) for v in VARIANTS}
    need = {v: max(0.05, 2.0 * seed_sd[v]) for v in VARIANTS}
    bar_pass = bool(all(margins[v] >= need[v] for v in VARIANTS) and folds_won >= 4
                    and all(0.8 <= r <= 1.25 for r in ratios.values()))
    bar = {"margins_vs_best_baseline": margins, "required_margin": need, "seed_sd": seed_sd,
           "baseline_used": baseline_used, "folds_won_ctrl_Dfull": folds_won, "n_folds": int(len(cellf)),
           "count_ratios": ratios, "bar_pass": bar_pass}
    return bar, base_rows


def real_plausibility(sim_dir, seeds, keys, base, grid, ranked, data_dir, out, top_k=REAL_TOP_K):
    """Decode the top-k cells (by mean rank) on the real match; label-free drift metrics per cell.

    Dead models (one per needed policy) and lengths come from ALL non-sealed sim halves, as in the sealed/candidate
    path. Writes summary_real.csv; returns {cell index: {metric: value, "real_plausible": bool, ...}}."""
    top = [int(c) for c in ranked["cell"].head(top_k)]
    train = load_sims(sim_dir, seeds, keys)
    lengths = estimate_lengths(train, base, train_dead)
    cfgs = policy_cfgs(base, [grid[i] for i in top], lengths)
    dead = {p: train_dead(train, policy=p) for p in cfgs}
    del train
    m = load_tactical(data_dir); tags = load_tags(data_dir); mask = blackout_mask_from_match(m)
    ev = {p: compute_evidence(m, tags, cfg, dead[p], cache_key="real_tactical") for p, cfg in cfgs.items()}
    strong = compute_strong(m, next(iter(cfgs.values())))
    mr = dict(zip(ranked["cell"].astype(int), ranked["mean_rank"]))
    rows, plaus = [], {}
    for rank, i in enumerate(top, 1):
        cfg = apply_cell(base, grid[i]); cfg.decoder.lengths = lengths
        lab = predict(m, tags, cfg, None, evidence=ev[cfg.dead_policy], strong=strong)
        d = drift_report(lab, m, tags, mask, code_table=cfg.tag.code_table)
        rec = {"rank": rank, "mean_rank": float(mr[i]), **{k: float(d[k]) for k in REAL_KEYS}}
        rec["real_plausible"] = is_real_plausible(rec["non_blackout_D_share"])
        plaus[i] = rec
        rows.append({"cell": i, **grid[i], **rec})
    tab = pd.DataFrame(rows); tab.to_csv(out / "summary_real.csv", index=False)
    print(f"REAL-MATCH PLAUSIBILITY (top {len(top)}; plausible = non_blackout_D_share in {list(REAL_D_SHARE_BAND)})")
    print(tab.to_string(index=False), flush=True)
    return plaus


GRID_PARAM_COLS = ("tau_HA", "tau_D", "w_dead", "aba_s", "w_pathcrf", "dead_policy")


def check_resume_grid(df, grid, out):
    """A resumed fold CSV must have been produced with the current grid: parameter columns equal grid[i] for every i."""
    bad = SystemExit(f"fold CSVs in {out} were produced with a different grid; use a fresh --out")
    rows = df[~df["cell"].isin(list(BASELINES))].copy()
    if len(rows) != len(grid):
        raise bad
    rows["cell"] = rows["cell"].astype(int)
    rows = rows.set_index("cell")
    for i, cell in enumerate(grid):
        if i not in rows.index:
            raise bad
        for c in GRID_PARAM_COLS:
            if c in rows.columns and c in cell:
                v = rows.at[i, c]
                same = (v == cell[c]) if isinstance(cell[c], str) else bool(np.isclose(float(v), float(cell[c])))
                if not same:
                    raise bad


def run_grid(sim_dir, seeds, keys, base, out, grid, real_data_dir=None, grid_name=None, top_k=REAL_TOP_K):
    fold_tables, lengths_by_fold, dead_by_fold = [], {}, {}
    policies = sorted({c.get("dead_policy", "full") for c in grid})
    for held, train_keys, test_keys in lomo_folds(keys):
        fold_csv = out / f"fold_{held}.csv"; halves_csv = out / f"fold_{held}_halves.csv"
        if fold_csv.exists() and halves_csv.exists():
            df = pd.read_csv(fold_csv); check_resume_grid(df, grid, out)
            fold_tables.append(df)
            lengths_by_fold[held] = json.loads(df["lengths"].iloc[0])
            print(f"fold {held} resumed", flush=True)
            continue
        train = load_sims(sim_dir, seeds, train_keys); test = load_sims(sim_dir, seeds, test_keys)
        lengths = estimate_lengths(train, base, train_dead)  # inner split: see estimate_lengths (policy-independent)
        dead = {p: train_dead(train, policy=p) for p in policies}  # final fold dead models: all train halves, one per policy
        lengths_by_fold[held] = lengths; dead_by_fold[held] = dead
        cfgs = policy_cfgs(base, grid, lengths)
        test_ev = {p: [compute_evidence(s.match, s.tags, cfg, dead[p], cache_key=f"{s.key}_s{s.seed}") for s in test]
                   for p, cfg in cfgs.items()}  # one evidence set per dead policy; otherwise grid-independent (tested)
        test_strong = [compute_strong(s.match, cfgs[policies[0]]) for s in test]  # touches do not depend on the grid
        rows, halves = [], []
        for i, cell in enumerate(grid):
            cfg = apply_cell(base, cell); cfg.decoder.lengths = lengths
            res, hdf = score_cell(test, test_ev[cfg.dead_policy], cfg, VARIANTS, strongs=test_strong)
            hdf.insert(0, "cell", i); halves.append(hdf)
            row = {"cell": i, **cell}
            for v in VARIANTS:
                row[v] = res[v]["spell_f1"]; row[f"{v}_ratio"] = res[v]["ratio"]
            rows.append(row)
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
        df.to_csv(fold_csv, index=False); fold_tables.append(df)
        grid_rows = df[~df["cell"].isin(list(BASELINES))]
        print(f"fold {held} done; best ctrl_Dfull {grid_rows['ctrl_Dfull'].max():.3f}", flush=True)
    allf = pd.concat(fold_tables)
    is_b = allf["cell"].isin(list(BASELINES))
    gcells = allf[~is_b].copy(); gcells["cell"] = gcells["cell"].astype(int)
    cells = gcells.groupby("cell")[list(VARIANTS) + [f"{v}_ratio" for v in VARIANTS]].mean().reset_index()
    ranked = mean_rank(cells, list(VARIANTS)); ranked.to_csv(out / "summary.csv", index=False)
    print(ranked.head(top_k).to_string())
    plaus = None
    if real_data_dir is not None:
        plaus = real_plausibility(sim_dir, seeds, keys, base, grid, ranked, real_data_dir, out, top_k=top_k)
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
    ref = out_of_sample_drift_reference(sim_dir, seeds, keys, base, best_cell, lengths_by_fold, dead_by_fold)
    (out / "drift_reference.json").write_text(json.dumps(ref, indent=1))
    selection = "mean rank over 4 variants, tie ctrl_Dfull"
    if plaus is not None:
        selection += (f"; highest-ranked real-plausible cell among the top {top_k} "
                      f"(real non-blackout D share in {list(REAL_D_SHARE_BAND)}; fallback rank 1)")
    real_json = None if plaus is None else {str(i): v for i, v in plaus.items()}
    reg = {"git": git_sha(), "seeds": seeds, "grid": grid_name, "grid_size": len(grid), "selection": selection,
           "scorer_md5": hashlib.md5(OFFICIAL_PATH.read_bytes().replace(b"\r\n", b"\n")).hexdigest(),
           "best_cell": best_cell, "best_cell_index": best_i, "chosen_rank": chosen_rank,
           "best": {v: best[v] for v in VARIANTS}, "baselines": base_rows.to_dict(),
           "seed_sd": bar["seed_sd"], "baseline_used": bar["baseline_used"], "spec_bar": bar,
           "use_pathcrf": bool(base.use_pathcrf), "pathcrf_threads": int(base.pathcrf.threads),
           "real_plausibility": real_json,
           "pathcrf_train_matches": [9001, 9002, 9003, 9004, 9005], "pathcrf_val": 9006, "clean": 9007,
           "sealed_match": SEALED_MATCH}
    (out / "registration.json").write_text(json.dumps(reg, indent=1))
    (out / "best.json").write_text(json.dumps({"cell": best_cell, "cell_index": best_i, "chosen_rank": chosen_rank,
                                               "scores": {v: best[v] for v in VARIANTS},
                                               "ratios": bar["count_ratios"], "seed_sd": bar["seed_sd"],
                                               "baseline_used": bar["baseline_used"], "spec_bar": bar,
                                               "real_plausibility": real_json}, indent=1))
    return best_cell


def run_selection(sim_dir, seeds, out_dir, grid=None, base=None, real_data_dir=None, grid_name=None):
    """Importable LOMO selection loop: fold tables, summary, best/registration, out-of-sample drift reference.

    real_data_dir=None skips the real-match plausibility step (the chosen cell is then mean-rank 1)."""
    grid = grid if grid is not None else sel.GRID
    grid_name = grid_name or next((n for n, g in sel.GRIDS.items() if g is grid), "custom")
    out = Path(out_dir); out.mkdir(parents=True, exist_ok=True)
    keys = [k for k in list_sim_keys(sim_dir) if match_id(k) != SEALED_MATCH]
    base = base if base is not None else base_for_grid(grid)
    return run_grid(sim_dir, list(seeds), keys, base, out, grid, real_data_dir=real_data_dir, grid_name=grid_name)


def run_pilot(args, keys, base, grid):
    """First fold only: print L_k and the eight tau/w_dead corner cells around grid[0]. Writes nothing."""
    held, train_keys, test_keys = lomo_folds(keys)[0]
    train = load_sims(args.sim, args.seeds, train_keys); test = load_sims(args.sim, args.seeds, test_keys)
    lengths = estimate_lengths(train, base, train_dead)
    print(f"PILOT fold {held} L_k:", json.dumps(lengths))
    cfg0 = apply_cell(base, grid[0]); cfg0.decoder.lengths = lengths
    dead = train_dead(train, policy=cfg0.dead_policy)
    ev = [compute_evidence(s.match, s.tags, cfg0, dead, cache_key=f"{s.key}_s{s.seed}") for s in test]
    strong = [compute_strong(s.match, cfg0) for s in test]
    for a, d, w in itertools.product((3.0, 30.0), (3.0, 30.0), (0.5, 2.0)):
        cell = {**grid[0], "tau_HA": a, "tau_D": d, "w_dead": w, "aba_s": 0.5 if grid is sel.GRID else grid[0]["aba_s"]}
        cfg = apply_cell(base, cell); cfg.decoder.lengths = lengths
        res, _ = score_cell(test, ev, cfg, ("ctrl_Dfull",), strongs=strong)
        r = res["ctrl_Dfull"]
        print(f"  tau_HA={a:>4} tau_D={d:>4} w_dead={w:>3}  n_pred/n_gt={r['ratio']:.3f}  spell_f1={r['spell_f1']:.3f}", flush=True)


def check_sealed_guard(best_cell):
    log = ROOT / "runs" / "sealed_reads.jsonl"; log.parent.mkdir(parents=True, exist_ok=True)
    sha = git_sha()
    if log.exists():
        for line in log.read_text().splitlines():
            if not line.strip():
                continue
            e = json.loads(line)
            if e.get("cell") == best_cell and e.get("git") == sha:
                raise SystemExit(f"refusing sealed read: cell already read at git {sha} ({e.get('time')})")
    return log


def check_bar_gate(best_json, force_reason):
    """Sealed reads need spec_bar.bar_pass in best.json, or an explicit --force-sealed REASON (logged)."""
    bar_pass = bool(json.loads(Path(best_json).read_text()).get("spec_bar", {}).get("bar_pass", False)) \
        if Path(best_json).exists() else False
    if not bar_pass and not force_reason:
        raise SystemExit(f"refusing sealed read: bar_pass is false in {best_json}; pass --force-sealed REASON to override")
    return bar_pass


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sim", default=str(ROOT / "runs" / "sim"))
    ap.add_argument("--seeds", type=int, nargs="+", default=[100, 101])
    ap.add_argument("--out", default=str(ROOT / "runs" / "select" / time.strftime("%Y%m%d-%H%M")))
    ap.add_argument("--grid", choices=sorted(sel.GRIDS), default="grid1",
                    help="grid1 (default) | grid2; the real-match plausibility step runs for every grid unless --no-real")
    ap.add_argument("--data", default=os.environ.get("MTAH_DATA_DIR"),
                    help="real-match data dir for the plausibility step (default $MTAH_DATA_DIR)")
    ap.add_argument("--no-real", action="store_true", help="skip the real-match plausibility step (choose mean-rank 1)")
    ap.add_argument("--best-from", default=None, help="selection dir whose best.json is reused (skips the grid loop)")
    ap.add_argument("--pilot", action="store_true", help="first fold only: print L_k and the 8 corner cells; writes nothing")
    ap.add_argument("--sealed", action="store_true")
    ap.add_argument("--force-sealed", default=None, metavar="REASON", help="allow the sealed read although bar_pass is false")
    args = ap.parse_args()
    grid = sel.GRIDS[args.grid]
    keys = [k for k in list_sim_keys(args.sim) if match_id(k) != SEALED_MATCH]
    if args.pilot:
        run_pilot(args, keys, base_for_grid(grid), grid)
        return
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    if args.best_from:
        best_cell = json.loads((Path(args.best_from) / "best.json").read_text())["cell"]
        print("reusing best cell from", args.best_from, best_cell)
    else:
        best_cell = run_selection(args.sim, args.seeds, out, grid=grid, base=base_for_grid(grid),
                                  real_data_dir=None if args.no_real else data_dir_or_exit(args.data), grid_name=args.grid)
    if args.sealed:
        reg_path = Path(args.best_from or out) / "registration.json"
        if not reg_path.exists():
            raise SystemExit(f"{reg_path} missing: cannot verify the registered seeds; refusing the sealed read")
        reg_seeds = json.loads(reg_path.read_text()).get("seeds")
        if reg_seeds != list(args.seeds):
            raise SystemExit(f"--seeds {list(args.seeds)} differ from the registered seeds {reg_seeds} in {reg_path}; "
                             "refusing the sealed read")
        bar_pass = check_bar_gate(Path(args.best_from or out) / "best.json", args.force_sealed)
        log = check_sealed_guard(best_cell)
        base = base_for_grid([best_cell])   # PathCRF only if the chosen cell weights it
        policy = best_cell.get("dead_policy", "full")
        train = load_sims(args.sim, args.seeds, keys); dead = train_dead(train, policy=policy)
        lengths = estimate_lengths(train, base, train_dead)
        cfg = apply_cell(base, best_cell); cfg.decoder.lengths = lengths
        ref_path = out / "drift_reference.json"
        if not ref_path.exists() and args.best_from and (Path(args.best_from) / "drift_reference.json").exists():
            ref_path.write_text((Path(args.best_from) / "drift_reference.json").read_text())
        if not ref_path.exists():
            print("WARNING: drift_reference.json absent; writing an IN-SAMPLE reference (train halves)")
            rows = [drift_report(predict(s.match, s.tags, cfg, dead, cache_key=f"{s.key}_s{s.seed}"), s.match, s.tags, blackout_mask_from_match(s.match))
                    for s in train]
            ref_path.write_text(json.dumps(_drift_summary(rows), indent=1))
        sealed = load_sims(args.sim, args.seeds, [k for k in list_sim_keys(args.sim) if match_id(k) == SEALED_MATCH])
        ev = [compute_evidence(s.match, s.tags, cfg, dead, cache_key=f"{s.key}_s{s.seed}") for s in sealed]
        res = {v: m["spell_f1"] for v, m in score_cell(sealed, ev, cfg, VARIANTS)[0].items()}
        rec = {"time": time.strftime("%Y-%m-%d %H:%M"), "git": git_sha(), "cell": best_cell, "sealed": res,
               "bar_pass": bar_pass, "force_reason": args.force_sealed}
        with open(log, "a") as f:
            f.write(json.dumps(rec) + "\n")
        dead.save(out / "dead_model.joblib"); (out / "config.json").write_text(json.dumps({**cfg.to_dict(), "lengths": lengths}, indent=1))
        print("SEALED", json.dumps(res))


if __name__ == "__main__":
    main()
