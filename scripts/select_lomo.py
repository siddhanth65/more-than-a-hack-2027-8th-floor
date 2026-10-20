"""LOMO selection over GRID on the simulated dataset; writes fold tables, summary and registration."""
import argparse, hashlib, itertools, json, subprocess, sys, time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd

from mtah import baselines
from mtah.eval.drift import blackout_mask_from_match, drift_report
from mtah.eval.scorer import OFFICIAL_PATH, score
from mtah.pipeline import PipelineConfig, compute_evidence, compute_strong, predict
import mtah.select as sel
from mtah.select import apply_cell, estimate_lengths, mean_rank, score_cell, train_dead
from mtah.sim.dataset import SEALED_MATCH, VARIANTS, list_sim_keys, load_half_sim, lomo_folds, match_id


def git_sha():
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT).decode().strip()
    except Exception:
        return "unknown"


def load_sims(sim_dir, seeds, keys):
    return [load_half_sim(sim_dir, k, s) for k in keys for s in seeds]


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
    for held, train_keys, test_keys in lomo_folds(keys):
        train = load_sims(sim_dir, seeds, train_keys); test = load_sims(sim_dir, seeds, test_keys)
        lengths = lengths_by_fold.get(held) or estimate_lengths(train, base, train_dead)
        dead = dead_by_fold.get(held) or train_dead(train)
        cfg = apply_cell(base, best_cell); cfg.decoder.lengths = lengths
        for s in test:
            ev = compute_evidence(s.match, s.tags, cfg, dead)
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


def run_grid(sim_dir, seeds, keys, base, out, grid):
    fold_tables, lengths_by_fold, dead_by_fold = [], {}, {}
    for held, train_keys, test_keys in lomo_folds(keys):
        fold_csv = out / f"fold_{held}.csv"; halves_csv = out / f"fold_{held}_halves.csv"
        if fold_csv.exists() and halves_csv.exists():
            df = pd.read_csv(fold_csv); fold_tables.append(df)
            lengths_by_fold[held] = json.loads(df["lengths"].iloc[0])
            print(f"fold {held} resumed", flush=True)
            continue
        train = load_sims(sim_dir, seeds, train_keys); test = load_sims(sim_dir, seeds, test_keys)
        lengths = estimate_lengths(train, base, train_dead)  # inner split: see estimate_lengths
        dead = train_dead(train)  # final fold dead model: all train halves
        lengths_by_fold[held] = lengths; dead_by_fold[held] = dead
        cfg0 = apply_cell(base, grid[0]); cfg0.decoder.lengths = lengths
        test_ev = [compute_evidence(s.match, s.tags, cfg0, dead) for s in test]  # grid-independent (tested)
        test_strong = [compute_strong(s.match, cfg0) for s in test]  # touches do not depend on the grid
        rows, halves = [], []
        for i, cell in enumerate(grid):
            cfg = apply_cell(base, cell); cfg.decoder.lengths = lengths
            res, hdf = score_cell(test, test_ev, cfg, VARIANTS, strongs=test_strong)
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
    best = ranked.iloc[0].to_dict(); best_i = int(best["cell"]); best_cell = grid[best_i]
    allf = allf.copy(); allf["cell"] = allf["cell"].astype(str)
    halves = pd.concat([pd.read_csv(out / f"fold_{h}_halves.csv") for h in sorted(allf["held"].unique())])
    halves["cell"] = halves["cell"].astype(str)
    bar, base_rows = compute_bar(allf, halves, best, str(best_i))
    print(ranked.head(5).to_string()); print(base_rows.to_string())
    print("SPEC BAR", json.dumps(bar, indent=1))
    ref = out_of_sample_drift_reference(sim_dir, seeds, keys, base, best_cell, lengths_by_fold, dead_by_fold)
    (out / "drift_reference.json").write_text(json.dumps(ref, indent=1))
    reg = {"git": git_sha(), "seeds": seeds, "grid_size": len(grid), "selection": "mean rank over 4 variants, tie ctrl_Dfull",
           "scorer_md5": hashlib.md5(OFFICIAL_PATH.read_bytes().replace(b"\r\n", b"\n")).hexdigest(),
           "best_cell": best_cell, "best": {v: best[v] for v in VARIANTS}, "baselines": base_rows.to_dict(),
           "seed_sd": bar["seed_sd"], "baseline_used": bar["baseline_used"], "spec_bar": bar}
    (out / "registration.json").write_text(json.dumps(reg, indent=1))
    (out / "best.json").write_text(json.dumps({"cell": best_cell, "scores": {v: best[v] for v in VARIANTS},
                                               "ratios": bar["count_ratios"], "seed_sd": bar["seed_sd"],
                                               "baseline_used": bar["baseline_used"], "spec_bar": bar}, indent=1))
    return best_cell


def run_selection(sim_dir, seeds, out_dir, grid=None, base=None):
    """Importable LOMO selection loop: fold tables, summary, best/registration, out-of-sample drift reference."""
    grid = grid if grid is not None else sel.GRID
    out = Path(out_dir); out.mkdir(parents=True, exist_ok=True)
    keys = [k for k in list_sim_keys(sim_dir) if match_id(k) != SEALED_MATCH]
    return run_grid(sim_dir, list(seeds), keys, base or PipelineConfig(), out, grid)


def run_pilot(args, keys, base):
    """First fold only: print L_k and the eight tau/w_dead corner cells. Writes nothing."""
    held, train_keys, test_keys = lomo_folds(keys)[0]
    train = load_sims(args.sim, args.seeds, train_keys); test = load_sims(args.sim, args.seeds, test_keys)
    lengths = estimate_lengths(train, base, train_dead)
    print(f"PILOT fold {held} L_k:", json.dumps(lengths))
    dead = train_dead(train)
    cfg0 = apply_cell(base, sel.GRID[0]); cfg0.decoder.lengths = lengths
    ev = [compute_evidence(s.match, s.tags, cfg0, dead) for s in test]
    strong = [compute_strong(s.match, cfg0) for s in test]
    for a, d, w in itertools.product((3.0, 30.0), (3.0, 30.0), (0.5, 2.0)):
        cell = {"tau_HA": a, "tau_D": d, "w_dead": w, "aba_s": 0.5}
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sim", default=str(ROOT / "runs" / "sim"))
    ap.add_argument("--seeds", type=int, nargs="+", default=[100, 101])
    ap.add_argument("--out", default=str(ROOT / "runs" / "select" / time.strftime("%Y%m%d-%H%M")))
    ap.add_argument("--best-from", default=None, help="selection dir whose best.json is reused (skips the grid loop)")
    ap.add_argument("--pilot", action="store_true", help="first fold only: print L_k and the 8 corner cells; writes nothing")
    ap.add_argument("--sealed", action="store_true")
    args = ap.parse_args()
    keys = [k for k in list_sim_keys(args.sim) if match_id(k) != SEALED_MATCH]
    base = PipelineConfig()
    if args.pilot:
        run_pilot(args, keys, base)
        return
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    if args.best_from:
        best_cell = json.loads((Path(args.best_from) / "best.json").read_text())["cell"]
        print("reusing best cell from", args.best_from, best_cell)
    else:
        best_cell = run_selection(args.sim, args.seeds, out)
    if args.sealed:
        log = check_sealed_guard(best_cell)
        train = load_sims(args.sim, args.seeds, keys); dead = train_dead(train)
        lengths = estimate_lengths(train, base, train_dead)
        cfg = apply_cell(base, best_cell); cfg.decoder.lengths = lengths
        ref_path = out / "drift_reference.json"
        if not ref_path.exists() and args.best_from and (Path(args.best_from) / "drift_reference.json").exists():
            ref_path.write_text((Path(args.best_from) / "drift_reference.json").read_text())
        if not ref_path.exists():
            print("WARNING: drift_reference.json absent; writing an IN-SAMPLE reference (train halves)")
            rows = [drift_report(predict(s.match, s.tags, cfg, dead), s.match, s.tags, blackout_mask_from_match(s.match))
                    for s in train]
            ref_path.write_text(json.dumps(_drift_summary(rows), indent=1))
        sealed = load_sims(args.sim, args.seeds, [k for k in list_sim_keys(args.sim) if match_id(k) == SEALED_MATCH])
        ev = [compute_evidence(s.match, s.tags, cfg, dead) for s in sealed]
        res = {v: m["spell_f1"] for v, m in score_cell(sealed, ev, cfg, VARIANTS)[0].items()}
        rec = {"time": time.strftime("%Y-%m-%d %H:%M"), "git": git_sha(), "cell": best_cell, "sealed": res}
        with open(log, "a") as f:
            f.write(json.dumps(rec) + "\n")
        dead.save(out / "dead_model.joblib"); (out / "config.json").write_text(json.dumps({**cfg.to_dict(), "lengths": lengths}, indent=1))
        print("SEALED", json.dumps(res))


if __name__ == "__main__":
    main()
