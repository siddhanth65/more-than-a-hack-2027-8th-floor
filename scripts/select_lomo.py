"""LOMO selection over GRID on the simulated dataset; writes fold tables, summary and registration."""
import argparse, hashlib, json, subprocess, sys, time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd

from mtah import baselines
from mtah.eval.drift import drift_report
from mtah.eval.scorer import OFFICIAL_PATH, score
from mtah.pipeline import PipelineConfig, compute_evidence, predict
from mtah.select import GRID, apply_cell, estimate_lengths, mean_rank, score_cell, train_dead
from mtah.sim.dataset import SEALED_MATCH, VARIANTS, list_sim_keys, load_half_sim, lomo_folds, match_id


def git_sha():
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT).decode().strip()
    except Exception:
        return "unknown"


def load_sims(sim_dir, seeds, keys):
    return [load_half_sim(sim_dir, k, s) for k in keys for s in seeds]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sim", default=str(ROOT / "runs" / "sim"))
    ap.add_argument("--seeds", type=int, nargs="+", default=[100, 101])
    ap.add_argument("--out", default=str(ROOT / "runs" / "select" / time.strftime("%Y%m%d-%H%M")))
    ap.add_argument("--sealed", action="store_true")
    args = ap.parse_args()
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    keys = [k for k in list_sim_keys(args.sim) if match_id(k) != SEALED_MATCH]
    base = PipelineConfig()
    fold_tables = []
    for held, train_keys, test_keys in lomo_folds(keys):
        train = load_sims(args.sim, args.seeds, train_keys); test = load_sims(args.sim, args.seeds, test_keys)
        dead = train_dead(train)
        lengths = estimate_lengths(train[: 4], base, dead)
        cfg0 = apply_cell(base, GRID[0]); cfg0.decoder.lengths = lengths
        test_ev = [compute_evidence(s.match, s.tags, cfg0, dead) for s in test]
        rows = []
        for i, cell in enumerate(GRID):
            cfg = apply_cell(base, cell); cfg.decoder.lengths = lengths
            row = {"cell": i, **cell}
            for v in VARIANTS:
                r = score_cell(test, test_ev, cfg, v); row[v] = r["spell_f1"]; row[f"{v}_ratio"] = r["ratio"]
            rows.append(row)
        for name, fn in {"smooth50": lambda m: baselines.smooth_vote(m, 50), "b4": baselines.b4_blackout_dead}.items():
            row = {"cell": name}
            for v in VARIANTS:
                row[v] = float(np.mean([score(s.truth[v], fn(s.match))["spell_f1"] for s in test]))
            rows.append(row)
        df = pd.DataFrame(rows); df["held"] = held; df["lengths"] = json.dumps(lengths)
        df.to_csv(out / f"fold_{held}.csv", index=False); fold_tables.append(df)
        grid_rows = df[~df["cell"].isin(["smooth50", "b4"])]
        print(f"fold {held} done; best ctrl_Dfull {grid_rows['ctrl_Dfull'].max():.3f}", flush=True)
    allf = pd.concat(fold_tables)
    cells = allf[~allf["cell"].isin(["smooth50", "b4"])].groupby("cell")[list(VARIANTS) + [f"{v}_ratio" for v in VARIANTS]].mean().reset_index()
    ranked = mean_rank(cells, list(VARIANTS)); ranked.to_csv(out / "summary.csv", index=False)
    best = ranked.iloc[0].to_dict(); best_cell = GRID[int(best["cell"])]
    base_rows = allf[allf["cell"].isin(["smooth50", "b4"])].groupby("cell")[list(VARIANTS)].mean()
    print(ranked.head(5).to_string()); print(base_rows.to_string())
    reg = {"git": git_sha(), "seeds": args.seeds, "grid_size": len(GRID), "selection": "mean rank over 4 variants, tie ctrl_Dfull",
           "scorer_md5": hashlib.md5(OFFICIAL_PATH.read_bytes().replace(b"\r\n", b"\n")).hexdigest(),
           "best_cell": best_cell, "best": {v: best[v] for v in VARIANTS}, "baselines": base_rows.to_dict()}
    (out / "registration.json").write_text(json.dumps(reg, indent=1))
    (out / "best.json").write_text(json.dumps({"cell": best_cell, "scores": {v: best[v] for v in VARIANTS},
                                               "ratios": {v: best[f"{v}_ratio"] for v in VARIANTS}}, indent=1))
    if args.sealed:
        train = load_sims(args.sim, args.seeds, keys); dead = train_dead(train)
        lengths = estimate_lengths(train[: 4], base, dead)
        cfg = apply_cell(base, best_cell); cfg.decoder.lengths = lengths
        ref_vals = {}
        for s in train:
            rep = drift_report(predict(s.match, s.tags, cfg, dead), s.match, s.tags, s.blackout)
            for k, v in rep.items():
                ref_vals.setdefault(k, []).append(v)
        ref = {k: {"min": float(np.nanmin(v)), "median": float(np.nanmedian(v)), "max": float(np.nanmax(v))}
               for k, v in ref_vals.items() if not np.all(np.isnan(v))}
        (out / "drift_reference.json").write_text(json.dumps(ref, indent=1))
        sealed = load_sims(args.sim, args.seeds, [k for k in list_sim_keys(args.sim) if match_id(k) == SEALED_MATCH])
        ev = [compute_evidence(s.match, s.tags, cfg, dead) for s in sealed]
        res = {v: score_cell(sealed, ev, cfg, v)["spell_f1"] for v in VARIANTS}
        rec = {"time": time.strftime("%Y-%m-%d %H:%M"), "git": git_sha(), "cell": best_cell, "sealed": res}
        with open(ROOT / "runs" / "sealed_reads.jsonl", "a") as f:
            f.write(json.dumps(rec) + "\n")
        dead.save(out / "dead_model.joblib"); (out / "config.json").write_text(json.dumps({**cfg.to_dict(), "lengths": lengths}, indent=1))
        print("SEALED", json.dumps(res))


if __name__ == "__main__":
    main()
