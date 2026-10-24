"""Refit a registered selection's final models (dead / team / merge, lengths) in the CURRENT environment, reusing its
best.json cell: the final-fit branch of select_lomo.run_selection without the grid loop. Used when the registered
.joblib files were pickled by another scikit-learn version and no longer load (a13: 1.8.0 vs the pinned 1.9.1).
best.json, registration.json and drift_reference.json are copied unchanged; the source dir is never written.
    python scripts/refit_selection.py --src runs/select/a13 --out runs/select/a13r --grid grid8"""
import argparse, json, os, shutil, sys
from pathlib import Path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import select_lomo as sl
import mtah.select as sel
from mtah.sim.dataset import SEALED_MATCH, list_sim_keys, match_id

ap = argparse.ArgumentParser()
ap.add_argument("--src", required=True); ap.add_argument("--out", required=True)
ap.add_argument("--grid", choices=sorted(sel.GRIDS), required=True)
ap.add_argument("--sim", default=str(sl.ROOT / "runs" / "sim"))
ap.add_argument("--seeds", type=int, nargs="+", default=[100, 101])
args = ap.parse_args()
src, out = Path(args.src), Path(args.out)
if out.resolve() == src.resolve():
    raise SystemExit("--out must differ from --src")
out.mkdir(parents=True, exist_ok=True)
grid = sel.GRIDS[args.grid]; base = sl.base_for_grid(grid)
best_cell = sl.load_best(src, args.seeds)["cell"]
print("refitting cell", best_cell)
keys = [k for k in list_sim_keys(args.sim) if match_id(k) != SEALED_MATCH]
cbase = sl.base_for_grid([best_cell]); need_merge = sl.uses_merge(best_cell)
if need_merge:
    cbase.use_teamctrl = bool(base.use_teamctrl); cbase.use_pathcrf = bool(base.use_pathcrf)
if cbase.use_switch:
    raise SystemExit("switch cells not supported here")
train = sl.load_sims(args.sim, args.seeds, keys)
lengths = sl.estimate_lengths(train, cbase, sl.train_dead)
dead = sl.train_dead(train, policy=best_cell.get("dead_policy", "full"))
team = sl.train_team_ctrl(train, policy=best_cell.get("team_policy", "full"), touch_params=cbase.touch) if cbase.use_teamctrl else None
merge = sl.train_merge(train, base, grid, lengths) if need_merge else None
sl.write_candidate_artifacts(out, cbase, best_cell, lengths, dead, team, merge, None)
for name in ("best.json", "registration.json", "drift_reference.json"):
    if (src / name).exists():
        shutil.copyfile(src / name, out / name)
old = json.loads((src / "config.json").read_text()); new = json.loads((out / "config.json").read_text())
print("config.json identical to source:", old == new)
if old != new:
    print("differing keys:", sorted(k for k in set(old) | set(new) if old.get(k) != new.get(k)))
