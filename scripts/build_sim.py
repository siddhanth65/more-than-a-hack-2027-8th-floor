"""Build the simulated DFL dataset for all halves and seeds; print aggregates only."""
import os
import argparse, json
from pathlib import Path

import numpy as np

from mtah.io.dfl_proxy import list_halves
from mtah.io.tags import coverage
from mtah.sim.blackout_sim import BlackoutParams
from mtah.sim.corrupt import CorruptParams
from mtah.sim.dataset import VARIANTS, build_half, match_id, save_half
from mtah.sim.tag_sim import TagSimParams

ROOT = Path(__file__).resolve().parents[1]


def load_or_write(path, cls):
    if path.exists():
        return cls.from_dict(json.loads(path.read_text()))
    p = cls(); path.write_text(json.dumps(p.to_dict(), indent=1)); return p


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dfl", default=os.environ.get("MTAH_DFL_DIR", os.path.join(os.path.expanduser("~"), "dfl-open", "work", "fp1d")))
    ap.add_argument("--extra-dir", default=None, help="directory with extra game_*.npz halves + its own fp1d_census.json")
    ap.add_argument("--only", type=int, nargs="+", default=None, help="build only these match ids (e.g. 9101 9201)")
    ap.add_argument("--keys", nargs="+", default=None, help="build only these half keys (e.g. game_9101_H1)")
    ap.add_argument("--no-dfl", action="store_true", help="skip the --dfl directory (extra-dir halves only)")
    ap.add_argument("--seeds", type=int, nargs="+", default=[100, 101, 102, 103, 104])
    ap.add_argument("--out", default=str(ROOT / "runs" / "sim"))
    args = ap.parse_args()
    dfl = Path(args.dfl)
    bp = BlackoutParams.from_dict(json.loads((ROOT / "configs" / "blackout_sim.json").read_text()))
    cp = load_or_write(ROOT / "configs" / "corrupt.json", CorruptParams)
    tp = load_or_write(ROOT / "configs" / "tag_sim.json", TagSimParams)
    agg = {"masked": [], "cov_in": [], "cov_out": [], **{f"D_{v}": [] for v in VARIANTS}, "dnarrow_changed_share": []}
    sources = [] if args.no_dfl else [(key, p, dfl / "fp1d_census.json") for key, p in list_halves(dfl)]
    if args.extra_dir:
        ex = Path(args.extra_dir)
        sources += [(key, p, ex / "fp1d_census.json") for key, p in list_halves(ex)]
    if args.only is not None:
        sources = [t for t in sources if match_id(t[0]) in args.only]
    if args.keys is not None:
        sources = [t for t in sources if t[0] in args.keys]
    for key, p, census in sources:
        for seed in args.seeds:
            sim = build_half(key, p, census, seed, bp, cp, tp)
            save_half(sim, args.out)
            cov = coverage(sim.tags, sim.match.n_frames)
            agg["masked"].append(sim.blackout.mean())
            agg["cov_in"].append(cov[sim.blackout].mean() if sim.blackout.any() else np.nan)
            agg["cov_out"].append(cov[~sim.blackout].mean())
            for v in VARIANTS:
                agg[f"D_{v}"].append((sim.truth[v] == "D").mean())
            full = sim.truth["raw_Dfull"] == "D"
            agg["dnarrow_changed_share"].append(
                (sim.truth["raw_Dfull"][full] != sim.truth["raw_Dnarrow"][full]).mean() if full.any() else np.nan)
        print(key, "done", flush=True)
    print(json.dumps({k: round(float(np.nanmean(v)), 3) for k, v in agg.items()}, indent=1))


if __name__ == "__main__":
    main()
