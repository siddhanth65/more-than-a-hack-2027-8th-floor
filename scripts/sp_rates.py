"""Set-piece tag rates per graded minute by type: simulated tags vs the hard-coded real reference.

    python scripts/sp_rates.py --sim runs/sim --seed 100

Reads only the sim tag tables (*.tags.parquet) and the frame count of each half (the `half` array of the .npz; sims
are fully graded). The sealed match is never read. Prints aggregates only.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from mtah.sim.dataset import SEALED_MATCH, list_sim_keys, match_id  # noqa: E402
from mtah.sim.tag_sim import REAL_SP_RATES, REAL_SP_TOTAL, SP_TYPES, sp_rate_by_type  # noqa: E402


def sim_rates(sim_dir, seed):
    """(rates dict, graded minutes, n halves) pooled over all non-sealed halves of one seed."""
    tabs, frames, n = [], 0, 0
    for key in list_sim_keys(sim_dir):
        if match_id(key) == SEALED_MATCH:
            continue
        stem = Path(sim_dir) / f"{key}_s{seed}"
        if not Path(str(stem) + ".tags.parquet").is_file():
            continue
        meta = json.loads(Path(str(stem) + ".json").read_text())
        with np.load(str(stem) + ".npz", allow_pickle=False) as z:
            frames_h = len(z["half"])
        frames += frames_h
        tabs.append(pd.read_parquet(str(stem) + ".tags.parquet", columns=["code", "type"]))
        n += 1
        fps = float(meta["fps"])
    if not n:
        raise SystemExit(f"no sim halves for seed {seed} in {sim_dir}")
    minutes = frames / fps / 60.0
    return sp_rate_by_type(pd.concat(tabs, ignore_index=True), minutes), minutes, n


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--sim", default="runs/sim")
    ap.add_argument("--seed", type=int, default=100)
    a = ap.parse_args()
    rates, minutes, n = sim_rates(a.sim, a.seed)
    print(f"sim {a.sim} seed {a.seed}: {n} halves, {minutes:.1f} graded min (sealed match {SEALED_MATCH} excluded)")
    print(f"{'type':<10} {'sim/min':>8} {'real/min':>9} {'ratio':>6}")
    rows = [(t, rates[t], REAL_SP_RATES[t]) for t in SP_TYPES]
    rows += [(t, v, float("nan")) for t, v in rates.items() if t not in SP_TYPES and t != "total"]
    rows.append(("total", rates["total"], REAL_SP_TOTAL))
    for t, s, r in rows:
        ratio = s / r if np.isfinite(r) and r > 0 else float("nan")
        print(f"{t:<10} {s:>8.3f} {r:>9.3f} {ratio:>6.2f}")


if __name__ == "__main__":
    main()
