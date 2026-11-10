"""PathCRF posterior for given sim-format halves (cache file per half), for running a share of the keys per process.

  MTAH_PATHCRF_PY=<venv python> MTAH_PATHCRF_REPO=<repo> python scripts/pathcrf_keys.py --sim runs/kh/sim \
      --cache runs/kh/cache/pathcrf --seed 100 --part 0 --of 8
Same code path as precompute_pathcrf (pathcrf_posterior, threads pinned to 4 in the cache key). Resumable (cached halves skipped).
Prints per-half seconds only.
"""
import argparse, sys, time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from mtah.evidence.pathcrf import PathCRFParams, pathcrf_posterior
from mtah.sim.dataset import SEALED_MATCH, list_sim_keys, load_half_sim, match_id

ap = argparse.ArgumentParser()
ap.add_argument("--sim", required=True); ap.add_argument("--cache", required=True)
ap.add_argument("--seed", type=int, default=100); ap.add_argument("--part", type=int, default=0)
ap.add_argument("--of", type=int, default=1); ap.add_argument("--keys", nargs="*", default=None)
a = ap.parse_args()
params = PathCRFParams(cache_dir=a.cache)
keys = a.keys or list_sim_keys(a.sim)
keys = [k for k in keys if match_id(k) != SEALED_MATCH][a.part::a.of]
for k in keys:
    t0 = time.time()
    s = load_half_sim(a.sim, k, a.seed)
    pathcrf_posterior(s.match, f"{k}_s{a.seed}", params)
    print(k, f"{time.time() - t0:.0f}s", flush=True)
print("DONE part", a.part, flush=True)
