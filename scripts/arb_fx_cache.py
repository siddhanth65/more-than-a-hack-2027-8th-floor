"""Compute PathCRF posteriors with input smoothing (smooth_s) into runs/cache/pathcrf_fx and report accuracy on the
pseudo-label frames per half.   python scripts/arb_fx_cache.py real | sim KEY SEED"""
import json, os, sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "scripts"))
import numpy as np
import arb_common as ac
from mtah.evidence.pathcrf import PathCRFParams, pathcrf_posterior
rule = ac.RuleParams(**json.load(open(ROOT / "runs/arb/rule.json"))["rule"])
p = PathCRFParams(smooth_s=1.0, cache_dir=str(ROOT / "runs/cache/pathcrf_fx"))
if sys.argv[1] == "real":
    from mtah.io.metrica_loader import load_tactical
    m = load_tactical(os.environ["MTAH_DATA_DIR"]); key = "real_tactical"
else:
    from mtah.sim.dataset import load_half_sim
    m = load_half_sim(ROOT / "runs/sim", sys.argv[2], int(sys.argv[3])).match; key = f"{sys.argv[2]}_s{sys.argv[3]}"
r = pathcrf_posterior(m, key, p)
ph = r["p_home"]; old = np.load(ROOT / f"runs/cache/pathcrf/{key}.npz")["p_home"]; c = ac.confident_control(m, rule)
for h in (1, 2):
    i = np.flatnonzero((c >= 0) & (m.half == h) & np.isfinite(ph) & np.isfinite(old))
    if len(i): print(key, "half", h, "n", len(i), "acc new", round(float(((ph[i] > .5) == (c[i] == 0)).mean()), 3), "old", round(float(((old[i] > .5) == (c[i] == 0)).mean()), 3), flush=True)
