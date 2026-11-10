"""Attacking-direction inference per half from the tracking alone (aggregates only).
Rules compared: (A) mean x of each team over the half (what run_infer.py uses); (B) goalkeeper rule: the slot with the
most extreme median x towards each end (>= 3 min visible) is that end's keeper; its team defends that end."""
import os, sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
from mtah.io.match import half_runs
from mtah.io.metrica_loader import load_tactical

def side_by_mean(pos, team):
    mx = [np.nanmean(pos[:, team == k, 0]) for k in (0, 1)]
    return (0 if mx[0] < mx[1] else 1), mx

def side_by_keepers(pos, team, fps=25.0, min_s=180):
    med = np.nanmedian(pos[:, :, 0], axis=0); n = np.isfinite(pos[:, :, 0]).sum(0)
    ok = n >= min_s * fps; med = np.where(ok, med, np.nan)
    lo, hi = int(np.nanargmin(med)), int(np.nanargmax(med))
    return int(team[lo]), int(team[hi]), float(med[lo]), float(med[hi])   # team at the left end, team at the right end

if __name__ == "__main__":
    m = load_tactical(os.environ["MTAH_DATA_DIR"])
    for h, a, b in half_runs(m.half):
        l, mx = side_by_mean(m.pos[a:b], m.team)
        kl, kr, ml, mr = side_by_keepers(m.pos[a:b], m.team)
        print(f"H{h}: mean-x rule: left team = {'home' if l == 0 else 'away'} (mean x {mx[0]:+.1f}/{mx[1]:+.1f}); keeper rule: left-end slot team {'home' if kl == 0 else 'away'} (median x {ml:+.1f}), right-end slot team {'home' if kr == 0 else 'away'} ({mr:+.1f})")
