"""Team-label coherence of tracking slots per half: share of slot-frames closer to the leave-one-out centroid of the
assigned team than to the other team's centroid, per slot class (aggregates only)."""
import os, sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
from mtah.io.match import half_runs
from mtah.io.metrica_loader import load_tactical
m = load_tactical(os.environ["MTAH_DATA_DIR"]); team = m.team
for h, a, b in half_runs(m.half):
    pos = m.pos[a:b:5]                       # 5 fps
    vis = np.isfinite(pos[:, :, 0]); P = np.nan_to_num(pos)
    cnt = [(vis & (team == k)).sum(1) for k in (0, 1)]
    sm = [(P * (vis & (team == k))[..., None]).sum(1) for k in (0, 1)]
    score = np.full(pos.shape[1], np.nan); nvis = vis.sum(0)
    for s in range(pos.shape[1]):
        k = team[s]; v = vis[:, s]
        if v.sum() < 500: continue
        cown = (sm[k][v] - P[v, s]) / np.maximum(cnt[k][v] - 1, 1)[:, None]
        coth = sm[1 - k][v] / np.maximum(cnt[1 - k][v], 1)[:, None]
        score[s] = np.mean(np.linalg.norm(P[v, s] - cown, axis=1) < np.linalg.norm(P[v, s] - coth, axis=1))
    ok = np.isfinite(score)
    print(f"H{h}: slots with data {int(ok.sum())}; coherence (share closer to own centroid) per slot: min {np.nanmin(score):.2f} p10 {np.nanpercentile(score,10):.2f} median {np.nanmedian(score):.2f}; slots < 0.5: {int((score[ok] < 0.5).sum())} (home {int(((score < .5) & (team == 0)).sum())}, away {int(((score < .5) & (team == 1)).sum())}); frame-weighted mean {np.nansum(score * nvis) / nvis[ok].sum():.3f}")
