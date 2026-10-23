"""Small label-free sweep of the rel detector's gate / scale on the real match (alignment lift vs chance; aggregates only).
Selection criterion (fixed before the run): mean over {a5c, a10} boundaries of (share touch within +-0.5 s) / chance,
subject to a rate in [6, 14] touches per visible minute.   -> runs/touch/sweep.json"""
import itertools, json, os, sys
os.environ.setdefault("OMP_NUM_THREADS", "4")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
from touch_common import DATA, OUT, load_real
from touch_validate import *   # noqa: F401,F403
from touch_validate import CACHE, FPS, Index, boundary_stats, event_stats, ha_boundaries, tag_handovers, pathcrf_handovers, pathcrf_stats
from mtah.evidence.touches import TouchParams, detect_touches
from mtah.io.tags import load_tags

m = load_real(); tags = load_tags(DATA); F = m.n_frames
z = np.load(CACHE, allow_pickle=False)
vis_graded = m.ball_present & m.graded; vis_c = np.r_[0, np.cumsum(m.ball_present)]; vmin = vis_graded.sum() / FPS / 60
ho = tag_handovers(tags); b5 = ha_boundaries(z["lab_a5c"], m.half); b10 = ha_boundaries(z["lab_a10"], m.half)
rows = []
for k, g, r, zt in itertools.product((5, 8, 12), (0, 4, 8, 12), (1.5, 2.0), (2.0, 2.5, 3.0)):
    p = TouchParams(mode="rel", k_rel=k, gate_frames=g, r_rel_m=r, z_thr=zt, z_min=min(1.5, zt))
    t = detect_touches(m, p); idx = Index(t, F); rng = np.random.default_rng(0)
    s5 = boundary_stats(b5, t, idx, vis_c, F, rng, vis_graded); s10 = boundary_stats(b10, t, idx, vis_c, F, rng, vis_graded)
    sh = event_stats(ho, idx, vis_c, F, FPS, rng, vis_graded)
    lift = np.mean([s["share_touch_within_0.5s"] / s["chance_touch_within_0.5s"] for s in (s5, s10)])
    rows.append({"k": k, "gate": g, "r": r, "z": zt, "rate": len(t) / vmin, "lift_boundary_0.5s": float(lift),
                 "lift_tag_gain": sh["share_gain_team_touch"] / sh["chance_gain_team"]})
    print(rows[-1], flush=True)
(OUT / "sweep.json").write_text(json.dumps(rows, indent=1))
