"""PathCRF team posterior vs truth per half (SkillCorner halves), first vs second halves: on frames whose ctrl_Dfull truth is H/A
and p_home finite, share where (p_home > 0.5) == (truth == H); also on frames with the ball within 3 m of the nearest player
(confident control) and the nearest-player proxy. Aggregates only."""
import sys, json
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT))
import numpy as np, pandas as pd
from mtah.sim.dataset import list_sim_keys, load_half_sim
rows = []
for build, sim, cache in (("nat", ROOT / "runs/kh/sim", ROOT / "runs/kh/cache_nat"), ("sim", ROOT / "runs/sim_extra2", ROOT / "runs/kh/cache_sim")):
    for k in list_sim_keys(sim):
        h = load_half_sim(sim, k, 100); m = h.match
        p = np.load(cache / f"{k}_s100.npz")["p_home"].astype(float)
        t = h.truth["ctrl_Dfull"]; ok = np.isin(t, ("H", "A")) & np.isfinite(p)
        agree = float(np.mean((p[ok] > 0.5) == (t[ok] == "H")))
        d = np.hypot(m.pos[..., 0] - m.ball[:, None, 0], m.pos[..., 1] - m.ball[:, None, 1]); d = np.where(np.isfinite(d), d, np.inf)
        j = d.argmin(1); dm = d[np.arange(len(j)), j]
        c = ok & (dm <= 3.0)
        conf = float(np.mean((p[c] > 0.5) == (t[c] == "H"))) if c.sum() else np.nan
        rows.append({"build": build, "key": k, "half": int(k[-1]), "agree_HA": agree, "agree_ball_within3m": conf, "n_conf": int(c.sum())})
df = pd.DataFrame(rows); df.to_csv(ROOT / "runs/kh/pathcrf_agree.csv", index=False)
g = df.groupby(["build", "half"])[["agree_HA", "agree_ball_within3m"]].agg(["mean", "std"]).round(3)
print(g.to_string())
paired = df.pivot_table(index=["build", df.key.str[:9]], columns="half", values="agree_ball_within3m").dropna()
print("H1-H2 paired (within-3m agreement):", paired.assign(d=paired[1] - paired[2]).groupby("build").d.agg(["mean", "std", "count"]).round(3).to_string())
