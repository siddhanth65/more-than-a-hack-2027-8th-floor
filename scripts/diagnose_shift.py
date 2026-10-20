"""Aggregate-only diagnosis of real-vs-sim shift for the selected dead model."""
import json, sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from mtah.evidence.dead import DEAD_FEATURES, DeadModel, dead_features
from mtah.io.metrica_loader import load_tactical
from mtah.io.tags import load_tags
from mtah.sim.dataset import SEALED_MATCH, list_sim_keys, load_half_sim, match_id
from mtah.sim.blackout_sim import runs_of
from mtah.eval.drift import blackout_mask_from_match

ROOT = Path(__file__).resolve().parents[1]
DATA = r"HOME/mtah-data"
model = DeadModel.load(ROOT / "runs/select/a1/dead_model.joblib")

def q(x):
    x = x[np.isfinite(x)]
    if not len(x): return dict(mean=None, p10=None, p50=None, p90=None)
    p = np.percentile(x, [10, 50, 90])
    return dict(mean=float(x.mean()), p10=float(p[0]), p50=float(p[1]), p90=float(p[2]))

def stats(X, sel):
    out = {}
    for i, n in enumerate(DEAD_FEATURES):
        c = X[sel, i]
        d = q(c.astype(float)); d["nan_share"] = float(np.isnan(c).mean()) if len(c) else None
        out[n] = d
    return out

def pd_stats(p):
    if not len(p): return None
    a = np.percentile(p, [10, 50, 90])
    return dict(p10=float(a[0]), p50=float(a[1]), p90=float(a[2]), share_gt_half=float((p > 0.5).mean()), mean=float(p.mean()))

def collect(m, tags, bmask, gmask):
    fps = m.fps
    X = dead_features(m, tags)
    p = model.predict_proba(X)
    runs = [(s, e) for s, e in runs_of(bmask)]
    sp = np.array([int(r.start) for r in tags.itertuples(index=False) if r.code == "SET PIECES"])
    spd = X[bmask, 14]
    return dict(X=X, p=p, bm=bmask, nb=gmask & ~bmask, nsp=len(sp), gmin=gmask.sum() / fps / 60,
                nblk=len(runs), blk_sp=sum(bool(((sp >= s) & (sp < e)).any()) for s, e in runs) if len(sp) else 0,
                entry5=sum(1 for s, e in runs if np.isfinite(X[s, 8]) and X[s, 8] <= 5),
                entry_n=sum(1 for s, e in runs if np.isfinite(X[s, 8])), spd=spd)

def pool(parts):
    X = np.concatenate([d["X"] for d in parts]); p = np.concatenate([d["p"] for d in parts])
    bm = np.concatenate([d["bm"] for d in parts]); nb = np.concatenate([d["nb"] for d in parts])
    gm = sum(d["gmin"] for d in parts); nsp = sum(d["nsp"] for d in parts)
    nblk = sum(d["nblk"] for d in parts)
    spd = np.concatenate([d["spd"] for d in parts])
    return dict(
        features_blackout=stats(X, bm), features_nonblackout=stats(X, nb),
        pD_blackout=pd_stats(p[bm]), pD_nonblackout=pd_stats(p[nb]),
        sp_tags_per_graded_min=nsp / gm if gm else None, n_blackouts=nblk,
        share_blackouts_with_sp_start=sum(d["blk_sp"] for d in parts) / max(1, nblk),
        sp_dist_s_blackout=q(spd.astype(float)),
        share_entry_line_dist_le5=sum(d["entry5"] for d in parts) / max(1, sum(d["entry_n"] for d in parts)),
        blackout_frames=int(bm.sum()), nonblackout_frames=int(nb.sum()),
        speed_ratio_mean_blackout=float(np.nanmean(X[bm, 10])) if bm.any() else None)

m = load_tactical(DATA); tags = load_tags(DATA)
bm = blackout_mask_from_match(m); gm = m.graded.astype(bool)
real = pool([collect(m, tags, bm & gm, gm)])
keys = [k for k in list_sim_keys(ROOT / "runs/sim") if match_id(k) != SEALED_MATCH][:6]
parts = []
for k in keys:
    s = load_half_sim(ROOT / "runs/sim", k, 100)
    g = s.match.graded.astype(bool)
    parts.append(collect(s.match, s.tags, s.blackout.astype(bool) & g, g))
sim = pool(parts)
flag = {}
for n in DEAD_FEATURES:
    r = real["features_blackout"][n]; s = sim["features_blackout"][n]
    if r["p50"] is None or s["p10"] is None: flag[n] = "no-data"; continue
    if r["p50"] < s["p10"] or r["p50"] > s["p90"]: flag[n] = True
res = dict(sim_keys=keys, real=real, sim=sim, flagged=[n for n, v in flag.items() if v])
out = ROOT / "runs/select/a1/shift.json"; out.write_text(json.dumps(res, indent=1))
f = lambda v: "  nan" if v is None else f"{v:7.2f}"
print("FLAGGED (real blackout p50 outside sim p10-p90)")
print(f"{'feature':20s} real[p10 p50 p90 nan] | sim[p10 p50 p90 nan]")
for n in DEAD_FEATURES:
    if n in res["flagged"]:
        r = real["features_blackout"][n]; s = sim["features_blackout"][n]
        print(f"{n:20s} {f(r['p10'])}{f(r['p50'])}{f(r['p90'])} {r['nan_share']:.2f} | {f(s['p10'])}{f(s['p50'])}{f(s['p90'])} {s['nan_share']:.2f}")
print("P(D)"); 
for k in ("pD_blackout", "pD_nonblackout"):
    print(k, "real", {a: round(b, 3) for a, b in real[k].items()}, "sim", {a: round(b, 3) for a, b in sim[k].items()})
for k in ("sp_tags_per_graded_min", "share_blackouts_with_sp_start", "share_entry_line_dist_le5", "speed_ratio_mean_blackout", "n_blackouts"):
    print(k, "real", real[k], "sim", sim[k])
print("sp_dist_s blackout real", real["sp_dist_s_blackout"], "sim", sim["sp_dist_s_blackout"])
