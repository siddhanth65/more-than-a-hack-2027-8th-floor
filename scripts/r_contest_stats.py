"""Task R step 2 (label-free): per-type SET PIECES tag rates on the contest match (per graded minute) and the offset
between each tag start and the candidate's D->live edge (a5c, a14; window [-3, +12] s), pooled and per type. Aggregates only.
These two things set how the SkillCorner synthetic tags are thinned and shifted. No contest ground truth is read."""
import json, os, sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT))
import numpy as np, pandas as pd
from mtah.io.metrica_loader import load_tactical
from mtah.io.tags import load_tags
D = Path(os.environ.get("MTAH_DATA_DIR", os.path.join(os.path.expanduser("~"), "mtah-data"))); FPS = 25
TAG2TYPE = {"Throw-in": "throw_in", "Free Kick": "free_kick", "Goal Kick": "goal_kick", "Corner Kick": "corner", "Kick Off (After Goal)": "goal"}
t = load_tactical(D); tags = load_tags(D); n = t.n_frames
sp = tags[tags.code == "SET PIECES"].copy(); sp["ty"] = sp["type"].map(TAG2TYPE)
gm = t.graded.sum() / FPS / 60
sp = sp[[bool(t.graded[min(max(int(s), 0), n - 1)]) for s in sp.start]]
rates = {k: float((sp.ty == k).sum() / gm) for k in TAG2TYPE.values()}
out = {"graded_min": float(gm), "tag_counts": {k: int((sp.ty == k).sum()) for k in TAG2TYPE.values()},
       "tag_rate_per_graded_min": rates, "other_types": sp[sp.ty.isna()].type.value_counts().to_dict(), "edge_offset_s": {}}
for cand in ("a5c", "a14"):
    s = pd.read_csv(ROOT / "runs" / "candidates" / f"{cand}.csv")
    lab = np.full(n, "", object); lab[s.frame.to_numpy()] = s.label.to_numpy(); isD = lab == "D"
    offs = {k: [] for k in TAG2TYPE.values()}
    for r in sp.itertuples():
        if r.ty is None or pd.isna(r.ty):
            continue
        a, b = max(0, int(r.start) - 3 * FPS), min(n - 2, int(r.start) + 12 * FPS)
        e = [i for i in range(a, b) if isD[i] and not isD[i + 1]]
        if e:
            i = min(e, key=lambda i: abs(i + 1 - r.start)); offs[r.ty].append((i + 1 - r.start) / FPS)
    allo = sum(offs.values(), [])
    q = lambda x: {"n": len(x), **({} if not x else {"p10": float(np.percentile(x, 10)), "p25": float(np.percentile(x, 25)), "median": float(np.median(x)), "p75": float(np.percentile(x, 75)), "p90": float(np.percentile(x, 90))})}
    out["edge_offset_s"][cand] = {"pooled": q(allo), **{k: q(v) for k, v in offs.items()}}
(ROOT / "runs" / "kh" / "R" / "contest_tag_stats.json").write_text(json.dumps(out, indent=1))
print(json.dumps({k: out[k] for k in ("tag_counts", "tag_rate_per_graded_min")}), json.dumps(out["edge_offset_s"]["a5c"]["pooled"]), json.dumps(out["edge_offset_s"]["a14"]["pooled"]))
