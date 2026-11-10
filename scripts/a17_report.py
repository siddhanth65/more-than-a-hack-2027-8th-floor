"""Label-free comparison of a candidate against a reference on the real match: frames changed (overall / per half / inside
blackouts), tag agreement per half, H/A changes inside blackouts per half. Aggregates only; no ground truth.
  python scripts/a17_report.py NEW REF [NEW2 ...]  (names of runs/candidates/<name>.csv) -> prints JSON, writes runs/kh/a17/compare_<NEW>.json"""
import json, os, sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT))
import numpy as np, pandas as pd
from mtah.eval.drift import blackout_mask_from_match, drift_report
from mtah.io.metrica_loader import load_tactical
from mtah.io.tags import load_tags
D = os.environ.get("MTAH_DATA_DIR", os.path.join(os.path.expanduser("~"), "mtah-data"))
m = load_tactical(D); tags = load_tags(D); mask = blackout_mask_from_match(m); n = m.n_frames


def lab(name):
    s = pd.read_csv(ROOT / "runs" / "candidates" / f"{name}.csv"); x = np.full(n, "", object); x[s.frame.to_numpy()] = s.label.to_numpy(); return x


def ha_changes_in_blackout(x, h):
    sel = np.flatnonzero(m.half == h); a, b = sel[0], sel[-1] + 1; segs = []; s = a
    for i in range(a + 1, b + 1):
        if i == b or x[i] != x[s]:
            segs.append((s, x[s])); s = i
    return int(sum(1 for i in range(1, len(segs)) if segs[i][1] in "HA" and segs[i - 1][1] in "HA" and segs[i][1] != segs[i - 1][1] and mask[segs[i][0]]))


def stats(name):
    x = lab(name); out = {}
    d = drift_report(x, m, tags, mask)
    out["overall"] = {k: d[k] for k in ("non_blackout_D_share", "spells_per_graded_min", "tag_agreement_implied", "tag_agreement", "pred_D_share", "blackout_D_share", "n_spells")}
    for h in (1, 2):
        xh = np.where(m.half == h, x, "D"); dh = drift_report(xh, m, tags, mask)
        out[f"H{h}"] = {"tag_agreement_implied": dh["tag_agreement_implied"], "HA_changes_in_blackout": ha_changes_in_blackout(x, h)}
    return x, out


names = sys.argv[1:]; new = names[0]; xn, sn = stats(new); res = {new: sn}
for ref in names[1:]:
    xr, sr = stats(ref); g = m.graded; ch = (xn != xr)
    res[ref] = sr
    res[f"{new}_vs_{ref}"] = {"changed_graded": float(ch[g].mean()), "changed_H1": float(ch[m.half == 1].mean()), "changed_H2": float(ch[m.half == 2].mean()),
                              "changed_inside_blackouts": float(ch[mask & g].mean()), "changed_outside_blackouts": float(ch[~mask & g].mean())}
(ROOT / "runs/kh/a17").mkdir(parents=True, exist_ok=True); (ROOT / "runs/kh/a17" / f"compare_{new}.json").write_text(json.dumps(res, indent=1))
print(json.dumps(res, indent=1))
