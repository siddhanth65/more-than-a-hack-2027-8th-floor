"""Label-free aggregates of the real inputs. Never prints frame-level data."""
import json
import os
from pathlib import Path

import numpy as np

from mtah.io.match import distance_to_line
from mtah.io.tags import coverage
from mtah.sim.blackout_sim import runs_of


def real_match_census(match, tags, long_s=10.0):
    fps = match.fps; F = match.n_frames; g = match.graded
    present = match.ball_present
    runs = [(s, e) for s, e in runs_of(~present & g)]
    long = [(s, e) for s, e in runs if (e - s) / fps > long_s]
    inside = np.zeros(F, bool)
    for s, e in long:
        inside[s:e] = True
    lens = np.array([(e - s) / fps for s, e in long]) if long else np.array([0.0])
    near, outward = [], []
    for s, e in long:
        k = s - 1
        while k >= 0 and not present[k]:
            k -= 1
        if k < 0:
            continue
        d = distance_to_line(match.ball[k], match.pitch); near.append(d <= 5.0)
        j = k - 5
        if j >= 0 and present[j]:
            outward.append(distance_to_line(match.ball[k], match.pitch) < distance_to_line(match.ball[j], match.pitch))
    cov = coverage(tags, F)
    sp = tags[tags["code"] == "SET PIECES"]
    sp_in = sum(1 for s, e in long if ((sp["start"] >= s - 2 * fps) & (sp["start"] <= e + 2 * fps)).any())
    throw = sp[sp["type"].fillna("").str.lower().str.startswith("throw")]
    missing_total = int((~present & g).sum())
    return {
        "ball_present_graded": float(present[g].mean()),
        "n_blackouts": len(long),
        "blackouts_per_graded_min": float(len(long) / (int(g.sum()) / fps / 60.0)),
        "blackout_median_s": float(np.median(lens)),
        "blackout_max_s": float(lens.max()),
        "long_share": float(inside.sum() / max(1, missing_total)),
        "entry_near_line_share": float(np.mean(near)) if near else float("nan"),
        "entry_outward_share": float(np.mean(outward)) if outward else float("nan"),
        "tag_coverage_inside": float(cov[inside].mean()) if inside.any() else float("nan"),
        "tag_coverage_outside": float(cov[g & ~inside].mean()),
        "setpiece_in_blackout_share": float(sp_in / max(1, len(long))),
        "throwins_with_team": int(throw["team"].notna().sum()),
    }


def real_blackout_lengths(match, long_s=10.0):
    """Sorted lengths in seconds of the graded ball gaps longer than long_s (aggregates, no frame data)."""
    runs = runs_of(~match.ball_present & match.graded)
    return sorted((e - s) / match.fps for s, e in runs if (e - s) / match.fps > long_s)


def main():
    from mtah.io.metrica_loader import load_tactical
    from mtah.io.tags import load_tags
    data = Path(os.environ.get("MTAH_DATA_DIR", r"HOME/mtah-data"))
    tactical = load_tactical(data)
    c = real_match_census(tactical, load_tags(data))
    runs_dir = Path(__file__).resolve().parents[1] / "runs"
    runs_dir.mkdir(exist_ok=True)
    (runs_dir / "census.json").write_text(json.dumps(c, indent=1))
    (runs_dir / "real_blackout_lengths.json").write_text(json.dumps(real_blackout_lengths(tactical)))
    print(json.dumps(c, indent=1))


if __name__ == "__main__":
    main()
