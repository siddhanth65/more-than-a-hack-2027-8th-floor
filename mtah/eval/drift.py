"""Label-free checks of a prediction on the real match against DFL-sim reference values."""
import numpy as np

from mtah.io.match import half_runs


def _segments(x, s0, e0):
    out = []; s = s0
    for i in range(s0 + 1, e0 + 1):
        if i == e0 or x[i] != x[s]:
            out.append((s, i, x[s])); s = i
    return out


def drift_report(labels, match, tags, blackout_mask):
    fps = match.fps; g = match.graded
    segs = []
    for _, s, e in half_runs(match.half):
        segs.extend(_segments(labels, s, e))
    spells = [(s, e) for s, e, l in segs if l in ("H", "A")]
    ha_frames = sum(e - s for s, e in spells)
    changes = [s for i, (s, e, l) in enumerate(segs) if i > 0 and l in ("H", "A") and segs[i - 1][2] in ("H", "A") and segs[i - 1][2] != l]
    d_starts = [s for s, e, l in segs if l == "D"]
    sp_starts = np.sort(np.array([int(r.start) for r in tags.itertuples(index=False) if r.code == "SET PIECES"]))
    near = [bool(sp_starts.size and np.min(np.abs(sp_starts - s)) <= 2 * fps) for s in d_starts]
    agree = []
    for r in tags.itertuples(index=False):
        if r.code != "SET PIECES" and r.team in ("H", "A"):
            a = max(0, int(r.start)); b = min(len(labels) - 1, int(r.end))
            sel = labels[a:b + 1]; alive = sel != "D"
            if alive.any():
                agree.append((sel[alive] == r.team).mean())
    return {
        "n_spells": len(spells),
        "spells_per_HA_min": len(spells) / max(1e-9, ha_frames / fps / 60),
        "pred_D_share": float((labels[g] == "D").mean()),
        "spell_len_median_s": float(np.median([(e - s) / fps for s, e in spells])) if spells else 0.0,
        "share_changes_in_blackout": float(np.mean([blackout_mask[s] for s in changes])) if changes else float("nan"),
        "D_starts_near_sp_share": float(np.mean(near)) if near else float("nan"),
        "tag_agreement": float(np.mean(agree)) if agree else float("nan"),
    }
