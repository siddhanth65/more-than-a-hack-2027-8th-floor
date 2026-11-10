"""Label-free checks of a prediction on the real match against DFL-sim reference values."""
import numpy as np

from mtah.io.match import half_runs


def _segments(x, s0, e0):
    out = []; s = s0
    for i in range(s0 + 1, e0 + 1):
        if i == e0 or x[i] != x[s]:
            out.append((s, i, x[s])); s = i
    return out


def blackout_mask_from_match(match, long_s=10.0):
    """(F,) bool: runs of ~ball_present & graded frames longer than long_s seconds."""
    from mtah.sim.blackout_sim import runs_of
    mask = np.zeros(match.n_frames, bool)
    for s, e in runs_of(~match.ball_present & match.graded):
        if e - s > long_s * match.fps:
            mask[s:e] = True
    return mask


def drift_report(labels, match, tags, blackout_mask, code_table=None):
    table_given = code_table is not None
    if code_table is None:
        from mtah.evidence.tag_prior import load_tag_codes
        code_table = load_tag_codes()
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
    agree = []; agree_imp = []
    for r in tags.itertuples(index=False):
        if r.code != "SET PIECES" and r.team in ("H", "A"):
            a = max(0, int(r.start)); b = min(len(labels) - 1, int(r.end))
            sel = labels[a:b + 1]; alive = sel != "D"
            if alive.any():
                agree.append((sel[alive] == r.team).mean())
                entry = code_table.get(r.code)
                if entry is None:
                    if table_given:
                        continue
                    sign = 1   # default table: unknown codes count as sign +1 (as in tag_prior_evidence)
                else:
                    sign = int(entry["sign"])
                if sign == 0:
                    continue
                exp = r.team if sign > 0 else ("A" if r.team == "H" else "H")
                agree_imp.append((sel[alive] == exp).mean())
    inb = g & blackout_mask; outb = g & ~blackout_mask
    return {
        "n_spells": len(spells),
        "spells_per_HA_min": len(spells) / max(1e-9, ha_frames / fps / 60),
        # spells are segmented inside the graded halves only, so every counted spell lies on graded frames; the real
        # spell-rate band (select.REAL_SPELL_RATE_BAND = [2.28, 4.10], other-game ctrl truth rates, applied to R / rho) is in this unit
        "spells_per_graded_min": len(spells) / max(1e-9, int(g.sum()) / fps / 60),
        "pred_D_share": float((labels[g] == "D").mean()),
        "spell_len_median_s": float(np.median([(e - s) / fps for s, e in spells])) if spells else 0.0,
        "share_changes_in_blackout": float(np.mean([blackout_mask[s] for s in changes])) if changes else float("nan"),
        "D_starts_near_sp_share": float(np.mean(near)) if near else float("nan"),
        "tag_agreement": float(np.mean(agree)) if agree else float("nan"),
        "tag_agreement_implied": float(np.mean(agree_imp)) if agree_imp else float("nan"),
        "non_blackout_D_share": float((labels[outb] == "D").mean()) if outb.any() else float("nan"),
        "blackout_D_share": float((labels[inb] == "D").mean()) if inb.any() else float("nan"),
    }
