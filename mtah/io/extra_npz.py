"""Shared writer + aggregates for extra (non-DFL) sources converted into the dfl_proxy npz schema.

The npz layout is the one `scripts/parse_dfl_match.py` writes and `mtah.io.dfl_proxy.load_half/load_events`
read: raw (N,14) player rows [frame, pid, ltr, shirt, role, xn, yn, -, -, 1, ...], ball_n (contiguous frame
numbers), ball_xy (centred metres on the 105 x 68 DFL pitch), ball_z, ball_status (1 alive / 0 dead),
ball_poss_ltr (ltr side 0/1 of the team in possession), truth (frame, ltr, shirt, cls, 1, 0), poss_only,
actor_pid. Extra keys (ignored by the loaders): ball_visible (F,) bool, interp (F,) bool, raw_detected (N,) bool.
Coordinates are rescaled from the source pitch to 105 x 68 so the loader's fixed DFL_PITCH holds.
"""
import json
from pathlib import Path

import numpy as np

from mtah.io.dfl_proxy import DFL_PITCH, SLOTS_PER_TEAM

FPS = 25.0
# event classes, same numbering as scripts/parse_dfl_match.py
DRIVE, PASS, CROSS, THROW_IN, SHOT, HEADER, TACKLE, BLOCK = range(1, 9)


def ltr_from_gk(gk_mean_xn):
    """{team: 0/1}: the team whose goalkeeper stands on the low-x side plays left-to-right (ltr 0)."""
    lo = min(gk_mean_xn, key=gk_mean_xn.get)
    return {t: (0 if t == lo else 1) for t in gk_mean_xn}


def player_rows(frames, xy_m, det, pid, ltr, shirt, role, pitch):
    """raw rows for one player: drops NaN frames; xy_m in centred source metres -> normalised 0..1."""
    ok = ~np.isnan(xy_m[:, 0])
    n = int(ok.sum())
    L, W = pitch
    r = np.full((n, 14), np.nan, np.float32)
    r[:, 0] = frames[ok]; r[:, 1] = pid; r[:, 2] = ltr; r[:, 3] = shirt; r[:, 4] = role
    r[:, 5] = (xy_m[ok, 0] + L / 2) / L
    r[:, 6] = (xy_m[ok, 1] + W / 2) / W
    r[:, 9] = 1.0
    return r, det[ok].astype(bool)


def to_dfl_metres(xy_m, pitch):
    L, W = pitch
    out = np.asarray(xy_m, np.float64).copy()
    out[..., 0] *= DFL_PITCH[0] / L
    out[..., 1] *= DFL_PITCH[1] / W
    return out


def write_half(out_dir, key, *, raw_parts, det_parts, ball_n, ball_xy, ball_z, status, poss_ltr, truth,
               poss_only, actor_pid, ball_visible, interp):
    raw = np.concatenate(raw_parts) if raw_parts else np.zeros((0, 14), np.float32)
    det = np.concatenate(det_parts) if det_parts else np.zeros(0, bool)
    for side in (0, 1):
        n = len(set(raw[raw[:, 2] == side, 1].astype(int).tolist()))
        if n > SLOTS_PER_TEAM:
            raise ValueError(f"{key}: {n} players on side {side} > {SLOTS_PER_TEAM}")
    if not np.isin(status, (0, 1)).all() or not np.isin(poss_ltr, (0, 1)).all():
        raise ValueError(f"{key}: status/poss outside {{0,1}}")
    np.savez_compressed(Path(out_dir) / f"{key}.npz", raw=raw, ball_n=np.asarray(ball_n, np.int64),
                        ball_xy=np.asarray(ball_xy, np.float32), ball_z=np.asarray(ball_z, np.float32),
                        ball_status=np.asarray(status, np.int8), ball_poss_ltr=np.asarray(poss_ltr, np.int8),
                        truth=np.asarray(truth, np.int64).reshape(-1, 6),
                        poss_only=np.asarray(poss_only, np.int64), actor_pid=np.asarray(actor_pid, np.int64),
                        ball_visible=np.asarray(ball_visible, bool), interp=np.asarray(interp, bool),
                        raw_detected=det)


def merge_census(out_dir, match_key, entry):
    """Add/replace one match in out_dir/fp1d_census.json (shared by every extra source in that dir)."""
    p = Path(out_dir) / "fp1d_census.json"
    c = json.loads(p.read_text()) if p.exists() else {}
    c[match_key] = entry
    p.write_text(json.dumps(c, indent=1))


def hda_labels(status, poss_ltr, ltr_home):
    return np.where(np.asarray(status) == 0, "D", np.where(np.asarray(poss_ltr) == ltr_home, "H", "A")).astype("<U1")


def spells(labels):
    """[(start, stop)] maximal H or A runs (the official possession spells; one half = one valid run)."""
    lab = np.asarray(labels)
    if lab.size == 0:
        return []
    cut = np.flatnonzero(lab[1:] != lab[:-1]) + 1
    starts = np.r_[0, cut]; stops = np.r_[cut, lab.size]
    return [(int(a), int(b)) for a, b in zip(starts, stops) if lab[a] in ("H", "A")]


def aggregate(halves, fps=FPS):
    """halves: list of dicts {labels (F,) U1, ball_visible (F,) bool}. Aggregates only."""
    F = sum(len(h["labels"]) for h in halves)
    if F == 0:
        return {"n_halves": 0}
    lab = np.concatenate([h["labels"] for h in halves])
    vis = np.concatenate([h["ball_visible"] for h in halves])
    lens = np.array([b - a for h in halves for a, b in spells(h["labels"])], float) / fps
    minutes = F / fps / 60.0
    q = np.percentile(lens, [10, 25, 50, 75, 90]) if lens.size else [np.nan] * 5
    return {
        "n_halves": len(halves),
        "minutes": round(minutes, 1),
        "ball_visible_share": round(float(vis.mean()), 3),
        "share_H": round(float((lab == "H").mean()), 3),
        "share_A": round(float((lab == "A").mean()), 3),
        "share_D": round(float((lab == "D").mean()), 3),
        "visible_dead_share": round(float((lab[vis] == "D").mean()), 3) if vis.any() else None,
        "spells_per_min": round(lens.size / minutes, 2),
        "spell_s_q10_25_50_75_90": [round(float(x), 1) for x in q],
        "spells_lt3s_share": round(float((lens < 3.0).mean()), 3) if lens.size else None,
    }
