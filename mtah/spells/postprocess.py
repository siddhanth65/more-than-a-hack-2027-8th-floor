"""Metric-aware clean-up of a decoded label array: islands, D holes, A-B-A merges."""
from dataclasses import asdict, dataclass

import numpy as np

from mtah.io.match import half_runs


@dataclass
class PostParams:
    island_s: float = 0.4
    d_fill_s: float = 0.0
    aba_s: float = 0.5
    snap_w: int = 50       # frames; 0 disables H<->A boundary snapping

    def to_dict(self):
        return asdict(self)

    @classmethod
    def from_dict(cls, d):
        return cls(**d)


def _one_half(x, fps, p, strong):
    """O(segments) clean-up on a run-length list.

    Rules are applied in the order fill -> island -> A-B-A, each as a full sweep
    (with the merged run re-examined), repeated until nothing changes. Filling D
    holes first makes the fixed point independent of scan direction: e.g.
    D H5 D3 H5 D with fill 5 / island 10 becomes one 13-frame H run instead of
    losing the first H as an island.
    """
    x = x.copy()
    if len(x) == 0:
        return x
    island = p.island_s * fps; fill = p.d_fill_s * fps; aba = p.aba_s * fps   # float thresholds: no rounding
    cuts = np.flatnonzero(x[1:] != x[:-1]) + 1
    starts = [0] + cuts.tolist(); stops = cuts.tolist() + [len(x)]
    segs = [[s, e, str(x[s])] for s, e in zip(starts, stops)]

    def fires(rule, i):
        s, e, lab = segs[i]
        prev = segs[i - 1][2] if i > 0 else None
        nxt = segs[i + 1][2] if i + 1 < len(segs) else None
        n = e - s
        if prev is None or nxt is None or prev != nxt:
            return None
        if rule == 0:
            return prev if lab == "D" and n < fill and prev in ("H", "A") else None
        if rule == 1:
            return "D" if lab in ("H", "A") and n < island and prev == "D" else None
        if lab in ("H", "A") and n < aba and prev in ("H", "A") and prev != lab:
            st = strong.get(lab)
            if st is None or not st[s:e].any():
                return prev
        return None

    any_change = True
    while any_change:
        any_change = False
        for rule in (0, 1, 2):
            i = 0
            while i < len(segs):
                new = fires(rule, i)
                if new is None:
                    i += 1
                    continue
                segs[i - 1:i + 2] = [[segs[i - 1][0], segs[i + 1][1], new]]
                any_change = True
                i = max(i - 1, 0)
    for s, e, lab in segs:
        x[s:e] = lab
    return x


def snap_team_boundaries(labels, half, touches, snap_w):
    """Move each H<->A boundary to the nearest touch (within +-snap_w, same half) by the new team.

    Left to right on already-updated labels; a move never crosses another boundary
    (both adjacent runs keep >= 1 frame); D boundaries are never moved. Ties go earlier.
    """
    out = np.asarray(labels).astype("<U1").copy()
    touches = np.asarray(touches)
    if snap_w <= 0 or len(touches) == 0:
        return out
    tf = touches[:, 0].astype(int)
    tt = touches[:, 2].astype(int)
    for _, hs, he in half_runs(half):
        x = out[hs:he]
        if len(x) == 0:
            continue
        cuts = np.flatnonzero(x[1:] != x[:-1]) + 1
        starts = [0] + cuts.tolist(); stops = cuts.tolist() + [len(x)]
        labs = [str(x[s]) for s in starts]
        for i in range(1, len(starts)):
            new, prev = labs[i], labs[i - 1]
            if new not in ("H", "A") or prev not in ("H", "A"):
                continue
            f = starts[i]                                   # local frame of the boundary
            code = 0 if new == "H" else 1
            m = (tt == code) & (tf >= hs + f - snap_w) & (tf <= hs + f + snap_w) & (tf >= hs) & (tf < he)
            if not m.any():
                continue
            c = tf[m] - hs
            fp = int(c[np.lexsort((c, np.abs(c - f)))[0]])
            fp = min(max(fp, starts[i - 1] + 1), stops[i] - 1)
            if fp == f:
                continue
            if fp < f:
                x[fp:f] = new
            else:
                x[f:fp] = prev
            starts[i] = fp; stops[i - 1] = fp
        out[hs:he] = x
    return out


def postprocess(labels, half, fps, params, strong_H=None, strong_A=None, touches=None):
    out = np.asarray(labels).astype("<U1").copy()
    strong = {"H": strong_H, "A": strong_A}
    for _, s, e in half_runs(half):
        sub = {k: (v[s:e] if v is not None else None) for k, v in strong.items()}
        out[s:e] = _one_half(out[s:e], fps, params, sub)
    if touches is not None and params.snap_w > 0:
        out = snap_team_boundaries(out, half, touches, params.snap_w)
    return out
