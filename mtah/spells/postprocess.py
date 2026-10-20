"""Metric-aware clean-up of a decoded label array: islands, D holes, A-B-A merges."""
from dataclasses import asdict, dataclass

import numpy as np

from mtah.io.match import half_runs


@dataclass
class PostParams:
    island_s: float = 0.4
    d_fill_s: float = 0.0
    aba_s: float = 0.5

    def to_dict(self):
        return asdict(self)

    @classmethod
    def from_dict(cls, d):
        return cls(**d)


def _segments(x):
    out = []; s = 0
    for i in range(1, len(x) + 1):
        if i == len(x) or x[i] != x[s]:
            out.append([s, i, x[s]]); s = i
    return out


def _one_half(x, fps, p, strong):
    x = x.copy()
    island = int(p.island_s * fps); fill = int(p.d_fill_s * fps); aba = int(p.aba_s * fps)
    changed = True
    while changed:
        changed = False
        segs = _segments(x)
        for i, (s, e, lab) in enumerate(segs):
            prev = segs[i - 1][2] if i > 0 else None; nxt = segs[i + 1][2] if i + 1 < len(segs) else None
            n = e - s
            if lab in ("H", "A") and n < island and prev == "D" and nxt == "D":
                x[s:e] = "D"; changed = True; break
            if lab == "D" and n < fill and prev is not None and prev == nxt and prev in ("H", "A"):
                x[s:e] = prev; changed = True; break
            if lab in ("H", "A") and n < aba and prev is not None and prev == nxt and prev in ("H", "A") and prev != lab:
                st = strong.get(lab)
                if st is None or not st[s:e].any():
                    x[s:e] = prev; changed = True; break
    return x


def postprocess(labels, half, fps, params, strong_H=None, strong_A=None):
    out = np.asarray(labels).astype("<U1").copy()
    strong = {"H": strong_H, "A": strong_A}
    for _, s, e in half_runs(half):
        sub = {k: (v[s:e] if v is not None else None) for k, v in strong.items()}
        out[s:e] = _one_half(out[s:e], fps, params, sub)
    return out
