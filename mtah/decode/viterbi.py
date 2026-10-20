"""Three-state (H, A, D) min-cost Viterbi over weighted, scale-corrected evidence."""
from dataclasses import dataclass, field

import numpy as np

from mtah.io.match import half_runs

LABELS = np.array(["H", "A", "D"], dtype="<U1")


@dataclass
class DecoderParams:
    weights: dict = field(default_factory=dict)
    lengths: dict = field(default_factory=dict)
    tau_HA: float = 10.0
    tau_HD: float = 10.0
    tau_DH: float = 10.0


def transition_matrix(p):
    T = np.zeros((3, 3))
    T[0, 1] = T[1, 0] = p.tau_HA
    T[0, 2] = T[1, 2] = p.tau_HD
    T[2, 0] = T[2, 1] = p.tau_DH
    return T


def cost_matrix(evidence, p):
    F = next(iter(evidence.values())).shape[0]
    C = np.zeros((F, 3), np.float64)
    for k, ev in evidence.items():
        w = p.weights.get(k, 1.0) / max(1.0, p.lengths.get(k, 1.0))
        C -= w * np.nan_to_num(ev.astype(np.float64), nan=0.0)
    return C


def viterbi_path(C, T):
    n = len(C)
    back = np.zeros((n, 3), np.int8)
    cur = C[0].copy()
    for t in range(1, n):
        tot = cur[:, None] + T          # tot[i, j]: from i to j
        bi = tot.argmin(0)
        back[t] = bi
        cur = tot[bi, np.arange(3)] + C[t]
    path = np.zeros(n, np.int8); path[-1] = int(cur.argmin())
    for t in range(n - 1, 0, -1):
        path[t - 1] = back[t, path[t]]
    return path


def decode(evidence, half, p):
    C = cost_matrix(evidence, p); T = transition_matrix(p)
    out = np.full(len(C), "D", dtype="<U1")
    for _, s, e in half_runs(half):
        out[s:e] = LABELS[viterbi_path(C[s:e], T)]
    return out
