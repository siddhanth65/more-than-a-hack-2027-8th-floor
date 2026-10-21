"""Three-state (H, A, D) min-cost Viterbi over weighted, scale-corrected evidence.

Switch hazard (lever A2/B). The evidence dict may carry auxiliary keys (AUX_STREAMS = {"switch"}) that are not
evidence: `cost_matrix` skips them. The "switch" entry is the hazard h (column 0 of an (F, 3) array, NaN
elsewhere; `mtah.evidence.switch.switch_evidence`). The decoder itself never reads it: the caller
(`pipeline.predict`) converts it to a per-frame H<->A switch cost tau_HA(t) = max(tau_floor, tau_HA -
w_switch * log(h_t / mean h)) and passes `decode(..., tau_HA_t=...)`. tau_HA_t[t] is the cost of an H<->A
transition between frames t - 1 and t; tau_HD / tau_DH stay constant. With tau_HA_t=None (the default, and what
predict does when w_switch == 0) the path is computed exactly as before (bit-identical labels)."""
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
    w_switch: float = 0.0     # per-frame tau_HA from the switch hazard; 0 = constant tau_HA (old behaviour)
    tau_floor: float = 3.0    # lower bound of the per-frame tau_HA


AUX_STREAMS = frozenset({"switch"})     # evidence-dict keys that are not evidence (skipped by cost_matrix)


def transition_matrix(p):
    T = np.zeros((3, 3))
    T[0, 1] = T[1, 0] = p.tau_HA
    T[0, 2] = T[1, 2] = p.tau_HD
    T[2, 0] = T[2, 1] = p.tau_DH
    return T


def cost_matrix(evidence, p):
    """Each stream is centred per frame over the states it has an opinion on, so a state it is
    silent about costs the same as its average opinion; this is what makes NaN mean
    'no opinion' rather than 'free'."""
    F = next(iter(evidence.values())).shape[0]
    C = np.zeros((F, 3), np.float64)
    for k, ev in evidence.items():
        if k in AUX_STREAMS:
            continue
        w = p.weights.get(k, 1.0) / max(1.0, p.lengths.get(k, 1.0))
        e = ev.astype(np.float64)
        fin = np.isfinite(e)
        n = fin.sum(1)
        with np.errstate(invalid="ignore", divide="ignore"):
            mean = np.where(fin, e, 0.0).sum(1) / np.maximum(n, 1)
        centred = np.where(fin, e - mean[:, None], 0.0)   # NaN = "average opinion" for that frame
        C -= w * centred
    return C


def viterbi_path(C, T, tau_HA_t=None):
    """Min-cost path; `tau_HA_t` (len(C),) optionally overrides T[0, 1] = T[1, 0] per frame (entry t: t-1 -> t)."""
    n = len(C)
    back = np.zeros((n, 3), np.int8)
    cur = C[0].copy()
    if tau_HA_t is not None:
        T = np.array(T, np.float64, copy=True); tau = np.asarray(tau_HA_t, np.float64)
    for t in range(1, n):
        if tau_HA_t is not None:
            T[0, 1] = T[1, 0] = tau[t]
        tot = cur[:, None] + T          # tot[i, j]: from i to j
        bi = tot.argmin(0)
        back[t] = bi
        cur = tot[bi, np.arange(3)] + C[t]
    path = np.zeros(n, np.int8); path[-1] = int(cur.argmin())
    for t in range(n - 1, 0, -1):
        path[t - 1] = back[t, path[t]]
    return path


def decode(evidence, half, p, tau_HA_t=None):
    """Labels per frame; `tau_HA_t` (F,) is an optional per-frame H<->A switch cost (module docstring)."""
    C = cost_matrix(evidence, p); T = transition_matrix(p)
    if tau_HA_t is not None and len(tau_HA_t) != len(C):
        raise ValueError(f"tau_HA_t has {len(tau_HA_t)} frames, evidence has {len(C)}")
    out = np.full(len(C), "D", dtype="<U1")
    for _, s, e in half_runs(half):
        out[s:e] = LABELS[viterbi_path(C[s:e], T, None if tau_HA_t is None else tau_HA_t[s:e])]
    return out
