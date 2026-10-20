"""Evidence-scale correction: how many frames one evidence sample is really worth."""
import numpy as np


def correlation_length(x, max_lag=2000):
    x = np.asarray(x, dtype=np.float64); x = x[np.isfinite(x)]
    n = len(x)
    if n < 10:
        return 1.0
    x = x - x.mean(); var = (x * x).sum()
    if var <= 0:
        return 1.0
    total = 0.0
    for lag in range(1, min(max_lag, n - 1)):
        r = (x[:-lag] * x[lag:]).sum() / var
        if r <= 0:
            break
        total += r
    return float(max(1.0, 1.0 + 2.0 * total))


def evidence_margin(ev):
    ev = np.asarray(ev, np.float64)
    out = ev[:, 0] - ev[:, 1]
    out[~(np.isfinite(ev[:, 0]) & np.isfinite(ev[:, 1]))] = np.nan
    return out


def residual_margin(margin, truth_labels, states):
    """Margin minus its per-state mean (over finite frames whose truth is in `states`).

    NaN where the margin is NaN or the truth is not in `states`. What is left is the stream's
    error given the truth, which is what the correlation length must be measured on."""
    m = np.asarray(margin, np.float64); t = np.asarray(truth_labels)
    out = np.full(m.shape, np.nan)
    fin = np.isfinite(m)
    for st in states:
        sel = (t == st) & fin
        if sel.any():
            out[sel] = m[sel] - m[sel].mean()
    return out
