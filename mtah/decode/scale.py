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
