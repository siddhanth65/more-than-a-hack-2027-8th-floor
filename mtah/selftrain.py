"""Cross-half self-training of the BiGRU stream B (candidate S; Plan 9 amendment 3). Numpy only.

Pseudo-labels are algorithmic (no ground truth of the target). For a target half h the fine-tuning data are the
original simulator windows plus pseudo-labelled windows of the OTHER half of the same match, never h itself.

Step labels use the codes of the sequence data: 0 H, 1 A, 2 D, 3 ignore (no label / ungraded).
Frame pseudo-labels use int8: 0 H, 1 A, 2 D, -1 none.
"""
import numpy as np

STEP = 5
IGNORE = 3
# the organisers' public window (frames 11185..18685 inclusive of the contest match): never pseudo-labelled
PUBLIC_WINDOW = (11185, 18685)
PSEUDO_MIN_LABELLED = 5           # a pseudo window needs at least this many labelled steps (of 600); amendment 3b
PSEUDO_STRIDE = 150               # stride of pseudo windows (simulator windows keep the base stride)
SIM_PER_PSEUDO = 1                # simulator windows drawn per pseudo window


def cross_half_source(target, halves):
    """The half whose pseudo-labels may train the model that predicts `target`: the other half of the match.
    `halves` is the collection of half ids of the match (exactly two)."""
    hs = sorted(set(halves))
    if len(hs) != 2 or target not in hs:
        raise ValueError(f"cross-half needs exactly two halves containing the target; got {hs}, target {target}")
    return hs[1] if target == hs[0] else hs[0]


def exclude_frames(frame_labels, ranges):
    """Copy of `frame_labels` with every frame inside the inclusive (lo, hi) frame ranges set to -1."""
    out = np.array(frame_labels, np.int8, copy=True)
    for lo, hi in ranges:
        out[max(int(lo), 0):int(hi) + 1] = -1
    return out


def combine_frame_labels(primary, secondary=None):
    """Frame labels of `primary` (confident ball control: 0/1/-1), filled by `secondary` (0/1/2/-1) where primary is -1."""
    out = np.array(primary, np.int8, copy=True)
    if secondary is not None:
        s = np.asarray(secondary, np.int8)
        take = (out < 0) & (s >= 0)
        out[take] = s[take]
    return out


def merged_posterior_labels(cost, decoded, thr=0.9):
    """V3 labels: the final decoded label (array of 'H'/'A'/'D') on frames where the softmax of minus the decoder's
    per-frame cost row gives that label a posterior of at least `thr`; -1 elsewhere. cost: (F, 3)."""
    c = np.asarray(cost, np.float64)
    z = -(c - c.min(1, keepdims=True))
    p = np.exp(z); p /= p.sum(1, keepdims=True)
    code = np.full(len(decoded), -1, np.int8)
    for i, ch in enumerate("HAD"):
        code[np.asarray(decoded) == ch] = i
    ok = code >= 0
    pl = np.where(ok, p[np.arange(len(p)), np.maximum(code, 0)], 0.0)
    out = np.where(ok & (pl >= thr), code, -1).astype(np.int8)
    return out


def to_step_labels(frame_labels, starts, step=STEP):
    """Label of the centre frame of each step; -1 becomes IGNORE."""
    fl = np.asarray(frame_labels)
    lab = fl[np.minimum(np.asarray(starts) + step // 2, len(fl) - 1)].astype(np.int64)
    return np.where(lab < 0, IGNORE, lab)


def pseudo_windows(y, win, stride, offset, min_labelled=PSEUDO_MIN_LABELLED):
    """Start indices of windows of `win` steps (stride `stride`, first at `offset`) holding at least `min_labelled`
    labelled steps (y != IGNORE)."""
    lab = (np.asarray(y) != IGNORE).astype(np.int64)
    c = np.concatenate(([0], np.cumsum(lab)))
    return [a for a in range(int(offset), max(len(y) - win, 0) + 1, int(stride)) if c[a + win] - c[a] >= min_labelled]


def step_weights(y, w_pseudo, is_pseudo):
    """Per-step loss weights of a window: 0 on ignored steps, `w_pseudo` on labelled steps of a pseudo window, 1 on
    labelled steps of a simulator window."""
    y = np.asarray(y)
    base = np.where(y == IGNORE, 0.0, 1.0)
    return base * (float(w_pseudo) if is_pseudo else 1.0)


def weighted_ce(logp, y, w):
    """Weighted mean cross-entropy: sum(w * -logp[y]) / sum(w) over steps with w > 0 (logp: (T, 3) log-probs)."""
    logp = np.asarray(logp, np.float64); y = np.asarray(y); w = np.asarray(w, np.float64)
    m = w > 0
    if not m.any():
        return 0.0
    return float((w[m] * -logp[m, y[m]]).sum() / w[m].sum())


def leakage_check(target_half, source_half, train_halves):
    """Raise if the pseudo-label source is the target half itself or a half the target model may not use."""
    if source_half == target_half:
        raise AssertionError("a half's own pseudo-labels would train the model that predicts it")
    if target_half in train_halves:
        raise AssertionError("target half listed among training halves")


def build_epoch(sim_seqs, pseudo_seq, w_pseudo, rng, win, stride, mirror, swap_labels, min_labelled=PSEUDO_MIN_LABELLED,
                sim_per_pseudo=SIM_PER_PSEUDO, pseudo_stride=PSEUDO_STRIDE):
    """Windows of one fine-tuning epoch (shuffled): every pseudo window of `pseudo_seq` (the OTHER half's normalised
    steps and pseudo step labels; None = no pseudo data) plus `sim_per_pseudo` times as many simulator windows drawn
    at random from `sim_seqs` [(Xn, y)] (stride `stride`; pseudo windows use `pseudo_stride`). Each window is team-swapped with probability 1/2 (`mirror(x)` on the
    features, `swap_labels(y)` on the labels). Returns a list of dicts x, y, w (per-step weights) and pseudo (bool).
    A pseudo window with weight 0 carries no gradient (the V0 control)."""
    out = []
    n_p = 0
    if pseudo_seq is not None:
        Xp, yp = pseudo_seq
        for a in pseudo_windows(yp, win, pseudo_stride, rng.integers(0, pseudo_stride), min_labelled):
            x, y = Xp[a:a + win], yp[a:a + win]
            if rng.random() < 0.5:
                x, y = mirror(x), swap_labels(y)
            out.append({"x": x, "y": y, "w": step_weights(y, w_pseudo, True), "pseudo": True}); n_p += 1
    pool = []
    for si, (Xs, ys) in enumerate(sim_seqs):
        for a in range(int(rng.integers(0, stride)), max(len(ys) - win, 0) + 1, stride):
            pool.append((si, a))
    k = min(len(pool), sim_per_pseudo * max(n_p, 1))
    for j in rng.choice(len(pool), size=k, replace=False):
        si, a = pool[j]; Xs, ys = sim_seqs[si]
        x, y = Xs[a:a + win], ys[a:a + win]
        if rng.random() < 0.5:
            x, y = mirror(x), swap_labels(y)
        out.append({"x": x, "y": y, "w": step_weights(y, 1.0, False), "pseudo": False})
    order = rng.permutation(len(out))
    return [out[i] for i in order]
