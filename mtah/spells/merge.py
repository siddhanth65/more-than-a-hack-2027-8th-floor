"""Per-spell merge model (Plan 2d T3, design lever C1).

A predicted H/A spell is a merge candidate when at least one neighbour (inside the same half) is an H/A spell. Since
spells are maximal runs, every H/A neighbour belongs to the OTHER team, so "merge" always relabels the spell's frames
to the other team: an A-B-A spell collapses into its surroundings (n_pred - 2), a spell next to one H/A spell and a
D run (or a half edge) extends that neighbour (n_pred - 1). D neighbours are never extended: the dead decision stays
with the dead stream and the real D-share band. Spells without an H/A neighbour are not mergeable.

`spell_table` builds one row per predicted H/A spell with label-free features (length, neighbour kinds/lengths,
per-stream mean/min evidence margin signed towards the spell's own team, touches inside, blackout / ball-absent
fractions, dead logit). `merge_labels` gives the training label: 1 iff merging that spell ALONE raises the official
spellF1 of the half vs truth. The F1 is computed by `SpellF1`, an exact re-implementation of the organisers'
`spell_scores` mean F1 on precomputed truth spells (vectorised overlaps, one greedy pass for all IoU thresholds);
tests check it equals `mtah.eval.scorer.score(...)["spell_f1"]`. Cost: one evaluation is O(n_spells) numpy plus an
O(n_overlaps) Python greedy pass, ~2-5 ms on a 70k-frame half with ~400 spells, i.e. ~1-2 s per half for all its
spells (`max_rows` caps the evaluated spells per half when needed).

`apply_merges` merges every mergeable spell with P(merge) > theta, shortest first; before each merge the spell is
re-derived on the current labels (skipped when an earlier merge absorbed or grew it) and so are its neighbours.
theta >= 1 is "off" and returns the labels unchanged.
"""
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier

from mtah.eval.official import possession_metrics as pm
from mtah.io.match import half_runs

MERGE_STREAMS = ("ball", "tag", "pathcrf", "teamctrl")
NB_NONE, NB_OTHER, NB_DEAD = 0, 1, 2          # neighbour kind relative to the spell (H/A neighbours are always OTHER)
FEATURES = (["length_s", "prev_kind", "next_kind", "prev_len_s", "next_len_s", "aba", "dead_prev", "dead_next",
             "target_len_s", "n_touch_own", "n_touch_other", "n_strong_own", "touch_own_rate", "blackout_frac",
             "ball_absent_frac", "dead_logit_mean"]
            + [f"{k}_{s}" for k in MERGE_STREAMS for s in ("mean", "min")])
META = ["start", "stop", "team", "target", "mergeable"]


def _spans(n, half):
    if half is None:
        return [(0, n)]
    return [(s, e) for _, s, e in half_runs(half)]


def _segments(labels, spans):
    """[(start, stop, label, span_index)] constant runs inside each span."""
    out = []
    for k, (s0, e0) in enumerate(spans):
        x = labels[s0:e0]
        if not len(x):
            continue
        cuts = np.flatnonzero(x[1:] != x[:-1]) + 1
        for a, b in zip(np.r_[0, cuts] + s0, np.r_[cuts, len(x)] + s0):
            out.append((int(a), int(b), str(labels[a]), k))
    return out


def _signed_margin(ev):
    """H - A log-evidence per frame (NaN where either is NaN)."""
    ev = np.asarray(ev, np.float64)
    m = ev[:, 0] - ev[:, 1]
    m[~(np.isfinite(ev[:, 0]) & np.isfinite(ev[:, 1]))] = np.nan
    return m


def spell_table(labels, evidence, touches, fps, half=None, blackout=None, ball_present=None, strong=None):
    """DataFrame, one row per predicted H/A spell: META columns + FEATURES (NaN = stream absent / no finite frame).

    labels: (F,) H/A/D; evidence: {stream: (F, 3) H, A, D log-evidence}; touches: (n, >=3) [frame, _, team 0=H/1=A, ...];
    half: (F,) half ids (None = one span); blackout / ball_present: (F,) bool or None; strong: (strong_H, strong_A[, ...])."""
    labels = np.asarray(labels).astype("<U1")
    n = len(labels)
    segs = _segments(labels, _spans(n, half))
    margins = {k: _signed_margin(evidence[k]) for k in MERGE_STREAMS if evidence is not None and k in evidence}
    dead = None
    if evidence is not None and "dead" in evidence:
        e = np.asarray(evidence["dead"], np.float64); dead = e[:, 2] - e[:, 0]
    touches = np.asarray(touches) if touches is not None else np.zeros((0, 4))
    if len(touches):
        order = np.argsort(touches[:, 0], kind="stable")
        tf = touches[order, 0].astype(int); tt = touches[order, 2].astype(int)
    else:
        tf = np.zeros(0, int); tt = np.zeros(0, int)
    cs_t = {c: np.r_[0, np.cumsum(tt == c)] for c in (0, 1)}
    rows = []
    for i, (s, e, lab, k) in enumerate(segs):
        if lab not in ("H", "A"):
            continue
        own = 0 if lab == "H" else 1
        sign = 1.0 if own == 0 else -1.0
        nb = []
        for j in (i - 1, i + 1):
            if 0 <= j < len(segs) and segs[j][3] == k:
                lj = segs[j][2]
                kind = NB_DEAD if lj == "D" else NB_OTHER
                nb.append((kind, (segs[j][1] - segs[j][0]) / fps, lj))
            else:
                nb.append((NB_NONE, 0.0, None))
        (pk, pl, plab), (nk, nl, nlab) = nb
        ha = [(ln, lb) for kd, ln, lb in nb if kd == NB_OTHER]
        if ha:
            if pk == nk == NB_OTHER and plab == nlab:
                target, tlen = plab, pl + nl
            else:
                tlen, target = max(ha, key=lambda t: t[0])
        else:
            target, tlen = "", 0.0
        L = e - s
        a, b = np.searchsorted(tf, s, "left"), np.searchsorted(tf, e, "left")
        n_own = int(cs_t[own][b] - cs_t[own][a]); n_oth = int(cs_t[1 - own][b] - cs_t[1 - own][a])
        r = {"start": s, "stop": e, "team": lab, "target": target, "mergeable": bool(ha),
             "length_s": L / fps, "prev_kind": pk, "next_kind": nk, "prev_len_s": pl, "next_len_s": nl,
             "aba": float(pk == nk == NB_OTHER), "dead_prev": float(pk == NB_DEAD), "dead_next": float(nk == NB_DEAD),
             "target_len_s": tlen, "n_touch_own": n_own, "n_touch_other": n_oth,
             "n_strong_own": (int(np.asarray(strong[own][s:e]).sum()) if strong is not None and strong[own] is not None
                              else np.nan),
             "touch_own_rate": n_own / (L / fps),
             "blackout_frac": float(np.mean(blackout[s:e])) if blackout is not None else np.nan,
             "ball_absent_frac": float(np.mean(~np.asarray(ball_present[s:e], bool))) if ball_present is not None else np.nan}
        if dead is not None:
            d = dead[s:e]; d = d[np.isfinite(d)]
            r["dead_logit_mean"] = float(d.mean()) if d.size else np.nan
        else:
            r["dead_logit_mean"] = np.nan
        for kname in MERGE_STREAMS:
            m = margins.get(kname)
            if m is None:
                r[f"{kname}_mean"] = r[f"{kname}_min"] = np.nan
                continue
            x = sign * m[s:e]; x = x[np.isfinite(x)]
            r[f"{kname}_mean"] = float(x.mean()) if x.size else np.nan
            r[f"{kname}_min"] = float(x.min()) if x.size else np.nan
        rows.append(r)
    return pd.DataFrame(rows, columns=META + FEATURES)


def match_spell_table(labels, match, evidence, strong):
    """spell_table with the match-derived context: halves, long ball-absent blackout mask, ball presence, touches."""
    from mtah.eval.drift import blackout_mask_from_match
    touches = strong[2] if strong is not None and len(strong) > 2 else None
    return spell_table(labels, evidence, touches, match.fps, half=match.half, blackout=blackout_mask_from_match(match),
                       ball_present=match.ball_present, strong=strong)


def _run_bounds(x, i, lo, hi):
    """[a, b) of the constant run of x containing index i, inside [lo, hi)."""
    v = x[i]
    left = x[lo:i][::-1] != v
    a = i - (int(np.argmax(left)) if left.any() else i - lo)
    right = x[i:hi] != v
    b = i + (int(np.argmax(right)) if right.any() else hi - i)
    return a, b


def apply_merges(labels, table, p, theta, half=None):
    """Relabel each mergeable spell with p > theta to its merge target, shortest first (ties: earlier start).

    Each spell is re-derived on the current labels first: when an earlier merge absorbed or grew it, it is skipped;
    its neighbours (and hence the target) are taken from the current labels. theta >= 1 returns an unchanged copy."""
    out = np.asarray(labels).astype("<U1").copy()
    p = np.asarray(p, float)
    if theta >= 1.0 or len(table) == 0:
        return out
    spans = _spans(len(out), half)
    lo_of = np.array([s for s, _ in spans]); hi_of = np.array([e for _, e in spans])
    st = table["start"].to_numpy(int); sp = table["stop"].to_numpy(int)
    sel = np.flatnonzero((p > theta) & table["mergeable"].to_numpy(bool))
    sel = sel[np.lexsort((st[sel], (sp - st)[sel]))]
    for i in sel:
        s, e, team = int(st[i]), int(sp[i]), str(table["team"].iat[i])
        k = int(np.searchsorted(lo_of, s, "right") - 1)
        lo, hi = int(lo_of[k]), int(hi_of[k])
        if out[s] != team or _run_bounds(out, s, lo, hi) != (s, e):
            continue                                   # absorbed or grown by an earlier merge
        nb = []
        if s > lo and out[s - 1] in ("H", "A"):
            a, _ = _run_bounds(out, s - 1, lo, hi); nb.append((s - a, str(out[s - 1])))
        if e < hi and out[e] in ("H", "A"):
            _, b = _run_bounds(out, e, lo, hi); nb.append((b - e, str(out[e])))
        if not nb:
            continue
        target = nb[0][1] if len(nb) == 2 and nb[0][1] == nb[1][1] else max(nb, key=lambda t: t[0])[1]
        out[s:e] = target
    return out


class SpellF1:
    """Exact spell_f1_mean (organisers' spell_scores on the abstention-filled prediction) for one truth array, with
    the truth spells precomputed; `f1(pred_codes)` takes int codes (pm.label_codes)."""

    def __init__(self, gt_codes):
        self.g = np.asarray(gt_codes)
        self.runs = pm.valid_runs(self.g)
        gs = pm.possession_spells(self.g, self.runs)
        self.gs = np.array(gs, np.int64).reshape(-1, 3)
        self.n_gt = len(gs)

    def f1(self, pred_codes):
        filled = pm.fill_abstentions(np.asarray(pred_codes), self.runs)
        ps = np.array(pm.possession_spells(filled, self.runs), np.int64).reshape(-1, 3)
        n_pred = len(ps)
        gs = self.gs
        lo = np.searchsorted(gs[:, 1], ps[:, 0], "right")     # first gt spell ending after the pred start
        hi = np.searchsorted(gs[:, 0], ps[:, 1], "left")      # gt spells starting before the pred stop
        cnt = np.maximum(hi - lo, 0)
        j = np.repeat(np.arange(n_pred), cnt)
        i = (np.repeat(lo - np.r_[0, np.cumsum(cnt)[:-1]], cnt) + np.arange(cnt.sum())) if cnt.sum() else np.zeros(0, int)
        keep = gs[i, 2] == ps[j, 2]
        i, j = i[keep], j[keep]
        inter = np.minimum(gs[i, 1], ps[j, 1]) - np.maximum(gs[i, 0], ps[j, 0])
        union = np.maximum(gs[i, 1], ps[j, 1]) - np.minimum(gs[i, 0], ps[j, 0])
        iou = inter / union.astype(np.float64)
        order = np.lexsort((j, i, -iou))
        thr = sorted(pm.SPELL_IOUS, reverse=True)
        tp_at = {}; tp = 0; t = 0
        used_g = np.zeros(self.n_gt, bool); used_p = np.zeros(n_pred, bool)
        for o in order:
            v = iou[o]
            while t < len(thr) and v < thr[t]:
                tp_at[thr[t]] = tp; t += 1
            if t == len(thr):
                break
            gi, pj = i[o], j[o]
            if used_g[gi] or used_p[pj]:
                continue
            used_g[gi] = True; used_p[pj] = True; tp += 1
        for th in thr[t:]:
            tp_at[th] = tp
        f1s = []
        for th in pm.SPELL_IOUS:
            tp = tp_at[th]; fp = n_pred - tp; fn = self.n_gt - tp
            precision = tp / float(tp + fp) if tp + fp else 0.0
            recall = tp / float(tp + fn) if tp + fn else 0.0
            f1s.append(2.0 * precision * recall / (precision + recall) if precision + recall else 0.0)
        return float(np.mean(f1s)) if f1s else 0.0


def merge_labels(truth, labels, table, max_rows=None, rng=None):
    """(n,) float: 1.0 iff merging that spell alone (labels otherwise unchanged) raises spellF1 vs truth, 0.0 if not,
    NaN for non-mergeable spells (and for spells beyond the `max_rows` cap, chosen with `rng`, default seed 0)."""
    gt = pm.label_codes(np.asarray(truth)); p0 = pm.label_codes(np.asarray(labels))
    sc = SpellF1(gt)
    base = sc.f1(p0)
    y = np.full(len(table), np.nan)
    idx = np.flatnonzero(table["mergeable"].to_numpy(bool)) if len(table) else np.zeros(0, int)
    if max_rows is not None and len(idx) > max_rows:
        rng = rng if rng is not None else np.random.default_rng(0)
        idx = np.sort(rng.choice(idx, max_rows, replace=False))
    code = {lab: i for i, lab in enumerate(pm.LABELS)}
    st = table["start"].to_numpy(int) if len(table) else np.zeros(0, int)
    sp = table["stop"].to_numpy(int) if len(table) else np.zeros(0, int)
    for r in idx:
        p = p0.copy(); p[st[r]:sp[r]] = code[table["target"].iat[r]]
        y[r] = 1.0 if sc.f1(p) > base + 1e-12 else 0.0
    return y


class MergeModel:
    """HGB on FEATURES -> P(merging the spell raises spellF1). A single-class training set gives a constant.
    NaN = missing (HGB native); a feature that is NaN on every training row (stream absent) is fed as 0 at fit and
    predict (`dead_cols`), since HGB cannot bin an all-missing column."""

    def __init__(self):
        self.model = HistGradientBoostingClassifier(max_iter=200, max_leaf_nodes=15, learning_rate=0.05,
                                                    min_samples_leaf=40, l2_regularization=1.0, random_state=0,
                                                    early_stopping=False)
        self.const = None
        self.dead_cols = []
        self.info = {}

    def _X(self, table):
        X = table[FEATURES].to_numpy(np.float64)
        if self.dead_cols:
            X[:, self.dead_cols] = 0.0
        return X

    def fit(self, table, y):
        y = np.asarray(y, float); ok = np.isfinite(y)
        X0 = table[FEATURES].to_numpy(np.float64)[ok]
        self.dead_cols = [int(c) for c in np.flatnonzero(~np.isfinite(X0).any(0))] if len(X0) else []
        X, y = self._X(table)[ok], y[ok].astype(int)
        if len(np.unique(y)) < 2:
            self.const = float(y.mean()) if len(y) else 0.0
        else:
            self.model.fit(X, y); self.const = None
        self.info.update(n_rows=int(len(y)), pos_rate=float(y.mean()) if len(y) else float("nan"))
        return self

    def predict_proba(self, table):
        if len(table) == 0:
            return np.zeros(0)
        if self.const is not None:
            return np.full(len(table), self.const)
        return self.model.predict_proba(self._X(table))[:, 1]

    def save(self, path):
        joblib.dump({"model": self.model, "const": self.const, "features": list(FEATURES), "dead_cols": self.dead_cols,
                     "info": self.info}, path)

    @classmethod
    def load(cls, path):
        obj = joblib.load(path)
        if list(obj["features"]) != list(FEATURES):
            raise ValueError(f"{path}: merge model features differ from mtah.spells.merge.FEATURES")
        m = cls(); m.model = obj["model"]; m.const = obj["const"]; m.info = obj.get("info", {})
        m.dead_cols = list(obj.get("dead_cols", []))
        return m


def apply_merge_model(labels, match, evidence, strong, model, theta):
    """Spell table on the post-processed labels -> P(merge) -> apply_merges (theta >= 1: unchanged copy)."""
    labels = np.asarray(labels).astype("<U1")
    if theta >= 1.0:
        return labels.copy()
    tab = match_spell_table(labels, match, evidence, strong)
    return apply_merges(labels, tab, model.predict_proba(tab), theta, half=match.half)
