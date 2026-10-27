"""Switch hazard: p(an H<->A possession change within +-0.25 s | frame), fed to the decoder as a per-frame tau_HA.

The hazard is NOT an evidence stream for `cost_matrix`. It reaches the decoder in two ways, both through the
evidence dict so cached evidence keeps working:
  * `compute_evidence` stores it under the key "switch" as an (F, 3) array (hazard in column 0, NaN elsewhere;
    `switch_evidence`); `cost_matrix` skips every key in `viterbi.AUX_STREAMS` (= {"switch"});
  * `pipeline.predict` turns column 0 into tau_HA(t) = max(floor, tau_HA - w_switch * log(h_t / mean h))
    (`hazard_to_tau`) and passes it to `decode(..., tau_HA_t=...)` when cfg.decoder.w_switch > 0;
    `PostParams.snap_mode = "hazard"` snaps H<->A boundaries to the hazard argmax.

Features (`switch_features`). Team-free by construction, so no mirror augmentation is needed: a team swap
(home <-> away) maps every column to itself (tested).
  * The T1 per-team columns (`team_features`) are re-indexed by the frame's nearest team instead of H/A:
    near{k}_* / far{k}_*, ball_dir_near/far, touch_{since,until}_{near,far}_s (ties: (min, max) of the pair).
  * Signed home-minus-away columns enter as absolute values (|gap|) or as products with the current sign
    (gap_mean_w * sign(nearest_team): negative when the rolling gap still favours the other team).
  * Temporal dynamics over L in {0.25, 0.5, 1} s: |s(t+L) - s(t-L)| for the nearest-team indicator s, |g(t+L) -
    g(t-L)| and g(t-L) * g(t+L) for the gap g, |m(t+L) - m(t-L)| and |m(t)| for the team-control margin
    m = log p - log(1 - p) when a "teamctrl" (F, 3) stream is given (NaN columns otherwise), |speed(t+L) -
    speed(t-L)| of the ball and the k = 3 velocity kink |v_out - v_in|.
  * Touches within +-0.25 / +-0.5 s (`detect_touches`): count, contested count, both-teams flag.
All columns are NaN where the ball is absent.
"""
import re

import joblib
import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier

from mtah.evidence.team_ctrl import TEAM_FEATURES, _swap_name, team_features
from mtah.evidence.touches import TouchParams, detect_touches, velocity_change

LAGS_S = (0.25, 0.5, 1.0)
TOUCH_WINDOWS_S = (0.25, 0.5)
PROB_CLIP = 1e-6


def _pair_name(n):
    """H-side T1 column name -> near/far stem (H1_dist -> {}1_dist, ball_dir_H -> ball_dir_{}, ...)."""
    return re.sub(r"(?<![A-Za-z])H(?=\d|_|$)", "{}", n)


_PAIRS = [(n, _swap_name(n)) for n in TEAM_FEATURES if _pair_name(n) != n]
_PAIR_STEMS = [_pair_name(h) for h, _ in _PAIRS]
_TEAM_FREE = ("ball_speed", "ball_accel", "touch_since_any_s", "touch_until_any_s")

SWITCH_FEATURES = (tuple(s.format(side) for s in _PAIR_STEMS for side in ("near", "far")) + _TEAM_FREE
                   + ("abs_gap_m",) + tuple(f"gap_mean_{w}_signed" for w in ("0.5s", "1s", "2s"))
                   + tuple(f"d_nearest_{L:g}s" for L in LAGS_S)
                   + tuple(f"d_gap_{L:g}s" for L in LAGS_S)
                   + tuple(f"gap_prod_{L:g}s" for L in LAGS_S)
                   + ("tc_abs_margin",) + tuple(f"tc_d_margin_{L:g}s" for L in LAGS_S)
                   + tuple(f"d_ball_speed_{L:g}s" for L in LAGS_S) + ("ball_kink",)
                   + tuple(f"touch_n_{w:g}s" for w in TOUCH_WINDOWS_S)
                   + tuple(f"touch_contested_{w:g}s" for w in TOUCH_WINDOWS_S)
                   + tuple(f"touch_both_{w:g}s" for w in TOUCH_WINDOWS_S))
TEAMCTRL_COLS = tuple(n for n in SWITCH_FEATURES if n.startswith("tc_"))

FEATURE_POLICIES = {
    "full": (),
    "no_touch": tuple(n for n in SWITCH_FEATURES if n.startswith("touch_")),
    "no_team": TEAMCTRL_COLS,   # no teamctrl margin features: needs no (out-of-fold) teamctrl stream (selection T2b)
}


def _shift(x, k):
    """y[t] = x[t + k] (NaN outside the array)."""
    y = np.full(x.shape, np.nan, np.float64)
    n = len(x)
    if k >= 0:
        if k < n:
            y[: n - k] = x[k:]
    elif -k < n:
        y[-k:] = x[: n + k]
    return y


def _window_count(frames, n, h, weights=None):
    """Per frame t: number (or weight sum) of event frames in [t - h, t + h]."""
    c = np.zeros(n + 1, np.float64)
    if len(frames):
        np.add.at(c, np.asarray(frames, int) + 1, 1.0 if weights is None else weights)
    c = np.cumsum(c)
    t = np.arange(n); lo = np.maximum(0, t - h); hi = np.minimum(n, t + h + 1)
    return c[hi] - c[lo]


def switch_features(match, team_stream=None, touch_params=None):
    """(F, len(SWITCH_FEATURES)) float32, team-swap invariant (module docstring); all-NaN rows where the ball is
    absent. `team_stream` is an (F, 3) "teamctrl" evidence array (H = log p, A = log(1 - p)) or None."""
    fps = float(match.fps); F = match.n_frames
    present = match.ball_present
    tp = touch_params or TouchParams()
    T = team_features(match, tp).astype(np.float64)
    tc = {n: i for i, n in enumerate(TEAM_FEATURES)}
    X = np.full((F, len(SWITCH_FEATURES)), np.nan, np.float64)
    col = {n: i for i, n in enumerate(SWITCH_FEATURES)}
    nt = T[:, tc["nearest_team"]]
    near_H = nt > 0; near_A = nt < 0; tie = ~(near_H | near_A)
    with np.errstate(invalid="ignore"):
        for (h, a), stem in zip(_PAIRS, _PAIR_STEMS):
            xh, xa = T[:, tc[h]], T[:, tc[a]]
            lo, hi = np.fmin(xh, xa), np.fmax(xh, xa)
            X[:, col[stem.format("near")]] = np.where(near_H, xh, np.where(near_A, xa, lo))
            X[:, col[stem.format("far")]] = np.where(near_H, xa, np.where(near_A, xh, hi))
    for n in _TEAM_FREE:
        X[:, col[n]] = T[:, tc[n]]
    gap = T[:, tc["nearest_gap_m"]]
    s = np.sign(nt)
    X[:, col["abs_gap_m"]] = np.abs(gap)
    for w in ("0.5s", "1s", "2s"):
        X[:, col[f"gap_mean_{w}_signed"]] = T[:, tc[f"gap_mean_{w}"]] * s
    sv = np.where(present, nt, np.nan); gv = np.where(present, gap, np.nan)
    m = None
    if team_stream is not None:
        e = np.asarray(team_stream, np.float64)
        m = np.where(present, e[:, 0] - e[:, 1], np.nan)
        X[:, col["tc_abs_margin"]] = np.abs(m)
    bs = np.where(present, T[:, tc["ball_speed"]], np.nan)
    for L in LAGS_S:
        k = max(1, int(round(L * fps)))
        X[:, col[f"d_nearest_{L:g}s"]] = np.abs(_shift(sv, k) - _shift(sv, -k))
        X[:, col[f"d_gap_{L:g}s"]] = np.abs(_shift(gv, k) - _shift(gv, -k))
        X[:, col[f"gap_prod_{L:g}s"]] = _shift(gv, k) * _shift(gv, -k)
        X[:, col[f"d_ball_speed_{L:g}s"]] = np.abs(_shift(bs, k) - _shift(bs, -k))
        if m is not None:
            X[:, col[f"tc_d_margin_{L:g}s"]] = np.abs(_shift(m, k) - _shift(m, -k))
    X[:, col["ball_kink"]] = velocity_change(match.ball.astype(np.float64), tp.k, fps)
    t = detect_touches(match, tp)
    fr = t[:, 0].astype(int) if len(t) else np.zeros(0, int)
    tm = t[:, 2].astype(int) if len(t) else np.zeros(0, int)
    for w in TOUCH_WINDOWS_S:
        h = max(1, int(round(w * fps)))
        X[:, col[f"touch_n_{w:g}s"]] = _window_count(fr, F, h)
        X[:, col[f"touch_contested_{w:g}s"]] = _window_count(fr[tm < 0], F, h)
        X[:, col[f"touch_both_{w:g}s"]] = ((_window_count(fr[tm == 0], F, h) > 0)
                                            & (_window_count(fr[tm == 1], F, h) > 0)).astype(np.float64)
    X[~present] = np.nan
    return X.astype(np.float32)


def switch_labels(truth, fps, half_s=0.25, ball_present=None, max_d_s=1.0):
    """(F,) int8: 1 within +-half_s of a truth H<->A change, 0 elsewhere, -1 (ignore) where truth is D or the ball
    is absent (`ball_present`, optional). Changes: direct H->A / A->H, and X -> D -> Y (X != Y in {H, A}) across
    a D run shorter than max_d_s; longer D runs and same-team D gaps are not changes. With h = round(half_s * fps)
    frames, a change whose new team starts at frame e after a D run starting at s (s = e when direct) marks
    [s - h, e + h - 1]."""
    x = np.asarray(truth).astype("<U1")
    n = len(x)
    y = np.zeros(n, np.int8)
    if n:
        h = max(1, int(round(half_s * fps)))
        cuts = np.flatnonzero(x[1:] != x[:-1]) + 1
        starts = np.r_[0, cuts]; stops = np.r_[cuts, n]; labs = x[starts]
        for i in range(1, len(starts)):
            if labs[i] not in ("H", "A"):
                continue
            if labs[i - 1] in ("H", "A") and labs[i - 1] != labs[i]:
                s = e = int(starts[i])
            elif (labs[i - 1] == "D" and i >= 2 and labs[i - 2] in ("H", "A") and labs[i - 2] != labs[i]
                  and (stops[i - 1] - starts[i - 1]) < max_d_s * fps):
                s, e = int(starts[i - 1]), int(starts[i])
            else:
                continue
            y[max(0, s - h): min(n, e + h)] = 1
    y[x == "D"] = -1
    if ball_present is not None:
        y[~np.asarray(ball_present, bool)] = -1
    return y


def apply_policy(X, policy):
    """Copy of X with the policy's dropped columns set to 0.0 (all-NaN rows stay all NaN)."""
    Y = np.array(X, np.float32, copy=True)
    keep = ~np.isnan(Y).all(1)
    for n in FEATURE_POLICIES[policy]:
        Y[keep, SWITCH_FEATURES.index(n)] = 0.0
    return Y


class SwitchModel:
    """Switch hazard p(change within +-half_s); `policy` travels with the model. Class-balanced sample weights,
    so the hazard is not calibrated to the base rate (only h / mean h matters to the decoder).
    `with_teamctrl` records whether the teamctrl columns were informative in training (then the stream is
    required at prediction time)."""

    def __init__(self, policy="full"):
        if policy not in FEATURE_POLICIES:
            raise KeyError(policy)
        self.policy = policy
        self.const_cols = ()
        self.with_teamctrl = False
        self.model = HistGradientBoostingClassifier(max_iter=300, max_leaf_nodes=31, learning_rate=0.1,
                                                    min_samples_leaf=200, random_state=0, early_stopping=False)

    def _prep(self, X):
        Y = apply_policy(X, self.policy)
        keep = ~np.isnan(Y).all(1)
        for j in self.const_cols:
            Y[keep, j] = 0.0
        return Y

    def fit(self, X, y, sample_weight=None):
        """y in {0, 1}; rows with y < 0 or all-NaN features are dropped. Classes are reweighted to equal mass."""
        X = np.asarray(X, np.float32); y = np.asarray(y, int)
        ok = (y >= 0) & ~np.isnan(X).all(1)
        X, y = X[ok], y[ok]
        sw = np.ones(len(y)) if sample_weight is None else np.asarray(sample_weight, float)[ok]
        for c in (0, 1):
            tot = sw[y == c].sum()
            if tot > 0:
                sw = np.where(y == c, sw * (0.5 * sw.sum() / tot), sw)
        self.const_cols = tuple(int(j) for j in np.flatnonzero(np.isnan(X).all(0)))
        tci = [SWITCH_FEATURES.index(n) for n in TEAMCTRL_COLS]
        self.with_teamctrl = (bool(np.isfinite(X[:, tci]).any())
                              and not set(TEAMCTRL_COLS) <= set(FEATURE_POLICIES[self.policy]))
        self.model.fit(self._prep(X), y, sample_weight=sw)
        return self

    def predict_hazard(self, X):
        """(F,) float32 in (0, 1); NaN on all-NaN rows."""
        X = np.asarray(X, np.float32)
        out = np.full(len(X), np.nan, np.float32)
        ok = ~np.isnan(X).all(1)
        if ok.any():
            p = self.model.predict_proba(self._prep(X[ok]))[:, 1]
            out[ok] = np.clip(p, PROB_CLIP, 1.0 - PROB_CLIP)
        return out

    def save(self, path):
        joblib.dump({"model": self.model, "policy": self.policy, "const_cols": list(self.const_cols),
                     "with_teamctrl": self.with_teamctrl}, path)

    @classmethod
    def load(cls, path):
        obj = joblib.load(path)
        m = cls(obj.get("policy", "full")); m.model = obj["model"]
        m.const_cols = tuple(obj.get("const_cols", ())); m.with_teamctrl = bool(obj.get("with_teamctrl", False))
        return m


def train_switch(sims, variant="ctrl_Dfull", policy="full", stride=2, touch_params=None, team_streams=None,
                 half_s=0.25):
    """SwitchModel on labelled (y >= 0) frames of `sims` (every `stride`-th frame per half). `team_streams`: one
    (F, 3) teamctrl evidence per sim or None; for an honest hazard they must be out-of-sample predictions."""
    Xs, ys, ws = [], [], []
    for i, s in enumerate(sims):
        w = float(getattr(s, "weight", 1.0))   # training-row weight of the half (selection: extra halves)
        if w <= 0:
            continue
        ts = None if team_streams is None else team_streams[i]
        y = switch_labels(s.truth[variant], s.match.fps, half_s, s.match.ball_present)
        idx = np.flatnonzero(y >= 0)[::stride]
        Xs.append(switch_features(s.match, ts, touch_params)[idx]); ys.append(y[idx]); ws.append(np.full(len(idx), w))
    sw = None if all(float(getattr(s, "weight", 1.0)) == 1.0 for s in sims) else np.concatenate(ws)
    return SwitchModel(policy=policy).fit(np.concatenate(Xs), np.concatenate(ys), sample_weight=sw)


def switch_evidence(h):
    """(F, 3) float32 carrier for the evidence dict: hazard in column 0, NaN elsewhere (skipped by cost_matrix)."""
    h = np.asarray(h, np.float32)
    ev = np.full((len(h), 3), np.nan, np.float32)
    ev[:, 0] = h
    return ev


def hazard_to_tau(h, tau_HA, w_switch, floor):
    """(F,) float64 tau_HA(t) = max(floor, tau_HA - w_switch * log(h_t / nanmean h)); tau_HA where h is NaN (or
    everywhere when w_switch == 0 or h has no finite value). mean h is over the whole array passed in."""
    h = np.asarray(h, np.float64)
    tau = np.full(len(h), float(tau_HA), np.float64)
    ok = np.isfinite(h) & (h > 0)
    if w_switch == 0 or not ok.any():
        return tau
    mu = h[ok].mean()
    tau[ok] = np.maximum(float(floor), float(tau_HA) - float(w_switch) * np.log(h[ok] / mu))
    return tau
