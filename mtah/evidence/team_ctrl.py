"""Learned team-in-control evidence on visible-ball frames ("teamctrl" stream).

Per frame with a visible ball: geometry and kinematics of the k = 3 nearest players of each team to the ball, ball
kinematics, ball direction against each team's attacking direction, touch timing per team, and the nearest-team
indicator/gap (raw and centred rolling means). A gradient-boosted classifier trained on ctrl truth H/A frames gives
p_home; the stream is H = log p, A = log(1 - p), D = NaN (no opinion), all NaN where the ball is absent.

Team-swap mirror. Swapping home and away maps the feature vector x to mirror(x):
  * every per-team column swaps with its counterpart: H{k}_* <-> A{k}_*, ball_dir_H <-> ball_dir_A,
    touch_{since,until}_H_s <-> touch_{since,until}_A_s;
  * home-minus-away columns change sign: nearest_team, nearest_gap_m, gap_mean_*;
  * team-free columns stay: ball_speed, ball_accel, touch_{since,until}_any_s.
Pitch mirroring (x -> -x) leaves every column unchanged: directions are taken against each team's own attacking
direction (`direction_of_play`), everything else is distances, speeds and angles. mirror is an involution.
Training uses (X, y) together with (mirror(X), 1 - y); prediction averages p(x) and 1 - p(mirror(x)), so
p(mirror x) = 1 - p(x) holds exactly (no home bias can be learned).
"""
import re

import joblib
import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier

from mtah.evidence.dead import direction_of_play
from mtah.evidence.touches import TouchParams, detect_touches

K_NEAREST = 3
TOUCH_CAP_S = 5.0
MIN_SPEED = 0.3          # m/s: below this a velocity direction is undefined (cosine NaN)
GAP_WINDOWS_S = (0.5, 1.0, 2.0)
PER_PLAYER = ("dist", "dd_dt", "speed", "cos")

TEAM_FEATURES = tuple(f"{t}{k}_{q}" for t in "HA" for k in range(1, K_NEAREST + 1) for q in PER_PLAYER) + (
    "ball_speed", "ball_accel", "ball_dir_H", "ball_dir_A",
    "touch_since_any_s", "touch_since_H_s", "touch_since_A_s",
    "touch_until_any_s", "touch_until_H_s", "touch_until_A_s",
    "nearest_team", "nearest_gap_m") + tuple(f"gap_mean_{w:g}s" for w in GAP_WINDOWS_S)

_SIGN_FLIP = {"nearest_team", "nearest_gap_m"} | {f"gap_mean_{w:g}s" for w in GAP_WINDOWS_S}


def _swap_name(n):
    """Feature name after a home/away swap (an uppercase H/A token not glued to other letters)."""
    return re.sub(r"(?<![A-Za-z])([HA])(?=\d|_|$)", lambda m: "A" if m.group(1) == "H" else "H", n)


MIRROR_PERM = np.array([TEAM_FEATURES.index(_swap_name(n)) for n in TEAM_FEATURES])
MIRROR_SIGN = np.array([-1.0 if n in _SIGN_FLIP else 1.0 for n in TEAM_FEATURES], np.float32)


def mirror_features(X):
    """Feature table of the same frames with home and away swapped (see the module docstring)."""
    X = np.asarray(X, np.float32)
    return (X[:, MIRROR_PERM] * MIRROR_SIGN).astype(np.float32)


def _central_diff(x, fps):
    """Central difference along axis 0 in units per second; NaN at both ends and next to missing samples."""
    out = np.full(x.shape, np.nan, np.float64)
    if len(x) >= 3:
        out[1:-1] = (x[2:] - x[:-2]) * (fps / 2.0)
    return out


def _centred_mean(x, h):
    """NaN-aware mean over [t - h, t + h]; NaN where the window has no finite sample."""
    n = len(x); f = np.isfinite(x)
    cv = np.concatenate([[0.0], np.cumsum(np.where(f, x, 0.0))]); cf = np.concatenate([[0.0], np.cumsum(f)])
    idx = np.arange(n); lo = np.maximum(0, idx - h); hi = np.minimum(n, idx + h + 1)
    cnt = cf[hi] - cf[lo]
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(cnt > 0, (cv[hi] - cv[lo]) / np.maximum(cnt, 1), np.nan)


def _since(frames, n, fps, cap):
    """Seconds since the last event frame at or before t (cap when there is none, capped at `cap`)."""
    mark = np.full(n, -1, np.int64)
    if len(frames):
        mark[np.asarray(frames, int)] = np.asarray(frames, int)
    last = np.maximum.accumulate(mark)
    t = np.arange(n)
    return np.where(last >= 0, np.minimum((t - last) / fps, cap), cap)


def _until(frames, n, fps, cap):
    """Seconds until the next event frame at or after t (capped)."""
    rev = (n - 1 - np.asarray(frames, int)) if len(frames) else np.zeros(0, int)
    return _since(rev, n, fps, cap)[::-1]


def team_features(match, touch_params=None):
    """(F, len(TEAM_FEATURES)) float32; all-NaN rows where the ball is absent.

    Per team T in (H, A), the k-th nearest tracked player to the ball (k = 1..3, by distance): distance (m),
    d/dt of that player's distance to the ball (m/s, negative = closing), player speed (m/s), cosine between the
    player's and the ball's velocity. Ball speed and |acceleration| (central differences), cosine between the
    ball velocity and T's attacking direction, seconds since the last / until the next detected touch (any team,
    by H, by A; capped at TOUCH_CAP_S), nearest-team indicator (+1 home nearer, -1 away nearer, 0 tie/none),
    gap d1_A - d1_H (m) and its centred rolling means over +-0.5/1/2 s."""
    fps = float(match.fps); F = match.n_frames
    present = match.ball_present
    X = np.full((F, len(TEAM_FEATURES)), np.nan, np.float32)
    ball = match.ball.astype(np.float64); pos = match.pos.astype(np.float64)
    vb = _central_diff(ball, fps); sb = np.hypot(vb[:, 0], vb[:, 1])
    ab = _central_diff(vb, fps)
    vp = _central_diff(pos, fps); sp = np.hypot(vp[..., 0], vp[..., 1])
    D = np.hypot(pos[..., 0] - ball[:, None, 0], pos[..., 1] - ball[:, None, 1])        # (F, S)
    dD = _central_diff(D, fps)
    with np.errstate(invalid="ignore", divide="ignore"):
        cos = (vp[..., 0] * vb[:, None, 0] + vp[..., 1] * vb[:, None, 1]) / (sp * sb[:, None])
    cos = np.where((sp >= MIN_SPEED) & (sb[:, None] >= MIN_SPEED), cos, np.nan)
    col = {n: i for i, n in enumerate(TEAM_FEATURES)}
    d1 = {}
    rows = np.arange(F)[:, None]
    for code, T in ((0, "H"), (1, "A")):
        sel = np.flatnonzero(match.team == code)
        key = np.where(np.isfinite(D[:, sel]), D[:, sel], np.inf)
        kk = min(K_NEAREST, len(sel))
        order = sel[np.argsort(key, axis=1, kind="stable")[:, :kk]] if kk else np.zeros((F, 0), int)
        for k in range(kk):
            j = order[:, k]
            d = D[rows[:, 0], j]
            ok = np.isfinite(d)
            for q, src in (("dist", D), ("dd_dt", dD), ("speed", sp), ("cos", cos)):
                X[:, col[f"{T}{k + 1}_{q}"]] = np.where(ok, src[rows[:, 0], j], np.nan)
        d1[T] = X[:, col[f"{T}1_dist"]].astype(np.float64)
    X[:, col["ball_speed"]] = sb
    X[:, col["ball_accel"]] = np.hypot(ab[:, 0], ab[:, 1])
    dop = direction_of_play(match)
    with np.errstate(invalid="ignore", divide="ignore"):
        for code, T in ((0, "H"), (1, "A")):
            X[:, col[f"ball_dir_{T}"]] = np.where(sb >= MIN_SPEED, vb[:, 0] * dop[:, code] / sb, np.nan)
    t = detect_touches(match, touch_params or TouchParams())
    fr = t[:, 0].astype(int) if len(t) else np.zeros(0, int)
    tm = t[:, 2] if len(t) else np.zeros(0)
    for name, sel in (("any", np.ones(len(fr), bool)), ("H", tm == 0), ("A", tm == 1)):
        X[:, col[f"touch_since_{name}_s"]] = _since(fr[sel], F, fps, TOUCH_CAP_S)
        X[:, col[f"touch_until_{name}_s"]] = _until(fr[sel], F, fps, TOUCH_CAP_S)
    dH = np.where(np.isfinite(d1["H"]), d1["H"], np.inf); dA = np.where(np.isfinite(d1["A"]), d1["A"], np.inf)
    X[:, col["nearest_team"]] = np.where(dH < dA, 1.0, np.where(dA < dH, -1.0, 0.0))
    gap = d1["A"] - d1["H"]
    X[:, col["nearest_gap_m"]] = gap
    gap_vis = np.where(present, gap, np.nan)
    for w in GAP_WINDOWS_S:
        X[:, col[f"gap_mean_{w:g}s"]] = _centred_mean(gap_vis, int(round(w * fps)))
    X[~present] = np.nan
    return X


FEATURE_POLICIES = {
    "full": (),
    "no_touch": tuple(n for n in TEAM_FEATURES if n.startswith("touch_")),
}


def apply_policy(X, policy):
    """Copy of X with the policy's dropped columns set to 0.0 (all-NaN rows stay all NaN)."""
    Y = np.array(X, np.float32, copy=True)
    keep = ~np.isnan(Y).all(1)
    for n in FEATURE_POLICIES[policy]:
        Y[keep, TEAM_FEATURES.index(n)] = 0.0
    return Y


class TeamCtrlModel:
    """p(home in control | visible ball); `policy` travels with the model (like DeadModel)."""

    def __init__(self, policy="full"):
        if policy not in FEATURE_POLICIES:
            raise KeyError(policy)
        self.policy = policy
        self.const_cols = ()      # columns all-NaN in training: set to 0.0 at fit and predict (HGB cannot bin them)
        self.model = HistGradientBoostingClassifier(max_iter=300, max_leaf_nodes=31, learning_rate=0.1,
                                                    min_samples_leaf=200, random_state=0, early_stopping=False)

    def _prep(self, X):
        Y = apply_policy(X, self.policy)
        keep = ~np.isnan(Y).all(1)
        for j in self.const_cols:
            Y[keep, j] = 0.0
        return Y

    def fit(self, X, y, sample_weight=None):
        """Fit on (X, y) plus the team-swapped copy (mirror(X), 1 - y); all-NaN rows are dropped. y = 1: home."""
        X = np.asarray(X, np.float32); y = np.asarray(y, int)
        ok = ~np.isnan(X).all(1)
        X, y = X[ok], y[ok]
        Xa = np.concatenate([X, mirror_features(X)]); ya = np.concatenate([y, 1 - y])
        sw = None if sample_weight is None else np.tile(np.asarray(sample_weight, float)[ok], 2)
        self.const_cols = tuple(int(j) for j in np.flatnonzero(np.isnan(Xa).all(0)))
        self.model.fit(self._prep(Xa), ya, sample_weight=sw)
        return self

    def predict_proba(self, X):
        """(F,) float32 p_home, symmetrised: 0.5 * (p(x) + 1 - p(mirror x)); NaN on all-NaN rows."""
        X = np.asarray(X, np.float32)
        out = np.full(len(X), np.nan, np.float32)
        ok = ~np.isnan(X).all(1)
        if ok.any():
            p1 = self.model.predict_proba(self._prep(X[ok]))[:, 1]
            p2 = self.model.predict_proba(self._prep(mirror_features(X[ok])))[:, 1]
            out[ok] = 0.5 * (p1 + 1.0 - p2)
        return out

    def save(self, path):
        joblib.dump({"model": self.model, "policy": self.policy, "const_cols": list(self.const_cols)}, path)

    @classmethod
    def load(cls, path):
        obj = joblib.load(path)
        if isinstance(obj, dict):
            m = cls(obj.get("policy", "full")); m.model = obj["model"]; m.const_cols = tuple(obj.get("const_cols", ()))
        else:                                  # bare sklearn model: trained on full features
            m = cls(); m.model = obj
        return m


def team_ctrl_evidence(p_home, eps=1e-3):
    """(F, 3) float32: H = log(p + eps), A = log(1 - p + eps), D = NaN; all-NaN rows where p is NaN."""
    p = np.asarray(p_home, np.float32)
    ev = np.full((len(p), 3), np.nan, np.float32)
    ok = np.isfinite(p)
    ev[ok, 0] = np.log(p[ok] + eps); ev[ok, 1] = np.log(1.0 - p[ok] + eps)
    return ev
