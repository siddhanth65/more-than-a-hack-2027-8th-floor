"""Dead-ball evidence: hand-built restart/blackout features + a gradient-boosted classifier."""
import joblib
import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier

from mtah.io.match import distance_to_line, half_runs
from mtah.sim.blackout_sim import runs_of

DEAD_FEATURES = ("ball_present", "ball_speed", "ball_line_dist", "ball_out", "ball_static_1s",
                 "since_gap_start_s", "until_gap_end_s", "gap_len_s", "entry_line_dist", "entry_outward",
                 "player_speed_ratio", "n_players", "tag_covered", "sp_inside", "sp_dist_s", "sp_type",
                 "team_phase_covered", "touchline_isolated", "corner_cluster", "kickoff_shape")
SP_TYPES = {"throw": 1, "corner": 2, "goal": 3, "free": 4, "kick": 5}


def _rolling_mean(x, w):
    """Trailing mean over the last w samples (NaN-aware)."""
    f = np.isfinite(x); v = np.where(f, x, 0.0)
    cv = np.concatenate([[0.0], np.cumsum(v)]); cf = np.concatenate([[0.0], np.cumsum(f)])
    idx = np.arange(len(x)) + 1; lo = np.maximum(0, idx - w)
    with np.errstate(invalid="ignore", divide="ignore"):
        return (cv[idx] - cv[lo]) / (cf[idx] - cf[lo])


def _span_direction(pos, team):
    out = np.array([1, -1], np.int8)       # fallback when a team is never tracked in the span
    for k in (0, 1):
        x = pos[:, team == k, 0]
        rows = ~np.isnan(x).all(1) if x.shape[1] else np.zeros(len(x), bool)
        if rows.any():
            x = x[rows]
            mn = np.nanmean(np.nanmin(x, axis=1)); mx = np.nanmean(np.nanmax(x, axis=1))
            out[k] = 1 if (-mn) > mx else -1   # deepest player on the negative side -> attacks +x
    return out


def direction_of_play(match):
    """(F, 2) int8: +1 if the team attacks +x, per half (frames outside any half take the nearest span's value)."""
    F = match.n_frames
    spans = [(a, b) for _, a, b in half_runs(match.half)] or [(0, F)]
    out = np.zeros((F, 2), np.int8)
    for a, b in spans:
        out[a:b] = _span_direction(match.pos[a:b], match.team)
    covered = np.zeros(F, bool)
    for a, b in spans:
        covered[a:b] = True
    for f in np.flatnonzero(~covered):
        a, b = min(spans, key=lambda ab: min(abs(f - ab[0]), abs(f - (ab[1] - 1))))
        out[f] = out[a]
    return out


def _sp_code(t):
    t = t.lower() if isinstance(t, str) else ""
    for key, code in SP_TYPES.items():
        if t.startswith(key):
            return code
    return 4


def dead_features(match, tags, fps=None):
    """(F, len(DEAD_FEATURES)) float32 table. sp_dist_s: signed seconds to the nearest set-piece start,
    positive = the next set-piece start is ahead, negative = it already began (clipped to +-60 s)."""
    fps = fps or match.fps; F = match.n_frames; L, W = match.pitch
    X = np.full((F, len(DEAD_FEATURES)), np.nan, np.float32)
    present = match.ball_present
    X[:, 0] = present
    X[:, 1] = match.ball_speed
    ld = distance_to_line(match.ball, match.pitch); X[:, 2] = ld
    X[:, 3] = np.where(present, ld < -0.5, 0)
    slow = np.where(present, match.ball_speed < 0.5, False).astype(float)
    X[:, 4] = _rolling_mean(slow, int(fps)) >= 0.999
    X[:, 5] = 0; X[:, 6] = 0; X[:, 7] = 0; X[:, 8] = np.nan; X[:, 9] = np.nan
    for s, e in runs_of(~present):
        t = np.arange(s, e)
        X[s:e, 5] = (t - s) / fps; X[s:e, 6] = (e - 1 - t) / fps; X[s:e, 7] = (e - s) / fps
        k = s - 1
        while k >= 0 and not present[k]:
            k -= 1
        if k >= 0:
            X[s:e, 8] = ld[k]
            j = k - 5
            if j >= 0 and present[j]:
                X[s:e, 9] = float(ld[k] < ld[j])
    on = ~np.isnan(match.pos[..., 0])
    ok = on & np.isfinite(match.speed)
    cnt = ok.sum(1)
    msp = np.where(cnt > 0, np.where(ok, np.nan_to_num(match.speed), 0).sum(1) / np.maximum(cnt, 1), np.nan)
    if F > int(fps):
        recent = _rolling_mean(msp, int(fps)); earlier = _rolling_mean(np.r_[np.full(int(fps), np.nan), msp[:-int(fps)]], int(5 * fps))
        with np.errstate(all="ignore"):
            ratio = recent / earlier
        X[:, 10] = np.where(np.isfinite(ratio), ratio, np.nan)
    X[:, 11] = on.sum(1)
    cov = np.zeros(F, bool); team_cov = np.zeros(F, bool); sp_in = np.zeros(F, bool); sp_type = np.zeros(F, np.float32)
    sp_starts = []
    for r in tags.itertuples(index=False):
        a = max(0, int(r.start)); b = min(F - 1, int(r.end))
        if b < a:
            continue
        cov[a:b + 1] = True
        if r.code == "SET PIECES":
            sp_in[a:b + 1] = True; sp_type[a:b + 1] = _sp_code(r.type); sp_starts.append(a)
        elif r.team in ("H", "A"):
            team_cov[a:b + 1] = True
    X[:, 12] = cov; X[:, 13] = sp_in; X[:, 15] = sp_type; X[:, 16] = team_cov
    if sp_starts:
        st = np.sort(np.array(sp_starts)); t = np.arange(F)
        i = np.searchsorted(st, t); prev = st[np.clip(i - 1, 0, len(st) - 1)]; nxt = st[np.clip(i, 0, len(st) - 1)]
        d_prev = t - prev; d_next = nxt - t
        d = np.where(np.abs(d_next) <= np.abs(d_prev), d_next, -d_prev).astype(float) / fps
        X[:, 14] = np.clip(d, -60, 60)
    else:
        X[:, 14] = 60
    # shapes
    px, py = match.pos[..., 0], match.pos[..., 1]
    near_tl = (np.abs(py) > W / 2 - 1.5) & on
    iso = np.zeros(F, bool)
    for f in np.flatnonzero(near_tl.any(1)):
        for j in np.flatnonzero(near_tl[f]):
            d = np.hypot(px[f] - px[f, j], py[f] - py[f, j]); d[j] = np.inf
            if np.nanmin(np.where(on[f], d, np.inf)) >= 8.0:
                iso[f] = True; break
    X[:, 17] = iso
    corners = np.array([[L / 2, W / 2], [L / 2, -W / 2], [-L / 2, W / 2], [-L / 2, -W / 2]])
    cc = np.zeros(F)
    for c in corners:
        cc = np.maximum(cc, ((np.hypot(px - c[0], py - c[1]) <= 20) & on).sum(1))
    X[:, 18] = cc
    dop = direction_of_play(match)
    own_half = np.ones(F, bool)
    for k in (0, 1):
        sel = match.team == k
        own = (px[:, sel] * dop[:, k:k + 1] < 0) | ~on[:, sel]     # attacking +x -> own half is x < 0
        own_half &= own.all(1)
    X[:, 19] = own_half & (on.sum(1) >= 16)
    return X


# ---- long-context, ball-independent player-motion features (Plan 2e T1) ----
# Computed from players only (match.pos / match.speed / match.team), per half (half_runs; the whole match when no
# half is graded; frames outside every half stay NaN). Per-frame quantities are NaN when fewer than
# LONG_MIN_PLAYERS players are visible (n_visible excepted: it is the raw count); each is then averaged over a
# centred window of +-w s (NaN frames ignored; NaN when the window holds no valid frame), clipped at the half edges.
LONG_WINDOWS_S = (2, 5, 10)
LONG_MIN_PLAYERS = 6
LONG_SLOW_MPS = 1.0
LONG_BASE = ("lc_speed",          # mean player speed (m/s)
             "lc_slow_share",     # share of visible players slower than LONG_SLOW_MPS
             "lc_accel",          # mean |speed change| (m/s^2) over players tracked on both frames
             "lc_team_spread",    # mean over teams (>= 2 visible players) of (std x + std y) / 2
             "lc_centroid_dist",  # distance between the two team centroids
             "lc_compactness",    # mean distance of the visible players to their own team centroid
             "lc_n_visible")      # number of visible players
# Column order: window-major (all LONG_BASE at +-2 s, then +-5 s, then +-10 s), then the two onset trends
# (+-2 s value minus +-10 s value) for mean speed and slow share.
LONG_FEATURES = (tuple(f"{n}_{w}s" for w in LONG_WINDOWS_S for n in LONG_BASE)
                 + ("lc_speed_trend", "lc_slow_trend"))

_NO_BALL_AT_ALL = ("ball_speed", "ball_line_dist", "ball_out", "ball_static_1s", "ball_present",
                   "entry_line_dist", "entry_outward")
FEATURE_POLICIES = {
    "full": (),
    "no_ball_cues": ("ball_speed", "ball_line_dist", "ball_out", "ball_static_1s"),
    "no_ball_at_all": _NO_BALL_AT_ALL,
    "long": (),                       # full + LONG_FEATURES
    "long_no_ball": _NO_BALL_AT_ALL,  # no_ball_at_all + LONG_FEATURES
}
LONG_POLICIES = frozenset({"long", "long_no_ball"})


def policy_feature_names(policy):
    """Column names of the feature table a model with this policy is trained on / predicts from:
    DEAD_FEATURES for the base policies, DEAD_FEATURES + LONG_FEATURES for LONG_POLICIES."""
    if policy not in FEATURE_POLICIES:
        raise KeyError(policy)
    return DEAD_FEATURES + LONG_FEATURES if policy in LONG_POLICIES else DEAD_FEATURES


def apply_policy(X, policy):
    """Copy of X with the policy's dropped columns set to 0.0 (constant: HGB cannot take all-NaN columns).
    X must be the policy's table (policy_feature_names(policy) columns)."""
    names = policy_feature_names(policy)
    if X.shape[1] != len(names):
        raise ValueError(f"policy {policy!r} expects {len(names)} feature columns, got {X.shape[1]}")
    Y = np.array(X, np.float32, copy=True)
    for n in FEATURE_POLICIES[policy]:
        Y[:, names.index(n)] = 0.0
    return Y


def _centred_mean(x, w):
    """Centred mean over [t - w, t + w] clipped to the array (NaN-aware; NaN when no finite sample)."""
    x = np.asarray(x, np.float64); n = len(x)
    f = np.isfinite(x)
    cv = np.concatenate([[0.0], np.cumsum(np.where(f, x, 0.0))]); cf = np.concatenate([[0], np.cumsum(f)])
    t = np.arange(n); lo = np.maximum(0, t - w); hi = np.minimum(n, t + w + 1)
    cnt = cf[hi] - cf[lo]
    out = np.full(n, np.nan)
    ok = cnt > 0
    out[ok] = (cv[hi] - cv[lo])[ok] / cnt[ok]
    return out


def _frame_motion(pos, speed, team, fps):
    """Per-frame LONG_BASE quantities for one contiguous span: dict name -> (n,) float64."""
    n = len(pos)
    pos = pos.astype(np.float64); speed = speed.astype(np.float64)
    on = np.isfinite(pos[..., 0]) & np.isfinite(pos[..., 1])
    nvis = on.sum(1)
    enough = nvis >= LONG_MIN_PLAYERS
    ok = on & np.isfinite(speed)
    cnt = ok.sum(1); has = enough & (cnt > 0)
    sp = np.where(ok, speed, 0.0)
    out = {k: np.full(n, np.nan) for k in LONG_BASE}
    out["lc_speed"][has] = sp.sum(1)[has] / cnt[has]
    out["lc_slow_share"][has] = (ok & (sp < LONG_SLOW_MPS)).sum(1)[has] / cnt[has]
    if n > 1:
        okd = ok[1:] & ok[:-1]; cd = okd.sum(1)
        dsum = np.where(okd, np.abs(sp[1:] - sp[:-1]) * fps, 0.0).sum(1)
        g = (cd > 0) & enough[1:]
        a = out["lc_accel"][1:]
        a[g] = dsum[g] / cd[g]
    x = np.where(on, pos[..., 0], 0.0); y = np.where(on, pos[..., 1], 0.0)
    cx = np.full((n, 2), np.nan); cy = np.full((n, 2), np.nan); spread = np.zeros((n, 2)); nk = np.zeros((n, 2), int)
    dist_sum = np.zeros(n); dist_cnt = np.zeros(n)
    for k in (0, 1):
        sel = team == k
        m = on[:, sel]; c = m.sum(1); nk[:, k] = c
        xs, ys = x[:, sel], y[:, sel]
        g = c > 0
        cx[g, k] = xs[g].sum(1) / c[g]; cy[g, k] = ys[g].sum(1) / c[g]
        g2 = c >= 2
        vx = np.maximum(0.0, (xs[g2] ** 2).sum(1) / c[g2] - cx[g2, k] ** 2)
        vy = np.maximum(0.0, (ys[g2] ** 2).sum(1) / c[g2] - cy[g2, k] ** 2)
        spread[g2, k] = (np.sqrt(vx) + np.sqrt(vy)) / 2
        if g.any():
            d = np.hypot(xs[g] - cx[g, k:k + 1], ys[g] - cy[g, k:k + 1])
            dist_sum[g] += np.where(m[g], d, 0.0).sum(1); dist_cnt[g] += c[g]
    nsp = (nk >= 2).sum(1)
    g = enough & (nsp > 0)
    out["lc_team_spread"][g] = spread.sum(1)[g] / nsp[g]
    g = enough & (nk >= 1).all(1)
    out["lc_centroid_dist"][g] = np.hypot(cx[g, 0] - cx[g, 1], cy[g, 0] - cy[g, 1])
    g = enough & (dist_cnt > 0)
    out["lc_compactness"][g] = dist_sum[g] / dist_cnt[g]
    out["lc_n_visible"] = nvis.astype(np.float64)
    return out


def long_features(match, fps=None):
    """(F, len(LONG_FEATURES)) float32 long-context player-motion table (see LONG_FEATURES for the column order)."""
    fps = fps or match.fps; F = match.n_frames
    X = np.full((F, len(LONG_FEATURES)), np.nan, np.float32)
    spans = [(a, b) for _, a, b in half_runs(match.half)] or [(0, F)]
    col = {n: i for i, n in enumerate(LONG_FEATURES)}
    lo, hi = LONG_WINDOWS_S[0], LONG_WINDOWS_S[-1]
    for a, b in spans:
        q = _frame_motion(match.pos[a:b], match.speed[a:b], np.asarray(match.team), fps)
        for w in LONG_WINDOWS_S:
            hw = int(round(w * fps))
            for name in LONG_BASE:
                X[a:b, col[f"{name}_{w}s"]] = _centred_mean(q[name], hw)
        X[a:b, col["lc_speed_trend"]] = X[a:b, col[f"lc_speed_{lo}s"]] - X[a:b, col[f"lc_speed_{hi}s"]]
        X[a:b, col["lc_slow_trend"]] = X[a:b, col[f"lc_slow_share_{lo}s"]] - X[a:b, col[f"lc_slow_share_{hi}s"]]
    return X


def dead_features_long(match, tags, fps=None):
    """(F, len(DEAD_FEATURES) + len(LONG_FEATURES)) float32: the base table, then the long-context columns."""
    return np.hstack([dead_features(match, tags, fps), long_features(match, fps)])


def dead_table(match, tags, policy, fps=None):
    """The feature table a dead model with `policy` reads: dead_features_long for LONG_POLICIES, else dead_features."""
    return dead_features_long(match, tags, fps) if policy in LONG_POLICIES else dead_features(match, tags, fps)


class DeadModel:
    def __init__(self, policy="full"):
        if policy not in FEATURE_POLICIES:
            raise KeyError(policy)
        self.policy = policy
        self.feature_names = policy_feature_names(policy)   # the table it reads: dead_table(match, tags, policy)
        self.model = HistGradientBoostingClassifier(max_iter=300, max_leaf_nodes=31, learning_rate=0.1,
                                                    min_samples_leaf=200, random_state=0, early_stopping=False)

    def features(self, match, tags):
        """The feature table this model reads (base or base + long, by policy)."""
        return dead_table(match, tags, self.policy)

    def fit(self, X, y, sample_weight=None):
        self.model.fit(apply_policy(X, self.policy), y, sample_weight=sample_weight); return self

    def predict_proba(self, X):
        return self.model.predict_proba(apply_policy(X, self.policy))[:, 1].astype(np.float32)

    def save(self, path):
        joblib.dump({"model": self.model, "policy": self.policy}, path)

    @classmethod
    def load(cls, path):
        obj = joblib.load(path)
        if isinstance(obj, dict):
            m = cls(obj.get("policy", "full")); m.model = obj["model"]
        else:                                  # old format: bare sklearn model, trained on full features
            m = cls(); m.model = obj
        return m


def dead_evidence(p_dead, eps=1e-3):
    p = np.asarray(p_dead, np.float32)
    ev = np.empty((len(p), 3), np.float32)
    ev[:, 2] = np.log(p + eps); ev[:, 0] = np.log(1 - p + eps); ev[:, 1] = ev[:, 0]
    return ev
