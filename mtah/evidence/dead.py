"""Dead-ball evidence: hand-built restart/blackout features + a gradient-boosted classifier."""
from dataclasses import dataclass

import joblib
import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier

from mtah.io.match import distance_to_line
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


def direction_of_play(match):
    out = np.zeros(2, np.int8)
    for k in (0, 1):
        x = match.pos[:, match.team == k, 0]
        with np.errstate(all="ignore"):
            mn = np.nanmean(np.nanmin(x, axis=1)); mx = np.nanmean(np.nanmax(x, axis=1))
        out[k] = 1 if (-mn) > mx else -1   # deepest player on the negative side -> attacks +x
    return out


def _sp_code(t):
    t = (t or "").lower()
    for key, code in SP_TYPES.items():
        if t.startswith(key):
            return code
    return 4


def dead_features(match, tags, fps=None):
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
    with np.errstate(all="ignore"):
        msp = np.nanmean(np.where(on, match.speed, np.nan), axis=1)
    recent = _rolling_mean(msp, int(fps)); earlier = _rolling_mean(np.r_[np.full(int(fps), np.nan), msp[:-int(fps)]], int(5 * fps))
    with np.errstate(all="ignore"):
        X[:, 10] = recent / earlier
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
        own = (px[:, sel] * dop[k] < 0) | ~on[:, sel]     # attacking +x -> own half is x < 0
        own_half &= own.all(1)
    X[:, 19] = own_half & (on.sum(1) >= 16)
    return X


class DeadModel:
    def __init__(self):
        self.model = HistGradientBoostingClassifier(max_iter=300, max_leaf_nodes=31, learning_rate=0.1,
                                                    min_samples_leaf=200, random_state=0, early_stopping=False)

    def fit(self, X, y, sample_weight=None):
        self.model.fit(X, y, sample_weight=sample_weight); return self

    def predict_proba(self, X):
        return self.model.predict_proba(X)[:, 1].astype(np.float32)

    def save(self, path):
        joblib.dump(self.model, path)

    @classmethod
    def load(cls, path):
        m = cls(); m.model = joblib.load(path); return m


def dead_evidence(p_dead, eps=1e-3):
    p = np.asarray(p_dead, np.float32)
    ev = np.empty((len(p), 3), np.float32)
    ev[:, 2] = np.log(p + eps); ev[:, 0] = np.log(1 - p + eps); ev[:, 1] = ev[:, 0]
    return ev
