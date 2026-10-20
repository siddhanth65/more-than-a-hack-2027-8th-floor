"""PathCRF robustness on corrupted DFL tracking (aggregates only).
Usage (pathcrf venv): python experiment_pathcrf_robust.py run <V0..V4> <H1|H2>
                      python experiment_pathcrf_robust.py merge   (any venv with numpy)
Variants: V0 clean | V1 corrupted, no imputation | V2 +carry-forward imputation |
          V3 +identity switches every ~8 s | V4 = V3 data, nodes sorted by x at chunk start.
Each alive episode is cut into ~10 s chunks (the training window of PathCRF); per chunk the 11 most-present
slots per team become nodes (two-digit names so the sorted() node order of PathCRF = our order)."""
import contextlib
import io
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

os.environ["TQDM_DISABLE"] = "1"
POSS = Path(__file__).resolve().parents[1]
OUT = POSS / "runs/dfl_extra"
REPO = os.environ.get("MTAH_PATHCRF_REPO", "HOME/mtah-ext/pathcrf")
CHUNK = 250
SEED = 100


def impute(pos, fps=25.0):
    pos = pos.copy()
    F, S, _ = pos.shape
    ar = np.arange(F)
    for s in range(S):
        valid = ~np.isnan(pos[:, s, 0])
        if not valid.any():
            continue
        last = np.maximum.accumulate(np.where(valid, ar, -1))
        first = int(np.flatnonzero(valid)[0])
        k = 5
        li = np.clip(last, 0, None)
        prev = np.clip(last - k, 0, None)
        okv = (last >= k) & valid[prev] & valid[li]
        v = np.zeros((F, 2), np.float32)
        v[okv] = (pos[li[okv], s] - pos[prev[okv], s]) / k * fps
        sp = np.hypot(v[:, 0], v[:, 1])
        v[sp > 9] = 0
        dt = np.minimum(ar - last, int(fps))  # velocity continues 1 s, then 0
        fill = ~valid & (last >= 0)
        pos[fill, s] = pos[li[fill], s] + v[fill] * (dt[fill, None] / fps)
        pos[:first, s] = pos[first, s]
    pos[..., 0] = np.clip(pos[..., 0], -52.5, 52.5)
    pos[..., 1] = np.clip(pos[..., 1], -34, 34)
    return pos


def fragment(pos, team, rng, fps=25.0, mean_s=8.0):
    F, S, _ = pos.shape
    smap = np.tile(np.arange(S), (F, 1))
    cps = np.flatnonzero(rng.random(F) < 1.0 / (mean_s * fps))
    cur = np.arange(S)
    have = ~np.isnan(pos[:, :, 0]).all(0)  # never-tracked slots are not part of the anonymous pool
    prev = 0
    for c in list(cps) + [F]:
        smap[prev:c] = cur
        if c < F:
            new = cur.copy()
            for k in (0, 1):
                idx = np.flatnonzero((team == k) & have)
                new[idx] = cur[rng.permutation(idx)]
            cur = new
        prev = c
    return np.take_along_axis(pos, smap[:, :, None], axis=1), len(cps)


def chunks_of(alive):
    idx = np.flatnonzero(alive)
    eps = []
    if len(idx):
        br = np.flatnonzero(np.diff(idx) >= 5)
        st = np.concatenate([[0], br + 1])
        en = np.concatenate([br, [len(idx) - 1]])
        eps = [(int(idx[a]), int(idx[b]) + 1) for a, b in zip(st, en)]
    out = []
    for s, e in eps:
        for a in range(s, e, CHUNK):
            b = min(e, a + CHUNK)
            if e - b < CHUNK // 2:
                b = e
            out.append((a, b))
            if b == e:
                break
    return out


def run(variant, half):
    import pandas as pd
    import torch
    sys.path.insert(0, str(POSS))
    sys.path.insert(0, REPO)
    os.chdir(REPO)
    from mtah.io.dfl_proxy import load_half
    from mtah.sim import half_rng
    from mtah.sim.dataset import half_index
    from mtah.sim.blackout_sim import BlackoutParams, blackout_mask, apply_blackouts
    from mtah.sim.corrupt import CorruptParams, corrupt
    from datatools import utils
    from models.utils import build_model, load_trial_args, resolve_model_path
    from inference import inference_episode
    torch.set_num_threads(int(os.environ.get("NT", "2")))
    key = f"game_9007_{half}"
    t_all = time.time()
    m, labels = load_half(OUT / f"{key}.npz", OUT / "fp1d_census.json")
    bp = BlackoutParams.from_dict(json.loads((POSS / "configs/blackout_sim.json").read_text()))
    rng = half_rng(SEED, half_index(key))
    mask = blackout_mask(labels, m.ball, m.pitch, m.fps, bp, rng)
    pos = m.pos
    if variant != "V0":
        sim = corrupt(apply_blackouts(m, mask), mask, CorruptParams(), rng)
        pos = sim.pos
    n_cp = 0
    if variant in ("V2", "V3", "V4"):
        pos = impute(pos)
    if variant in ("V3", "V4"):
        pos, n_cp = fragment(pos, m.team, np.random.default_rng([SEED, 7, half_index(key)]))
    t_prep = time.time() - t_all
    team = m.team
    F = len(labels)
    ta = load_trial_args("saved/140")
    model = build_model(ta, device="cpu")
    model.load_state_dict(torch.load(resolve_model_path("saved/140", "state_dict_best_acc.pt"),
                                     map_location="cpu", weights_only=False))
    model.eval()
    left_team = 0 if np.nanmean(pos[:, team == 0, 0]) < np.nanmean(pos[:, team == 1, 0]) else 1
    names = ["home", "away"]
    pred = np.full(F, "X", dtype="<U1")  # X = no prediction
    chunks = chunks_of(labels != "D")
    t0 = time.time()
    for (a, b) in chunks:
        seg = pos[a:b]
        cols, gk = {}, {}
        for k in (0, 1):
            sl = np.flatnonzero(team == k)
            cnt = (~np.isnan(seg[:, sl, 0])).sum(0)
            o = np.argsort(-cnt, kind="stable")[:11]
            sel = sl[o][cnt[o] > 0]
            if variant == "V4":
                x0 = np.array([np.nanmean(seg[:5, s, 0]) if (~np.isnan(seg[:5, s, 0])).any() else 1e9 for s in sel])
                sel = sel[np.argsort(x0, kind="stable")]
            else:
                sel = np.sort(sel)
            xs = []
            for j, s in enumerate(sel):
                nm = f"{names[k]}_{j:02d}"
                cols[nm + "_x"] = seg[:, s, 0] + 52.5
                cols[nm + "_y"] = seg[:, s, 1] + 34.0
                xs.append(np.nanmean(seg[:, s, 0]) if (~np.isnan(seg[:, s, 0])).any() else np.nan)
            xs = np.array(xs)
            gk[k] = f"{names[k]}_{(np.nanargmin(xs) if k == left_team else np.nanargmax(xs)):02d}"
        n = b - a
        df = pd.DataFrame({"frame_id": np.arange(a, b), "period_id": 1, "timestamp": np.arange(n) / 25.0,
                           "phase_id": 1, "episode_id": 1, "ball_state": "alive", "ball_owning_team_id": "home"})
        df = pd.concat([df, pd.DataFrame(cols), pd.DataFrame({"ball_x": np.nan, "ball_y": np.nan}, index=df.index)], axis=1)
        try:
            with contextlib.redirect_stdout(io.StringIO()):
                df = utils.calculate_running_features(df, fps=25)
        except ValueError:
            continue
        df = df.set_index("frame_id")
        try:
            res = inference_episode(model, df, left_gk=gk[left_team], right_gk=gk[1 - left_team],
                                    use_crf=ta.get("crf_weight", 0) > 0, decode="indep", evaluate=False,
                                    window_seconds=ta.get("window_seconds"), device="cpu")
        except ValueError:  # chunk shorter than the savgol window: no prediction
            continue
        micro = res[2]
        if micro is None:
            continue
        s = np.array([str(v) for v in micro["edge_src"].values], dtype="<U12")
        pred[a:b] = np.where(np.char.startswith(s, "home_"), "H", np.where(np.char.startswith(s, "away_"), "A", "N"))
    t_inf = time.time() - t0
    alive = labels != "D"

    def agree(sel):
        sel = sel & alive
        got = sel & (pred != "X")
        return {"agree": float((pred[got] == labels[got]).mean()) if got.any() else None,
                "agree_nopred_wrong": float((pred[sel] == labels[sel]).mean()) if sel.any() else None,
                "nopred": float((pred[sel] == "X").mean()) if sel.any() else None, "n": int(sel.sum())}

    out = {"variant": variant, "half": half, "all": agree(np.ones(F, bool)), "in_blackout": agree(mask),
           "outside": agree(~mask), "blackout_share_alive": float(mask[alive].mean()), "chunks": len(chunks),
           "identity_switch_points": int(n_cp), "t_prep_s": round(t_prep, 1), "t_pathcrf_s": round(t_inf, 1),
           "t_total_s": round(time.time() - t_all, 1)}
    (OUT / f"pathcrf_robust_{variant}_{half}.json").write_text(json.dumps(out))
    print(json.dumps(out))


def merge():
    res = {}
    for p in sorted(OUT.glob("pathcrf_robust_V?_H?.json")):
        d = json.loads(p.read_text())
        res[f"{d['variant']}_{d['half']}"] = d
    (OUT / "pathcrf_robust.json").write_text(json.dumps(res, indent=1))

    def f(x):
        return "  n/a" if x is None else f"{100 * x:5.1f}"
    for k, d in res.items():
        print(k, f(d["all"]["agree"]), f(d["in_blackout"]["agree"]), f(d["outside"]["agree"]),
              f"nopred {100 * d['all']['nopred']:.1f}%", f"{d['t_total_s']}s")


if __name__ == "__main__":
    run(sys.argv[2], sys.argv[3]) if sys.argv[1] == "run" else merge()
