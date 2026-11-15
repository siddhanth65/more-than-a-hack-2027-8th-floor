"""PathCRF players-only team posterior for ONE half (run under the PathCRF venv, not the mtah venv).

    python run_infer.py --in half.in.npz --out half.out.npz [--ckpt 140] [--repo <pathcrf dir>] [--threads N]

Input npz: pos (F,S,2) float32 metres centred on the pitch (x along length, NaN = missing), team (S,) int8
(0 home / 1 away), fps, optional pitch (L, W). Output npz, all (F,):
  p_home       float32  P(home | a player owns the ball) = home / (home + away) mass of the source-node
                        marginal of PathCRF's per-frame edge posterior (softmax emissions, use_crf=False);
                        NaN = no prediction
  p_none       float32  source-node mass on the four out-of-pitch nodes; NaN = no prediction
  imputed_frac float32  share of the 22 node slots whose position is imputed on that frame (unfilled = imputed)
  chunk_id     int32    10 s window the frame belongs to (-1 = none)
  n_chunks, n_ok  int32 scalars: windows attempted / windows with a prediction

Why the emission marginal: on 15 000 frames of the corrupted sim half game_9007_H1 s100 it beat the other soft
outputs against the ball-owning team (acc / mean log-lik: emission marginal 0.885 / -0.340, macro heads
poss_prev 0.849 / -0.366, mean of poss_prev and poss_next 0.833 / -0.377; the CRF-decoded hard edge_src scores
0.898 but is 0/1, log-lik -0.70, -0.50 after a 1 s box filter), at the cost of a single model pass.

The whole half is one in-play episode (the ball is unknown), cut into 10 s windows (PathCRF's training window;
a short remainder is merged into the previous window). Missing positions are imputed (hold last position,
extrapolate at the last velocity for 1 s, then hold; back-fill before the first sighting). Per window the 11
slots per team with the most real (not imputed) frames become nodes, in slot order (node order is immaterial:
runs/dfl_extra/pathcrf_robust.json V3 == V4); keeper = the node deepest towards its own goal.
Self-contained on purpose (numpy/pandas/torch + the PathCRF repo); prints only counts and timings.
PathCRF (Kim et al. 2026) is MPL-2.0 and used unmodified; see SETUP.md.
"""
import argparse
import contextlib
import io
import os
import sys
import time

import numpy as np

os.environ.setdefault("TQDM_DISABLE", "1")
CHUNK_S = 10.0
TEAM_SIZE = 11
PITCH = (105.0, 68.0)   # PathCRF coordinates: corner origin, 105 x 68 m


def impute(pos, fps=25.0):
    """Fill NaN slots: last position + last velocity for 1 s, then hold; back-fill the start; clip to pitch."""
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
        v[np.hypot(v[:, 0], v[:, 1]) > 9] = 0
        dt = np.minimum(ar - last, int(fps))  # velocity continues 1 s, then 0
        fill = ~valid & (last >= 0)
        pos[fill, s] = pos[li[fill], s] + v[fill] * (dt[fill, None] / fps)
        pos[:first, s] = pos[first, s]
    pos[..., 0] = np.clip(pos[..., 0], -PITCH[0] / 2, PITCH[0] / 2)
    pos[..., 1] = np.clip(pos[..., 1], -PITCH[1] / 2, PITCH[1] / 2)
    return pos


def chunks_of(F, chunk):
    """[(a, b)] windows of `chunk` frames covering [0, F); a remainder < chunk/2 is merged into the last one."""
    out = []
    for a in range(0, F, chunk):
        b = min(F, a + chunk)
        if F - b < chunk // 2:
            b = F
        out.append((a, b))
        if b == F:
            break
    return out


def load_model(repo, ckpt):
    import torch
    from models.utils import build_model, load_trial_args, resolve_model_path
    d = f"saved/{ckpt:03d}"
    ta = load_trial_args(d)
    model = build_model(ta, device="cpu")
    model.load_state_dict(torch.load(resolve_model_path(d, "state_dict_best_acc.pt"), map_location="cpu",
                                     weights_only=False))
    model.eval()
    return model, ta


def run(pos, team, fps, model, window_seconds):
    import pandas as pd
    import torch
    from datatools import utils
    from inference import inference_episode

    F = len(pos)
    miss = np.isnan(pos[:, :, 0])
    pos = impute(pos, fps)
    seen = ~np.isnan(pos[:, :, 0]).any(0)       # never-tracked slots stay NaN and are never chosen
    left_team = 0 if np.nanmean(pos[:, team == 0, 0]) < np.nanmean(pos[:, team == 1, 0]) else 1
    names = ("home", "away")
    out = {"p_home": np.full(F, np.nan, np.float32), "p_none": np.full(F, np.nan, np.float32),
           "imputed_frac": np.ones(F, np.float32), "chunk_id": np.full(F, -1, np.int32)}
    chunks = chunks_of(F, int(round(CHUNK_S * fps)))
    n_ok, n_failed, row_err = 0, 0, 0.0
    for ci, (a, b) in enumerate(chunks):
        out["chunk_id"][a:b] = ci
        seg, mseg = pos[a:b], miss[a:b]
        cols, gk, n_imp = {}, {}, np.zeros(b - a)
        for k in (0, 1):
            sl = np.flatnonzero((team == k) & seen)
            cnt = (~mseg[:, sl]).sum(0)
            sel = np.sort(sl[np.argsort(-cnt, kind="stable")[:TEAM_SIZE]])
            n_imp += mseg[:, sel].sum(1) + (TEAM_SIZE - len(sel))
            xs = []
            for j, s in enumerate(sel):
                nm = f"{names[k]}_{j:02d}"
                cols[nm + "_x"] = seg[:, s, 0] + PITCH[0] / 2
                cols[nm + "_y"] = seg[:, s, 1] + PITCH[1] / 2
                xs.append(seg[:, s, 0].mean())
            if not xs:
                break
            gk[k] = f"{names[k]}_{(int(np.argmin(xs)) if k == left_team else int(np.argmax(xs))):02d}"
        out["imputed_frac"][a:b] = n_imp / (2 * TEAM_SIZE)
        if len(gk) < 2:
            continue
        n = b - a
        df = pd.DataFrame({"frame_id": np.arange(a, b), "period_id": 1, "timestamp": np.arange(n) / fps,
                           "phase_id": 1, "episode_id": 1, "ball_state": "alive", "ball_owning_team_id": "home"})
        df = pd.concat([df, pd.DataFrame(cols), pd.DataFrame({"ball_x": np.nan, "ball_y": np.nan}, index=df.index)],
                       axis=1)
        try:
            with contextlib.redirect_stdout(io.StringIO()):
                df = utils.calculate_running_features(df, fps=fps)
            df = df.set_index("frame_id")
            with torch.no_grad():
                _, _, micro, _, _ = inference_episode(
                    model, df, left_gk=gk[left_team], right_gk=gk[1 - left_team],
                    use_crf=False, decode="indep", evaluate=False,
                    window_seconds=window_seconds, fps=fps, sample_freq=max(1, int(round(fps / 5))),
                    device="cpu")
        except ValueError as e:     # e.g. window shorter than the smoothing filter: no prediction
            n_failed += 1
            print(f"pathcrf: chunk {ci} failed: {type(e).__name__}", flush=True)
            continue
        if micro is None:
            continue
        P = micro.reindex(index=df.index).to_numpy(np.float64)          # (n, 26*26) edge "src-dst" softmax
        cn = np.array([str(c).split("-")[0] for c in micro.columns])    # source node of each edge
        real = np.char.find(cn, "_pad_") < 0                             # padded nodes carry no team
        home = P[:, np.char.startswith(cn, "home_") & real].sum(1)
        away = P[:, np.char.startswith(cn, "away_") & real].sum(1)
        none = P[:, np.char.startswith(cn, "out_")].sum(1)
        row_err = max(row_err, float(np.abs(home + away + none - 1).max()))
        with np.errstate(invalid="ignore", divide="ignore"):
            out["p_home"][a:b] = home / (home + away)
        out["p_none"][a:b] = none
        n_ok += 1
    return out, len(chunks), n_ok, row_err


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--ckpt", type=int, default=140)
    ap.add_argument("--repo", default=os.environ.get("MTAH_PATHCRF_REPO", os.path.join(os.path.expanduser("~"), "mtah-ext", "pathcrf")))
    ap.add_argument("--threads", type=int, default=4)
    a = ap.parse_args()
    inp, outp = os.path.abspath(a.inp), os.path.abspath(a.out)
    t0 = time.time()
    with np.load(inp) as z:
        pos = z["pos"].astype(np.float32)
        team = z["team"].astype(np.int8)
        fps = float(z["fps"])
        pitch = tuple(float(v) for v in z["pitch"]) if "pitch" in z.files else PITCH
    pos = pos * np.array([PITCH[0] / pitch[0], PITCH[1] / pitch[1]], np.float32)   # to PathCRF's 105 x 68
    sys.path.insert(0, a.repo)
    os.chdir(a.repo)
    import torch
    torch.set_num_threads(a.threads)
    model, ta = load_model(a.repo, a.ckpt)
    out, n_chunks, n_ok, row_err = run(pos, team, fps, model, ta.get("window_seconds"))
    np.savez(outp, n_chunks=np.int32(n_chunks), n_ok=np.int32(n_ok), **out)
    print(f"pathcrf: frames={len(pos)} chunks={n_chunks} predicted_chunks={n_ok} "
          f"nopred={np.isnan(out['p_home']).mean():.3f} imputed_frac={out['imputed_frac'].mean():.3f} "
          f"p_none={np.nanmean(out['p_none']) if n_ok else float('nan'):.3f} max|rowsum-1|={row_err:.1e} "
          f"t={time.time() - t0:.1f}s", flush=True)


if __name__ == "__main__":
    main()
