# PathCRF (external tool)

PathCRF (Kim et al. 2026): a players-only model of which node (player or one of four out-of-pitch nodes)
owns the ball. We use it as an evidence stream (`mtah/evidence/pathcrf.py`, stream name `pathcrf`).

- Upstream: https://github.com/hyunsungkim-ds/pathcrf
- Commit: `1b7c6f85a6b78c92e4543fde72ca2b86a933f997`
- Checkpoint: `saved/140` (`saved/140/model/state_dict_best_acc.pt`, trial args `saved/140/args.json`:
  set-transformer + dynamic sparse CRF, team size 11, 5 fps, 10 s windows, trained on Sportec/DFL)
- Licence: MPL-2.0. PathCRF is used **unmodified** as an external tool, run in its own venv through a
  subprocess. No PathCRF code is copied into `mtah/`; `run_infer.py` here is our own wrapper that imports the
  upstream repo from its checkout.

## Install

PathCRF needs torch, which the main venv (Python 3.12) does not have. It runs in a separate venv (Python 3.14.0).
`$PC` below is any directory you choose (the scripts default to `~/mtah-ext`; override with `MTAH_PATHCRF_REPO` and `MTAH_PATHCRF_PY`).

```
git clone https://github.com/hyunsungkim-ds/pathcrf $PC/pathcrf
git -C $PC/pathcrf checkout 1b7c6f85a6b78c92e4543fde72ca2b86a933f997
py -3.14 -m venv $PC/pathcrf-venv
$PC/pathcrf-venv/Scripts/python.exe -m pip install torch==2.11.0 --index-url https://download.pytorch.org/whl/cpu
$PC/pathcrf-venv/Scripts/python.exe -m pip install -r requirements-torch.txt        # from the root of this repository
```

Versions verified working (Python 3.14.0): torch 2.11.0 (the generating machine had the +cu128 build; inference runs on
the CPU), torch-geometric 2.8.0, pandas 3.0.0, numpy 2.4.0, scipy 1.17.0, tqdm 4.67.3, all pinned in
`requirements-torch.txt`. The environment that generated the submission was created with `--system-site-packages` over a
system Python 3.14.0, so the pins list the packages that matter rather than a full freeze; the minimal set was not
bisected. The checkpoint (`saved/140`, 37 MB) ships inside the upstream repository; no separate weights download is needed.

## Use

- Wrapper: `third_party/pathcrf/run_infer.py --in <npz> --out <npz> [--ckpt 140] [--repo <dir>] [--threads N]`,
  run under the PathCRF venv with `TQDM_DISABLE=1`. One half per call. The in-repo file is the source of
  truth; the wrapper is run in place and its sha256 is part of the cache key.
- `mtah.evidence.pathcrf.pathcrf_posterior(match, cache_key, params)` calls it per half and caches the merged
  result in `runs/cache/pathcrf/<cache_key>.npz` (`<sim.key>_s<seed>` for sims, `real_tactical` for the
  real match). Hand-off npz files go to a temporary directory that is deleted after the run. The cache is keyed on the input hash,
  the wrapper hash, checkpoint and threads. The wrapper also writes `n_chunks` and `n_ok` (windows with a prediction);
  `pathcrf_posterior` raises `RuntimeError` when `n_ok / n_chunks < 0.5` for a half.
- Environment overrides of the defaults in `PathCRFParams`:
  - `MTAH_PATHCRF_REPO` - PathCRF checkout (default `~/mtah-ext/pathcrf`)
  - `MTAH_PATHCRF_PY` - PathCRF venv python (default `~/mtah-ext/pathcrf-venv/Scripts/python.exe`)
- Enable with `PipelineConfig.use_pathcrf = True` (or a grid cell with `w_pathcrf > 0`).
- Runtime: about 3.5 min per 45-min half (68 354 frames at 25 fps: 190-205 s measured) at the pinned 4 CPU threads (the code pins 4, see below; about 4 min per simulated half measured), of which ~20 s is
  importing torch/torch_geometric and loading the model.

## Robustness (runs/dfl_extra/pathcrf_robust.json, DFL match 9007, agreement with the ball-owning team on alive frames)

| variant | H1 | H2 |
|---|---|---|
| V0 clean tracking | 92.6 % | 92.0 % |
| V1 corrupted (blackouts, dropouts), no imputation | no prediction | no prediction |
| V2 corrupted + carry-forward imputation | 87.3 % | 75.6 % |
| V3 V2 + identity switches every ~8 s | 68.0 % | 54.1 % |
| V4 V3 with nodes sorted by x | 68.0 % | 54.1 % |

V3 == V4 to every digit: node ordering is immaterial, so the wrapper keeps slot order. Imputed nodes and
identity switches are the dominant damage, which is why the evidence is down-weighted by
`1 - conf_alpha * imputed_frac`.

## Thread count is pinned

The torch thread count is pinned to 4 (`PathCRFParams.threads`; `run_infer.py` calls `torch.set_num_threads` with the
`--threads` value the Python side passes, and nothing else overrides it). The posterior can vary with the thread
count, so it is part of the cache key and is not exposed as a CLI option in `scripts/precompute_pathcrf.py` or
`scripts/select_lomo.py`; this keeps the cached posteriors byte-reproducible. `registration.json` records
`pathcrf_threads: 4`.

## Disclosure: 9007 label exposure

Match 9007 (DFL J03WR9) is the sealed match of the selection pipeline (`select_lomo` never scores it), but its labels
were used for PathCRF-side work before the seal: the clean PathCRF evaluation (92.6 % agreement), the PathCRF
robustness experiment (table above), the choice of the PathCRF output head (`run_infer.py`: "on 15 000 frames of the
corrupted sim half game_9007_H1 s100 it beat the other soft outputs against the ball-owning team", emission marginal
chosen), and the ctrl-truth diagnostic (66 % disagreement on 9007 H1). Any sealed a2 result on 9007 must be reported
with this caveat. The PathCRF integration tests use the label-free nearest-player proxy, not 9007 truth.

## Training-data overlap and the real-feed smoothing (added for the final submission)

- PathCRF was trained by its authors on DFL open-data matches. Our simulator matches 9001-9005 are PathCRF's training matches
  and 9006 its validation match (9007 is the clean test match). PathCRF's accuracy on the simulator folds is therefore optimistic.
- On the real tactical feed PathCRF is run on player positions smoothed with a 1 s centred moving average
  (`PathCRFParams.smooth_s = 1.0`, `make_candidate.py --pathcrf-smooth 1.0`, own cache `runs/cache/pathcrf_fx`). Without it the
  second half is near chance; see `docs/arb/FIX_NOTES.md`. The simulator inputs are not smoothed.
- `third_party/pathcrf/run_infer.py` defaults `--repo` to `MTAH_PATHCRF_REPO` or `~/mtah-ext/pathcrf`; the caller (`mtah/evidence/pathcrf.py`) always passes `--repo` explicitly. The sha256 of this file is part of the PathCRF cache key, so editing it invalidates a cache written by an earlier copy (the cache is only a speed-up; a clean build recomputes it).
