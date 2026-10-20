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

PathCRF needs torch, which the mtah venv (Python 3.12) does not have. It runs in a separate venv:

```
git clone https://github.com/hyunsungkim-ds/pathcrf HOME/mtah-ext\pathcrf
git -C HOME/mtah-ext\pathcrf checkout 1b7c6f85a6b78c92e4543fde72ca2b86a933f997
py -3.14 -m venv HOME/mtah-ext\pathcrf-venv
HOME/mtah-ext\pathcrf-venv\Scripts\python.exe -m pip install torch==2.11.0 --index-url https://download.pytorch.org/whl/cpu
HOME/mtah-ext\pathcrf-venv\Scripts\python.exe -m pip install torch-geometric==2.8.0 pandas==3.0.0 numpy==2.4.0 scipy==1.17.0 tqdm==4.67.3
```

Versions verified working (Python 3.14.0): torch 2.11.0 (the machine has the +cu128 build; inference runs on
CPU), torch-geometric 2.8.0, pandas 3.0.0, numpy 2.4.0, scipy 1.17.0, tqdm 4.67.3. Note: the existing venv on this
machine was created with `--system-site-packages` over `C:\Python314`, so these packages live in the system
interpreter; the commands above give an equivalent self-contained venv. The checkpoint ships with the repo.

## Use

- Wrapper: `third_party/pathcrf/run_infer.py --in <npz> --out <npz> [--ckpt 140] [--repo <dir>] [--threads N]`,
  run under the PathCRF venv with `TQDM_DISABLE=1`. One half per call. The in-repo file is the source of
  truth; a copy lives at `HOME/mtah-ext\pathcrf-eval\run_infer.py` (re-copy after edits).
- `mtah.evidence.pathcrf.pathcrf_posterior(match, cache_key, params)` calls it per half and caches the merged
  result in `runs/cache/pathcrf/<cache_key>.npz` (`<sim.key>_s<seed>` for sims, `real_tactical` for the
  real match). Hand-off npz files go to a temporary directory that is deleted after the run. The cache is keyed on the input hash,
  the wrapper hash, checkpoint and threads. The wrapper also writes `n_chunks` and `n_ok` (windows with a prediction);
  `pathcrf_posterior` raises `RuntimeError` when `n_ok / n_chunks < 0.5` for a half.
- Environment overrides of the defaults in `PathCRFParams`:
  - `MTAH_PATHCRF_REPO` - PathCRF checkout (default `HOME/mtah-ext\pathcrf`)
  - `MTAH_PATHCRF_PY` - PathCRF venv python (default `HOME/mtah-ext\pathcrf-venv\Scripts\python.exe`)
- Enable with `PipelineConfig.use_pathcrf = True` (or a grid cell with `w_pathcrf > 0`).
- Runtime: about 3.5 min per 45-min half (68 354 frames at 25 fps: 190-205 s measured) on 8 CPU threads, of which ~20 s is
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
