# mtah — MTAH 2027 possession pipeline

Install: `python -m pip install -e ".[dev]"` (Python 3.12). Data: set `MTAH_DATA_DIR` to the unpacked bundle (`tactical/`, `broadcast/`). Tests: `python -m pytest -q`.
Regenerate the submission: `python -m mtah.submission --out runs/submission.csv` (Plan 2 wires the model; Plan 1 writes baselines).

## Reproduce the submission

`scripts\reproduce.ps1` (Windows PowerShell 5.1) regenerates the candidate CSV and prints its md5.

- `scripts\reproduce.ps1 -CandidateOnly` rebuilds `runs/candidates/a2.csv` from the committed `models/a2/` (config + dead model); `-Name` selects another candidate. Without `-CandidateOnly` it runs the full chain (simulator, PathCRF cache, leave-one-match-out selection, candidate).
- Environment: `MTAH_DATA_DIR` (unpacked competition bundle, required), `MTAH_VENV` (Python 3.12 venv, default `HOME/mtah-venv`; dependencies are pinned in `pyproject.toml`).
- PathCRF (external, MPL-2.0, separate Python 3.14 venv) is set up per `third_party/pathcrf/SETUP.md`; override locations with `MTAH_PATHCRF_REPO` and `MTAH_PATHCRF_PY`.
- Dependencies are pinned to exact versions in `pyproject.toml` (numpy 2.5.4, pandas 3.0.6, scipy 1.18.1, scikit-learn 1.9.1, pyarrow 26.0.0, joblib 1.6.0); the committed dead model is a pickle, so scikit-learn must match. Byte-for-byte regeneration of the submission is verified on the same machine, Python and pinned packages with the PathCRF threads pinned to 4; cross-machine identity is not yet shown.
- The technical document is `docs/technical-document.md`.
