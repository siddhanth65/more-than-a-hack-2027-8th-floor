# mtah — MTAH 2027 possession pipeline

Install: `python -m pip install -e ".[dev]"` (Python 3.12). Data: set `MTAH_DATA_DIR` to the unpacked bundle (`tactical/`, `broadcast/`). Tests: `python -m pytest -q`.
Regenerate the submission: `python -m mtah.submission --out runs/submission.csv` (Plan 2 wires the model; Plan 1 writes baselines).

## Reproduce the submission

`scripts\reproduce.ps1` (Windows PowerShell 5.1) regenerates the candidate CSV and prints its md5.

- `scripts\reproduce.ps1 -CandidateOnly` rebuilds `runs/candidates/a2.csv` from the committed `models/a2/` (config + dead model); `-Name` selects another candidate. Without `-CandidateOnly` it runs the full chain (simulator, PathCRF cache, leave-one-match-out selection, candidate).
- Environment: `MTAH_DATA_DIR` (unpacked competition bundle, required), `MTAH_VENV` (Python 3.12 venv, default `HOME/mtah-venv`; dependencies are pinned in `pyproject.toml`).
- PathCRF (external, MPL-2.0, separate Python 3.14 venv) is set up per `third_party/pathcrf/SETUP.md`; override locations with `MTAH_PATHCRF_REPO` and `MTAH_PATHCRF_PY`.
- The technical document is `docs/technical-document.md`.
