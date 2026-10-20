# mtah — MTAH 2027 possession pipeline

Install: `python -m pip install -e ".[dev]"` (Python 3.12). Data: set `MTAH_DATA_DIR` to the unpacked bundle (`tactical/`, `broadcast/`). Tests: `python -m pytest -q`.
Regenerate the submission: `python -m mtah.submission --out runs/submission.csv` (Plan 2 wires the model; Plan 1 writes baselines).
