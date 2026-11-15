# More Than A Hack 2027, team 8th floor: frame-level possession labelling

The registered team name is **team 8th floor**: Siddhanth Susheel, Rishitej Reddy Vyalla, Anirudh Srinivas Dronamraju.
Repository: https://github.com/siddhanth65/more-than-a-hack-2027-8th-floor . Licence: MIT (`LICENSE`); third-party components in `THIRD_PARTY.md`.

Code that turns the provided tactical tracking and Smart Tagging of one match into a per-frame possession file
(`frame,label`, labels H / A / D, 199,319 rows). The submitted file is candidate **s_ko**: candidate S (a17 with a cross-half self-trained BiGRU stream B, plus two moved offsets) with the second-half kick-off rule of Amendment 4 (graded frames before a half's kick-off are D). S and a17 stay documented and reproducible.
The method is described in [`docs/technical-document.md`](docs/technical-document.md) (PDF: `docs/technical-document.pdf`).

**This repository contains no match data**: no tracking, tags, video, ground truth, or frame-level exports. Everything the
pipeline reads from the competition bundle is read from a directory you point to with `MTAH_DATA_DIR`.

## What is committed

- `mtah/`: the pipeline (evidence streams, decoder, post-processing, simulator, scoring wrapper).
- `models/<candidate>/`: the trained artefacts and configuration of every registered candidate (dead-ball model, team-control model,
  BiGRU stream B; 0.5 to 1.7 MB each). Trained on DFL open data only, except the two stream-B networks of `models/s`, which were
  additionally fine-tuned on algorithmic pseudo-labels of the provided game (technical document, section 3).
  The submitted candidate is built from `models/s` (`--name s_ko --kickoff-dead`); `models/a17` is its predecessor and `models/a5c_fx2` the tracking-only fallback.
- `third_party/pathcrf/`: our wrapper around the external PathCRF model (MPL-2.0) and setup notes. PathCRF itself is not copied here.
- `mtah/eval/official/`: the organisers' scorer, vendored unmodified, with their MIT notice (`LICENSE-organisers`).
- `scripts/`: candidate generation (`make_candidate.py`), reproduction (`reproduce.ps1`), calibration, experiments and diagnostics.
- `tests/`: unit tests. `docs/`: technical document, pre-registrations, read logs (`docs/reads/`), reports.

## Reproduce the submission file

Expected result: `runs/candidates/s_ko.csv`, **md5 `2b908887901aad725558de040f3874b6`** (S without the kick-off rule: `4c2184b6807ae6f2b0c110c3dc164388`; a17: `afb9e7b4ede118268fd0222fd5e11068`), 199,319 data rows, passes the organisers'
`validate_submission.py`. This md5 was reproduced from scratch on two unrelated machines (see Determinism).

### 0. Input data

Four files of the competition bundle, in a folder `tactical/`; `MTAH_DATA_DIR` must point at the folder that **contains** `tactical/`:

```
<MTAH_DATA_DIR>/tactical/BAR-ATH_tactical_atd.xml
<MTAH_DATA_DIR>/tactical/BAR-ATH_tactical_atd.txt
<MTAH_DATA_DIR>/tactical/BAR-ATH_tactical_homography.csv
<MTAH_DATA_DIR>/tactical/BAR-ATH_tactical_smart_tagging.xml
```

The video (`.mp4`) is not needed.

### 1. Clone (line endings)

```
git -c core.autocrlf=false clone https://github.com/siddhanth65/more-than-a-hack-2027-8th-floor.git
```

`core.autocrlf=false` keeps every file byte-identical to the one we built from (on Windows the default `autocrlf=true` would rewrite the
files with CRLF; that does not change the labels, but it changes the PathCRF cache key and is not what we tested).

### 2. Environments

Two Python environments are used. The pipeline process (numpy, pandas, scikit-learn) runs in the main one; PathCRF and the BiGRU
stream B (torch) run as subprocesses in the second one.

| | main environment | torch / PathCRF environment |
|---|---|---|
| Python | 3.12.10 | 3.14.0 |
| pins | `requirements-main.txt` (same as `pyproject.toml`) | `requirements-torch.txt` plus `torch==2.11.0` from the PyTorch **CPU** wheel index |
| env var | `MTAH_VENV` (venv directory, used by `reproduce.ps1`; default `~/mtah-venv`) | `MTAH_PATHCRF_PY` (its Python executable) |

scikit-learn must be exactly 1.9.1 (the committed dead-ball and team-control models are pickles). Details of the torch environment:
[`third_party/pathcrf/SETUP.md`](third_party/pathcrf/SETUP.md).

### 3a. Windows (PowerShell 5.1 or 7)

```
py -3.12 -m venv $HOME\mtah-venv
& $HOME\mtah-venv\Scripts\python.exe -m pip install -r requirements-main.txt
& $HOME\mtah-venv\Scripts\python.exe -m pip install -e . --no-deps

New-Item -ItemType Directory -Force $HOME\mtah-ext | Out-Null
git clone https://github.com/hyunsungkim-ds/pathcrf $HOME\mtah-ext\pathcrf
git -C $HOME\mtah-ext\pathcrf checkout 1b7c6f85a6b78c92e4543fde72ca2b86a933f997
py -3.14 -m venv $HOME\mtah-ext\pathcrf-venv
& $HOME\mtah-ext\pathcrf-venv\Scripts\python.exe -m pip install torch==2.11.0 --index-url https://download.pytorch.org/whl/cpu
& $HOME\mtah-ext\pathcrf-venv\Scripts\python.exe -m pip install -r requirements-torch.txt

$env:MTAH_DATA_DIR = "C:\path\to\bundle"        # the folder that contains tactical\
# optional overrides: $env:MTAH_VENV, $env:MTAH_PATHCRF_REPO, $env:MTAH_PATHCRF_PY
powershell -NoProfile -ExecutionPolicy Bypass -File scripts\reproduce.ps1      # builds s_ko, prints the md5 and compares it
```

`scripts\reproduce.ps1 -Name s` regenerates S, `-Name a17` the predecessor and `-Name a5c_fx2` the fallback (md5 `107e7fc1157c05ad3ba8f893734606b7`).

### 3b. Linux (the sequence of the clean build of 15 Nov 2026)

```
python3 -m pip install --user uv                      # or any other way to get uv
uv python install 3.12.10 3.14.0
git -c core.autocrlf=false clone https://github.com/siddhanth65/more-than-a-hack-2027-8th-floor.git repo && cd repo

uv venv --python 3.12.10 ../venv-main
uv pip install --python ../venv-main/bin/python -r requirements-main.txt
uv pip install --python ../venv-main/bin/python -e . --no-deps

uv venv --python 3.14.0 ../venv-torch
uv pip install --python ../venv-torch/bin/python torch==2.11.0 --index-url https://download.pytorch.org/whl/cpu
uv pip install --python ../venv-torch/bin/python -r requirements-torch.txt

git clone https://github.com/hyunsungkim-ds/pathcrf ../pathcrf
git -C ../pathcrf checkout 1b7c6f85a6b78c92e4543fde72ca2b86a933f997      # checkpoint saved/140 ships in the repository

export MTAH_DATA_DIR=/path/to/bundle                  # contains tactical/
export MTAH_PATHCRF_REPO=$PWD/../pathcrf
export MTAH_PATHCRF_PY=$PWD/../venv-torch/bin/python
export TQDM_DISABLE=1
../venv-main/bin/python scripts/make_candidate.py --select models/s --name s_ko --pathcrf-smooth 1.0 --kickoff-dead
md5sum runs/candidates/s_ko.csv                       # expected 2b908887901aad725558de040f3874b6
```

The same `make_candidate.py` line works on Windows too (use the venv interpreters). For S drop `--kickoff-dead` and use `--name s`; for a17 use
`--select models/a17 --name a17 --pathcrf-smooth 1.0`.

### 4. Validate

The organisers' validator needs `games.py` one folder up and `possession_metrics.py` beside it, so run it from a checkout of the organisers' repository:

```
python <organisers-repo>/evaluation/validate_submission.py runs/candidates/s_ko.csv
```

(Our vendored copy of the scorer, `mtah/eval/official/possession_metrics.py`, is the same file; the validator itself and `games.py` are not vendored.)

**Runtime** (8 CPU cores, CPU only): about 6.5 to 7 min for PathCRF on the real match the first time (4 threads, cached afterwards in
`runs/cache/pathcrf_fx`), plus 2 to 3 min for the rest; 433 s in total on the Linux machine below.

**Determinism.** The md5 was defined on a Windows laptop (torch 2.11.0, CUDA build used on the CPU). On 15 Nov 2026 the file was rebuilt on a
different machine (Linux, Intel Xeon Platinum 8592+, 8 CPUs, fresh clone, fresh environments exactly as above with the CPU wheel, empty PathCRF cache,
no thread or BLAS variables set) and the md5 matched. The laptop also rebuilt S and s_ko from fresh clones with empty caches. This is a
positive observation on two machines, not a proof for every CPU: PathCRF's output differed by about 6e-6 between machines in an earlier comparison
(the code pins 4 torch threads), without changing the labels in the one comparison made. If your md5 differs, `docs/technical-document.md`, section 8, lists what we used.

### 5. Tests

```
python -m pytest -m "not dfl and not real"        # unit suite; needs no data
```

Tests marked `real` need `MTAH_DATA_DIR`; tests marked `dfl` need the parsed DFL proxy (`MTAH_DFL_DIR`).

## External assets

| asset | what for | where to get it | licence |
|---|---|---|---|
| competition data bundle (`tactical/`) | the input; not redistributable | organisers (see their repository) | competition terms |
| PathCRF code + checkpoint `saved/140` (commit `1b7c6f85a6b78c92e4543fde72ca2b86a933f997`) | players-only team evidence | https://github.com/hyunsungkim-ds/pathcrf (checkpoint ships in the repository; set `MTAH_PATHCRF_REPO`) | MPL-2.0 (used unmodified) |

Only these two are needed to regenerate the file. Everything below is needed **only to retrain or re-validate**:

| asset | what for | where | licence |
|---|---|---|---|
| DFL / Sportec open data (7 matches); Bassek et al., Scientific Data 12, 195 (2025), doi:10.1038/s41597-025-04505-y | training and selection proxy; one match sealed | https://doi.org/10.6084/m9.figshare.28196177 | CC BY 4.0 |
| SkillCorner open data (20 A-League 2024/25 matches) | other-game hold-out and band anchors, never training for a17 | https://github.com/SkillCorner/opendata | MIT |
| Metrica Sports sample games 1 to 3 | other-game validation | https://github.com/metrica-sports/sample-data | see that repository |

## Data and scripts that are not needed to reproduce

Retraining uses simulator halves built from DFL open data (`scripts/build_sim.py`, `scripts/select_lomo.py`). The six training matches
were converted to the `fp1d` npz layout by an earlier private converter of the team; `scripts/parse_dfl_match.py` is a standalone copy
of its tracking part and writes the same layout, but it has not been checked to reproduce those six files byte for byte. The two
stream-B checkpoints of `models/s` were fine-tuned on pseudo-labels of the provided game and cannot be regenerated from this repository alone.
Many files in `scripts/` are one-off experiment drivers that were run on a layout of directories (`runs/...`) that is not committed;
they are kept for transparency, not as a supported interface.

The set-piece tag shift of Amendment 5 (`MTAH_SP_SHIFT`, `MTAH_SP_SHIFT_REAL`) is an opt-in hook that is off unless the environment variable is set; it is not used by any submitted build.

## Public-window use

The organisers' 5-minute public window was read 11 times in total (reads 1 to 7 and 10 to 11 scored, reads 8 and 9 booleans only; reads 10 and 11 of 14 Nov were report-only and deviate from our own pre-registration; in addition one simulator fitting target was measured on the window, see the technical document, section 6, item 6). All reads are in
`docs/reads/public.jsonl` and listed in the technical document (section 6). `scripts/public_read.py` is disabled; `scripts/final_check.py`
needs the path of the public ground-truth file as an argument or in `MTAH_PUBLIC_GT`; the file is not in this repository.

## Use of LLMs

LLM-based coding assistants were used during development. No video, tracking, tagging or homography data, nor any frame-level extract of them,
was given to a hosted LLM; the assistants saw code and aggregate scores only.

## Licence

MIT, see `LICENSE`. PathCRF (MPL-2.0), the DFL data (CC BY 4.0), the SkillCorner data (MIT) and the organisers' scorer (MIT, Metrica Sports) are third-party
and keep their own terms (`THIRD_PARTY.md`).
