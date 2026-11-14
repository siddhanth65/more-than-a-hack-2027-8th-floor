# mtah: frame-level possession labelling (More Than A Hack 2027)

Code that turns the provided tactical tracking and Smart Tagging of one match into a per-frame possession file
(`frame,label`, labels H / A / D, 199,319 rows). The submitted file is candidate **s_ko**: candidate S (a17 with a cross-half self-trained BiGRU stream B, plus two moved offsets) with the second-half kick-off rule of Amendment 4 (graded frames before a half's kick-off are D). S and a17 stay documented and reproducible.
The method is described in [`docs/technical-document.md`](docs/technical-document.md) (PDF: `docs/technical-document.pdf` if present).

**This repository contains no match data**: no tracking, tags, video, ground truth, or frame-level exports. Everything the
pipeline reads from the competition bundle is read from a directory you point to with `MTAH_DATA_DIR`.

## What is committed

- `mtah/`: the pipeline (evidence streams, decoder, post-processing, simulator, scoring wrapper).
- `models/<candidate>/`: the trained artefacts and configuration of every registered candidate (dead-ball model, team-control model,
  BiGRU stream B; 0.5 to 1.7 MB each). They were trained on DFL open data only (see the technical document, section 3).
  The submitted candidate is built from `models/s` (`--name s_ko --kickoff-dead`); `models/a17` is its predecessor and `models/a5c_fx2` the tracking-only fallback.
- `third_party/pathcrf/`: our wrapper around the external PathCRF model (MPL-2.0) and setup notes. PathCRF itself is not copied here.
- `scripts/`: candidate generation (`make_candidate.py`), reproduction (`reproduce.ps1`), calibration, experiments and diagnostics.
- `tests/`: unit tests. `docs/`: technical document, pre-registrations, read logs (`docs/reads/`), reports.

## Reproduce the submission file

Expected result: `runs/candidates/s_ko.csv`, **md5 `2b908887901aad725558de040f3874b6`** (S without the kick-off rule: `4c2184b6807ae6f2b0c110c3dc164388`; a17: `afb9e7b4ede118268fd0222fd5e11068`), 199,319 data rows, passes the organisers'
`validate_submission.py`.

### 1. Environments

Two Python environments are used. The pipeline process (numpy, pandas, scikit-learn) runs in the main one; PathCRF and the BiGRU
stream B (torch) run as subprocesses in the second one.

| | main environment | torch / PathCRF environment |
|---|---|---|
| Python | 3.12.10 | 3.14.0 |
| pins | `requirements-main.txt` (same as `pyproject.toml`) | `requirements-torch.txt` (+ torch 2.11.0 from the PyTorch CPU index) |
| env var | `MTAH_VENV` (venv directory; default `~/mtah-venv`) | `MTAH_PATHCRF_PY` (its `python.exe`; default `~/mtah-ext/pathcrf-venv/Scripts/python.exe`) |

```
py -3.12 -m venv mtah-venv
mtah-venv\Scripts\python -m pip install -r requirements-main.txt
mtah-venv\Scripts\python -m pip install -e . --no-deps
```

scikit-learn must be exactly 1.9.1 (the committed dead-ball and team-control models are pickles). The torch environment is set up in
[`third_party/pathcrf/SETUP.md`](third_party/pathcrf/SETUP.md).

### 2. External assets

| asset | what for | where to get it | licence |
|---|---|---|---|
| competition data bundle (`tactical/`) | the input; not redistributable | organisers (see their repository) | competition terms |
| PathCRF code + checkpoint `saved/140` (commit `1b7c6f85a6b78c92e4543fde72ca2b86a933f997`) | players-only team evidence | https://github.com/hyunsungkim-ds/pathcrf (checkpoint ships in the repository; set `MTAH_PATHCRF_REPO`) | MPL-2.0 (used unmodified) |

Only these two are needed to regenerate the file. Everything below is needed **only to retrain or re-validate**, not to reproduce the CSV:

| asset | what for | where | licence |
|---|---|---|---|
| DFL open data (Sportec, 7 matches) | training and selection proxy; one match sealed | the public DFL / Sportec open-data release (check the exact download page and citation before publishing; not verified here) | CC BY 4.0 |
| SkillCorner open data (20 A-League 2024/25 matches) | other-game hold-out and band anchors, never training for a17 | https://github.com/SkillCorner/opendata | MIT |
| Metrica Sports sample games 1 to 3 | other-game validation | https://github.com/metrica-sports/sample-data | see that repository |

### 3. Run

```
set MTAH_DATA_DIR=<folder that contains tactical/>
set MTAH_VENV=<main venv>                 (optional)
set MTAH_PATHCRF_REPO=<pathcrf checkout>  (optional, default ~/mtah-ext/pathcrf)
set MTAH_PATHCRF_PY=<torch venv python>   (optional)
powershell -File scripts\reproduce.ps1            # candidate s_ko; prints the md5 and compares it with the expected one
```

`scripts\reproduce.ps1 -Name s` regenerates S, `scripts\reproduce.ps1 -Name a17` the predecessor and `scripts\reproduce.ps1 -Name a5c_fx2` the fallback (md5 `107e7fc1157c05ad3ba8f893734606b7`). The script is a thin wrapper around

```
python scripts/make_candidate.py --select models/s --name s_ko --pathcrf-smooth 1.0 --kickoff-dead
```

which can be run directly on any platform (set `MTAH_PATHCRF_PY` to the torch venv interpreter). Then validate:

```
python <organisers>/evaluation/validate_submission.py runs/candidates/s_ko.csv
```

**Runtime** (8-core laptop, CPU only): about 6.5 min for PathCRF on the real match the first time (4 threads, cached afterwards in
`runs/cache/pathcrf_fx`), plus 2 to 3 min for the rest.

**Line endings and the PathCRF cache.** The PathCRF cache key includes the sha256 of `third_party/pathcrf/run_infer.py`. The cache was written from a Windows working copy (CRLF), so a clone must be made with `git config core.autocrlf true` (the Windows default). On Linux, `git clone -c core.autocrlf=true` gives the same bytes; with LF files the cache misses and PathCRF would be recomputed.

**Determinism.** a17 was regenerated byte-identically twice and from a clean clone on the laptop that defines its md5. S was regenerated byte-identically from two fresh clones on a Linux server (using the laptop-computed PathCRF cache); the same server environment rebuilds a17 to its laptop md5. On 14 Nov the laptop rebuilt S from a fresh clone with an empty PathCRF cache and obtained the expected md5; on 15 Nov a fresh laptop clone rebuilt S and s_ko to their md5s.
The PathCRF posterior depends on the torch build and thread count (pinned to 4 in the code); on another machine PathCRF's output was
seen to differ by about 6e-6, which did not change the labels in the one cross-machine comparison made, but byte identity across machines
is not guaranteed.

### 4. Tests

```
python -m pytest -m "not dfl and not real"        # unit suite; needs no data
```

Tests marked `real` need `MTAH_DATA_DIR`; tests marked `dfl` need the parsed DFL proxy (`MTAH_DFL_DIR`).

## Data and scripts that are not needed to reproduce

Retraining uses simulator halves built from DFL open data (`scripts/build_sim.py`, `scripts/select_lomo.py`). The six training matches
were converted to the `fp1d` npz layout by an earlier private converter of the team; `scripts/parse_dfl_match.py` is a standalone copy
of its tracking part and writes the same layout, but it has not been checked to reproduce those six files byte for byte.
Many files in `scripts/` are one-off experiment drivers that were run on a layout of directories (`runs/...`) that is not committed;
they are kept for transparency, not as a supported interface.

## Public-window use

The organisers' 5-minute public window was read 11 times in total (reads 1 to 7 and 10 to 11 scored, reads 8 and 9 booleans only; reads 10 and 11 of 14 Nov were user-requested, report-only, and deviate from our own pre-registration, see the technical document, section 6). All reads are in
`docs/reads/public.jsonl` and listed in the technical document (section 6). `scripts/public_read.py` is disabled; `scripts/final_check.py`
needs the path of the public ground-truth file as an argument or in `MTAH_PUBLIC_GT`; the file is not in this repository.

## Licence

No licence file has been chosen yet; see the note in the technical document's reproduction section. PathCRF (MPL-2.0), the DFL data
(CC BY 4.0) and the SkillCorner data (MIT) are third-party and keep their own terms.
