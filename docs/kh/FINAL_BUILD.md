# FINAL_BUILD (exp-kh, 11 Nov): a17 and a5c_fx2

## 1. PathCRF fix merged
Cherry-picked exp-arb "pathcrf: optional input smoothing ... make_candidate --pathcrf-smooth" (11 Nov 03:29) + "pathcrf: optional input smoothing (smooth_s ...)" (exp-arb, 03:30) (PathCRFParams.smooth_s, cache keyed on it, make_candidate --pathcrf-smooth, tests) onto exp-kh; no merge conflicts. One test needed updating (tests/test_pathcrf_remote.py lists the cache file keys, now including smooth_s). Unit suite `pytest -m "not dfl and not real"`: 336 passed, 11 deselected (laptop, 2 threads, 125 s).

## 2. a17 calibration (scripts/calib_a17.py, runs/kh/a17/calib_a17.csv/.json)
a17 = a14f configuration (seqB 12, input_dead_offset -2.0, merge off) + PathCRF smoothing 1.0 s on the REAL feed (simulator inputs unchanged, smooth_s 0, no retraining). Grid = union of a14's two calibration grids (20 cells), stored 6-fold LOMO scores (B weight 12), real side recomputed on the fixed evidence (cache runs/cache/pathcrf_fx). Admissible: visible-dead in [0.116, 0.181] and R/rho in [2.28, 4.10]. Chosen = best mean rank (4 variants) among admissible cells.

| dead | seq_d | LOMO ctrl | LOMO raw | visible-dead | R | rho | R/rho | admissible | mean rank |
|---|---|---|---|---|---|---|---|---|---|
| -3.0 | +0.5 | 0.5695 | 0.5821 | 0.2089 | 2.880 | 0.876 | 3.29 | no | 1.0 |
| -3.5 | +0.5 | 0.5681 | 0.5805 | 0.2031 | 2.871 | 0.878 | 3.27 | no | 2.5 |
| -4.0 | +0.5 | 0.5677 | 0.5797 | 0.1994 | 2.880 | 0.880 | 3.27 | no | 2.5 |
| -3.0 | 0.0 | 0.5663 | 0.5780 | 0.1923 | 2.899 | 0.882 | 3.29 | no | 4 |
| -3.5 | 0.0 | 0.5644 | 0.5765 | 0.1874 | 2.899 | 0.885 | 3.28 | no | 5 |
| -4.0 | 0.0 | 0.5632 | 0.5752 | 0.1837 | 2.927 | 0.886 | 3.30 | no | 6 |
| **-3.0** | **-0.5** | **0.5622** | **0.5740** | **0.1778** | **2.871** | **0.888** | **3.23** | **yes** | **7 (chosen)** |
| -3.5 | -0.5 | 0.5605 | 0.5719 | 0.1747 | 2.871 | 0.892 | 3.22 | yes | 8.25 |
| -2.0 | -1.0 | 0.5604 | 0.5717 | 0.1746 | 2.871 | 0.892 | 3.22 | yes | 8.75 |
| -4.0 | -0.5 | 0.5590 | 0.5703 | 0.1694 | 2.852 | 0.895 | 3.18 | yes | 10.25 |
| -2.5 | -1.0 | 0.5588 | 0.5701 | 0.1690 | 2.852 | 0.896 | 3.18 | yes | 10.75 |
| -3.0 | -1.0 (a14f) | 0.5568 | 0.5676 | 0.1666 | 2.861 | 0.898 | 3.18 | yes | 12 |
| -3.5 | -1.0 | 0.5546 | 0.5645 | 0.1644 | 2.899 | 0.901 | 3.22 | yes | 13 |
| -4.0 | -1.0 | 0.5524 | 0.5620 | 0.1614 | 2.956 | 0.901 | 3.28 | yes | 14 |
| -2.0 | -2.0 | 0.5465 | 0.5564 | 0.1549 | 2.993 | 0.908 | 3.30 | yes | 15 |
| -2.5 | -2.0 | 0.5443 | 0.5544 | 0.1517 | 2.993 | 0.910 | 3.29 | yes | 16 |
| -3.0 | -2.0 | 0.5416 | 0.5513 | 0.1483 | 2.974 | 0.912 | 3.26 | yes | 17 |
| -2.0 | -3.0 | 0.5303 | 0.5404 | 0.1385 | 3.078 | 0.922 | 3.34 | yes | 18 |
| -2.5 | -3.0 | 0.5264 | 0.5367 | 0.1351 | 3.059 | 0.926 | 3.31 | yes | 19 |
| -3.0 | -3.0 | 0.5239 | 0.5343 | 0.1310 | 3.078 | 0.928 | 3.32 | yes | 20 |

Chosen cell: dead_offset -3.0, seq_d_offset -0.5 (a14f: -3.0 / -1.0). It differs from a14f by +0.5 in seq_d_offset only (the smoothed PathCRF lowers real visible-dead by about 0.011 at the same cell, so the rule moves to the less negative cell). Simulator LOMO ctrl_Dfull 0.5622 (a14f 0.5568), raw 0.5740 (0.5676).

## 3. a17 registration
models/a17 (select dir, manifest, drift JSON with both band sets), runs/candidates/a17.csv, 199,319 rows (+ header), official validator pass, md5 afb9e7b4ede118268fd0222fd5e11068; byte-identical on a second run. Clean clone: fresh `git clone --branch exp-kh` (commit "a17: a14f config + PathCRF smoothing 1.0 s ..." (11 Nov 04:23)) into a temp dir on the LAPTOP, PathCRF real feed regenerated from scratch (smooth 1.0, ckpt 140, threads 4, 6.5 min, empty cache), `make_candidate.py --select models/a17 --name a17 --pathcrf-smooth 1.0` -> same md5 afb9e7b4.... (Server not used: PathCRF differs from the laptop by ~6e-6, so the registered md5 is defined by the laptop environment.)

Drift (real match, label-free): visible-dead 0.1778, R 2.871 spells/graded min, rho 0.888, R/rho 3.23 (admissible under the new bands, also under the old: yes), predicted D share 0.404, blackout D share 0.831, 304 spells. Tag agreement (implied) overall 0.817, H1 0.841, H2 0.794 (a14f: 0.777 / 0.837 / 0.719; a14: 0.773 / 0.833 / 0.715). H/A changes inside blackouts: H1 12, H2 11 (a14f 13 / 15; a14 18 / 16).

## 4. a5c_fx2
a5c (models/a5c, merge 0.7) + smoothing 1.0 s; dead_offset over a5c's grid8 cells {-2, -2.5, -3}: all three admissible (R/rho 3.05 / 3.13 / 3.12; visible-dead 0.170 / 0.156 / 0.137), best simulator rank is -2.0, i.e. a5c's own offset. LOMO ctrl 0.4944 raw 0.5242 (old build). models/a5c_fx2, runs/candidates/a5c_fx2.csv md5 107e7fc1157c05ad3ba8f893734606b7, byte-identical twice, validator pass. Drift: visible-dead 0.1697, R 2.663, rho 0.872, R/rho 3.05, tag agreement 0.787 (H1 0.805, H2 0.770), H/A changes inside blackouts H1 6 / H2 1. Clean-clone for a5c_fx2 not done (same code path as a17).

## 5. Label-free sanity
Frames changed, a17 vs a14f (old evidence): graded 8.5 %; H1 2.4 %, H2 14.6 %; inside blackouts 5.5 %; outside blackouts 10.1 %. vs a14: 9.6 % (H1 4.0 %, H2 15.1 %, blackouts 7.7 %). a5c_fx2 vs a5c: 10.6 % (H1 2.3 %, H2 19.0 %, blackouts 3.6 %).
Pre-restart D-run table (window [-3, +10] s from the tag; run length p25-median-p75 in s, [share with a D->live edge], diag_prerestart.py from exp-diag, runs/kh/a17/diag):
| type (n) | a14 | a14f | a17 |
|---|---|---|---|
| throw-in (26) | 6.2-10.1-18.4 [.85] | 6.4-9.9-18.6 [.85] | 6.6-10.0-18.1 [.81] |
| free kick (5) | 21.4-24.2-28.3 [.60] | 21.7-24.2-28.3 [.60] | 22.0-24.0-28.3 [.60] |
| goal kick (10) | 5.6-9.0-33.4 [.80] | 6.5-10.6-33.9 [.80] | 0.0-8.4-35.2 [.60] |
| corner (12) | 24.0-28.4-31.5 [1.0] | 24.0-28.6-33.0 [1.0] | 23.9-28.9-33.9 [1.0] |
| kick-off after goal (4) | 16.2-44.6-76.7 [.75] | 64.6-82.8-102.8 [.75] | 82.8-105.7-119.3 [.75] |
Deficit to the SkillCorner p25 as share of graded frames: a14 0.052, a14f 0.038, a17 0.040. Pre-restart runs are essentially unchanged by the fix; throw-ins and goal kicks remain short (SkillCorner p25: 10 / 24 s).

## 6. final_check.py and public_read.py
scripts/final_check.py (NOT run): validator + row count (exit 1 with reason if invalid), then prints exactly two booleans: best offset among {-50..50 step 5} equals 0 (ties with 0 count), and swapping H/A is strictly worse. Prints no score, writes nothing. scripts/public_read.py is disabled (exits with a message pointing to final_check.py; imports no scorer, no ground-truth path); tests/test_readlog.py updated (disabled-guard test and helper test; 5 passed).

## Commits (exp-kh)
the cherry-pick of the smoothing option (03:30)/the cherry-pick with make_candidate --pathcrf-smooth (03:29) (cherry-picks), tests fix + calib_a17, "final_check.py (two booleans), public_read.py disabled + tests" (04:07) (final_check, public_read), "a17: a14f config + PathCRF smoothing 1.0 s ..." (11 Nov 04:23) (a17), "a5c_fx2: a5c + PathCRF smoothing 1.0 s ..." (04:35) (a5c_fx2). No public-window truth or 9007 used.
