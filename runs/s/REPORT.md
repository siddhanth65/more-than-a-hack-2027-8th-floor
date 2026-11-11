# S (cross-half self-training of stream B): report

Status: ACCEPTED (simulator bar passed, admissible on the real match, clean-clone md5 reproducible on the server; laptop rebuild pending, see 'Laptop' below).

## Stage log
- 2026-11-11 11:26 IST  Amendment 3 committed on exp-s (839d70e) before any S run.
- 11:33  library `mtah/selftrain.py`, `scripts/s_selftrain.py`, `tests/test_selftrain.py` committed (99047ca).
- 11:34  harness check: re-decoding the registered a17 fold-9001 predictions with the new script reproduces the stored
  calibration row exactly (ctrl_Dfull 0.5623 fold 9001).
- 11:35  base B (fold-final) retraining for folds 9001-9006 on compute-server (CPU torch, ~/projects/mtah/s).
- 11:45  Amendment 3b committed (a8e593c): fine-tuning schedule of item 4 was degenerate for V1/V2 (about 2.6 % of
  steps labelled; 0-19 qualifying windows per half), found from label counts and job logs before any prediction was
  decoded; partial outputs deleted unread. New schedule: stride 150, min 5 labelled steps, 1:1 simulator windows, 8 epochs.
- 11:50-12:05  Simulator acceptance run (compute-server, CPU, ~/projects/mtah/s/out): base B retrained per fold, pseudo-labels,
  96 fine-tunes (4 variants x 6 folds x 2 halves x 2 seeds), decode at the a17 cell, scored with the vendored scorer.

## Simulator acceptance run (LOMO folds 9001-9006, seeds 100/101, cell dead -3.0 / seq_d -0.5 / B weight 12)

| system | ctrl_Dfull | raw_Dfull | ctrl per fold 9001..9006 |
|---|---|---|---|
| a17 reference, registered fold predictions re-decoded in this harness | 0.5622 | 0.5740 | .5623 .4461 .5944 .5456 .6100 .6145 |
| a17 base retrained here (starting point of the fine-tunes; not bit-identical, argmax of 0.02-0.04 % of steps differs) | 0.5617 | 0.5736 | .5620 .4463 .5934 .5453 .6088 .6145 |
| V0 diagnostic (same schedule, pseudo weight 0) | 0.5616 | 0.5746 | .5619 .4421 .5972 .5450 .6085 .6152 |
| V1 (confident control, w 1.0) | 0.5591 | 0.5760 | .5602 .4387 .5883 .5510 .6039 .6124 |
| V2 (confident control, w 0.5) | 0.5595 | 0.5765 | .5611 .4410 .5895 .5456 .6052 .6143 |
| V3 (confident control + a17 decode >= 0.9, w 0.5) | 0.5614 | 0.5769 | .5689 .4373 .5940 .5540 .6028 .6115 |

raw_Dfull per fold is in `runs/s/lomo/summary.json`; per-half scores in `runs/s/lomo/lomo_scores.csv`.
The re-decoded reference equals the registered 0.5622 / 0.5740 exactly (difference < 0.0005), so the bar is 0.5622 - 0.005 = 0.5572.

Pseudo-label quality on the simulator (aggregates, `pseudo` stage): confident control labels about 2.6 % of steps at raw
precision 0.83-1.00 per half; the a17-posterior >= 0.9 rule adds about 13,700 labelled steps per half (D about 1/3) at raw precision 0.84-0.92.

VERDICT (simulator): chosen variant = V3 (best mean ctrl_Dfull among V1-V3, 0.5614 vs 0.5595 and 0.5591); 0.5614 >= 0.5572, so
the simulator part of acceptance is PASSED. It is non-inferiority only: V3 is -0.0008 against a17 on ctrl_Dfull (better in 2 of 6
folds) and +0.0029 on raw_Dfull; fine-tuning with no pseudo-labels (V0) is also -0.0006. In the simulator the held-out
match is in-distribution, so no gain was expected. Admissibility on the real match is the remaining condition.

## Real-match S (accepted branch)

Everything that touches the contest data ran on compute-server (`~/projects/mtah/s`, CPU), because the laptop's data drive (E:, target of
the `~/mtah-data` link) is not attached at the moment. No label of the contest match, no score and nothing of the public window was used or computed.

- `scripts/s_real.py prep`: a17's pipeline on the contest match (PathCRF smoothing 1.0 s, cache `runs/cache/pathcrf_fx`), step features of B,
  a17 decode and merged posterior, pseudo-labels with the public window (frames 11185-18685) excluded (asserted: no labelled step has its centre
  frame in the window; 1,500 steps excluded). Pseudo-label counts (steps): confident control H1 426, H2 392 of about 15,900 per half;
  V3 labelled steps H1 13,998 (D 6,080), H2 15,235 (D 6,376).
- `scripts/s_real.py ft`: for each target half, a17's B fine-tuned on the simulator training sequences of all six matches plus the pseudo-labels of
  the OTHER half only (V3, weight 0.5, schedule of amendment 3b). Checkpoint for half 1 uses half 2's pseudo-labels and vice versa.
- `scripts/calib_s.py sim|real`: offsets by the usual rule. Grid = a17's 21 cells (dead {-2,-2.5,-3} x seq_d {-1,-2,-3} union dead {-3,-3.5,-4} x
  seq_d {-1,-0.5,0,0.5}); simulator side = S's V3 LOMO predictions decoded at each cell; real side = the registered pipeline per cell, label-free.
  Chosen = best simulator mean rank among admissible cells: **dead offset -4.0, stream-B D offset +0.5** (`runs/s/calib/calib_s.csv`, `.json`).
  Cells (-3.0, +0.5) and (-3.5, +0.5) rank higher in the simulator but exceed the visible-dead band (0.1876, 0.1835 > 0.181). The chosen cell
  has little margin to the band edge (visible-dead 0.1793 against 0.181).
- `scripts/register_s.py` -> `models/s` (a17's dead/team models, decoder and drift reference unchanged; `modelB` is a per-half dict;
  `mtah/evidence/seqstream.py` and `scripts/make_candidate.py` accept it). Commit 1795bb3.

### S at its registered cell and at the a17 cell (simulator LOMO, 6 folds, ctrl_Dfull / raw_Dfull)

| cell | S (V3) | a17 |
|---|---|---|
| a17's cell (dead -3.0, seq_d -0.5) | 0.5614 / 0.5769 | 0.5622 / 0.5740 |
| S's cell by the rule (dead -4.0, seq_d +0.5) | 0.5665 / 0.5836 | (a17's own cell is its rule-chosen one) |

The acceptance test was fixed at a17's cell (0.5614, passes the 0.5572 bar). The higher number at S's own cell is a consequence of the offset
rule, not an improvement of the stream: at identical cells S and a17 differ by under 0.001 on ctrl_Dfull.

### Admissibility (real match, label-free; `runs/candidates/s.drift.json`, `scripts/a17_report.py s a17`)

| measure | S | a17 | band / note |
|---|---|---|---|
| visible-dead share | 0.1793 | 0.1778 | [0.116, 0.181] pass (margin 0.0017) |
| R (spells per graded min) | 2.946 | 2.871 | |
| rho (simulator count ratio of the cell) | 0.8921 | 0.8883 | |
| R / rho | 3.302 | 3.232 | [2.28, 4.10] pass |
| tag agreement (implied) overall / H1 / H2 | 0.8203 / 0.8399 / 0.8014 | 0.8167 / 0.8406 / 0.7936 | |
| predicted D share / blackout D share | 0.4146 / 0.8576 | 0.4044 / 0.8312 | |
| spells | 312 | 304 | |
| H/A changes inside blackouts H1 / H2 | 11 / 9 | 12 / 11 | |
| frames changed vs a17 (graded) | 3.04 % (H1 2.58 %, H2 3.50 %) | | inside blackouts 3.44 %, outside 2.83 % |

The `out_of_range` list in the manifest (share_changes_in_blackout, tag_agreement, tag_agreement_implied, non_blackout_D_share, blackout_D_share) is
identical to a17's: those are the legacy simulator-reference ranges in `drift_reference.json`, not the admissibility bands.

### Reproducibility

- `runs/candidates/s.csv`: 199,319 rows, vendored official validator OK (H 55,768 / A 37,231 / D 106,320).
- **md5 4c2184b6807ae6f2b0c110c3dc164388**, byte-identical from two independent fresh clones of the bundle of `exp-s` at commit 1795bb3 (server,
  `git clone -c core.autocrlf=true`, PathCRF real feed read from the laptop-computed smoothed cache whose md5 equals the server copy
  6b10de0f...; B subprocess in the server torch 2.11.0+cpu venv).
- Sanity check of the server environment: a17 rebuilt there from `models/a17` has md5 afb9e7b4ede118268fd0222fd5e11068, the laptop-registered value.
- Note for clean clones: the PathCRF cache key includes the sha256 of `third_party/pathcrf/run_infer.py`; the cache was written from a Windows
  working copy (CRLF), so a clone needs `core.autocrlf=true` (as on the laptop) or it misses the cache and tries to run PathCRF.
- Laptop: a rebuild "on the laptop from a clean clone" was NOT done, because the laptop currently has no contest data (E:\C-offload\mtah-data is not
  mounted, `~/mtah-data` is a dangling link). Command once it is attached:
  `git clone --branch exp-s <repo> t && cd t && (copy or link runs/cache/pathcrf_fx) && python scripts/make_candidate.py --select models/s --name s --pathcrf-smooth 1.0`
  and compare with 4c2184b6.... Expected equal (a17 transfers server-to-laptop byte-identically here), but not verified.

### Tests
`pytest -m "not dfl and not real"` on the final tree: 363 passed (352 on `final` + 11 new in `tests/test_selftrain.py`: cross-half split / no leakage, public-window
exclusion, pseudo weighting, window selection, swap augmentation, merged-posterior rule, per-half checkpoint dispatch).

### Caveats worth the controller's attention
1. S is non-inferior, not better, in the simulator (-0.0008 ctrl_Dfull, +0.0029 raw_Dfull at a17's cell; V0, the no-pseudo-label control with the same schedule, is -0.0006).
   The held-out simulated match is in-distribution, so the simulator cannot show the benefit self-training is meant to give (adaptation to the real feed).
2. The schedule was corrected once (Amendment 3b) before any score; the corrected schedule is the one used everywhere.
3. V3 pseudo-labels use a17's own decode: S is therefore partly a self-distillation of a17 on the contest match (13,998 / 15,235 labelled steps per half, about a third D).
4. S changes 3.0 % of graded frames against a17, raises the predicted D share by 1.0 point and the visible-dead share by 0.15 point (to within 0.0017 of the band edge).
5. Rule outcome per Amendment 3 item 6: accepted S replaces a17 as the L-family representative.
