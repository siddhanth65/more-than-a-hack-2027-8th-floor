# S (cross-half self-training of stream B): report

Status: IN PROGRESS (written in stages; the last stage listed below is where work stopped).

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
