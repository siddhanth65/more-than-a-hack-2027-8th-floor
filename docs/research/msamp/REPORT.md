# Metrica-sample other-game validation (games 1-3, Metrica convention)

Scores: candidates registered on the six DFL folds (no sample-game data in any training set), halves of games 1-3 corrupted with the runs/sim settings (ball smoothing 13, rel touches), seeds 100/101, truth = `raw_Dfull` (event + blank-ball derived H/A/D). Cell = mean over the two halves of a game; mean/sd over the 6 (game, seed) cells; sd_seeds = sd over the two seed means.

## spellF1 (raw_Dfull)

| system | G1 | G2 | G3 | mean | sd cells | sd seeds |
|---|---|---|---|---|---|---|
| a5c | 0.584 | 0.558 | 0.615 | 0.586 | 0.029 | 0.004 |
| a13r | 0.586 | 0.550 | 0.619 | 0.585 | 0.033 | 0.008 |
| a10 | 0.567 | 0.566 | 0.614 | 0.583 | 0.026 | 0.003 |
| a14 | 0.581 | 0.566 | 0.657 | 0.601 | 0.048 | 0.008 |
| a16 | 0.527 | 0.525 | 0.598 | 0.550 | 0.044 | 0.002 |
| smooth50 | 0.329 | 0.332 | 0.305 | 0.322 | 0.014 | 0.002 |
| b0 | 0.092 | 0.075 | 0.081 | 0.083 | 0.009 | 0.002 |
| b4 | 0.105 | 0.088 | 0.085 | 0.093 | 0.010 | 0.000 |

## macroF1 (raw_Dfull)

| system | G1 | G2 | G3 | mean | sd cells | sd seeds |
|---|---|---|---|---|---|---|
| a5c | 0.842 | 0.848 | 0.883 | 0.858 | 0.020 | 0.000 |
| a13r | 0.842 | 0.850 | 0.881 | 0.858 | 0.019 | 0.005 |
| a10 | 0.857 | 0.862 | 0.886 | 0.868 | 0.014 | 0.003 |
| a14 | 0.856 | 0.868 | 0.889 | 0.871 | 0.015 | 0.000 |
| a16 | 0.824 | 0.828 | 0.861 | 0.838 | 0.019 | 0.002 |
| smooth50 | 0.702 | 0.743 | 0.727 | 0.724 | 0.018 | 0.000 |
| b0 | 0.379 | 0.381 | 0.411 | 0.390 | 0.017 | 0.003 |
| b4 | 0.683 | 0.706 | 0.689 | 0.692 | 0.011 | 0.000 |

## Per game x seed spellF1 (raw_Dfull)

| system | G1_s100 | G1_s101 | G2_s100 | G2_s101 | G3_s100 | G3_s101 |
|---|---|---|---|---|---|---|
| a5c | 0.603 | 0.565 | 0.553 | 0.564 | 0.610 | 0.621 |
| a13r | 0.603 | 0.569 | 0.556 | 0.544 | 0.614 | 0.625 |
| a10 | 0.575 | 0.560 | 0.564 | 0.568 | 0.602 | 0.627 |
| a14 | 0.611 | 0.551 | 0.554 | 0.578 | 0.657 | 0.658 |
| a16 | 0.552 | 0.501 | 0.522 | 0.527 | 0.571 | 0.625 |
| smooth50 | 0.333 | 0.324 | 0.328 | 0.336 | 0.302 | 0.309 |
| b0 | 0.096 | 0.088 | 0.080 | 0.071 | 0.077 | 0.085 |
| b4 | 0.107 | 0.103 | 0.089 | 0.087 | 0.082 | 0.087 |

## Per half spellF1 (raw_Dfull, mean over seeds)

| system | G1H1 | G1H2 | G2H1 | G2H2 | G3H1 | G3H2 |
|---|---|---|---|---|---|---|
| a5c | 0.557 | 0.611 | 0.561 | 0.556 | 0.611 | 0.619 |
| a13r | 0.561 | 0.611 | 0.557 | 0.543 | 0.617 | 0.622 |
| a10 | 0.568 | 0.567 | 0.609 | 0.524 | 0.616 | 0.613 |
| a14 | 0.555 | 0.608 | 0.589 | 0.542 | 0.657 | 0.657 |
| a16 | 0.493 | 0.560 | 0.557 | 0.493 | 0.617 | 0.579 |
| smooth50 | 0.301 | 0.356 | 0.325 | 0.340 | 0.317 | 0.293 |
| b0 | 0.093 | 0.091 | 0.078 | 0.073 | 0.073 | 0.089 |
| b4 | 0.111 | 0.099 | 0.085 | 0.092 | 0.080 | 0.089 |

a14 - a5c over the 6 halves (seed-averaged): mean +0.0155, sd 0.0250; a14 wins 3, a5c wins 3, ties 0. Per half: G1H1 -0.003, G1H2 -0.003, G2H1 +0.028, G2H2 -0.014, G3H1 +0.046, G3H2 +0.038


## Label-free drift (mean over the 12 half x seed runs) vs the real match

| system | visible-dead share (sim) | spells/graded min (sim) | tag agr. (sim) | visible-dead (real) | spells/min (real) | tag agr. (real) |
|---|---|---|---|---|---|---|
| truth | 0.056 | 3.71 | 0.789 | n/a | n/a | n/a |
| a5c | 0.081 | 2.66 | 0.794 | 0.178 | 3.10 | 0.555 |
| a13r | 0.078 | 2.75 | 0.793 | 0.173 | 2.87 | 0.549 |
| a10 | 0.047 | 2.88 | 0.777 | 0.166 | 3.52 | 0.573 |
| a14 | 0.057 | 3.02 | 0.781 | 0.169 | 3.32 | 0.575 |
| a16 | 0.072 | 2.85 | 0.758 | 0.179 | 3.05 | 0.599 |
| smooth50 | 0.049 | 4.83 | 0.874 | n/a | n/a | n/a |
| b0 | 0.001 | 15.55 | 0.847 | n/a | n/a | n/a |
| b4 | 0.001 | 16.05 | 0.832 | n/a | n/a | n/a |

Truth variants: max |spellF1(raw_Dfull) - spellF1(other variant)| per system = a5c 0.0045, a13r 0.0044, a10 0.0057, a14 0.0053, a16 0.0033, smooth50 0.0033, b0 0.0019, b4 0.0020.

## Truth-convention aggregates

| game | dead share | spells/min | <1 s | <3 s | <5 s | median s | direct handover | blank-ball share | event-dead | blank-dead | dead with ball visible |
|---|---|---|---|---|---|---|---|---|---|---|---|
| G1 | 0.418 | 3.65 | 0.062 | 0.323 | 0.496 | 5.04 | 0.725 | 0.391 | 0.381 | 0.389 | 0.029 |
| G2 | 0.414 | 3.43 | 0.065 | 0.322 | 0.483 | 5.52 | 0.681 | 0.41 | 0.358 | 0.406 | 0.008 |
| G3 | 0.321 | 4.04 | 0.088 | 0.341 | 0.491 | 5.12 | 0.736 | 0.322 | 0.278 | 0.319 | 0.001 |

Contest anchors (sanity only): median spell ~7.6 s, share <3 s ~0.18, <1 s 0.06, <5 s 0.35, 12/16 direct handovers.

## Download manifest (raw.githubusercontent.com/metrica-sports/sample-data, master)

| file | bytes | sha256 |
|---|---|---|
| data/Sample_Game_1/Sample_Game_1_RawEventsData.csv | 139446 | 097dda6345e17390dca095331458fd00e47960c480c76f4356fd4edf2575ef05 |
| data/Sample_Game_1/Sample_Game_1_RawTrackingData_Away_Team.csv | 32782073 | 0a3df68a92af0290ede472e62de3b6489ddee29223da09bdf0cf76c8e0ab0618 |
| data/Sample_Game_1/Sample_Game_1_RawTrackingData_Home_Team.csv | 32782961 | 0a3d0eff7785950379c587d3597b86a2d91c2492281b4652e1abb97c99b6b364 |
| data/Sample_Game_2/Sample_Game_2_RawEventsData.csv | 153897 | edf31a18599265b77a8baf150f2ce6d89456fb0324b62ea7a391229657a619ba |
| data/Sample_Game_2/Sample_Game_2_RawTrackingData_Away_Team.csv | 29627711 | aa639f40762eade46c389020fe4458266b5ba14027b1e96e22e75db62803718b |
| data/Sample_Game_2/Sample_Game_2_RawTrackingData_Home_Team.csv | 31890338 | 3f9990ac204b07133e4202441d6fcbb386897c7c7f81e1bc3ce4d35bc8644694 |

## Skipped / not done

- Candidates a14f and H: not available (no a14f directory anywhere; runs/kh/H holds calibration files only, no registered artefacts; STATUS.md last updated 10 Nov 22:40) - not scored.
- Server: not used. Scoring took ~10 s per half per system on the laptop (one process, 2 threads); PathCRF ran locally as ONE process, 4 threads, 12 halves, 2640 s total.
- Truth variants ctrl_Dfull / raw_Dnarrow / ctrl_Dnarrow are almost identical to raw_Dfull here (labels are event-derived and the ball is blank in dead time, so the ctrl filter and narrowing barely act); max spellF1 differences are listed in results.json.
- Brief-opponent-spell absorbing truth (metrica_Dfull of Plan 4) not evaluated.
- Pitch for games 1 and 2 assumed 105 x 68 m (files are normalised, no size); game 3 uses its metadata size rescaled.
- a13r, a14 scored from runs/select/<name>; a5c, a10, a16 from models/<name> (a14 registered pipeline has the known input_dead_offset mismatch).

