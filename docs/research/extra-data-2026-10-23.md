# Extra training matches: SkillCorner open data and Metrica Sample Game 3 (2026-10-23)

Levers 2 and 3 of `expensive-levers-2026-10-23.md`. This was code and conversion work only: one process with `OMP_NUM_THREADS=2`, no selection and no training. Sim match 9007 was never loaded and the public window was not touched. Everything below is an aggregate.

## What was built (branch `exp-data`)

- `mtah/io/extra_npz.py` is the shared writer for the `dfl_proxy` npz schema, plus a census merge and label aggregates (official spell definition: maximal H/A runs).
- `scripts/convert_skillcorner.py` writes the 20 SkillCorner A-League 2024/25 matches as `game_9101..9120_H{1,2}`, in sorted SkillCorner-id order.
- `scripts/convert_metrica_sample.py` writes Metrica Sample Game 3 as `game_9201_H{1,2}`. Run it with `python -I` from outside the download folder. It adds the user site and this checkout explicitly.
- `scripts/extra_data_report.py` computes per-source aggregates, raw and `ctrl_Dfull`. DFL uses 9001–9006 only and skips 9007 by name before loading.
- `scripts/build_sim.py` gets three new flags:
  - `--only` now takes several ids.
  - `--keys` filters to single halves.
  - `--no-dfl` builds extra-dir halves only.
- Outputs go to `possession/runs/extra/` (gitignored): 42 halves, a shared `fp1d_census.json` and `report.json`, about 490 MB compressed.

The schema is the same as `game_9007_*`: `raw`, `ball_n`, `ball_xy` (centred metres, rescaled to 105 × 68), `ball_z`, `ball_status`, `ball_poss_ltr`, `truth`, `poss_only` and `actor_pid`. Three extra keys are ignored by the loaders:
- `ball_visible` (F,): the real detection or visibility flag.
- `interp` (F,): the frame was produced by resampling.
- `raw_detected` (N,): the SkillCorner per-player `is_detected` flag.

`ltr` is set from the goalkeeper mean x, as in `parse_dfl_match`. On SkillCorner it agreed with `home_team_side` on 40 of 40 halves.

## Label rules

**SkillCorner** (10 fps, then resampled):
- *alive* means the frame is inside a `phases_of_play` interval, or `possession.group` is set.
- *team* is `possession.group`. Inside an alive run it is carried forward while the ball travels or is loose. The head of a run falls back to the phase's `team_in_possession`.
- *D* is everything else: stoppages, out of play, and replays or close-ups with no phase and no possessor.
- Resampling to 25 fps runs on the period clock. Positions are interpolated linearly (NaN if either neighbour is NaN). Labels and flags take the nearest source frame. 80 % of output frames are interpolated.
- Positions are SkillCorner's *extrapolated* ones. The real `is_detected` flags are kept alongside.
- Events come from `player_possession` rows at `frame_start`:
  - `end_type` pass or clearance gives PASS.
  - `end_type` shot gives SHOT.
  - Anything else gives a poss-only DRIVE.
  - A throw-in reception gives THROW_IN.
- About 470 events per half.

**Metrica sample** (25 fps, events only):
- *team* is the team of the latest possession event. Possession events are PASS, CARRY, RECOVERY, SHOT, SET PIECE, BALL LOST and BALL OUT. CHALLENGE and FAULT RECEIVED do not move possession.
- *D* runs from a trigger to the start of the next SET PIECE. A trigger is:
  - BALL OUT, or a SHOT with subtype OUT or GOAL (at its end frame);
  - a foul, OFFSIDE, HAND BALL or REFEREE HIT (at its start frame).
- A D run is kept only if the next possession event after the trigger is that SET PIECE. Otherwise play continued.
- The head of each half up to the kick-off is D, and so is its tail after the last event.
- Player slots follow the EPTS columns, because substitutes take the leaver's column, as in the challenge export. Team B has 17 distinct players in H2, which would exceed the loader's 16 slots if each player had a slot of their own.

## Aggregates

Visible-dead is the D share among frames with the ball visible. Spell quantiles are p10/p25/p50/p75/p90 in seconds.

| source | halves | min | ball visible | H / A / D | visible-dead | spells/min | spell s quantiles | <3 s |
|---|---|---|---|---|---|---|---|---|
| DFL sims 9001–9006, raw | 12 | 571 | 1.00 | .281/.265/.453 | .453 | 4.72 | 0.8/1.6/3.6/8.4/17.4 | 44 % |
| DFL sims, ctrl_Dfull | 12 | 571 | 1.00 | .276/.270/.453 | .453 | 3.69 | 1.2/2.4/5.2/10.8/21.6 | 31 % |
| SkillCorner, raw | 40 | 1970 | 0.60 | .284/.274/.442 | .145 | 3.39 | 1.0/2.4/5.9/13.2/23.9 | 30 % |
| SkillCorner, ctrl_Dfull | 40 | 1970 | 0.60 | .283/.275/.442 | .145 | 2.28 | 2.0/4.1/9.1/19.7/34.5 | 17 % |
| Metrica sample, raw | 2 | 96 | 0.68 | .408/.366/.226 | .001 | 4.16 | 0.9/2.0/5.4/15.3/31.2 | 34 % |
| Metrica sample, ctrl_Dfull | 2 | 96 | 0.68 | .408/.366/.226 | .001 | 4.10 | 1.0/2.0/5.5/15.3/31.4 | 32 % |
| real public window (known) | — | 5 | — | — | .155 | 3.4 | median 7.6 | 18 % |

## Convention caveats

- **SkillCorner possession is per player** (`group` is null on loose balls and passes) and broadcast-based. Forward-filling it gives the right spell rate (3.39 vs 3.4 per minute) but too many short spells (30 % vs 18 %). The ctrl filter gives the right short-spell share (17 %) and a median of 9.1 s, but too few spells (2.28 per minute). The real convention lies between the two.
- **SkillCorner D is broad.** At 44 % it includes replays and close-ups where nothing is tracked, so it reads as "not live on camera" more than "whistle to restart". Its visible-dead (0.145) is close to the real 0.155, but "visible" here means SkillCorner detected the ball on the broadcast.
- **Metrica Sample Game 3 leaves the ball NaN during stoppages** (visible-dead 0.001). The challenge feed tracks the ball through dead time (0.155). So this match cannot teach visible dead ball.
- **Metrica's event-derived D is narrow.** At 23 % it covers only out or foul to the restart. Its team labels flip on every BALL LOST / RECOVERY pair, giving 34 % short spells. These are our labels, not Metrica's.
- **SkillCorner pitches** are 104–106 × 68 m, rescaled to 105 × 68. Ball and positions are smoother than real 25 fps tracking, because 4 of every 5 frames are interpolated.

## Smoke test

`build_sim.py --no-dfl --extra-dir runs/extra --keys <one half> --seeds 100 --out runs/extra/sim_smoke` ran on one half from each source. It took about 4 s per half-seed, and both halves loaded and ran through `truth_variants`, blackouts, corruption and tag simulation without error.

| half | masked | tag coverage (in / out of blackouts) | D raw_Dfull | D raw_Dnarrow | narrow changed |
|---|---|---|---|---|---|
| game_9101_H1 | .389 | .613 / .740 | .397 | .386 | .028 |
| game_9201_H1 | .362 | .720 / .812 | .237 | .236 | .007 |

## Cost of the full use (estimates, single process)

| step | estimate |
|---|---|
| Full sim build of the 42 extra halves × 5 seeds | about 15 min |
| PathCRF precompute | about 3–4 h (the earlier 12-half DFL precompute took about 1 h) |
| Selection with the extras as training only (LOMO folds stay on DFL 9001–9006) | about 5–7 h (about 1.5 h before, with about 4.5× the training frames now) |
| Selection with the extras also as folds | over 20 h; not recommended |
| **Total** | **about 9–11 CPU hours** |

## Recommendation

- **SkillCorner: use for training** the dead model and the spell and duration priors, with `ctrl_Dfull` as truth. Its spell statistics bracket the real ones and its visible-dead is close to real. Also use its real `is_detected` masks to fit the blackout and corruption models.
  - Weight it below DFL, or flag the source, because of the broadcast and replay D convention.
  - It is also the volume a BiGRU/TCN head would need (lever 7).
- **Metrica Sample Game 3: validation only.** It has the same provider and format, so it is useful for team-flip and convention checks. Do not use it for dead-ball training, since the ball is NaN in stoppages. Its D and team labels are our event-derived ones.

## Plan 6: SkillCorner halves as training-only simulator data (2026-10-23)

### Part 1. Simulator build (done)

`build_sim.py --extra-dir runs/extra --no-dfl --only 9101 ... 9120 --seeds 100 --out runs/sim_extra` (main tree, not `runs/sim`): 40 halves, seed 100 only, 160 files, 508 MB, about 12 min with `OMP_NUM_THREADS=2`.

Aggregates over the 40 halves (the blackout, corruption and tag simulators are the DFL-fitted ones, applied unchanged):

| quantity | value |
|---|---|
| masked share (blackout) | .398 |
| tag coverage in / out of blackouts | .626 / .767 |
| D share, raw_Dfull = ctrl_Dfull | .441 |
| D share, raw_Dnarrow = ctrl_Dnarrow | .422 |
| share of D frames changed by the narrow rule | .043 |

Truth: the `ctrl_Dfull` variant is the training truth for these halves (the selection's `train_dead`, `train_team_ctrl` and `train_switch` already train on `ctrl_Dfull`). The raw and ctrl D shares coincide on these halves (.441); the controlled spell filter changes H/A spell structure, as in the table above.

### Part 2. PathCRF posteriors on compute-server (blocked)

Stopped at the environment check, as instructed: `~/miniconda3/envs/det12/bin/python` has torch 2.5.1 but **no `torch_geometric`**, and the PathCRF checkout imports it (`dataset.py`, `inference.py`, `models/gat_lstm.py`, `models/utils.py`). Nothing was installed or cloned on the server. `~/projects/mtah/` does not exist yet. The server was at load 85 / 128 cores and 722 GB free on `/` when checked.

Everything else is ready and tested:

- `scripts/pathcrf_remote_assemble.py prepare --sim runs/sim_extra --seeds 100 --out runs/pathcrf_remote` was run: 40 `<key>_s100_h0.in.npz` files (722 MB uncompressed), `manifest.json` (input and wrapper sha per half), and `run_all.sh` (xargs, 8 jobs by default, `--threads 4`, skips finished halves).
- Server steps once `torch_geometric` is available (a pure-Python package; `pip install torch_geometric` into a fresh venv or into det12, with approval): clone `hyunsungkim-ds/pathcrf` at 1b7c6f85a6b78c92e4543fde72ca2b86a933f997 into `~/projects/mtah/pathcrf`; copy `saved/140` (37 MB), `third_party/pathcrf/run_infer.py` (unchanged; its sha must equal the manifest's `wrapper_sha`) and `runs/pathcrf_remote/{in,run_all.sh,manifest.json}`; then `PY=... REPO=~/projects/mtah/pathcrf WRAPPER=.../run_infer.py tmux ... bash run_all.sh 8`.
- Back on the laptop: copy `out/` next to `manifest.json`, then `scripts/pathcrf_remote_assemble.py assemble --out runs/pathcrf_remote --cache-dir runs/cache/pathcrf`. It re-checks the sims against the manifest, applies the same `MIN_OK_RATIO` rule, and writes the cache through `mtah.evidence.pathcrf.write_cache`, the function `pathcrf_posterior` itself uses now (refactored out; behaviour unchanged). The cache format has `ckpt, threads, input_sha, wrapper_sha, p_home, p_none, imputed_frac`; `n_chunks` and `n_ok` are not part of it and go to `assemble_summary.json`.
- Tested: `tests/test_pathcrf_remote.py` (assembled cache equals the normal path's cache for faked wrapper outputs, is a cache hit for `pathcrf_posterior`, and stale or short outputs are refused). Real check on `game_9101_H1_s100`: the wrapper run locally took 317 s for 72 451 frames (290 of 290 windows predicted, imputed fraction .296), and the assembled cache matched a second, independent local `pathcrf_posterior` computation exactly (max abs difference 0.0 on `p_home`, `p_none`, `imputed_frac`; same machine, so deterministic at 4 threads).
- Cross-machine float differences (different torch build and BLAS) are expected to be small; they are acceptable for TRAINING data. The verification against the server run (max abs `p_home` difference on one half) still has to be done after the server run.

Selection as built needs no extra-half PathCRF at all (see Part 3: extras are never decoded), so Part 2 only matters if the extras are later decoded (as test halves, or in the merge table).

### Part 3. Selection support (done)

`scripts/select_lomo.py --train-extra DIR [--extra-weight W]`:

- `DIR` is a `build_sim` output directory; its halves (flagged `extra=True`, at the requested seeds that exist there) join the training set of every fold at the cell's `extra_weight` (cells without one use `--extra-weight`, default 1.0; 0 removes them, so an `extra_weight` 0 cell is the DFL-only control and is the unchanged a5c computation).
- Weights: dead, team-control and switch models take `sample_weight` (all three support it; no subsampling needed). The merge model has no row weights; its inner split trains the stream models on the extras (weighted) and then decodes only the held DFL match, so the merge table is DFL rows.
- Cells are grouped by `extra_weight`: each group trains its own dead/team/switch/merge models per fold, and its own set for the plausibility step and the registered models.
- Never in: test folds (`lomo_folds` runs on `--sim` only), the merge-held match (`mtah.select.inner_held_split` takes the last non-extra match), `estimate_lengths`, the drift reference (test halves only), the plausibility decode (the real match only).
- The fold fingerprint (`fold_*.meta.json`) records the extra set (half count, key hash, default weight); a run is never resumed with a different or missing extra set. `registration.json` records `train_extra` and the chosen cell's `extra_weight`.
- `GRID13` (`--grid grid13`): a5c's cell (`GRID8[0]`) x `extra_weight` {0, 0.5, 1.0} x `dead_offset` {-2.0, -1.0, 0.0} = 9 cells.

Tests: `tests/test_select_extra.py` (8 tests: grid shape, grouping, weights reach the three trainers and weight 0 drops the extras, extras never in test/decode/held, resume guard, registration record), `tests/test_pathcrf_remote.py`.
