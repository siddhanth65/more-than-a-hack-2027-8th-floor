# Technical document: frame-level possession labelling (More Than A Hack 2027)

*Status: skeleton with known facts filled in. Cells marked `FILL` are produced by selection a2, the sealed read on match 9007 and the candidate drift check, and do not exist yet.*

## 1. Summary

For each of the 199,319 tactical frames we output H, A, D or X. The pipeline is deterministic (numpy, pandas, scikit-learn) and has two parts. Where the tracked ball exists it follows the ball (carrier, flight, contested). Where the ball is missing, which is 41 % of graded frames, it combines a dead-ball classifier, a Smart-Tagging prior and a players-only team model (PathCRF, an external tool) in one centred 3-state Viterbi decoder per half, then applies spell-level post-processing. X is never emitted.

The central data fact is that the ball is missing in long blackouts, not in flicker. Parameters are chosen by leave-one-match-out (LOMO) selection on a DFL open-data proxy with simulated blackouts fitted to this match, scored with the official spellF1 scorer.

## 2. Approach

### 2.1 Data facts that drive the design

| fact (graded frames, aggregates) | value |
|---|---|
| ball missing | 41 % |
| blackouts longer than 10 s | 79, holding 85 % of missing frames |
| duration of those blackouts | median 20.8 s, p90 50.5 s |
| gaps between 0.5 s and 5 s | none (the tracker needs at least 5 s to re-acquire) |

Interpolation and smoothing cannot fill a 20 s gap. The task inside a blackout is: is the ball dead, and if live, which team has it. Both are decided from player geometry, tags and restart structure.

### 2.2 Scoring

The official `evaluation/possession_metrics.py` is vendored unmodified and md5-pinned. spellF1 matches predicted possessions to true ones one-to-one by IoU (greedy, same team), averaged over IoU thresholds 0.3 to 0.7. The scorer forward-fills X, so abstaining is not a way to avoid a hand-over. The count and boundaries of possessions dominate the score; per-frame accuracy is secondary. Differences below about 0.04 on one match are noise.

### 2.3 Evidence streams

Each stream gives per-frame log-scores for H, A, D, or abstains.

| stream | what it uses | role |
|---|---|---|
| ball control | tracked ball and nearest player; carrier, flight, contested model; loose ball stays with last owner | ball frames |
| tag prior | Smart-Tagging phases, using the per-code table `configs/tag_codes.json` | team prior through blackouts |
| dead-ball model | gradient-boosted trees (HGB) on ball-out geometry at blackout entry, time since blackout start, set-piece tags, restart shapes, player-speed collapse; trained on the DFL proxy with a feature policy (`full` or `no_ball_at_all`) | D versus live |
| touches | ball-velocity kinks | strong boundary evidence |
| PathCRF team posterior | players-only model of which node owns the ball (5 fps, Kim et al. 2026), run unmodified | team inside blackouts |

Smart-Tagging `Team` semantics: it is the team performing the event. Attacking codes name the team with the ball; defensive codes name the team without it. Our first candidate treated every code's team as the possessor, which gave 0.60 tag agreement against nearest-player on the real match (0.86 to 0.96 on the simulation). The per-code sign and reliability table fixed this.

### 2.4 Decoder and post-processing

A centred 3-state (H, A, D) Viterbi per half, with switch costs `tau_HA` and `tau_D` and a per-frame centring of the cost matrix so that smoothed evidence does not swamp the switch costs. Post-processing is O(segments): fill short gaps, remove islands, merge A-B-A below a minimum duration unless a strong touch says otherwise, and snap team-change boundaries to touches within 50 frames (the snap was worth about +0.02 spellF1 in a fold experiment).

## 3. Assets used and dropped

| asset | decision | note |
|---|---|---|
| tactical tracking (players + ball) | used | main input |
| Smart-Tagging phases | used | per-code team semantics, see 2.3 |
| tactical homography | used | pitch coordinates |
| DFL open data, 6 matches (9001 to 9006) | used as proxy truth | per-frame ball-alive flag and possession team |
| DFL match J03WR9 (= 9007) | sealed test match | not touched until selection is registered |
| PathCRF (external) | used | see 3.2 |
| broadcast video and broadcast tracking | **dropped (no-go)** | census in 3.1 |

### 3.1 Broadcast fusion: tried and dropped

We measured whether the broadcast recording could supply the ball inside tactical blackouts. The two feeds share one constant clock offset: tactical frame = broadcast frame + 4014 (+160.58 s, within 0.3 s). Of the valid broadcast frames, 23 % are out of sync (replays).

| on tactical blackout frames | share |
|---|---|
| a broadcast frame exists | 87 % |
| broadcast homography valid | 41 % |
| broadcast ball present | 18.7 % |
| plausibility-checked usable broadcast ball | 1.3 % (692 frames) |

Both trackers lose the ball at the same times, so there is nothing to borrow. Broadcast fusion was dropped. The offset is recorded in case a video-based ball detector is ever tried.

### 3.2 PathCRF dependency and licence

PathCRF (https://github.com/hyunsungkim-ds/pathcrf, commit `1b7c6f85a6b78c92e4543fde72ca2b86a933f997`, checkpoint `saved/140`) is licensed MPL-2.0. It is used unmodified as an external tool through a subprocess in its own Python 3.14 venv; no PathCRF code is copied into this repository. `third_party/pathcrf/run_infer.py` is our own wrapper. Torch threads are pinned to 4 because the posterior can vary with thread count; the cache is keyed on input hash, wrapper hash, checkpoint and threads. Setup is in `third_party/pathcrf/SETUP.md`.

Robustness on match 9007, agreement with the ball-owning team on alive frames:

| variant | H1 | H2 |
|---|---|---|
| clean tracking | 92.6 % | 92.0 % |
| corrupted + carry-forward imputation | 87.3 % | 75.6 % |
| corrupted + imputation + identity switches every ~8 s | 68.0 % | 54.1 % |

Missing players must be imputed (without imputation no prediction is produced). Identity switches are the dominant risk, so the evidence is down-weighted by the imputed fraction. 9001 to 9005 are PathCRF's training matches and 9006 its validation match; 9007 is the clean test and is therefore the sealed match for PathCRF-era candidates.

## 4. Possession rules implemented

- **H / A**: the team that owns the ball while it is in play. **D**: dead ball (stoppage). A possession is a maximal run of H or of A; D breaks it.
- **Loose and contested balls** stay with the last owner. A change of team requires controlled possession; a touch, deflection or aerial duel without control is not a change.
- **X** is never written. The scorer forward-fills X with the last label in the half, which is a free repeat inside a possession and costly across a hand-over, so a label is always committed.
- **Controlled-possession notion**: DFL raw truth counts every touch as a possession (about 8.8 per minute), while the competition's ground truth is closer to 3.4 per minute in the public window. We therefore score against four truth variants: raw and `ctrl` (touch-only possessions removed), each on the full and on a narrowed dead definition. `ctrl` is the variant that matches the competition's definition and is the one on which the possession-count ratio is judged.
- **Restarts**: kick-offs and throw-ins carry no team in the tags; the restart structure (taker position, cluster shapes) feeds the dead-ball model. Inferring the taker's team after the restart is not implemented.

## 5. Validation protocol

One real match with no private ground truth cannot be used for tuning, so:

1. **Proxy**: 6 DFL matches plus the sealed 9007, 5 seeds of simulated blackouts each.
2. **Blackout simulator** fitted to the real gap statistics (79 blackouts, median 20.8 s). Dead-ball behaviour on visible frames was recalibrated to the real tracker (ball static 2 %, out of pitch 3 %, median speed about 5 m/s).
3. **Corruption model** calibrated label-free to the real tracker: player outages (inside blackouts mean 18.3 players, p10 11.4; real 18.8 and 12), mass outages, identity-swap rate (`swap_rate_per_team_min` 0.2 per team-minute; `flip_rate_per_min` 0.02 is the team-flip rate), teleports about 0.025 per player-minute (real about 0).
4. **LOMO selection** over grid `grid2` (tau_HA, tau_D, dead weight, PathCRF weight in {0, 0.5, 1, 2}, dead policy). Cells are ranked by mean rank over the four truth variants. Spec bar: margin over the best baseline `>= max(0.05, 2x seed SD)` on every variant, AND folds won on ctrl_Dfull `>= n_folds - 1`, AND the ctrl_Dfull and ctrl_Dnarrow count ratios (n_pred/n_gt) in [0.8, 1.25] (raw ratios are recorded only).
5. **Sealed match**: 9007 is read once, after selection is registered; the read is logged. Disclosure: 9007 is sealed for the pipeline (never scored by `select_lomo`), but its labels were used for the PathCRF clean evaluation (92.6 %), the PathCRF robustness experiment, the choice of the PathCRF output head (`run_infer.py`: the emission marginal was picked on 15 000 frames of the corrupted sim half game_9007_H1 s100 against the ball-owning team), and the ctrl-truth diagnostic (66 % disagreement on 9007 H1). The sealed a2 result is reported with this caveat.
6. **Real-match plausibility (label-free)**: the candidate's non-blackout D share on the real match must lie in [0.10, 0.18] (the public window implies about 0.15 on visible frames). The first candidate (a1) had 0.028 on visible frames with a dead model trained on unrealistic visible-ball dead time; this criterion is what exposed it.
7. **Public 5-minute window**: reads are budgeted at 3, used only for conventions and plausibility, never for tuning. 2 of 3 are used; the last is reserved for the final candidate.

### Earlier candidate a1 (for context)

| item | value |
|---|---|
| best a1 cell, ctrl_Dfull on proxy | 0.344 (smooth50 baseline 0.174) |
| a1 on sealed 9006 | raw 0.445 / ctrl 0.375 |
| a1 public window read | spellF1 0.290 (smooth50 0.326) |
| a1 real-match visible-frame D share | 0.028 (out of range) |

The public read showed a1 below the baseline on the 5-minute window. The causes found were (a) the tag-team semantics and (b) the visible-ball dead realism, both addressed in plan 2b.

## 6. Results

### 6.1 Candidate a2

Selection a2 used grid2 (64 cells), leave-one-match-out (LOMO) over DFL matches 9001 to 9006, seeds 100 and 101, ranking by mean rank over the four truth variants. The chosen cell is 60: tau_HA 30, tau_D 100, w_dead 2, w_pathcrf 1.0, dead policy full, aba_s 1.0.

**LOMO spellF1 on the simulated-blackout proxy, chosen cell vs the organisers' baseline_smooth50**

| truth variant | a2 | baseline_smooth50 |
|---|---|---|
| raw_Dfull | 0.527 | 0.233 |
| ctrl_Dfull | 0.471 | 0.208 |
| raw_Dnarrow | 0.503 | 0.248 |
| ctrl_Dnarrow | 0.444 | 0.220 |

Folds won (ctrl_Dfull, margin of at least 0.05): 6 of 6. Seed SD at most 0.002. Count ratios: ctrl 1.13 / 1.13 (raw 0.89 / 0.89). Spec bar: pass.

**Effect of the PathCRF stream on the simulation** (mean ctrl_Dfull over all cells sharing that setting)

| w_pathcrf | 0 | 0.5 | 1.0 | 2.0 |
|---|---|---|---|---|
| ctrl_Dfull | 0.345 | 0.445 | 0.433 | 0.397 |

| dead policy | full | no_ball_at_all |
|---|---|---|
| ctrl_Dfull | 0.415 | 0.396 |

PathCRF agreement with the label-free nearest-player proxy: simulation mean 0.812 (range 0.62 to 0.89), real match 0.677.

**Sealed read 2** (match 9007, our own parse of DFL J03WR9, never used in selection; forced past the plausibility gate, reason logged)

| raw_Dfull | ctrl_Dfull | raw_Dnarrow | ctrl_Dnarrow |
|---|---|---|---|
| 0.528 | 0.464 | 0.510 | 0.438 |

**Real-match candidate a2** (label-free drift check)

| quantity | value |
|---|---|
| spells | 518 |
| spells per possession-minute | 5.96 |
| median spell | 5.3 s |
| predicted D share, overall | 0.180 |
| predicted D share, inside / outside blackouts | 0.425 / 0.050 |
| team changes inside blackouts | 15 % |
| implied-team tag agreement | 0.669 |

**Public 5-minute window** (7,500 labelled frames; used here only as an aggregate, no scoring read taken)

| D share | truth | candidate a2 |
|---|---|---|
| overall | 0.196 | 0.055 |
| visible-ball frames | 0.147 | n/a |
| outside blackouts | 0.155 | 0.023 |

For context, the earlier candidate a1 scored spellF1 0.290 on public read 2 (baseline 0.326 on that window) and 0.445 raw / 0.375 ctrl on sealed 9006.

**Known weakness.** a2 is the PathCRF-era baseline. Its known weakness is dead-ball under-detection on visible-ball frames: the candidate's D share outside blackouts is 0.023 against a truth of 0.155, and the real-match plausibility criterion (non-blackout D share in [0.10, 0.18]) failed for all top-5 cells (0.047 to 0.076). The next iteration targets this.

| table | content | status |
|---|---|---|
| selection a2 (grid2, 6 folds 9001-9006, 2 seeds) | best cell, four variant scores, margins over baselines, folds won, seed SD | FILL |
| sealed match 9007 | raw and ctrl spellF1 of the registered cell | FILL |
| candidate drift on the real match | spell count, non-blackout D share, tag agreement | FILL |
| final public window read (3 of 3) | spellF1, macroF1 | FILL |

## 7. Limitations

- 9007 label exposure: the sealed match was used for PathCRF evaluation, robustness, output-head choice and the ctrl-truth diagnostic (see section 5); the sealed a2 score is therefore not a fully untouched estimate.
- One real match: no real-data tuning set; the proxy is DFL with simulated blackouts, and DFL possession and dead definitions are not identical to the competition's (hence four truth variants).
- The simulator under-represents the worst real tracking: real long player outages (p90 about 102 s) versus simulated (p90 about 35 s). PathCRF may be over-rated by the simulation; its accuracy falls to 54 to 68 % under identity switches.
- Error analysis of a1 on the proxy: only 34 % of true possessions were matched at IoU 0.5; short possessions under 3 s are rarely matched; the largest loss is missed or false team changes on visible-ball frames and the team at blackout starts. Oracle boundaries would add about +0.18, so boundary placement remains the largest lever.
- Ball-control parameters were flat in a sweep; further gains need new evidence, not retuning.
- Dead-ball realism on visible frames is calibrated only on aggregates (static, out, speed p50).
- With more time: a video-based ball detector on the broadcast feed (the offset is known); a semi-Markov decoder; set-piece taker rules.

## 8. Reproduction

```
$env:MTAH_DATA_DIR = "<unpacked data bundle>"
scripts\reproduce.ps1 -CandidateOnly      # regenerate runs/candidates/a2.csv from the committed models/a2/
scripts\reproduce.ps1                     # full chain: simulator, PathCRF cache, selection, candidate
```

- `scripts/reproduce.ps1` checks the mtah venv (Python 3.12, `MTAH_VENV`; pins in `pyproject.toml`), `MTAH_DATA_DIR`, and PathCRF (`MTAH_PATHCRF_REPO`, `MTAH_PATHCRF_PY`), then prints the md5 of the candidate.
- PathCRF setup, pinned versions and the thread pin: `third_party/pathcrf/SETUP.md`.
- The full chain first runs `scripts/parse_dfl_match.py` (in the repository) on the raw DFL XML (`E:\football-synthesizer\data\dfl_open`, or `MTAH_DFL_RAW`) to produce `runs/dfl_extra` (match 9007); the raw XML is not in the repository; `-CandidateOnly` needs only the competition data and PathCRF.
- Output file: `runs/candidates/a2.csv` (format checked with the official `validate_submission.py`). md5: `FILL` (placeholder; the clean-clone regeneration check is in progress).

## 9. Rules compliance

- No hand labels. No private ground truth was used; the ranking metric was only run against the public 5-minute window (3 reads at most, none used for tuning) and DFL proxy truth.
- No competition data was uploaded to any cloud LLM service; the data stays on the local machine and is not in the repository.
- The repository regenerates the submission file byte-for-byte from the committed config and dead model, verified on the same machine, Python and pinned packages (`pyproject.toml`) with the PathCRF thread count pinned to 4. The PathCRF/torch CPU posterior is cached and hashed; identity across machines is not yet shown.
- External tool: PathCRF (MPL-2.0), unmodified, documented in 3.2. DFL open data is used as proxy under its own terms.
