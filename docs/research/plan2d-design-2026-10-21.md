# Plan 2d design study: next lever for spellF1 on the real match

Date 2026-10-21. Base: branch plan2c, candidate a3b (cell 52: tau_HA 30 / tau_D 100 / w_dead 4 / w_pathcrf 1 /
full / aba 1.0; sim LOMO ctrl 0.431, raw 0.488). Read-only study: code + ledger + run summaries; no match data opened.

## 0. Facts that drive the ranking

1. Metric: spells = maximal H or A runs; D splits spells. F1 ~ 2M/(Nt+Np) at each IoU. On the a1 basis
   (M 908, Nt 2692, Np 2698): +100 matched spells = +0.037; -10 % Np at constant M = +0.018.
2. Sim error is on visible frames: 1648 of 1790 unmatched predicted spells start outside blackouts; only 35 % of
   predicted H<->A changes have a true one within +-2 s; oracle boundaries +0.18 but snapping only +0.018.
   Spells < 3 s are 30 % of truth (800/2692) and are matched 2-13 %.
3. Ball-control parameters are flat (sweep 0.335-0.352). raw > ctrl by ~0.06 on every cell: the hand-crafted
   model tracks *proximity* possession (DFL raw), not *controlled* possession (ctrl, Metrica-like). That is a
   target mismatch, which a parameter sweep cannot fix and a model trained on ctrl truth can.
4. Real over-segmentation (label-free + the already-read window): a3b predicts 548 spells in 105.9 graded min =
   5.2/min; the public window truth has ~3.4/min (20 transitions in 5 min). Sim count ratio is 1.14, so the real
   ratio is ~1.5 (window = 17 spells, early match, 12 % blackout vs 41 % match-wide: CI roughly +-25 %).
   Spells per H/A-minute: real 7.3 vs sim drift median 6.16 (range 4.7-7.9). Real precision is the weak side.
5. Real evidence is weaker than sim: PathCRF proxy 0.68 vs 0.81; tag agreement implied 0.661 vs sim 0.69-0.83.
6. Real visible-dead recall is about half (window D on non-blackout frames: a3b 0.078 vs truth 0.155); D is
   fine on sim (missed 50 / false 69 of 1044 stoppages).
7. Tags: team-tagged phase coverage 0.78 non-blackout / 0.71 blackout (a1/shift.json, includes sign-0 codes, so
   signed coverage <= ~0.75); per-code nearest agreement 0.52-0.77. tag_sim emits only PROGRESSION and
   SET PIECES, so nothing that uses transition codes can be validated on the sim.
8. Single-match noise band of spellF1 is 0.04 (official SPELL_NOISE_BAND): real-only gains below that are not
   observable even at the final public read; selection must run on sim LOMO (seed SD ~0.007).

## 1. Levers

### A. Learned team-in-control stream on visible-ball frames ("teamctrl") + switch hazard (A2)
Design. `mtah/evidence/team_ctrl.py`:
- `TEAM_CTRL_FEATURES`, `team_ctrl_features(match, touches, ball_post) -> (F, K) float32`, NaN rows where the
  ball is absent. Per team (encoded as H-minus-A difference plus symmetric sum; x signed by
  `direction_of_play`): nearest and 2nd-nearest distance; relative speed |v_player - v_ball| and closing speed
  of the nearest player; cos(ball direction, nearest-player velocity); players within 2/5/10 m; ball speed,
  acceleration, `velocity_change` dv; seconds since last / until next detected touch by H and by A (capped
  10 s; offline labelling, so future context is legal) and that touch's dv; team of last strong touch; ball
  distance to line and attack-signed x; visible players per team (outage flag); seconds since the ball
  reappeared / until it disappears; centred rolling means of d1_H - d1_A over +-0.5/1/2 s; the hand-crafted
  `ball_ctrl_posteriors` p_team and p_loose (complement, not discard). No ball height exists (Match.ball is
  (F,2)); "aerial" proxy = speed > 20 m/s with no player within 3 m for >= 0.3 s.
- `TeamCtrlModel` (HGB like DeadModel: 300 iter, 31 leaves, min_samples_leaf 200), feature policies `full` /
  `no_touch` (fallback if real touch statistics drift). Target: ctrl_Dfull on visible frames whose truth is
  H/A (D rows excluded). Team-swap + pitch-mirror augmentation (no home bias; real home = Barcelona).
  Stride-5 frame subsampling (~200 k rows per fold, ~1 min fit).
- `team_ctrl_evidence(p_home, eps) -> (F,3)`: [log p, log(1-p), NaN]; all NaN where the ball is absent.
- A2 hazard: `SwitchModel` = same features + p_home deltas over +-0.5 s; target "ctrl H<->A change (or D->team
  different from the pre-D team) within +-0.25 s". Stream key "switch" consumed by the decoder, not by
  `cost_matrix`: tau_HA(t) = max(tau_floor, tau_HA - w_switch * log(h_t / mean_h)); h = mean_h (neutral) where
  NaN (blackouts keep today's behaviour). The same hazard also serves lever B.
Needs: per-fold training in `select_lomo` beside `train_dead`; `compute_evidence(..., team_model=None,
switch_model=None)` adds keys "teamctrl" / "switch"; grid keys `w_teamctrl`, `w_ball`, `w_switch`.
Expected gain. Targets facts 2 and 3 directly (false/missed H<->A changes on visible frames; ctrl-vs-proximity
mismatch). Removing 25 % of the 1648 visible false spells while restoring a third of them as matches (fixed
false splits) is +0.08 on the a1 basis; discounted for what PathCRF already fixed: sim +0.03 to +0.06, real
+0.02 to +0.05 (it also lowers real Np). A2 on top: +0.01 to +0.03. Precedent: the learned PathCRF stream gave
+0.10 where hand-tuning ball_ctrl gave nothing.
Sim-to-real risk: medium. Geometry is physical; real has no identity switches (helps velocity features); long
real outages are now in the sim (n_players p10 11.4 vs 12); real dead ball looks live (handled by D, and D rows
are excluded from training). Main risk: touch/kink statistics of the real ball (smoothing) -> compare
touches/min and dv quantiles real vs sim before trusting `full`; else `no_touch`.
Runtime: features vectorised, seconds per half; training ~1-2 min per fold per model; inference trivial.
Label-free real validation: adversarial AUC sim-vs-real on the feature table (visible frames; > 0.8 -> drop the
top shifted features); argmax agreement with PathCRF on real visible alive frames vs the same on sim (gap
<= 0.05 expected); tag_agreement_implied not below a3b's 0.661; spells per graded minute (fact 4);
touch-supported share of predicted H<->A changes (touch by the new team within +-1 s) vs its sim reference.

### B. Learned boundary placement in `snap_team_boundaries`
Ceiling is small: placement fixes are bounded by the 35 % of predicted changes that sit near a true one;
nearest-touch snap gave +0.018, a global offset +0.005. A learned start detector adds maybe +0.005 to +0.015.
Cheapest form: reuse A2's hazard: `PostParams.snap_mode in {"touch", "hazard"}` = argmax h within +-snap_w.
No separate model, zero extra training. Real risk low (bounded move). Fold into A2.

### C. Spell-structure correction
C0 real spell-rate gate (no new model). Add to `is_real_plausible`: spells per graded minute in [2.7, 4.3]
(= 3.4 x [0.8, 1.25], the band the sim bar already uses) and extend the grid with tau_HA 60 and aba_s 2.0.
Cost: an hour. Gain: sim 0 to -0.01 (it picks a stickier cell), real +0.02 to +0.05 if the 1.5x
over-segmentation holds (Np 1.5Nt -> 1.2Nt at constant M multiplies F1 by 2.5/2.2). Risk: the 3.4/min anchor
is 17 spells; use the band, never a point target.
C1 spell-level merge model (`mtah/spells/merge.py`). Per predicted H/A spell: length, team, neighbour teams and
lengths, D-adjacent flags, mean and min centred cost margin per stream (ball, teamctrl, pathcrf, tag) for its
team, blackout fraction, own-team strong touches, max hazard at start/end. Label: spurious if < 50 % of its
frames carry its team in ctrl truth. Apply greedily (highest P first) while P > theta: relabel to the
neighbours' team when both agree, else extend the longer neighbour. Honest training predictions need an inner
LOMO inside each fold (dead/teamctrl retrained on 4 of 5 train matches; PathCRF cached; HGBs cheap) ->
~5x model fits per fold, minutes. theta in the grid.
Gain: sim +0.01 to +0.03 (aba_s 1.0 already removes the easy A-B-A cases); real +0.02 to +0.05 when theta is
chosen under the C0 gate (merging is what real needs). Risk medium-low: features are relative margins, but
real margins are weaker (PathCRF gap) -> theta must be gated on real, not taken from sim.
C2 semi-Markov decoder (explicit durations at 5 fps, D_max 60 s, log-normal per state). Cost high (new
decoder + upsampling + tests); 30 % of true spells are < 3 s, so a duration prior trades recall for precision
much like a larger tau_HA. Expected +0.00 to +0.02; only if C1 leaves structure errors merges cannot express.

### D. Smart-Tagging on the real match
Already used: per-code sign/p prior (configs/tag_codes.json) and SP rows in dead features. Remaining idea:
ATTACKING TRANSITION start for X = turnover to X; DEFENSIVE TRANSITION start for X = turnover away from X.
Coverage upper bound ~0.75 of frames, per-frame agreement only 0.62-0.63 for these codes, start lag unknown.
Not testable on the sim (no transition codes in tag_sim; simulating them from truth would be circular).
Cheap label-free measurement (30 min): lag from each transition-tag start to the nearest detected touch by the
gaining team on visible frames, and to the nearest predicted change to that team. Only if |lag| < 1 s for
>= 60 % of tags: anchor stream (team bonus on [start, start + 2 s] + hazard bump at start). Expected +0.00 to
+0.02, unverifiable before the final read -> optional, last.

### E. Literature items not yet used
E1 long-context ball-free in-play features (Sci Rep 2022: player x/y/v/a at offsets +-0.4..80 s, ~3.6 s median
filter) for the dead model -> fact 6 (real visible-dead recall is half). A missed stoppage between two
same-team spells costs both matches. Real +0.01 to +0.04, sim ~0 (D already good there), validated only via
the D-share band -> strong Plan 2e candidate.
E2 MS-TCN lesson: track Edit score / over-segmentation ratio next to spellF1 (drift + selection output). A full
TCN segmenter is too heavy for 5 training matches.
E3 restart-geometry taker rule (spells starting inside blackouts match 15 % vs 40 %): partly covered by
PathCRF; enters later as a C1 / A2 feature.
E4 ELASTIC-style tag sync: only if D is pursued.

## 2. Ranking (expected gain on the REAL match)

| rank | lever | sim gain | real gain | cost | sim-to-real risk |
|---|---|---|---|---|---|
| 1 | A teamctrl (+A2 hazard, B folded in) | +0.03..0.06 (+0.01..0.03) | +0.02..0.05 | 1.5 d | medium |
| 2 | C0 spell-rate gate + C1 merge model | +0.00..0.03 | +0.02..0.05 | 1 d | medium-low |
| 3 | E1 long-context visible-dead features | ~0 | +0.01..0.04 | 1 d | medium, no sim check |
| 4 | B standalone learned snapping | +0.005..0.015 | same | 0.5 d | low |
| 5 | D transition-tag anchors | n/a | +0.00..0.02 | 0.5 d | unmeasurable |
| 6 | C2 semi-Markov decoder | +0.00..0.02 | same | 2 d | low |
A and C attack different terms (A raises M and lowers Np on sim; C lowers Np where real over-segments), so they
stack; the order between them is what the diagnostic decides.

## 3. First diagnostic (sim only; cheapest A-vs-C discriminator)

`scripts/diagnose_levers.py` -> `runs/select/a3b/levers_diag.json`. Cell 52, seed 100, all 6 LOMO test folds
(12 halves), dead model retrained per fold (as select_lomo does), PathCRF from cache. ~1-2 h CPU.
1. Error anatomy on visible alive frames (ctrl_Dfull H/A, ball present): team error rate of the ball-stream
   argmax and of the final labels; error-run length histogram (< 1 s, 1-3 s, > 3 s); for each unmatched
   predicted spell (IoU 0.3) starting on a visible frame, the mean centred ball+pathcrf margin for its own team.
   Confident-and-wrong (margin above the median margin of matched spells) => evidence problem (A);
   weak-margin flicker => structure problem (C).
2. Ceiling A: replace the ball stream on visible frames by truth log-probs with block noise at accuracy a
   (error runs resampled from the measured ball-stream error runs); a = 1.0 and a = step-4 accuracy; rescore.
3. Ceiling C: greedy oracle merge on the a3b labels (relabel a predicted spell to its neighbours' team, or
   extend the longer neighbour, iff the half's spellF1 rises; repeat to a fixed point). Merges only, i.e. no
   new information: exactly what C1 can learn.
4. Realism of A: one-fold HGB teamctrl (fold 9001, `full`, ~20 min) -> visible-frame accuracy vs the ball
   stream's; feed it into step 2.
5. Also log (label-free, sim and real): spells per graded minute, touches/min and dv quantiles, touch-supported
   share of predicted changes. These are the real-side checks reused in T4.
Decision: dA = step 2 at the step-4 accuracy, dC = step 3. dA >= 1.5 dC -> order T1, T2, T3; dC >= dA -> T3
first, then T1. Both < 0.02 -> the loss is short spells neither fixes: do E1 + A2 only.

## 4. Plan 2d task list (<= 5 tasks)

T0. Diagnostic (section 3). Script + JSON + ledger line; no pipeline change. Decides the order of T1-T3.

T1. teamctrl evidence (A).
- `mtah/evidence/team_ctrl.py`: TEAM_CTRL_FEATURES, team_ctrl_features, FEATURE_POLICIES {full, no_touch},
  TeamCtrlModel(policy).fit/predict_proba/save/load (dict {model, policy}), team_ctrl_evidence.
- `mtah/select.py`: `train_team_ctrl(sims, variant="ctrl_Dfull", policy="full")` (stride 5, swap+mirror aug).
- `mtah/pipeline.py`: PipelineConfig.use_team_ctrl, team_policy, team_model_path; `compute_evidence(...,
  team_model=None)` -> ev["teamctrl"]; policy mismatch raises like dead.
- `scripts/select_lomo.py`: per fold `team = {p: train_team_ctrl(train, policy=p)}`; ev_key gains team_policy;
  estimate_lengths "unit" covers the key; candidate artefacts save the team model.
- Tests: shapes and NaN on absent ball; swap antisymmetry (p(swapped) = 1 - p); D column NaN; policy mismatch;
  routing.

T2. Switch hazard in the decoder (A2 + B).
- `mtah/evidence/switch.py`: SwitchModel + switch_features (teamctrl features + p deltas); `train_switch`.
- `mtah/decode/viterbi.py`: AUX_STREAMS = {"switch"} skipped by cost_matrix; DecoderParams.w_switch, tau_floor;
  viterbi_path takes a per-frame tau_HA (only T[0,1] and T[1,0] vary); w_switch 0 is bit-identical to today.
- `mtah/spells/postprocess.py`: PostParams.snap_mode {"touch", "hazard"} (argmax h within +-snap_w).
- Tests: w_switch 0 identity; neutral h = old path; hazard snap respects neighbouring boundaries.

T3. Spell merge model + real spell-rate gate (C0 + C1).
- `mtah/spells/merge.py`: spell_table(labels, evidence, touches, half, ball_present) -> DataFrame, MergeModel,
  apply_merges(labels, half, proba, theta); PostParams.merge_theta (None = off), run after `_one_half`, before
  snapping.
- `mtah/select.py`: `train_merge(train_sims, cfg, ...)` on inner-LOMO honest predictions;
  REAL_SPELL_RATE_BAND = (2.7, 4.3) per graded minute in is_real_plausible (summary_real gets
  spells_per_graded_min next to spells_per_HA_min).
- Tests: merges never cross halves; theta None identity; gate band edges.

T4. Grid4 selection, sealed read, candidate a4.
- GRID4 around cell 52: w_ball {0, 1} x w_teamctrl {0, 1, 2} x tau_HA {30, 60} x w_switch {0, 1} x merge_theta
  {off, 0.6} x w_dead {3, 4} -> 96 cells (a3 ran 64 cells in ~50 min; evidence per fold once per
  (dead_policy, sp_bonus, team_policy)); team_policy no_touch only if the T0 touch check fails.
- Bar as now (ctrl variants, 5/6 folds, margin, ratio band). Real plausibility = D band [0.10, 0.18] AND spell
  rate band AND tag_agreement_implied >= 0.64; report adversarial AUC, PathCRF agreement gap, touch-supported
  share, Edit / over-segmentation ratio vs drift_reference.
- Sealed 9007 read once (logged), candidate a4, clean-clone reproduction. Public window stays unread until the
  very end (one read).

Deferred to Plan 2e: E1 long-context dead features; D transition-tag anchors (after the 30-min lag check); C2
semi-Markov only if T3 shows structure errors that merges cannot express.
