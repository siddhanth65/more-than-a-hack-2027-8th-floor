# Plan 9 pre-registration: window-free final selection (committed 2026-11-10)

Written before any result of the transfer test (Exp1), the masked-ball test (Exp2), the
Metrica-sample validation or the SkillCorner natural-visibility hold-out was seen.

## Why

The organisers' rule text of 23 Oct 2026 says the public 5-minute window is for checking that the
pipeline is correct, "never to choose between two versions of it", and is not a tuning set. We
pulled that text on 10 Nov. Our read 7 (27 Oct) applied the rule "a16 iff window spellF1 >= 0.62,
else a5c"; that decision is void. Our two real-match plausibility bands were anchored on window
aggregates; they are re-anchored below on other-game data. Audit:
`docs/research/plan9-compliance-and-selection-2026-11-10.md`.

From this commit on, no spellF1 or macroF1 is computed on the public window.

## Re-anchored admissibility bands (other games only, numbers recorded 23 Oct in `runs/extra/report.json`)

- Visible-dead share of the real-match prediction in [0.116, 0.181] = [0.8, 1.25] x 0.145
  (SkillCorner open data, 40 halves).
- Implied truth spell rate R / rho in [2.28, 4.10] spells per graded minute, where R is the
  predicted rate on the real match and rho is the same configuration's simulator count ratio
  (predicted / true, ctrl truth). The range is the span of controlled-possession truth rates in
  SkillCorner (2.28), DFL (3.69) and the Metrica sample (4.10).
- Dead-prior offsets are set by one rule for every candidate: the simulator-best admissible offset.

## Amendment 1 (10 Nov, 22:20 IST; still before any M/K/T/B result was seen)

A label-free diagnostic (`possession-diag`, `runs/diag/REPORT.md`) compared the length of the
dead run each candidate predicts before a tagged restart with other-game norms from SkillCorner
open data (dead time before a throw-in p25 10 s, free kick 19 s, goal kick 24 s, corner 31 s,
kick-off after a goal 64 s). Where the ball is visible before the tag, candidates call only
~4 s dead before throw-ins and goal kicks; corners are short too. The camera-motion cue failed
its pre-set bar and is dropped.

The pool gains one optional modifier, R: a type-specific pre-restart dead prior whose window
lengths and strength come ONLY from SkillCorner matches (fitted on 10 matches), applied to P, L
and H. R is accepted for a base candidate iff, on the other 10 SkillCorner matches of the K
hold-out (tags synthesised from SkillCorner's typed restart events at the contest match's tag
rate), base+R leads base by mean spellF1 >= 0.01 and wins >= 60 % of halves, AND base+R is
admissible on the contest match with its dead offset re-set by the usual rule. If accepted,
base+R replaces base everywhere in the procedure below. Same freeze and deadlines.

## Amendment 2 (10 Nov, late evening IST; before any K score was computed)

The K hold-out as built has the ball missing on only 5.7 % of graded frames (SkillCorner's
ball-missing frames coincide with its no-player frames, which are ungraded), so it tests the
visible-ball regime, not the contest match's 41 % blackout regime. K is therefore scored on two
builds of the same 40 SkillCorner halves: K-nat (natural gaps) and K-sim (the same halves with
our blackout simulator applied, `runs/sim_extra2`). K favours X iff X leads by mean spellF1
>= 0.01 and wins >= 60 % of halves on K-sim AND X does not trail by 0.01 or more on K-nat.
The acceptance test of modifier R uses K-nat and K-sim pooled over the 10 test matches.

## Procedure

1. Pool, frozen: {a5c, a13r, a10, a14, a16, H, S}.
   H = the a14 build with the learned-stream weights halved. S = a14 with cross-half
   self-training on algorithmic pseudo-labels, only if registered by Sat 14 Nov 12:00 IST.
   Nothing is added after that.
2. Admissible iff: both bands hold, the clean-clone md5 matches, the official validator passes.
3. Representatives, fixed now from the simulator only: P = a5c (hand-built streams);
   L = a14 (learned sequence streams). a16 replaces a14 only if within 0.01 on LOMO ctrl_Dfull
   (it is not). S replaces a14 iff S >= a14 - 0.005 on its simulator acceptance run and is
   admissible.
4. Signals; each favours P, favours L, or is "none":
   - M (Metrica sample games 1-3, simulated blackouts, truth derived from the public event data):
     favours X iff X leads by mean spellF1 >= 0.02 and wins >= 4 of 6 halves.
   - K (SkillCorner hold-out using its own ball-detection gaps as blackouts, >= 30 halves):
     favours X iff X leads by mean spellF1 >= 0.01 and wins >= 60 % of halves.
   - T (Exp1, players-only stream accuracy on confident ball-control frames, ball hidden,
     simulator vs real): favours P iff gap(B) - gap(PathCRF) >= 0.05 and B's real accuracy
     <= PathCRF's; favours L iff B's real accuracy >= PathCRF's + 0.03 and
     gap(B) <= gap(PathCRF) + 0.02. (B = the BiGRU sequence stream; gap = simulator accuracy
     minus real accuracy.)
   - B (Exp2, masked ball on the real match): counts only if the proxy passes on the simulator
     (mean Kendall tau >= 0.5 against true spellF1, tau > 0 on >= 5 of 6 folds, and a13r < a14
     ordered as in the truth on >= 5 of 6 folds). It favours X iff X beats the rival when scored
     against the RIVAL family's unmasked decode, with bootstrap probability >= 0.90.
5. Decision:
   - P iff at least 2 signals favour P and none favours L.
   - H iff exactly one signal favours P, or signals conflict, and H is admissible and within 0.01
     of L on LOMO ctrl_Dfull.
   - Otherwise L.
   - A signal not finished by Sun 15 Nov 18:00 IST counts as "none".
   - Residual ties: sealed 9007 ctrl_Dfull, then fewer streams.
6. Final file: official validator and row count; one boolean check on the public window
   ("best frame offset is 0 and H/A are not swapped"), with no score printed. Only a demonstrable
   alignment or orientation bug may change the file.

## Amendment 3 (11 Nov 2026, 11:26 IST; S recipe, written before any S training or scoring)

Selected for S in Procedure step 1 and step 3. Nothing in this amendment has been run. The sealed match 9007 is not
read for S.

1. **Definition of S.** S is the a17 build (same evidence streams, decoder, post-processing, PathCRF smoothing, offsets
   rule) with the BiGRU sequence stream B (`seqB`) fine-tuned by cross-half self-training. For a target match half h,
   B is fine-tuned on (a) the original simulator training data (the same windows and augmentation as B's own
   training) plus (b) pseudo-labelled frames from the OTHER half of the same match only, then used to predict half h.
   No frame's own pseudo-label ever trains the model that predicts it. Each half therefore has its own fine-tuned copy
   of B; the starting point of every copy is the same base B (in the simulator folds: the fold-final B trained on the
   five training matches; on the contest match: `models/a17/seq_modelB.pt`). Input normalisation constants of the base
   stay fixed.
2. **Pseudo-labels (algorithmic only: no human input, no ground truth of the target).**
   (i) H or A on confident-ball-control frames by the frozen rule of `docs/arb/rule.json` (`scripts/arb_common.py`,
   `confident_control`: nearest player within 1.5 m, nearest opponent at least 5 m, ball speed at least 1 m/s, ball
   moving at most 5 m/s relative to that player, same player for at least 1 s, middle of the run only, 0.5 s trimmed
   from both ends; precision 0.971 against simulator truth). A 5-frame step takes the label of its centre frame.
   (ii) V3 only: the labels of a17 itself (a17 decode with the base B, never a fine-tuned B) on frames whose merged
   posterior is at least 0.9. Merged posterior of frame t = softmax over {H, A, D} of minus the decoder's per-frame
   cost row (`mtah.decode.viterbi.cost_matrix` of all a17 streams including the base `seqB`, at the a17 weights and
   offsets); the frame qualifies iff the posterior of its final decoded label is at least 0.9; the label is the final
   decoded label. Where (i) and (ii) disagree, (i) wins.
   Frames 11185 to 18685 (inclusive) of the contest match, the organisers' public window, are EXCLUDED from every
   pseudo-label set; no score is ever computed on the window.
3. **Variants (at most three, fixed here).** V1 = (i) only, pseudo-label weight 1.0 relative to a simulator frame.
   V2 = (i) at weight 0.5. V3 = (i) plus (ii), weight 0.5 (this adds D labels). A fourth run, V0 = the identical
   fine-tuning schedule with no pseudo-labels, is a diagnostic only (separates the effect of pseudo-labels from the
   effect of extra training) and cannot be chosen. The variant is chosen on the simulator leave-one-match-out run only:
   best mean ctrl_Dfull. This is a simulator-only choice.
4. **Fine-tuning schedule (fixed in `scripts/s_selftrain.py`, not tuned).** Adam, learning rate 5e-4 (the base training
   used 3e-3), weight decay 1e-5, gradient clipping 1.0, batch 64, 4 epochs, torch and numpy seeds 0, deterministic
   CPU torch. Each epoch: every pseudo window (600 steps, stride 300, random offset) of the other half that holds at
   least 30 labelled steps, plus twice as many windows drawn at random from the simulator training sequences (random
   offsets, team-swap augmentation, as in base training); pseudo windows are team-swap augmented at random. Loss =
   cross-entropy over labelled steps with per-step weight (1 simulator, w pseudo) normalised by the batch's weight
   sum, plus 0.5 times the boundary BCE on simulator windows only.
5. **Acceptance (stricter than the original rule S >= a14 - 0.005).** The chosen S must reach leave-one-match-out
   ctrl_Dfull >= a17's 0.5622 - 0.005 = 0.5572, computed with the same harness as a17 (decode cell dead offset -3.0,
   stream-B D offset -0.5, B weight 12, `exp_seq_lomo.py` decoder; scored with the vendored organisers' scorer), the
   same folds 9001 to 9006 and the same simulated blackouts (seeds 100 and 101). In simulation the held-out match gets
   the same cross-half procedure, with pseudo-labels computed by the same rule from its simulated feed. The a17
   reference is re-decoded in the same harness from the registered a17 fold predictions; if it differs from 0.5622 by
   more than 0.0005, the re-run value minus 0.005 is the bar and both are reported. It must also be admissible on the
   real match: visible-dead share of the real-match prediction in [0.116, 0.181], R / rho in [2.28, 4.10], the dead
   offset set by the usual rule (simulator-best admissible offset), the official validator passes, and the clean-clone
   md5 reproduces.
6. **Consequence.** If S is accepted it replaces a17 as the L-family representative. The already computed signals
   M, K, T and B (all "none") stand, so the decision is S. If it is not accepted, a17 stays. Negative results are
   disclosed.
7. **Deadline.** S is registered by Sat 14 Nov 2026 12:00 IST or it is dropped.

## Amendment 3b (11 Nov 2026, 11:45 IST; correction of the fine-tuning schedule of Amendment 3 item 4, before any S prediction was scored)

What happened. After the base models were retrained and the pseudo-labels were built for the six simulator folds, the
first fine-tuning jobs were started. Their logs showed that the schedule of item 4 is degenerate for V1 and V2: the
confident-ball-control rule labels only about 2.6 % of the steps of a half (about 350 of 13,500), so a window needs at
least 30 labelled steps of 600 in only 0 to 19 of about 90 windows per half (median 10); in the jobs seen the epoch held
2 pseudo windows and 6 windows in all, i.e. one gradient step per epoch, a no-op. This was seen from label counts and
job logs only. No fine-tuned prediction has been decoded or scored; the partial outputs of those jobs were deleted
unread, and the pseudo-label files, base models and a17 reference are untouched. Amendment 3 items 1 to 3 and 5 to 7
(definition, pseudo-labels, variants and weights, acceptance, deadline) are unchanged.

Replacement for item 4 (fixed here, not tuned on any score): Adam, learning rate 5e-4, weight decay 1e-5, gradient
clipping 1.0, batch 64, 8 epochs, seeds 0, deterministic CPU torch. Each epoch: every pseudo window (600 steps, stride
150, random first offset below 150) of the other half that holds at least 5 labelled steps, plus the same number of
simulator windows drawn at random from the simulator training sequences (team-swap augmentation, as before); pseudo
windows are team-swap augmented at random. Loss as before: cross-entropy over labelled steps with per-step weight (1
simulator, w pseudo, i.e. per labelled frame, not per window) normalised by the batch's weight sum, plus 0.5 times the
boundary BCE on simulator windows only. V0 (diagnostic, weight 0) uses the same schedule.

## Amendment 4 (15 Nov 2026, 02:15 IST; period-handling bug fix, written before the rebuild)

What. Graded frames that precede a half's kick-off frame must be D: the organisers' FAQ states that the graded second
half starts at video frame 110,808 but kicks off at 113,742, and that those 2,934 graded frames are all D (the same
follows from the provided period data, the half video blocks of BAR-ATH). The pipeline defines the kick-off frames
(mtah/io/periods.py) but never used them, so the registered candidate S labels most of that block H or A.

Rule (post-decode, deterministic). For each half, every graded frame from the half's first graded frame up to, not
including, its kick-off frame is set to D. For half 1 the graded block starts at the kick-off frame, so the rule is a
no-op there. The kick-off frames are read from the existing constant, not from any score.

Status. This is a period-handling (alignment) bug of the kind step 6 of the Final-file section allows. It was found by a
label-free review of the predicted file against the FAQ, not by any window score. The public window (frames 11185 to
18685) lies in the first half and is unaffected by the rule. The fix applies to the selected candidate S only (new name
s_ko, built from the same models/s); it does not choose between candidates, and no other setting, offset, threshold or
cell changes. The new file must equal s.csv byte for byte outside the forced block; the official validator, the
admissibility statistics (reported both with and without the block) and the clean-clone md5 are reported afterwards.
Candidates s and a17 stay reproducible unchanged (the rule is behind a flag that is off by default).

## Amendment 5 (15 Nov 2026, 05:40 IST; set-piece tag lead, written before any stand-in was built or scored and before any corrected candidate was run)

Why. An external review suspects that `mtah/sim/tag_sim.py` (`sp_before_s = 5.0`, `sp_after_s = 30.0`: tag starts 5 s before the
restart and is 35 s long) writes set-piece Smart Tags at a different lead than the real match, where the two published kick-offs
(FAQ frames 11185 and 113742) sit 3.52 s and 3.28 s after their `Kick Off (Start)` tag starts. The dead-ball features (`sp_dist_s`,
`sp_inside`, `sp_type`, the set-piece team prior) are read from tag starts, so a lead mismatch could shift where restarts are
called live. No public-window truth is read, no window score is computed, and frames 11185 to 18685 are excluded from the
measurement below (any tag whose [start - 10 s, end + 10 s] touches the window is dropped).

Measurement (label-free, ball only; `scripts/lead_measure.py`, detector fixed before its output was looked at except that a first
speed-threshold version found too few static balls on this noisy ball track and was replaced by the displacement detector
described in the script header; aggregates only). Restart = last frame of a visible ball that stayed within 1.0 m of its median
position for at least 1.0 s before it left that disk and was at least 4 m away within 1.5 s; search window tag start +- 10 s;
when several qualify, the one closest to start + 3.4 s. Lead = restart - tag start, seconds, median [p25, p75], counts used / tags:

| tag type | used / tags | lead median [p25, p75] | tag length |
|---|---|---|---|
| Throw-in | 13 / 26 | 3.96 [2.04, 5.24] | 35.0 s |
| Goal Kick | 5 / 9 | 4.36 [3.44, 5.20] | 35.0 s |
| Corner Kick | 1 / 12 | 9.6 | 35.0 s |
| Free Kick | 0 / 4 | none | 35.0 s |
| Kick Off (After Goal) | 3 / 4 | 8.60 [6.72, 8.96] | 5.0 s |
| Kick Off (Start), outside the window | 0 / 1 | none | 5.0 s |
| types with fewer than 8 used, pooled | 9 | 5.20 [4.36, 8.60] | |
| all types pooled | 22 / 56 | 4.46 [2.31, 7.74] | |

The two published kick-off leads (3.52 s, 3.28 s) are the only ground-truth-like anchors; the detector does not find the
second-half kick-off (frame 113742: ball not visible and static), so it is unvalidated against them, and its three
kick-off-after-goal values (6.7 to 9.0 s, beyond the 5 s tag length) are not credible as leads. Only 39 % of the tags yield a
restart (the ball is missing on 41 % of graded frames). Both facts are reported as limits of this measurement, and the rule
below is nevertheless applied exactly as fixed: the measured value is used where the count rule says so, with no manual
override. Typed real tags are 35.0 s long (like the simulator's 5 + 30), kick-off tags 5.0 s (the simulator writes 35 s for
kick-offs as well; that second mismatch is outside this amendment and is not changed).

Rule for the leads (fixed): a type with at least 8 usable restarts gets its own median; every other type shares the pooled
median of the usable restarts of the types below 8. Therefore lead(Throw-in) = 3.96 s and lead(every other type) = 5.20 s.
Relative to the simulator's 5.0 s: throw-in tags start 1.04 s later than the simulator writes them (stand-in shift +1.04 s),
other types 0.2 s earlier (stand-in shift -0.2 s). The tag length stays 35 s (start and end move together).

Candidate (a'), chosen over (a). (a) would re-fit every tag-consuming model (dead model, team model, merge model, the six
cross-fitted pair models and fold models, six fold bases of B, the per-half self-training of S) on re-simulated tags; that chain
does not finish by Mon 16 Nov 18:00 IST. (a') = s_ko with the real-match set-piece tag times (start and end, `SET PIECES` rows
only) moved by the opposite of the stand-in shift (throw-in -1.04 s, other types +0.2 s) before any feature is built, with the
cross-half self-training of B and the post-decode kick-off rule redone on those tags and every other setting, model, offset and
threshold unchanged. For every model that reads tag timing, (a) trained and applied at the measured lead and (a') trained and
applied at the simulator lead 5.0 s see the same relative tag timing; on a stand-in whose tags use the measured lead, (a')
reads the stand-in with the shift undone, that is, exactly the lead-5.0 stand-in. Hence the corrected candidate's score on a
stand-in equals s_ko's score at lead 5.0 by construction (checked in the code: shift then unshift reproduces the tags
exactly). Variant (b), a kick-off law, is not built: the pipeline does not contain one, but the stand-in data cannot test it
independently of the lead and it was marked optional.

Stand-ins. Every simulated or other-game half of the sets below gets its SET PIECES rows moved by +1.04 s (throw-in) or -0.2 s
(other types) in `load_half_sim` (opt-in environment hook, off by default). Sets: LOMO DFL 9001-9006 (12 halves, ctrl_Dfull and
raw_Dfull, seeds 100 and 101 averaged per half), K-sim (40 halves), K-nat (40), sealed 9007 (2 halves), M (6 halves).
Reference = s_ko on the same halves at lead 5.0 (stored harness outputs, recomputed on LOMO from the stored fold predictions).
Mismatch = s_ko models (trained at lead 5.0, LOMO fold models and fold bases unchanged) evaluated on the shifted stand-ins, with
S's per-half self-training redone on the shifted held-out halves exactly as on the real match. LOMO: the held-out match's halves
are recomputed with the fold's dead and team models; training sequences of B stay the original cross-fitted ones.
Diagnostic only, cannot be adopted: D34 = the same mismatch test with every set-piece type at 3.4 s (the reviewer's
hypothesis), reported if time allows.

Adoption rule (fixed now, unchanged from the instruction): adopt (a') over s_ko only if all of
1. on LOMO DFL 9001-9006 (12 halves) AND on K-sim (40 halves), the mean ctrl_Dfull spellF1 of the corrected candidate minus the
   mismatch run is >= 0 on both sets and >= +0.003 on at least one;
2. no set (sealed 9007, K-nat, M) drops: corrected minus mismatch >= -0.003 on each;
3. on the real match the corrected candidate (the whole S chain re-run on the moved tags, kick-off rule applied) still passes the
   label-free checks: (i) tag agreement against the ORIGINAL tags not below s_ko's minus 0.02; (ii) visible-dead share, with the
   organiser-forced pre-kick-off block excluded as in Amendment 4, in [0.116, 0.181]; (iii) R / rho in [2.28, 4.10] with R
   block-excluded and rho = 0.8921 (S's registered simulator ratio; unchanged by (a')).
Reported for reference: s_ko at lead 5.0 on every set; per-half wins; per-set means for raw_Dfull. The public window is not read.
The decision is made by the caller, not by this amendment's code. If a step cannot finish by 16 Nov 18:00 IST the experiment
stops and reports; it does not substitute another rule.
