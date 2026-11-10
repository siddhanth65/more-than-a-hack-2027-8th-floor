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

## Amendment 1 (10 Nov, 23:00 IST; still before any M/K/T/B result was seen)

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
