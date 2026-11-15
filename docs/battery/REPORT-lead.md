# Set-piece tag lead experiment (Amendment 5), report

Branch exp-lead (from final). Pre-registration: docs/reads/plan9-preregistration.md, Amendment 5 (commit 2bca9e1, written before any
stand-in or corrected-candidate run). Public window not read; no score computed on the real match. Aggregates only.
Runs: a university compute server, CPU only. Aggregate files: docs/battery/lead/agg_main.json, agg_d34.json, real_labelfree.json.

## Measured leads (restart from the ball, tags start to restart, seconds)
Throw-in 3.96 [2.04, 5.24] (13 of 26 tags usable); Goal Kick 4.36 [3.44, 5.20] (5/9); Corner 1/12; Free Kick 0/4; Kick Off (After Goal)
8.60 [6.72, 8.96] (3/4, not credible: beyond the 5.0 s tag); types below 8 usable pooled 5.20 [4.36, 8.60] (n 9); all pooled 4.46
(n 22/56). Published kick-off leads 3.52 s and 3.28 s. The detector did not find the 2nd-half kick-off, so it is unvalidated against
the only anchors. Rule applied as fixed: lead(Throw-in) = 3.96, all other types 5.20 (stand-in shifts +1.04 s and -0.2 s).
Typed real tags are 35.0 s long, kick-off tags 5.0 s.

## Candidate
(a') chosen: refitting (a) cannot finish by 16 Nov 18:00 IST. Corrected candidate on a stand-in = s_ko at lead 5.0 by construction
(shift then unshift reproduces the tags exactly, unit-checked), so "corrected minus mismatch" = "s_ko at lead 5.0 minus s_ko on the
shifted stand-ins". Kick-off law (b) not built.

## Chain checks
- The server chain with no shift reproduces s_ko on the real match: md5 2b908887901aad725558de040f3874b6, both fine-tuned B
  checkpoints byte-identical to models/s.
- LOMO chain with no shift reproduces the stored LOMO harness (12 of 12 halves tied; ctrl_Dfull 0.5665, raw_Dfull 0.5836).

## Stand-in results (mean spellF1; per half = mean over seeds; wins/ties/losses = corrected vs mismatch)
ctrl_Dfull:

| set | halves | s_ko at lead 5.0 (= corrected) | s_ko on measured-lead stand-ins | corrected - mismatch | w/t/l |
|---|---|---|---|---|---|
| LOMO DFL 9001-9006 | 12 | 0.5665 | 0.56651 | -0.00003 | 5/1/6 |
| K-sim | 40 | 0.4151 | 0.41390 | +0.00121 | 20/6/14 |
| K-nat | 40 | 0.5345 | 0.53333 | +0.00114 | 22/8/10 |
| sealed 9007 | 2 | 0.5760 | 0.57663 | -0.00062 | 1/0/1 |
| M | 6 | 0.6148 | 0.61520 | -0.00041 | 2/0/4 |

raw_Dfull corrected - mismatch: LOMO -0.00020 (6/0/6), K-sim +0.00104 (21/7/12), K-nat +0.00109 (20/10/10), sealed -0.00141, M -0.00059.

Diagnostic D34 (every type at 3.4 s, never adoptable), ctrl_Dfull corrected - mismatch: LOMO -0.00073 (6/0/6), K-sim +0.00167 (24/4/12),
K-nat +0.00132 (21/7/12), sealed -0.00311 (0/0/2), M -0.00151 (2/0/4). Moving the sim tags to a shorter lead does not hurt the
stand-ins on LOMO, sealed and M (the stand-in is a little better there) and costs about 0.0013 to 0.0017 on K.

## Real match, label-free (original tags used for tag agreement; the pre-kick-off block excluded as in Amendment 4)
| | md5 | tag agreement | visible-dead | R / rho | graded frames differing from s_ko |
|---|---|---|---|---|---|
| s_ko (re-run) | 2b908887901aad725558de040f3874b6 | 0.82034 | 0.17686 | 3.257 | 0 |
| corrected (a') | 655a56260cc647a37d8598230676777c (server build, not validated, see below) | 0.82031 | 0.17526 | 3.224 | 459 of 199,319 (0.23 %), all outside the forced block |

All three checks pass (tag agreement not below s_ko minus 0.02; visible-dead in [0.116, 0.181]; R/rho in [2.28, 4.10]).

## Verdict of the rule
1. LOMO -0.00003 (< 0) and K-sim +0.00121, K-nat +0.00114: the >= 0 condition fails on LOMO and no set reaches +0.003 -> FAIL.
2. No set drops by more than 0.003 (sealed -0.00062, K-nat +0.00114, M -0.00041) -> pass.
3. Real-match label-free checks -> pass.
Overall: NOT adopted. s_ko stays. No submission file was built or validated; the corrected file above exists only on the server and
was written by make_candidate before its manifest step (the server tree is not a git repo) so it is neither a registered nor a
validated candidate.

## Reading
The lead mismatch costs at most about 0.001 spellF1 on K and nothing measurable on LOMO, so even a corrected lead is a small
effect; the measurement itself is weak (22 usable restarts, wide IQR, detector unvalidated), and the kick-off tags, whose lead is
known, are exactly where it cannot be measured. Kick-off tags are also 5 s long on the real match versus 35 s in the simulator, a
second mismatch that this experiment does not touch. Sealed 9007 was read for the rule (as pre-registered); no record was added to
docs/reads/sealed.jsonl on this branch.
