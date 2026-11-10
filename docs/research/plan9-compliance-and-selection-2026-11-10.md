# Plan 9: public-window compliance audit, band re-anchoring and a window-free selection rule (2026-11-10)

Read-only review. Sources: the ledger `.superpowers/sdd/progress.md` (gitignored, so it is not in the repository; "L" numbers below are its lines), `docs/technical-document.md`, `docs/reads/*`, `docs/research/*`, the `exp-seq`/`exp-seq2`/`exp-robust`/`exp-data` worktrees, aggregate JSON/CSV under `runs/select` and `runs/extra/report.json`, and `git -C ../repo diff HEAD origin/main`. No tracking or ground-truth rows were opened, and no read was taken.

## 0. Decisions

1. **The read-7 decision is void.** "a16 final iff window >= 0.62, else a5c" used the window to choose between two versions. The final entry must be re-chosen by the rule in section 3. Commit that rule before any Exp1, Exp2, Metrica-sample or SkillCorner-test number is printed.
2. **Re-anchor both bands on other-game data recorded on 23 Oct** (`runs/extra/report.json`, commit bd84575, written before we knew of the rule change):
   - visible-dead share in **[0.116, 0.181]**, which is 0.8–1.25 × the SkillCorner value of 0.145;
   - spell rate as an **implied truth rate** R/ρ in **[2.28, 4.10]**, the range of other-game controlled-possession truths.

   Under both, **every band-dependent setting of a5c stands**: offset −2.0, `long_rel`, and cell 41 (merge 0.7). So do all learned-family offsets. Constants and documentation change; no simulator rerun is required.
3. **The window-free default is a14** (the learned family). It switches to a5c only if at least 2 independent window-free signals favour a5c and none favours a14. If exactly one signal favours a5c, or the signals conflict, the choice is the hedge H (learned weights × 0.5).
4. **Remaining effort goes into evidence for the family choice, not into new models.** Build a SkillCorner natural-visibility hold-out (20 matches), finish the Metrica samples, and run the masked-ball test scored under the rival family's reference. Run self-training only if Exp1 shows a real transfer gap of at least 0.05. Do not do multi-source retraining.
5. **No more spellF1 on the window.** The final correctness check covers only validation, row count and frame-offset/H-A orientation.
6. **The technical document must list all 7 reads, the read-7 rule, the band history and the window-derived designs** (section 5). It currently says the opposite in its sections 5 and 9.

## 1. Compliance audit

Rule text (organisers' commit f8f2629, 23 Oct 15:42 CEST, i.e. 19:12 IST):
- EVALUATION.md: *"Use the public window to check that your pipeline is correct — frames aligned, file well-formed, labels meaning what you think — never to choose between two versions of it, and never as training data. That is a rule, not just advice."*
- RULES.md §2: *"They are not training data and not a tuning set."*
- FAQ: *"Can I train or tune a model on the 5 public minutes? No."*

The old EVALUATION.md already advised "never to choose between two versions of it". Reads 5, 6 and 7 were taken after f8f2629 was published; read 5 came 42 minutes after it.

| # | Window-truth use | Evidence | Effect on a5c or on a choice | Class | Severity |
|---|---|---|---|---|---|
| C1 | Read 1: the organisers' baselines scored on the window (smooth50 0.3256) | e65510f; `runs/baselines_public.json` | none | check | OK |
| C2 | Read 2 (a1, 0.290) diagnostics led to the tag-sign bug fix and to the finding "D recall 0.16" | 53f9918, 86cf2d0; L47–48 | The tag fix is "labels meaning what you think". The D finding set the direction of later work. | check / grey | Low |
| C3 | The ctrl truth variant was justified by the window's 3.4 possessions/min | L46, L58; tech doc §4 | The ranking variant. It is also justified by the written "controlled possession" rule. | grey (convention) | Low |
| C4 | **Visible-dead band [0.10, 0.18] anchored on the window's 0.155** | `mtah/select.py:85-86` (993c1b9); L59, L91 | It chose a3b cell 52 (sim rank 5), `long_rel` over `long`, and **a5c's offset −2.0** (grid8 was built from the band, 671ae2c; the sim prefers offset 0). It also chose the offsets of a6, a8r, a10, a14 and a16. | tuned / chose | **High** |
| C5 | **Spell-rate band [2.7, 4.3] = 17/5 × [0.8, 1.25]** | `select.py:87-91` (9336057); `eval/drift.py:63` | In a4 it rejected sim rank 1, cell 32 (merge off, 4.995/min), which gave **cell 41, a5c's base (merge 0.7)**. It also rejected a9. | tuned / chose | **High** (the outcome survives, §2) |
| C6 | Each candidate's window shares (D 0.103/0.078 vs truth 0.196/0.155; spell starts vs 17) set the next plan's goal | L91, L103, L121, L125, L135; tech doc §6.1–6.4; plan2d-design §0.4/0.6 | Plans 2c–2e targeted "the window gap". No parameter was fitted. | grey | Medium (disclose) |
| C7 | Plan 2f: a pre-restart prior designed from the window's 5 visible-dead runs; the sweep printed window catch rates | L140–143; afd9afd, e02b91d; tech doc §6.5 | a6 was not selected. The code stays, inert (bonus 0). | tuned, not shipped | Medium |
| C8 | Reads 3 and 4 (a5c and a8r, 0.656), report-only and pre-registered | 17cbff1, 063a290; `plan3r-preregistration.md` | No effect then. 0.656 later became the comparator behind read 7's 0.62. | grey | Medium |
| C9 | Replay cue checked against window truth-D (0.81 vs 0.12) | L158 | a8r is not final | grey | Low |
| C10 | Plan 4 Metrica-convention truth calibrated to the window's spell-length quantiles | c2d0065 (its commit subject says so); L163–168 | a9 was rejected. The code is still in the repository. | tuned, not shipped | Medium |
| C11 | **Reads 5 (a10, 0.594) and 6 (a14, 0.556)**, labelled report-only | 3fb349a, c93e9ea; L176–177 ("decision held pending"), L193 ("learned family lower twice") | They weighed in on the family choice in practice | chose (informally) | **High** |
| C12 | **Read 7: "a16 iff >= 0.62 else a5c"** | 420ccca (exp-seq2 `public.jsonl`) | It chose a5c over the learned family | chose | **High** (direct breach) |
| C13 | Window facts used in interpretation ("56 % D in blackouts", "Athletic losing long balls") | `candidates-analysis-2026-10-26.md` §3.3–3.4 | none | grey | Low |

Hygiene: the main tree's `docs/reads/public.jsonl` lacks reads 5 and 7, and has read 6 uncommitted. The final repository needs one complete log.

## 2. Remediation: re-anchored bands

**Anchors** (other games, all from `runs/extra/report.json`):
- **Visible-dead.** SkillCorner gives 0.145 over 40 halves (1,970 min). DFL (0.453; the ball is tracked through stoppages) and the Metrica sample (0.001; the ball is NaN in stoppages) are not comparable.
- **ctrl spell rate.** SkillCorner 2.28, DFL 3.69, Metrica sample 4.10. These span the convention range.
- **The bands:**
  - The D band uses the project's existing [0.8, 1.25] tolerance, which comes from the sim count-ratio bar.
  - The rate check is made in truth space: the real predicted rate R divided by the same configuration's sim ctrl count ratio ρ (pred/true). This removes the decoder's own count bias. The window's 3.4 then plays no part.

| candidate | real vis-dead | R (spells per graded min) | sim ρ | R/ρ | D band | rate check |
|---|---|---|---|---|---|---|
| a5c | 0.178 | 3.10 | 0.872 | 3.56 | in (margin 0.003) | in |
| a13r | 0.173 | 2.87 | 0.863 | 3.33 | in | in |
| a10 | 0.166 | 3.52 | 0.933 | 3.77 | in | in |
| a14 | 0.177 (registered artefact 0.169) | 3.29 | 0.897 | 3.67 | in | in |
| a16 | 0.179 | 3.05 | read from its LOMO table | ~3.3 | in | check |
| a4 cell 32 (old rank 1) | 0.094 | 4.995 | 1.100 | **4.54** | out | **out** |

**Does a5c survive?** Each band-dependent setting, in turn:
- **Offset −2.0 survives, but it is fragile.**
  - The sim prefers less negative offsets monotonically (grid7 `long_rel`: 0 gives 0.533, −1 gives 0.521, −1.5 gives 0.510, −2 gives 0.498; grid8: −2/−2.5/−3 give 0.494/0.479/0.466). The cause is that the sim's visible-dead prior is about 0.30, against about 0.15 on the real feed. **The offset is therefore set by the band's upper edge, not by evidence**, and the document should say so.
  - Real sweep: −1.5 gives 0.195 (out), −2.0 gives 0.178 (in). The offset flips to −2.5 (0.161; sim −0.015 ctrl) if the upper edge falls below 0.178 (anchor < 0.1424), and to −1.5 if the edge reaches 0.195.
  - The private-score stake is small: 0.017 of visible frames, about 1 % of graded frames, well inside the 0.04 noise band.
- **`long_rel` over `long` stands.** `long` gives 0.27–0.31 visible-dead, 1.9–2.1 × the anchor, so it fails under any anchor.
- **Cell 41 (merge 0.7, team-control 2, w_dead 4) stands without any band.** It is the ctrl_Dfull-best cell of grid4 (0.4626; cell 32 gives 0.4559 and wins only through the raw variants of the mean rank). Cell 32's implied rate of 4.54 is also above every other-game truth.
- **Label-free cross-check.** Predicted overall D is 0.406 for a5c and 0.392 for a10, against whole-match dead shares of 0.44–0.45 in DFL and SkillCorner. Consistent; corroboration only.

**Must re-run or do:**
1. Replace the band constants and comments (`select.py:85-91`, `drift.py:63`), add the R/ρ check, and update the tests.
2. Recompute admissibility for every sweep cell already on disk (`a5b/offset_sweep_real.json`, `exp_robust/variants/*/real_cells.csv`, the exp-seq2 drift files). Flag any cell in (0.180, 0.181] or [0.100, 0.116).
3. Recommended for the audit trail: re-run the grid8 registration with the new constants (about 45 min). The same cell is expected, so no sealed read is needed.
4. Remove or flag the window-calibrated code: Plan 4 truth (c2d0065) and the Plan 2f per-type pre-restart windows (afd9afd). Then re-verify the md5 of a5c (2a672be6) and of the chosen candidate.
5. Merge the read log from 3fb349a, c93e9ea and 420ccca, and set the guard to refuse all scoring reads.
6. Resolve the a14 artefact mismatch: the registered drift is 0.169/3.32, the calibrated one 0.177/3.29. Name one md5.

**What stands:** all LOMO tables, all sealed reads, the a5c artefacts, and the learned-family artefacts.

## 3. Window-free final selection rule

**Where the proposed checks can mislead, and the fixes:**
- **Self-agreement.**
  - The label-free agreement metrics already point both ways, because each family agrees with what it weights. On the frames where a10 and a5c disagree, a10 agrees more with the tags (0.62 vs 0.35). a5c agrees more with confident PathCRF (0.74 vs 0.41) and with replay-D (0.44 vs 0.16) (L186).
  - A masked-ball reference built from a candidate's own unmasked decode measures invariance: a model that ignores the ball scores perfectly. A consensus reference is 3:2 learned.
  - **Fix:** score each family against the rival family's unmasked decode. A family wins only if it wins under the rival's reference.
- **Regime mismatch.**
  - Masked visible stretches are live-heavy (about 17 % dead). Natural blackouts are dead-heavy: the candidates call 79–84 % D inside them, against a sim reference of 27–59 % (candidates-analysis §3.3).
  - The families differ most exactly there: in-blackout team changes are 5 for a5c and 34 for a14, and stream B is over-confident in real blackouts.
  - **Fix:**
    - Draw mask starts from the real blackout entry distribution (near-line share 0.52, entry ball speed) and lengths from the real one (median 20.8 s, p90 50.5 s).
    - Report the live and dead strata separately.
    - Take the dead-heavy regime from natural-blackout tests (the SkillCorner and Metrica samples).
- **Reference error.**
  - The visible-frame reference is wrong 10–25 % of the time (ball argmax 0.74, team-control 0.88 on the sim; worse on the real feed), and its errors correlate with a5c's streams.
  - **Fix:** score only frames where the reference is confident. Drop ±5 s around set-piece tags, because a player holding the ball for a restart is D.
- **Exp1 leakage and selection shift.**
  - Hide the ball over ±10 s, or stream B reads it from neighbouring steps. Exclude the team-control HGB, which is ball-based and circular with the confident label.
  - Select confident frames by a geometric rule and reweight the sim to the real mix (pitch third, players visible, ball speed). The real ball differs: touches 8 vs 17.7/min, speed p50 5.1 vs 3.4 m/s.
  - Use PathCRF's gap as the yardstick. Its known nearest-player proxy is 0.81 on the sim against 0.68 real (L87).
- **Proxy validity on the sim.**
  - With 5–7 candidates, a rank correlation has a standard error of about 0.3, and picking the best of many proxy definitions overfits.
  - **Fix:** pre-declare at most 3 definitions. Validate on at least 20 configurations (cells, λ, offsets) × 6 folds; choose on folds 9001–9003 and confirm on 9004–9006. Passing on the sim cannot certify the proxy on the real feed.
- **Other contamination.**
  - Self-trained candidates may be scored only on the half they did not train on.
  - The Metrica-sample truth is our own event labelling (34 % of spells < 3 s; it flips on every BALL LOST/RECOVERY), so it rewards over-segmentation. Never use it alone.

**Procedure** (commit as `docs/reads/plan9-preregistration.md` before any of these results is printed):

1. **Pool, frozen:** {a5c, a13r, a10, a14, a16, H, S}.
   - H = a14 build with learned weights × 0.5 (A 2, B 6). On the old build, λ 0.5 costs −0.005 on the sim against a10.
   - S = a14 with cross-half self-training, only if registered by **Sat 14 Nov 12:00**.
   - Nothing is added after that.
2. **Admissible** iff all of the following hold:
   - visible-dead in [0.116, 0.181];
   - R/ρ in [2.28, 4.10];
   - clean-clone md5;
   - the official validator passes.

   Every candidate's D offsets are set by one rule: the sim-best admissible offset.
3. **Representatives, sim-only, fixed now:**
   - P = a5c.
   - L = a14. a10 is excluded because its inputs are on the old build (adversarial AUC 0.997 vs 0.952). a16 replaces a14 only if it is within 0.01 on LOMO ctrl_Dfull; at 0.519 vs 0.557 it is not.
   - S replaces a14 iff S is at least a14 − 0.005 on its sim acceptance run and is admissible.
4. **Signals.** Each favours P, favours L, or is "none".
   - **M** (Metrica samples 1–3, 6 halves): favours X iff X leads by a mean spellF1 of at least 0.02 and wins at least 4 of 6 halves.
   - **K** (SkillCorner natural visibility, at least 30 halves, replay/no-player frames = X): favours X iff X leads by a mean of at least 0.01 and wins at least 60 % of halves.
   - **T** (Exp1):
     - favours P iff gap(B) − gap(PathCRF) ≥ 0.05 and B's real accuracy ≤ PathCRF's;
     - favours L iff B's real accuracy ≥ PathCRF's + 0.03 and gap(B) ≤ gap(PathCRF) + 0.02.
   - **B** (masked ball): counts only if it passes on the sim:
     - mean Kendall τ ≥ 0.5;
     - τ > 0 on at least 5 of 6 folds;
     - a13r < a14 ordered as in the truth on at least 5 of 6 folds.

     It favours X iff X beats the rival under the rival's reference with bootstrap P ≥ 0.90.
5. **Decision.**
   - Choose P iff at least 2 signals favour P and none favours L.
   - Choose H iff exactly one signal favours P, or the signals conflict, and H is admissible and within 0.01 of L on LOMO ctrl_Dfull.
   - Otherwise choose L.
   - A signal not finished by **Sun 15 Nov 18:00** counts as "none".
   - Residual ties go to sealed 9007 ctrl_Dfull, then to fewer streams.
6. **Final file.**
   - Run the validator and the row count.
   - Run one boolean window check, "best frame offset = 0 and H/A not swapped", with no spellF1 printed.
   - Only a demonstrable alignment or orientation bug may change the file.

## 4. Levers for the private spellF1, 6 days left

1. **SkillCorner natural-visibility hold-out.** This is about the quality of the decision. The family gap is +0.06 to +0.08 on the sim and the sealed match, so the decision is worth more than any model change now.
   - **Build:** use SkillCorner's own ball-detection flag as the blackout mask, not the DFL simulator, and set frames with no detected players to X. Run PathCRF on 40 halves (about 3.5 CPU-h in the laptop venv; the server lacks `torch_geometric`). Then decode the pool. About 1 day in all.
   - **Clean:** no pool member trained on SkillCorner.
   - **Caveats:** players are broadcast-extrapolated, the dead-ball convention is broad, and the league is the A-League.
2. **Hedge H (λ 0.5).** Decode-only, a few hours. It keeps about 90 % of the learned family's sim gain and roughly halves the weight on the shifted stream. It is the minimax-regret choice when the evidence is split. Check R/ρ: λ 0.25 raised the real spell rate to 3.92, so blends can flicker.
3. **Metrica samples 1–3** (agent C). Cheap, low power and biased by convention. Useful because all their dead time has no ball (about 70 % of the naturally ball-less frames are D), which is close to the real blackout regime.
4. **Cross-half self-training of the players-only stream.** Conditional.
   - The code exists (`exp-selftrain` e6a6bcc) but has never been run.
   - Run it only if Exp1 shows a real-minus-sim gap of at least 0.05 and items 1–3 finish by Saturday.
   - Pseudo-labels from confident control are settled-possession frames, so the stream learns the easy regime, not dead-heavy blackouts.
   - Expected gain +0.00 to +0.02. It needs the sim acceptance run and cross-half scoring.
5. **Multi-source training (DFL + SkillCorner + Metrica): not worth it.**
   - a12b gained only +0.006 ctrl, below the 0.01 bar (seed SD 0.004–0.009).
   - SkillCorner labels differ: dead ball includes replays, possession is per player, and 80 % of frames are interpolated.
   - It costs more than 10 CPU-h plus recalibration.
   - It would also use up SkillCorner, the only independent test set with real-like dead-ball visibility.
6. **Any further decoder, post-processing or offset tuning: not worth it.** Every one of these was measured flat or negative, or rejected:
   - MBR: no-go.
   - Metric rules, asymmetric snap, merge θ: flat.
   - Visibility gating: −0.001 to −0.012. Switch hazard: −0.024.
   - Metrica truth: window-calibrated and rejected.
   - Ball fill and whistle: rejected or no-go.
   - Replay stream: marginal.

   The operating point is already near the F/2 optimum (candidates-analysis §3.1).

## 5. What the technical document must disclose

- **A table of all 7 window reads:** date, commit, candidate, spellF1/macroF1, stated purpose, and whether it was pre-registered. The reads are: 1 baselines; 2 a1 0.290; 3 a5c 0.656; 4 a8r 0.656; 5 a10 0.594; 6 a14 0.556; 7 a16 0.565.
- **That read 7 applied a rule that chose between versions.** Say that reads 5–7 came after the organisers' 23 Oct clarification, which we had not pulled; that the old text already advised against this; and that the decision was discarded on 10 Nov and remade by the rule committed at <hash>.
- **The band history.** Both bands were anchored on window aggregates (0.155; 17 possessions in 5 minutes) and used to choose a3b, a4 (cell 41), the a5c offset and the learned-family offsets. Report their re-anchoring on SkillCorner, DFL and Metrica statistics, with the section 2 table showing which choices stand, and state that the offset is set by the band edge.
- **The other uses of window truth**, with what happened to each:
  - window D shares and spell starts per candidate, used as goals (C6);
  - the a6 prior designed from window dead runs (not shipped; code removed or inert);
  - the Plan 4 truth calibrated to window spell lengths (rejected; code removed or flagged);
  - the replay-cue cross-check against window truth.
- **Corrections to the current text:**
  - the status line (a8r);
  - §2.6 item 3;
  - §2.7.3;
  - §5 items 6–7 ("budgeted at 3… never for tuning");
  - §6.5 (the motivation and the window rows);
  - §9 ("3 reads at most, none used for tuning").
- **The new SUBMISSION "training data" item.** Cover DFL (7 matches, DFL labels), SkillCorner, the Metrica samples and our own derived truth variants. Say that none of it is from the provided game, and that the window was never used as training data but was used for calibration (as above).
- **The final correctness check and its limited, pre-declared use.**
