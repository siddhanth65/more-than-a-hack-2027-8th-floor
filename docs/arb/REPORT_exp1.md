# Experiment 1 - per-stream transfer gap on confident ball-control frames (interim)

Branch exp-arb, server job outputs: ~/projects/mtah/arb/code/runs/arb/ (exp1.json, exp1_centre.json). Aggregates only.

## Pseudo-label rule (frozen, tuned on sim folds 9001-9006 only, seeds 100/101)
Ball visible; the nearest player is within 1.5 m, the nearest opponent at least 5.0 m away, ball speed relative to that player <= 5 m/s (averaged over +-0.4 s),
ball itself moving >= 1 m/s, same player for >= 1 s, label the middle (0.5 s trimmed off both ends). Precision against the sim ball-owner truth (raw_Dfull,
a labelled true-D frame counts as wrong): 0.971 pooled (min fold 0.954). Against the event-confirmed truth ctrl_Dfull it is only 0.84: ctrl_Dfull
disagrees with the nearest player on ~25 % of near-player frames by construction, so 0.97 against it is unreachable for ANY tracking-only rule. The raw truth is
itself a proximity label, so the precision partly measures corruption robustness, not semantics. The relative-speed threshold had to be loosened to 5 m/s
(2 m/s left almost no frames: sim ball jitter) and "ball moves" was added (a still ball next to a player is mostly truth-D).
Coverage: sim 3.164% of visible graded frames, H share 0.52; real 4.678%, H share 0.55.
Cross-check on the real match (unmasked decodes, confident frames): a5c / a14 agree with the pseudo-label (on their H/A frames) 0.94 / 0.98 in H1 and 0.88 / 0.95 in H2, so the labels are sound in both halves.

## Ball hiding
For each unit, 2 mask seeds: the ball is hidden for +-10 s around randomly chosen confident frames (centres >= 45 s apart) and the FULL pipeline runs on the masked data.
Table A evaluates confident frames within +-3 s of a centre (ball hidden >= 7 s on both sides); Table B only the centre frames (hidden >= 10 s each side).
The learned team-control stream (teamctrl) is ball-based and outputs NOTHING when the ball is hidden (NaN: no opinion), so it has no row; it would be circular with the label anyway.
Sim is scored out of fold (fold dead/team/B models); the reweighting is to the real mix of (pitch third x visible players x ball-speed bin), 98.6% of the real mass is covered by sim cells.

## Table A (+-3 s of the centre)
| stream | sim acc (6 folds, mean ± sd) | sim acc reweighted to real mix | real acc | real H1 / H2 | gap raw | gap reweighted | mean loglik sim / real | n frames sim / real |
|---|---|---|---|---|---|---|---|---|
| a14/pathcrf | 0.960 ± 0.024 | 0.969 | 0.754 | 0.910 / 0.550 | +0.206 | +0.215 | -0.11 / -1.25 | 16598 / 4401 |
| a14/seqB | 0.956 ± 0.023 | 0.966 | 0.753 | 0.919 / 0.536 | +0.203 | +0.213 | -0.17 / -0.58 | 16598 / 4401 |
| a16/seqB | 0.786 ± 0.109 | 0.796 | 0.535 | 0.546 / 0.521 | +0.250 | +0.260 | -0.44 / -0.73 | 16598 / 4401 |

## Table B (centre frames only)
| stream | sim acc (6 folds, mean ± sd) | sim acc reweighted to real mix | real acc | real H1 / H2 | gap raw | gap reweighted | mean loglik sim / real | n frames sim / real |
|---|---|---|---|---|---|---|---|---|
| a14/pathcrf | 0.952 ± 0.028 | 0.960 | 0.709 | 0.884 / 0.542 | +0.243 | +0.250 | -0.12 / -1.39 | 585 / 141 |
| a14/seqB | 0.963 ± 0.021 | 0.966 | 0.730 | 0.928 / 0.542 | +0.232 | +0.236 | -0.16 / -0.67 | 585 / 141 |
| a16/seqB | 0.842 ± 0.085 | 0.840 | 0.574 | 0.551 / 0.597 | +0.267 | +0.265 | -0.35 / -0.67 | 585 / 141 |

PathCRF is one stream shared by all families (players only; the a5c/a13r/a14/a16 rows are identical). Stream A (GBT) exists only in a10 and has no saved fold model: not evaluated.

## Signal T (pre-registration: B = a14's BiGRU stream, gap = sim - real)
a14: gap(B) - gap(PathCRF) = -0.003 (centre-only -0.011); real B - real PathCRF = -0.001 (centre-only +0.021).
Favours P needs gap diff >= +0.05 and B real <= PathCRF real; favours L needs real B >= PathCRF + 0.03 and gap(B) <= gap(PathCRF) + 0.02. Table A: neither holds -> signal T = none
(centre-only: B is +0.021 above PathCRF on real, still below the +0.03 bar -> none). a16's B (per-half-normalised): gap +0.044 larger than PathCRF's and real accuracy 0.535 vs 0.754: a16's B is much worse on the real feed (and worse in sim, 0.79).

## Findings to flag
1. The transfer gap is ~0.20 for PathCRF AND for a14's B and it sits almost entirely in the real SECOND half: real H1 0.91 / 0.92, real H2 0.55 / 0.54 (sim 0.96). On the same confident frames the ball-based decodes agree with the label in H2 (0.88 / 0.95), so the label is fine; PathCRF is near chance in the real second half
(unmasked as well: 0.58). a14's B takes PathCRF as an input, so it inherits the failure: B and PathCRF are not independent evidence. A likely cause (not tested here) is the attacking-direction / side convention of PathCRF in period 2.
2. a14's B only equals PathCRF here; it adds nothing in the hidden-ball frames in sim (0.956 vs 0.960) and little in real (0.753 vs 0.754 pooled; +0.021 centre-only, n=141).
3. Provenance: sim numbers are out of fold. a14's fold Model B is the recipe-identical exp_seq3/base retrain (a14's own fold weights were not saved); unmasked it gives 6-fold ctrl_Dfull 0.539 vs the registered a14 0.557. a16 fold models are the exact recipe (0.5185 = registered). The a5c fold dead/team/merge models were RETRAINED
(saved pickles unloadable under sklearn 1.9.1; dead retrain identical to the loadable original, max|dp| = 0; unmasked 6-fold ctrl_Dfull 0.4944 = registered 0.494). a13r fold models: exp_seq2 (a13 refit), unmasked 0.482 (a13 registered 0.477).
Real-match numbers use the registered artefacts; their CSVs regenerate byte-identically (a5c, a10, a14, a16).
