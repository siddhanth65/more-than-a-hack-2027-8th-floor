# REPORT - label-free arbitration experiments (exp-arb)

Exp 1 details: REPORT_exp1.md. All numbers are aggregates; server outputs: ~/projects/mtah/arb/code/runs/arb/ (out/*.npz per unit and mask, exp2.csv, *.json), logs ~/projects/mtah/arb/logs.

## Exp 2 set-up (as run)
Masks: 5 seeds (0-4), ~30 % of the naturally visible graded frames hidden, window lengths drawn from the REAL blackout lengths (79 runs >= 10 s, median 20.8 s), window starts drawn so that the share near a line (0.52) and moving toward it (0.80) matches real blackout entries; on sim applied on top of the existing corruption (seed-100 halves of folds 9001-9006). Full pipeline re-run per candidate (out of fold on sim). 20 configurations on sim: a5c, a13r, a14, a16 + 16 pre-declared variants (a14/a16 learned-weight x0.25/0.5/0.75 and seq d_offset shifts, a5c/a13r dead offset -1/-3, a5c without teamctrl, a13r PathCRF x2); a10 is real-only (its fold Model A was never saved). Variant list fixed before any run. References: own unmasked decode, consensus (all four base candidates agree), ball-only (family pipeline with PathCRF/teamctrl/learned streams and merge off), rival (a5c-type scored against a14's unmasked decode; learned configs against a5c's). Scoring only inside masked windows, only where the reference is confident (>= 1 s from a reference label change, +-5 s around SET PIECES tags dropped); the unfiltered version is in exp2.csv (filtered=False). Scorer: md5-pinned official wrapper. Truth on sim = ctrl_Dfull.

## Proxy validity on the simulator (6 folds, 20 configs, per-fold Kendall tau vs true spellF1 in the masked windows)

| proxy | mean Kendall tau | folds tau>0 | mean Spearman | pooled Spearman (fold-demeaned) | same best config as truth | a13r<a14 ordered as truth |
|---|---|---|---|---|---|---|
| rival | +0.48 | 5/6 | +0.61 | +0.74 | 2/6 | 4/6 |
| cons | +0.49 | 5/6 | +0.64 | +0.74 | 1/6 | 4/6 |
| ball | +0.24 | 5/6 | +0.32 | +0.28 | 1/6 | 4/6 |
| own (not pre-declared) | +0.36 | 6/6 | +0.51 | +0.55 | 1/6 | 4/6 |

Pre-registered validity bar (mean tau >= 0.5, tau>0 on >= 5/6 folds, a13r<a14 as truth on >= 5/6): rival tau 0.48 and ordering 4/6 -> NOT passed (cons 0.49 / 4/6 also not). Signal B therefore does not count.

True spellF1 in the masked windows (sim, mean over folds, ctrl_Dfull): a14_sd0 0.563, a14_w50 0.557, a14_w75 0.556, a14 0.555, a14_w25 0.553, a16_w25 0.549, a14_sd-2 0.546, a16_w50 0.542, a5c_notc 0.536, a5c 0.536, a5c_do-1 0.535, a13r_do-1 0.535, a16_w75 0.535, a16_sd-2 0.530, a13r 0.527, a16 0.526, a5c_do-3 0.526, a13r_pc2 0.519, a16_sd-4 0.519, a13r_do-3 0.517

Strata (sim, pooled configs): {"live": {"kendall_pooled_means": 0.7157894736842105, "truth_a5c_a14": [0.5907411332697168, 0.6078287886455597]}, "dead": {"kendall_pooled_means": 0.5157894736842106, "truth_a5c_a14": [0.3519036577141541, 0.38618149421903475]}}

## Real match: spellF1 / macroF1 inside masked windows, mean (sd) over 5 mask seeds, filtered references

| candidate | ref | H1 spellF1 | H1 macroF1 | H2 spellF1 | H2 macroF1 |
|---|---|---|---|---|---|
| a5c | rival | 0.811 (0.032) | 0.889 | 0.401 (0.025) | 0.605 |
| a5c | own | 0.826 (0.046) | 0.898 | 0.550 (0.029) | 0.707 |
| a5c | cons | 0.933 (0.011) | 0.965 | 0.521 (0.049) | 0.726 |
| a5c | ball | 0.679 (0.049) | 0.877 | 0.356 (0.030) | 0.608 |
| a13r | rival | 0.767 (0.054) | 0.881 | 0.385 (0.031) | 0.591 |
| a13r | own | 0.792 (0.046) | 0.888 | 0.406 (0.022) | 0.611 |
| a13r | cons | 0.907 (0.066) | 0.951 | 0.466 (0.087) | 0.708 |
| a13r | ball | 0.675 (0.053) | 0.876 | 0.345 (0.049) | 0.573 |
| a14 | rival | 0.806 (0.056) | 0.926 | 0.518 (0.056) | 0.651 |
| a14 | own | 0.824 (0.026) | 0.911 | 0.414 (0.042) | 0.621 |
| a14 | cons | 0.938 (0.014) | 0.985 | 0.514 (0.096) | 0.719 |
| a14 | ball | 0.683 (0.018) | 0.891 | 0.343 (0.030) | 0.527 |
| a16 | rival | 0.652 (0.025) | 0.759 | 0.505 (0.030) | 0.637 |
| a16 | own | 0.732 (0.029) | 0.857 | 0.445 (0.045) | 0.617 |
| a16 | cons | 0.798 (0.026) | 0.844 | 0.535 (0.082) | 0.719 |
| a16 | ball | 0.580 (0.020) | 0.735 | 0.378 (0.047) | 0.571 |
| a10 | rival | 0.814 (0.043) | 0.911 | 0.486 (0.055) | 0.669 |
| a10 | own | 0.825 (0.034) | 0.932 | 0.415 (0.053) | 0.597 |
| a10 | cons | 0.965 (0.020) | 0.984 | 0.490 (0.096) | 0.697 |
| a10 | ball | 0.676 (0.038) | 0.885 | 0.314 (0.046) | 0.575 |

All 21 configurations (variants) and unfiltered versions: exp2.json / exp2.csv.

Real strata (rival ref, spellF1; dead = windows whose a5c decode is mostly D): {"a13r|dead": 0.366, "a13r|live": 0.583, "a14|dead": 0.254, "a14|live": 0.679, "a16|dead": 0.146, "a16|live": 0.597, "a5c|dead": 0.343, "a5c|live": 0.613}

## Paired bootstrap on the real match (rival references)
295 mask windows pooled over 5 seeds, 200 resamples. a5c scored against a14's unmasked decode: 0.567; a14 scored against a5c's: 0.627. P(a5c beats a14) = 0.005, P(a14 beats a5c) = 0.995; mean difference -0.059, 95 % interval [-0.098, -0.022]. NOTE: each candidate is scored against the OTHER family's decode, so the comparison is partly 'who is closer to the rival', and a14 (which agrees with a5c far more than the reverse in some windows) is favoured by construction when the learned family's output is the smoother one; because the sim validity bar failed this does not count under the pre-registration.

## Provenance and caveats
- Repro (real, server): a5c, a10, a14, a16 CSVs byte-identical to the pinned md5s; a13r (no pinned md5) md5 faf6a6d42f5c211458877b4d847a47de.
- Retrained (not registered) artefacts, sim folds only: a5c fold dead/team/merge models (old pickles unloadable; dead retrain verified identical on fold 9001), a14 fold Model B = exp_seq3/base retrain (unmasked 6-fold 0.539 vs registered 0.557). Unmasked 6-fold ctrl_Dfull reproduced: a5c 0.4944 (0.494), a16 0.5185 (0.519), a13r 0.482 (a13 0.477). Real numbers use registered models only.
- a5c was evaluated on the old (pre-smoothing) sim build it was trained on; the others on the new build; masks derive from the new build (ball visibility, positions and blackouts verified identical across builds).
- Seeds: mask seeds 0-4 (rng [seed, unit_id, 4711]), Exp1 centred masks 'c0','c1' (rng [seed, unit_id, 9157]), bootstrap seed 20261110. Sim units: seed-100 halves only (12); seed-101 halves unmasked only (repro).
- Rule and masks never touched the public-window ground truth or sealed match 9007.
