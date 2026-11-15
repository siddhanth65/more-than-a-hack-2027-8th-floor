# Battery report (partial, staged)

S == s_ko on sealed/K/M (kick-off rule is a no-op: hard-wired to the contest match's kick-off frames; these halves start at kick-off). Wins are per half vs S/s_ko (seed-averaged halves). Format: spellF1/macroF1.

- sealed ctrl_Dfull n=2: S 0.5760/0.8535 | a17 0.5676/0.8449 (w0-l2) | a5c_fx2 0.4719/0.7535 (w0-l2) | smooth50 0.1926/0.4762 (w0-l2)
- sealed raw_Dfull n=2: S 0.6009/0.8930 | a17 0.5990/0.8858 (w1-l1) | a5c_fx2 0.5048/0.7926 (w0-l2) | smooth50 0.2209/0.5079 (w0-l2)
- Knat ctrl_Dfull n=40: S 0.5345/0.8027 | a17 0.5316/0.7983 (w15-l25) | a5c_fx2 0.5380/0.7723 (w20-l20) | smooth50 0.3493/0.5194 (w0-l40)
- Knat raw_Dfull n=40: S 0.4314/0.7822 | a17 0.4276/0.7774 (w11-l29) | a5c_fx2 0.4345/0.7529 (w22-l18) | smooth50 0.3228/0.5154 (w0-l40)
- Ksim ctrl_Dfull n=40: S 0.4151/0.6555 | a17 0.4064/0.6558 (w11-l29) | a5c_fx2 0.3947/0.6178 (w12-l28) | smooth50 0.2089/0.6109 (w0-l40)
- Ksim raw_Dfull n=40: S 0.3377/0.6410 | a17 0.3287/0.6415 (w9-l31) | a5c_fx2 0.3207/0.6066 (w9-l31) | smooth50 0.2081/0.6088 (w1-l39)
- M ctrl_Dfull n=6: S 0.6148/0.8716 | a17 0.6065/0.8727 (w1-l5) | a5c_fx2 0.5859/0.8576 (w1-l5) | smooth50 0.3218/0.7240 (w0-l6)
- M raw_Dfull n=6: S 0.6149/0.8715 | a17 0.6070/0.8728 (w1-l5) | a5c_fx2 0.5860/0.8576 (w1-l5) | smooth50 0.3221/0.7241 (w0-l6)

## Simulator LOMO 9001-9006 (12 halves, seed-averaged), spellF1 / macroF1; wins vs S = s_ko
Kick-off rule: hard-wired to the contest match's frames (periods.py), so a no-op on simulator, K and M halves (they start at kick-off). s_ko == S on all of them.
Recorded numbers reused (harness outputs docs/s/lomo_scores.csv ref_stored for a17 at its cell; docs/battery/csv/calib_s_folds.csv for S at its registered cell dead -4.0 / seq_d +0.5, V3, both from the earlier harness). a5c_fx2 re-run on the current (new) sim build with select_lomo grid8 cell 0 (-2.0), LOMO-trained fold models (a5c was registered on the old build: 0.4944 ctrl there).
| variant | S = s_ko | a17 (wins/losses vs S) | a5c_fx2 (wins vs S) |
|---|---|---|---|
| ctrl_Dfull | 0.5665 / 0.8569 | 0.5622 / 0.8531 (3/9) | 0.4817 / 0.7792 (0/12) |
| raw_Dfull | 0.5836 / 0.8828 | 0.5740 / 0.8771 (2/10) | 0.5043 / 0.8077 (0/12) |
| ctrl_Dnarrow | 0.5260 / 0.7475 | 0.5235 / 0.7464 (4/8) | 0.4616 / 0.6992 (0/12) |
| raw_Dnarrow | 0.5357 / 0.7716 | 0.5280 / 0.7686 (2/10) | 0.4861 / 0.7292 (0/12) |
At a17's own cell S is 0.5614/0.5769 (docs/s): the S-over-a17 gain here comes from the offset rule (cell), not the stream.

## Real match, label-free (docs/battery/real_labelfree.json)
Public window: frames 11185-18685 differing between s_ko and S = 0 (1867 differ in total, all in the 2nd-half pre-kick-off block 110808-113741; 0 outside). Window reads copied as recorded: a5c 0.656, a17 0.6061, S 0.5765 (s_ko = S there). Nothing scored on the window; gt file not opened.
Smoother baseline: tag 0.698, vis-dead 0.040, 594 spells.

## Notes
- smooth_s = 0 on sealed/K/M (registered smoothing is a contest-feed property; caches are unsmoothed). S fine-tuned per target half from the other half's pseudo labels (V3), base = models/a17, 6 DFL matches' simulator sequences; ~27 s each.
- K sims are single-seed (100); sealed/M seeds 100,101 averaged per half. M truth variants are numerically identical for ctrl/raw.
- Sealed 9007: four records appended to docs/reads/sealed.jsonl (s_ko, S, a17, a5c_fx2).
