# exp_seq3 robustness variants (Plan 8 T2), 2026-10-27

Aggregates only. No ground truth, public or sealed read; sim match 9007 never loaded. Branch exp-seq2, scripts `exp_seq3_robust.sh` (driver), `exp_seq3_robust_real.py`, `exp_seq3_robust_table.py`, plus `exp_seq3_adv.py` (now merges into its JSON).

Decode = a14 configuration on the a13r base: Model B stream at weight 12, dead_offset -3.0, seq_d_offset -1.0, tau 30/100, aba 1.0. Sim = 6-fold LOMO held-out halves (ctrl/raw x Dfull/Dnarrow spellF1, mean over halves). Merge th0.7 rows come from the honest-LOMO spell model (`exp_seq3_spell.py <variant> 0`); "off" is its th1 row. Folds won = folds (of 6) with higher ctrl_Dfull than the reference. Real drift = the variant's FINAL model (all six matches, epochs = median fold choice; trained now, none existed) in the a14 decode on the real match, merge off. Half-CV AUC = leave-halves-out 2-fold adversarial AUC (sim vs real) of the variant's input step features (`touch` shares `base` inputs, `v1touch` shares `v1`).

Reference: a13r (merge on, same build) ctrl_Dfull 0.4817, raw 0.5043, ctrl_Dn 0.4616, raw_Dn 0.4861, macroF1 0.779, 151 spells. Every row below beats a13r in 6/6 folds.

| variant | merge | ctrl_Dfull | raw_Dfull | ctrl_Dnarrow | raw_Dnarrow | macroF1 | n_spells | won vs a14 off / same-merge | real vis-dead (band .10-.18) | spells/min (2.7-4.3) | tag implied | PathCRF conf. agree | replay-D | half-CV AUC |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| a14 (ref) | off | 0.5568 | 0.5676 | 0.5202 | 0.5235 | 0.849 | 157 | - | 0.177 ok | 3.29 ok | 0.777 | 0.796 | 0.471 | 0.929 |
| a14 | th0.7 | 0.5659 | 0.5682 | 0.5295 | 0.5252 | 0.849 | 145 | 6 / - | (merge off only) | | | | | |
| base (none, ctrl) | off | 0.5485 | 0.5612 | 0.5137 | 0.5196 | 0.845 | 161 | 0 / 0 | 0.177 ok | 3.29 ok | 0.777 | 0.796 | 0.471 | 0.929 |
| base | th0.7 | 0.5579 | 0.5601 | 0.5236 | 0.5206 | 0.845 | 146 | 4 / 3 | | | | | | |
| v1 (norm v1, ctrl) | off | 0.5460 | 0.5649 | 0.5104 | 0.5246 | 0.845 | 163 | 2 / 2 | 0.240 FAIL | 2.92 ok | 0.812 | 0.764 | 0.566 | 0.779 |
| v1 | th0.7 | 0.5551 | 0.5652 | 0.5204 | 0.5268 | 0.844 | 149 | 4 / 2 | | | | | | |
| touch (none, touch12) | off | 0.5388 | 0.5560 | 0.5045 | 0.5154 | 0.841 | 160 | 1 / 1 | 0.193 FAIL | 3.14 ok | 0.789 | 0.798 | 0.496 | 0.929 |
| touch | th0.7 | 0.5478 | 0.5551 | 0.5145 | 0.5170 | 0.840 | 143 | 3 / 0 | | | | | | |
| v1touch | off | 0.5447 | 0.5627 | 0.5095 | 0.5212 | 0.844 | 160 | 2 / 2 | 0.247 FAIL | 3.09 ok | 0.806 | 0.769 | 0.570 | 0.779 |
| v1touch | th0.7 | 0.5517 | 0.5572 | 0.5173 | 0.5177 | 0.843 | 145 | 3 / 2 | | | | | | |

Reading
- The real-match stream of `base` is bit-identical to a14's (every drift number equal), so the final-model training pipeline reproduces a14 exactly. The sim held-out predictions of `base` are 0.008 below a14 because the fold models were trained on the cluster under the per-run time budget (several folds stopped at 6-7 epochs). All variants carry this handicap, so variant-vs-base differences are the fair comparison; variant-vs-a14 is biased against the variants by about 0.008.
- Normalisation v1 cuts the input half-CV AUC from 0.929 (none) to 0.779 (v2 earlier: 0.778), so v1 inputs look much more real-like, but the learned stream then calls too much dead ball on the real match (vis-dead 0.24-0.25, outside the band) at the a14 calibration. Sim cost: -0.003 ctrl_Dfull (off), -0.003 (th0.7).
- touch12 targets lose 0.010 sim ctrl_Dfull and push vis-dead to 0.193 (just outside); no benefit.
- Merge th0.7 adds +0.009 on every variant (4-6 folds won vs the same variant's no-merge, see the sweep).
- Only the unchanged `base` (= a14) passes the real bands; v1 and v1touch would need a dead_offset / seq_d_offset recalibration on the real drift (not done here, a separate sim-vs-band trade).
