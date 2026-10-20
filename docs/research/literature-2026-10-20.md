# Prior-work brief: per-frame team possession (H/A/dead) from noisy tracking with long ball blackouts

Date: 2026-10-20. Compiled from web search plus partial reads (abstracts/HTML) of the papers. Numbers quoted are as reported by authors; items marked [UNVERIFIED] were not confirmed from a fetched source. Context: 25 fps tracking, ball missing on 41 % of frames (median blackout 21 s, starting when ball goes out near a line); metric spellF1 (maximal same-team runs, one-to-one temporal-IoU matching, F1 averaged over IoU 0.3-0.7).

## 1. Possession / ball-carrier inference from tracking

**Vidal-Codina et al., "Automatic event detection in football using tracking data"** (MIT/FIFA; Sports Engineering 2022), https://arxiv.org/abs/2202.00804
- Does: deterministic decision tree. Step 1 assigns per-frame ball control with four states (dead ball, no possession, possession, duel) from ball-player distance; step 2 turns possession changes + rules of football into passes, shots, crosses, receptions, interceptions, set pieces. Claims >90 % detection rate for most event types across providers/tournaments.
- Reusable: the 4-state per-frame schema (our H/A/dead plus an internal "contested/none" latent state resolved later); using player configuration during dead ball (taker + line-up) to detect set pieces; their supplementary notes that the in-play boolean is often off-sync, ball is missing during live play, and the taker may be untracked outside the lines (set-piece events imputed from partial triggers). Same failure modes as ours.
- Caveats: needs a ball; PDF text could not be extracted, so exact radius/time thresholds were not verified (read the PDF for them).

**Kim, Lee et al., "PathCRF: Ball-Free Soccer Event Detection via Possession Path Inference from Player Trajectories"** (KAIST/Fitogether/UoS, Feb 2026), https://arxiv.org/abs/2602.12080 ; code https://github.com/hyunsungkim-ds/pathcrf
- Does: no ball at all. Fully connected dynamic graph over players (+4 "outside" nodes, one per boundary line, absorbing); at each step one edge = current possessor/receiver; a masked CRF with Viterbi decoding forbids illegal transitions (violation rate 0 %). 5 fps, 10 s windows, trained on Sportec Open (5 train / 1 val / 1 test). Test match: edge acc 69.6 %, sender 80.4 %, receiver 82.3 %; event F1 75.7 % (+-1 s, same player) vs 65.3 % without CRF. Predicted home possession share 62.6 % vs 62.1 % true.
- Reusable: the most relevant paper. Possessor-path decoding with hard transition constraints is what is needed inside our blackouts; team label = team of the decoded possessor. The "outside node" idea maps to our blackouts that start at a line exit. Code is public.
- Caveats: single test match; GT partly derived from ball trajectories; in-play segments only (dead ball not modelled), so we need a separate dead/alive layer; 5 fps, so labels must be upsampled to 25 fps.

**Conventions**: DFL/Sportec position files carry a per-frame ball status (alive/dead) and possession (home/away) code (parsed by floodlight, kloppy, databallpy) - our proxy GT. Metrica sample data has no possession flag (derive from events). StatsBomb 360 / SkillCorner conventions: [UNVERIFIED, not fetched].

## 2. Ball position inference without the ball

**Kim, Choi, Kim, Yoon, Ko, "Ball Trajectory Inference from Multi-Agent Sports Contexts Using Set Transformer and Hierarchical Bi-LSTM"** (Ball Radar, KDD 2023), https://arxiv.org/abs/2306.08206 ; code https://github.com/pientist/ballradar
- Does: Set Transformer encodes the player set per frame; hierarchical Bi-LSTM first classifies the ball possessor, then regresses the trajectory conditioned on it; reality loss + postprocessing for physical plausibility. Demo on Metrica sample.
- Reusable: trainable on DFL + Metrica; possessor probabilities give a team signal even when ball xy is poor. Paper's suggested uses include missing-trajectory imputation and possession-wise metrics. A prior regressor it compares against had ~7.6 m / 5.0 m x/y error, so team-level possession is far more robust than exact ball xy.
- Caveats: trained on clean full-match data; our blackouts (median 21 s, starting at line exits) are much longer than typical masked windows, so train with whole-segment masks that start at ball-out.

**Capellera et al., "TranSPORTmer: A Holistic Approach to Trajectory Understanding in Multi-Agent Sports"** (ACCV 2024), https://arxiv.org/abs/2410.17785
- Does: masked Set-Attention transformer; one model for forecasting, imputation of missing players/ball, inferring unseen agents, and global-state classification via a CLS token (passes, possessions). Reports SOTA on ball inference and ball imputation.
- Reusable: masked-input training (simulate our blackout masks); CLS-token state classification as a direct home/away/dead head.
- Caveats: heavy; data differs from ours. No newer (2025-26) ball-imputation paper found in searches.

## 3. Dead-ball / in-play detection, restarts

**"Predicting the in-game status in soccer with machine learning using spatiotemporal player tracking data"** (Scientific Reports 2022; first author not verified), https://pmc.ncbi.nlm.nih.gov/articles/PMC9522646/
- Does: positions only. Per frame 22 players x (x, y, v, a) plus the same at offsets +-0.4, 0.8, 2, 4, 20, 40, 80 s (1,320 features); AdaBoost best: frame accuracy 92.0 %, F1 93.1 %; median filter 3.64 s (91 frames) tuned on validation; stoppage-level F1 79.8 % at IoU >= 0.5 (stoppages > 2 s); starts/ends within +-2 s for 77 % / 81 %. 102 Bundesliga matches (TRACAB), data not public.
- Reusable: a ball-free dead-ball detector. Long-context offsets (to 80 s) are the key idea. Its stoppage-level IoU matching and +-2 s tolerance is the closest published analogue to spellF1. The ~8 % frame errors sit at boundaries, which is exactly what IoU matching punishes.
- Caveats: stoppages are shorter than our 21 s ball-out blackouts.

**Set pieces from tracking**: Vidal-Codina (above) uses taker/line-up geometry; a thesis on corner/free-kick/throw-in classification showed up in search [UNVERIFIED]. Practical rule: ball exits near a touchline -> throw-in, goal line -> corner/goal kick; the taker identity ends the dead segment and fixes the next team (usually the opponent of the last toucher).

## 4. Temporal segmentation and sequence models

- **MS-TCN / MS-TCN++** (Farha & Gall CVPR 2019, https://arxiv.org/abs/1903.01945 ; Li et al. TPAMI 2020, https://arxiv.org/abs/2006.09220): multi-stage temporal conv nets with a smoothing loss against over-segmentation. Metrics: frame accuracy, Edit score, segmental F1@{10,25,50}. Our spellF1 is the same family (one-to-one IoU matching, averaged over thresholds). Lesson: frame accuracy hides fragmentation; every spurious short segment is a false positive and can split a true spell, losing a match at higher IoU.
- **Viterbi/CRF decoding**: PathCRF shows constrained Viterbi works for possession. A semi-Markov (explicit duration) layer with a minimum-duration prior and H<->A switch cost is the natural way to optimise spell structure. No prior art for HMM team-possession from tracking was found (HMMs exist for tactics: Ötting & Karlis, https://pub.uni-bielefeld.de/record/2962681 ; Markov possession models on event data: Lopez Pena, https://arxiv.org/abs/1403.7993).
- **Temporal action detection**: THUMOS14-style mAP over tIoU 0.3-0.7 is the likely origin of the spellF1 range [UNVERIFIED; no detection-specific source retrieved].

## 5. Aligning two feeds of the same match

- **ELASTIC** (Kim et al.; https://arxiv.org/abs/2508.09238 ; CIKM 2026 version with Needleman-Wunsch, https://arxiv.org/abs/2608.30227 ; code https://github.com/hyunsungkim-ds/elastic): event<->tracking sync using only tracking-derived features (ball-player distance, ball acceleration, kick distances, frame-delay penalty); 88.4 % exact-frame alignment vs 21.1 % for ETSY; includes a public frame-level re-annotated benchmark on 3 Sportec matches. Reusable: the DP (Needleman-Wunsch) alignment pattern for syncing SportsCode tags to tracking; kickoff sync first, then refine.
- **ETSY** rule-based sync: https://dtai.cs.kuleuven.be/sports/blog/etsy:-a-rule-based-approach-to-event-and-tracking-data-synchronization
- **Broadcast <-> tactical**: no peer-reviewed method found. Patent filings (US 11778244 / 12260789 family) describe aligning feeds via fixed pitch markings and matched player crossings [patent text only]. Practical recipe: map broadcast tracks to pitch coordinates via homography; estimate lag by cross-correlating aggregate signals (team centroid / width velocity, per-player speed) scanning +-several seconds per half, allowing linear drift (25 vs 29.97/30 fps); validate on kickoff and goal-kick restarts. Broadcast coverage is partial (a 2025 validity study found under half of player activity visible: https://arxiv.org/abs/2508.19477 ; sparse broadcast tracks reconstructed with 3.45 m mean error: https://arxiv.org/abs/2311.14642), so use it as ball-visibility / on-screen-team cue during blackouts rather than full tracking.

## 6. SoccerNet overlap and hackathon write-ups

- **SoccerNet challenge reports** 2023 https://arxiv.org/abs/2309.06006 , 2024 https://arxiv.org/abs/2409.10587 , 2025 https://arxiv.org/abs/2508.19182 : Ball Action Spotting (Average-mAP, tight tolerances 1-5 s), 2025 Team Ball Action Spotting (Team-mAP@1 s; best 60.03; assigns each action to a team - closest to us), Game State Reconstruction (pitch coordinates + role + team + jersey from broadcast). T-DEED (2024 winner) is the reusable spotting backbone if we derive team-labelled pass/receive anchors from the broadcast video. 2026 player-centric ball-action spotting / FOOTPASS (macro-F1 over 8 actions): https://www.alphaxiv.org/abs/2606.09679 . Review: https://arxiv.org/abs/2410.01304
- **More Than A Hack (Barca Innovation Hub / Mobile World Capital / Metrica Sports)**: 2nd edition framed as generative AI on sports data, six-month programme, final at Talent Arena 2027 (https://www.fcbarcelona.es/es/noticias/4584971/). NO public write-ups of a possession-labelling task or solutions found. Nearest hackathon write-up: SkillCorner x Zurich Sports Analytics Club winners (https://landing.skillcorner.com/articles/meet-the-winners-of-the-skillcorner-x-zurich-sports-analytics-club-hackathon), EPV-focused.

## 7. Evaluation design (noise floors, bootstrap)

No football-specific literature found. Transferable practice:
- Report spellF1 per IoU threshold as well as the mean; 0.3 vs 0.7 separates missed spells from boundary slop.
- GT boundary noise is ~1-2 s (stoppage study uses +-2 s; ELASTIC shows manual timestamps need frame-level re-annotation). Build a noise floor by jittering GT boundaries by U(-d, +d) s and scoring a perfect-but-jittered predictor; also score trivial baselines (always-home, carry last possession forward).
- Bootstrap CIs by resampling matches, then spells within match (block bootstrap; adjacent spells are dependent). Single-match tests (as in PathCRF) give wide CIs; use leave-one-match-out over all open matches.
- Track Edit score / over-segmentation ratio next to spellF1.

## Top 8 actionable ideas (ranked)

1. **Semi-Markov / constrained Viterbi over {H, A, D}** with minimum-duration prior and switch costs, tuned directly on spellF1 (merge spells shorter than ~2-3 s). Cheapest, biggest win on a segment metric; fragmentation is the main loss.
2. **Ball-free dead/alive classifier** from positions (Sci Rep recipe: x, y, v, a for all players + offsets to 80 s + ~3.6 s median filter, GBM/AdaBoost). Blackouts start with ball-out, so predict D at onset and find the restart end.
3. **Restart-geometry rule**: after a near-line exit, detect the taker (one player at the line, others in set shape) to end D and set the next team from last-touch alternation (last known possessor before blackout).
4. **PathCRF-style possessor-path inference** trained on DFL + Metrica with whole-blackout masks, with outside nodes for line exits; team label from decoded possessor.
5. **Ball Radar / TranSPORTmer masked training** with simulated 21 s blackouts starting at line exits; use possessor probability (not ball xy) as the feature.
6. **SportsCode tags as soft anchors** after an ELASTIC-style Needleman-Wunsch sync; first measure the tag-lag distribution per tag type before trusting boundaries.
7. **Broadcast feed as sparse side-information**: estimate offset by cross-correlating team-centroid velocity; inject ball/player evidence where visible. Optional until sync error < ~0.5 s.
8. **Evaluation harness**: leave-one-match-out, match-level block-bootstrap CIs, boundary-jitter noise floor, trivial baselines; select models by CI-aware spellF1, not frame accuracy.

## Open datasets with ball-status / possession labels

- **Sportec Open DFL (already used)**: Bassek, Rein, Weber, Memmert, "An integrated dataset of spatiotemporal and event data in elite soccer", Sci Data 2025, https://www.ncbi.nlm.nih.gov/pmc/articles/PMC11787359/ (CC-BY 4.0; 7 matches; ball status + possession; loaders in floodlight / kloppy / databallpy). Plus ELASTIC's frame-level re-annotated events for 3 of these matches (via the ELASTIC repo).
- **Metrica sample games (already used)**: events only; derive possession; no dead-ball flag.
- **PFF FC World Cup 2022**: 64 matches, broadcast-derived tracking + events (sign-up; check licence): https://www.blog.fc.pff.com/blog/enhanced-2022-world-cup-dataset . Closest noise profile to ours (missing players/ball).
- **SkillCorner open data** (github.com/SkillCorner/opendata; broadcast tracking + events) [UNVERIFIED: not retrieved; confirm URL/licence].
- **SoccerNet-GSR / SoccerNet Team Ball Action Spotting**: broadcast-side team labels and team-attributed passes/drives as possession anchors; no continuous possession labels.
- **StatsBomb open data (360)**: events + freeze frames, no continuous tracking; limited use.
- Loaders: https://floodlight.readthedocs.io , https://kloppy.pysport.org

## Gaps
Vidal-Codina PDF thresholds not verified; Sci Rep first author unverified; no public Metrica hackathon possession write-ups; no broadcast-vs-tactical sync paper; no football-specific bootstrap-evaluation literature.
