# Score tricks for spellF1: decoding, GT conventions, segmental post-processing (2026-10-23)

Web research only. No match data, file contents or per-frame numbers were sent to any service.
Builds on `literature-2026-10-20.md` and `plan2d-design-2026-10-21.md`. Those already cover the semi-Markov decoder (C2),
the merge model (C1) and long-context dead features (E1), so this note covers only what is new or changes a decision.

## 0. What the official scorer actually does

Sources: https://github.com/metrica-sports/more-than-a-hack-2027 (EVALUATION.md, CHALLENGE.md, FAQ.md,
`evaluation/possession_metrics.py`).
- `valid_runs` splits the graded frames into the runs where GT is not X (one per half). Spells, IoU and transitions are all computed
  inside a run, so the half-time gap creates no phantom transition.
- Predicted X is forward-filled within the run, or back-filled at the head of a half, **before** spells are built.
  For spellF1, X therefore means "repeat my last label". It never helps at a hand-over and costs macroF1, so never emit X.
- `possession_spells` is the maximal H or A blocks. D blocks are dropped, so `H H D D H H` is two spells.
- `spell_overlaps` pairs each prediction with every overlapping **same-team** GT spell. IoU is counted in frames. Pairs are sorted by IoU
  (descending) with index tie-breaks, and greedy one-to-one matching runs at each threshold in (0.3, 0.4, 0.5, 0.6, 0.7).
  F1 = 2PR/(P+R), and spellF1 is the unweighted mean over the five thresholds. macroF1 over H/A/D is the first tie-break.
- Official sensitivity (EVALUATION.md): 1 frame late = 1.0000, 3 frames = 0.9818, 8 frames = 0.8933, 1 s = 0.6539.
  The noise band on the full match is about 0.04.

**Closed-form surrogate (the key tool for everything below).** Let g(IoU) be the fraction of thresholds that IoU passes:
0 below 0.3, then 0.2 / 0.4 / 0.6 / 0.8 for each band, and 1.0 from 0.7 up. The matched sets are almost nested across thresholds,
so
  spellF1 ≈ 2·Σ_matched g(IoU_i) / (N_pred + N_gt).
- Check against our 5-min state: (17 + 19) · 0.656 / 2 ≈ 11.8 = 9 · 1.0 + 5 boundary spells averaging g ≈ 0.56. That fits.
- **Add or keep a predicted spell iff E[g] > S/N_pred = F/2 ≈ 0.33** at F ≈ 0.66. This is the same structure as Lipton et al.,
  "optimal threshold = F*/2" (https://arxiv.org/abs/1402.1892): each extra spell adds 1 to the denominator.
- **Split** a spell into two iff E[g_a] + E[g_b] − E[g_merged] > F/2. **Merge** under the reverse condition.
- Marginal values on the 5-min window: lifting one boundary-only spell from g 0.56 to 1 is worth +0.024; removing both false splits
  is worth +0.038; recovering one missed spell at g = 1 is worth +0.056.
- **Boundary asymmetry.** A spell of length L that is over-extended by e has IoU L/(L+e). Under-extended, it has (L−e)/L, which is lower.
  At a shared boundary only the short neighbour sits near a threshold, so an uncertain boundary should be **shifted into the longer
  neighbour**. Example: a 2 s spell with a 1 s error scores g 0.8 when extended and 0.6 when shrunk. The long neighbour stays ≥ 0.7 either way.
- **D interaction.** Predicting D on GT-alive frames shortens our spell. Predicting alive on GT-D frames lengthens it, and a lengthening
  error costs less IoU. When a D boundary's timing is uncertain, lean toward alive. **Exception:** a D that separates two same-team
  spells must be found. Missing it merges the two (one match lost, union inflated). A false short D inside a long spell
  is a split: +1 to N_pred, and the larger piece usually still matches. So a D should be emitted mid-spell only when E[gain] beats F/2.

## 1. Decoding for expected IoU-F1 instead of MAP

**Idea.** MAP/Viterbi returns the single most probable labelling, but its objective is the frame-joint likelihood. spellF1 rewards
spells with E[g] > F/2 and boundary positions that maximise E[g] for the short neighbour. Minimum Bayes risk (MBR) decoding picks
the output with the highest expected metric under the posterior.
- Lember & Koloydenko, "Bridging Viterbi and posterior decoding", JMLR 2014: https://jmlr.org/papers/v15/lember14a.html
- Bæk et al. 2025, hybrid decoding with FMCI. Posterior run length, dwell time and visit counts are computable, and hybrid beats Viterbi:
  https://arxiv.org/abs/2504.15156
- Nánási, Vinař, Brejová, HERD. Expected reward with boundary-position uncertainty in HMMs: https://arxiv.org/abs/1001.4499
- Gross et al., CONTRAST/Conrad, maximum expected (boundary) accuracy decoding for gene segments: https://genome.cshlp.org/content/17/9/1389 ;
  AUGUSTUS MEA (sampled exon posteriors → best feature set via shortest path): https://gensoft.pasteur.fr/docs/augustus/3.3.3/MEAposter.pdf
- Nowozin, CVPR 2014, optimal decisions for expected IoU. IoU decisions are combinatorial, and the approximation beats max-marginal:
  https://www.microsoft.com/en-us/research/wp-content/uploads/2016/02/nowozin2014intersectionoverunion.pdf
- Eikema & Aziz 2020, sampling-based MBR beats the mode when the mode carries tiny mass: https://arxiv.org/abs/2005.10283
- Goel/Kumar/Byrne, segmental MBR for ASR: https://www.isca-archive.org/icslp_2002/kumar02_icslp.html

**Concrete algorithm for our chain (MBR-S, sample-based).**
1. Treat the combined decoder as a linear-chain CRF. Per-frame score e_t(s) is the summed stream log-scores for s ∈ {H, A, D}.
   The transition score is −switch_cost. Scale everything by 1/T (temperature). The forward pass is O(T_frames · 9).
   At 25 fps for a full match (~2·10^5 frames) that takes well under a second in numpy, or downsample to 5 fps.
2. Forward-filtering backward-sampling (FFBS) draws K = 100–300 label paths. Apply the deterministic post-processing to each sample,
   or don't, and compare both.
3. Build the candidate pool:
   - the current pipeline output;
   - Viterbi under a grid of switch costs, τ_HA and island/merge thresholds (about 20–40 variants);
   - optionally a few samples.
4. Select argmax_c (1/K) Σ_k spell_f1_mean(c, sample_k), calling the **official scorer functions** with the sample as pseudo-GT.
   Cost: C·K scorer calls, about 40 × 200 = 8 000, each O(spells) → seconds to a few minutes per half on CPU.
5. **MBR-M (segment-level MEA, AUGUSTUS-style)** is the finer variant. Collect candidate spells (start, end, team) from all
   samples and variants. Estimate each candidate's E[g] as the mean g of its best-IoU same-team spell in each sample. Then choose a
   non-overlapping set that maximises Σ(E[g] − λ), with λ ≈ F/2, by weighted-interval-scheduling DP in O(C log C).
   Fill the gaps by extending neighbours, which uses the boundary asymmetry from section 0. This returns spells and boundaries
   directly instead of picking one of the variants.

**Calibration (the real risk).** Summed log-scores are a product of experts with double counting, so raw posteriors are overconfident.
Fit T, plus optionally a per-stream weight, on the training folds. Choose T so that the predicted P(spell matched at 0.5) is reliable,
or simply so that LOMO MBR spellF1 is best. Then evaluate on the held-out match.
First-order chains imply geometric durations, and samples will contain too many very short runs. MBR penalises those runs itself
because their E[g] is low, but if the calibration plot shows it, sample from an HSMM instead (C2 becomes a sampler, not a decoder).

**Gain mechanism.** (a) It drops locally probable but boundary-uncertain short spells whose E[g] < F/2: the false splits, and the
islands that the A-B-A rules miss. (b) It places boundaries at the position that maximises expected g, not at the most likely
switch frame. (c) It keeps the merge model's work: post-processed variants simply join the pool.
Expected +0.01..+0.04 on the simulator. It targets the 5 boundary-only spells and the 2 false splits. Cost 1–1.5 d. Risk medium,
from calibration and sim-to-real temperature drift. Mitigation: pick T on the simulator, then check that the spell count is
plausible on the real match.

**Cheap hybrid baseline (2 h).** "Posterior-Viterbi" (Fariselli 2005 style): run Viterbi on log forward-backward marginals with
the same switch costs. Lember–Koloydenko generalise it as a hybrid with weight k between MAP and posterior decoding. Add it as one
extra variant in the pool.

## 2. How providers define possession and dead ball, and what that implies

- **Metrica (this challenge, CHALLENGE.md/FAQ.md).**
  - Possession changes only when the opponent gains **controlled** possession. Contested or loose phases keep the last team.
    A touch without control, a deflection that does not intercept, or an aerial duel won without control does **not** change
    possession.
  - Proxies listed as possibilities, not rules: "keeps the ball for a few seconds" and "more than two consecutive passes".
  - Dead ball is always D and splits possessions.
  - GT is derived from **corrected tracking**: "a frame is D where the corrected tracking marks the ball dead".
    "Nothing is smoothed, interpolated or hand-adjusted." Public 5 min: 17 possessions, mean 14.2 s.
  - Metrica publishes no further documentation of its possession algorithm. Searches of metrica-sports.com, its blogs and Nexus
    or Smart Tagging pages found nothing.
- **Opta/Stats Perform.** A possession is ended by the opposition gaining control. Sequences break at stoppages, but a possession
  can span fouls and corners. That is unlike Metrica, where D splits.
  https://www.statsperform.com/resource/sequences-and-the-premier-league ; https://theanalyst.com/articles/opta-football-stats-definitions
- **StatsBomb.** `possession_team` is "the team that started this possession in control of the ball". The possession id
  increments on a change of control.
  https://socceraction.readthedocs.io/en/stable/api/generated/socceraction.data.statsbomb.StatsBombEventSchema.html
- **Wyscout.** A recovery ends the opponent's possession. Possessions that start after the ball goes out or after a foul are not
  recoveries. https://dataglossary.wyscout.com/recovery/
- **Link & Hoernig 2017** (DFL-type tracking, PLOS ONE). Team possession starts when an individual possession starts, "the moment
  a player is able to perform an action with the ball". There is no possession while the ball is out. Ping-pong and headed balls
  count as possession without control (IBC).
  F = .83 for IBP. https://pmc.ncbi.nlm.nih.gov/articles/PMC5503225/
- **Vidal-Codina et al. 2022.** Possession zone of about 1.5 m, plus a change in ball direction or speed to confirm a touch.
  Duel zone: two opponents inside it means a duel. Missing ball data can signal dead ball.
  https://arxiv.org/abs/2202.00804 ; databallpy implementation and the warning that "the definition of a ball possession does not
  align between our approach and that of the tracking data provider":
  https://databallpy.readthedocs.io/en/latest/features/team_and_player_possession.html
- **DFL open data** (our simulator truth). The positional files carry per-frame BallPossession and BallStatus flags
  (floodlight parser: https://floodlight.readthedocs.io/en/latest/modules/io/io.html ; dataset:
  https://www.dshs-koeln.de/en/institut-fuer-trainingswissenschaft-und-sportinformatik/detail/meldung/official-game-data-in-the-journal-scientific-data).
  I could not retrieve their exact semantics. They are a different definition from Metrica's controlled-possession rule.

**Implications.**
1. **Boundary convention.** Metrica places the hand-over at the gaining team's control, not at the loser's last touch.
   During pass flight, a loose ball, an aerial duel or a blackout, the label stays with the previous team. Boundary snapping should
   target the **first controlled touch of the gaining team** (a reception or interception followed by control), never the release.
   In blackouts, prefer the late end of the uncertainty window. This is also the safer side by the asymmetry in section 0 when the
   previous spell is the longer one.
2. **Short spells.** A control requirement suppresses one-touch "spells", but "nothing smoothed" means they can exist when control is
   reached quickly. The simulator has about 30 % of true spells < 3 s under the DFL definition, which may well be inflated relative
   to Metrica GT.
   Zero-cost local check (offline, on the public 5-min GT): the GT spell-length histogram (share < 3 s), and the offset of each GT
   H↔A transition from the nearest visible touch by the gaining player.
   If real short spells are rare, the simulator is teaching the selector to keep short spells it should drop.
3. **Across stoppages.** Every D run ends a spell. A same-team restart after D is a new spell, so missing a short D merges two spells.
   This raises the value of D recall specifically **between same-team spells** (E1 in plan2d).

## 3. Action-segmentation tricks for segmental F1 (CPU, no deep retraining)

- **ASRF (Ishikawa et al., WACV 2021).** A separate boundary-regression branch; frame labels are then refined by voting inside
  segments delimited by boundary peaks. Up to +16 F1 on Breakfast.
  https://arxiv.org/abs/2007.06866 , code https://github.com/yiskw713/asrf
  **Transfer:** a "boundary-first" decoder. Build a boundary hazard stream from touch detections, PathCRF edge changes, Smart-Tagging
  starts and ball-reappearance events. Take peak-picking with NMS (≥ 1 s apart) as candidate boundaries, label each interval with the
  argmax of summed log-scores, then decide which boundaries to keep with the F/2 rule or MBR-M.
  About 0.5 d. It overlaps with the existing snapping and is best used as a candidate generator for MBR-M.
- **BCN (barrier pooling)**, **MS-TCN T-MSE smoothing**, **OnlineTAS duration+confidence post-processing**. All three are "smooth
  within segments, allow change only at confident boundaries". Our switch costs and island removal already implement this, and
  T-MSE is a training loss. Low marginal value.
  Survey: https://arxiv.org/abs/2210.10352 ; resource list: https://github.com/nus-cvml/awesome-temporal-action-segmentation
- **Length-model decoding (FIFA, GCPR 2021; NN-Viterbi / HSMM).** https://arxiv.org/abs/2108.03894 . Same verdict as plan2d C2:
  more useful as an MBR sampler than as a MAP decoder.
- **Metric hygiene.** Track Edit score and the ratio N_pred/N_gt beside spellF1, as in the TAS literature. The surrogate in section 0
  makes N_pred/N_gt directly interpretable.

## 4. Football possession from tracking (2019–2026)

- **PathCRF** (Kim et al., arXiv 2602.12080, Feb 2026, code https://github.com/hyunsungkim-ds/pathcrf).
  - State = one edge per step: a self-loop means control, (u, v) means the ball is in flight from u to v, plus outside nodes.
  - Masked-transition Viterbi. Event F1 75.7 %, frame team-possession accuracy 92.4 % on the DFL test match.
  - **New use:** the in-flight edge (u, v) keeps the sender's team until the receiver's self-loop, which is exactly Metrica's
    "loose stays with last team".
  - Extract **"opponent receiver enters a self-loop" times** as boundary candidates inside ball blackouts. Snap boundaries to them,
    or feed them to the hazard stream in section 3. We already cache PathCRF, so cost is about 0.5 d.
- **Ball trajectory inference from player context** (Kim et al., KDD 2023; Set Transformer + hierarchical Bi-LSTM):
  https://arxiv.org/abs/2306.08206 . This fills ball blackouts, but deep training on 7 matches on CPU is costly. Low priority:
  PathCRF already captures most of that information.
- **Vidal-Codina 2022 and Link & Hoernig 2017** (above). Touch confirmation by a change in ball direction or speed is a cheap filter
  for "controlled" touches before snapping.
- **Sci Rep 2022** ball-free in-play features. Already E1 in plan2d.
- No 2024–2026 work was found that targets spell-level IoU metrics for team possession. The MBR angle appears to be unused in football.

## 5. Metric-specific rules of thumb (all follow from section 0)

1. **Never emit X.** Use H, A or D everywhere. X is forward-filled for spellF1 and counted as a miss in macroF1.
2. **Drop spells with E[g] < F/2.** The current merge-model θ can be set from this rule: keep if (1 − P_spurious) · E[g | real] > F/2.
   This replaces a grid and transfers better to real, because F/2 scales with the operating point.
3. **Boundary bias.** When uncertain, move the boundary **into the longer neighbour** by about half the boundary uncertainty.
   This lifts the short spell's IoU and is free for the long one. Learn the offset as a function of (L_short, uncertainty) on sim.
4. **Precision has a ceiling.** Beyond ±3 frames there is little left to win (0.98 at 3 frames). Gains come from errors > 8 frames,
   which mostly sit in blackouts and duels.
5. **IoU just under 0.7.** A spell at 0.69 is worth 0.8 and at 0.70 is worth 1.0. Borderline spells are the cheapest target for
   boundary tweaks.
6. **D placement.** Under-predict short D inside long spells. Prioritise D recall where it separates same-team spells. At D/alive
   edges of uncertain timing, lean toward alive.

## 6. Ranked shortlist (each runs on the simulator by LOMO)

| # | experiment | module | sim test | cost | exp. gain | risk |
|---|---|---|---|---|---|---|
| 1 | **MBR-S/M decoding.** FFBS samples from the tempered combined chain; choose the candidate (pipeline variants + MBR-M spell set) that maximises mean official spellF1 against the samples | after Viterbi, before or instead of post-proc | fit T and λ on train folds, score held-out match; compare with current output | 1–1.5 d CPU-min | +0.01..0.04 | calibration / T drift |
| 2 | **Surrogate-driven post-proc.** Keep or merge spells by the E[g] > F/2 rule (using merge-model P and sim-learned E[g]), plus a learned boundary shift into the longer neighbour | `spells/merge.py`, snapping | LOMO; ablate rule vs grid θ; boundary shift on/off | 0.5 d | +0.01..0.03 | low |
| 3 | **Metrica-convention alignment.** (a) offline 5-min GT check of short-spell share and hand-over offset vs gaining-team first touch; (b) relabel sim truth so flips occur only at the gaining team's first controlled touch, with loose, aerial and in-flight frames staying with the last team; re-select by LOMO; (c) PathCRF receiver-self-loop boundary candidates | simulator truth, snapping | LOMO with relabelled truth; check that the selected settings shift (τ, θ, snap side) | 0.5–1 d | real-side +0.01..0.05 | GT rule only partly known |

Order: run 3a first (minutes, local only). It decides whether the simulator's short-spell share is trustworthy, and therefore
whether 1 and 2 should be tuned on original or relabelled sim truth. Then run 2 (cheap, low risk), then 1.
