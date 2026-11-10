# spellF1 metric analysis: toy experiments (2026-10-23)

Everything here uses synthetic label sequences only. Code (pure functions + toy scripts) lives in the scratchpad:
(scratch directory of the analysis session; not part of the repository)
(`spellmetric.py` is the module proposal; `toys1.py` covers Q1-Q3, `toys2.py` timing, `toys3.py`/`toys4.py` the decoder sweeps, `synth.py` the posterior model).
`spellmetric.match_tp` works on spell arrays and matched the official `spell_scores` exactly on 300 random cases (0 mismatches).

## How the official scorer behaves (`mtah/eval/official/possession_metrics.py`)

- **Spells** are maximal runs of H or of A inside each GT-graded run (GT != X). D always ends a spell, so `H D H` is
  two spells. Predicted X is forward-filled (back-filled at a run head) *before* spells are built, so X cannot split a
  spell. Predicted labels on GT-X frames are ignored, and a predicted spell is cut at a GT-X gap.
- **Matching**: every same-team overlapping pair gets an IoU. Pairs are sorted by (-IoU, gt_idx, pred_idx), and
  greedy one-to-one matching stops below the threshold. The matches at threshold t are exactly the greedy prefix
  with IoU >= t, which is why one pass serves all thresholds (our `SpellF1` and `match_tp` rely on this).
- **Score**: F1_t = 2 TP_t / (n_gt + n_pred) for t in {0.3, 0.4, 0.5, 0.6, 0.7}, averaged with no weighting. D is not
  scored here at all, except that it splits spells. macro-F1 over H/A/D is only the first tie-break.
- **Consequences**: (a) at IoU >= 0.5 two disjoint predictions cannot both match one GT spell, so greedy order only
  matters at 0.3/0.4, where a fragment or a straddling prediction can still win the larger piece. There is nothing
  in the ordering to exploit. (b) A spell scores the same at IoU 0.70 as at 1.0, so boundary work only matters while
  a spell sits below IoU 0.7. (c) The marginal economics, linearised and checked exactly below:
  **dF1 = 2/(n_gt+n_pred) * (dTP - (F/2) dP)**, where F is the current spellF1. One more true match is worth +1
  unit (about 1/n_spells, roughly 0.3 points at 350 spells). Each extra predicted spell costs F/2 units, whether it
  matches or not.

## Q1: Boundary error vs credit (credit = fraction of the 5 thresholds matched, isolated spell)

Overshoot by a fraction f of L gives IoU = 1/(1+f). Undershoot gives IoU = 1-f.

| error f = x/L | IoU (too long) | credit | IoU (too short) | credit |
|---|---|---|---|---|
| 0.20 | 0.833 | 1.0 | 0.800 | 1.0 |
| 0.30 | 0.769 | 1.0 | 0.700 | 1.0 |
| 0.43 | 0.699 | 0.8 | 0.570 | 0.6 |
| 0.50 | 0.667 | 0.8 | 0.500 | 0.6 |
| 0.67 | 0.599 | 0.6 | 0.330 | 0.2 |
| 1.00 | 0.500 | 0.6 | 0 | 0 |
| 1.50 | 0.400 | 0.4 | 0 | 0 |
| 2.40 | 0.294 | 0 | 0 | 0 |

The cliffs fall at total undershoot of 0.30/0.40/0.50/0.60/0.70 L and total overshoot of 0.43/0.67/1.0/1.5/2.33 L.
**Overshooting is about 1.5-3x more tolerant than undershooting.** Long spells (L > 5 s with sigma < 0.1 L) are
always at full credit, so all the metric sensitivity sits in spells of 1-3 s.

**Isolated spell** (D on both sides, so extending it costs nothing on spellF1). Both ends get N(0, sigma) error, and
each end is moved outward by k*sigma:

| sigma/L | k=-0.5 | k=0 | k=0.5 | k=1 | k=1.5 | k=2 |
|---|---|---|---|---|---|---|
| 0.10 | 0.977 | 0.993 | **0.996** | 0.988 | 0.962 | 0.910 |
| 0.15 | 0.889 | 0.949 | **0.957** | 0.925 | 0.853 | 0.759 |
| 0.20 | 0.775 | 0.874 | **0.892** | 0.840 | 0.743 | 0.628 |
| 0.30 | 0.560 | 0.707 | **0.743** | 0.679 | 0.561 | 0.434 |

The best outward bias is about 0.5 sigma per end, worth +0.003 to +0.036 credit per spell. Over-extending
(k >= 1.5) is worse than no bias.

**Shared H|A boundary**: extending one spell shortens the other. The short spell Lb is 50 frames (2 s), the
boundary is N(0, sigma), and the shift is optimised by Monte Carlo (negative = into the longer spell A):

| La/Lb | sigma=5 | sigma=10 | sigma=20 | credit gain at sigma=20 (sum of 2 spells) |
|---|---|---|---|---|
| 1 | 0 | 0 | 0 | 0 |
| 2 | -5 | -4 (0.4 sigma) | -6 (0.3 sigma) | +0.013 |
| 4 | -5 | -4 | -9 (0.45 sigma) | +0.028 |
| 10 | -5 | -4 | -9 | +0.029 |

So under symmetric uncertainty, **err toward lengthening the shorter spell**, by 0.3-0.45 sigma, and only when the
lengths are clearly unequal. Under asymmetric uncertainty, place the boundary on the skewed side's quantile, then
apply the same rule.

Probability q that a real short spell is matched (credit) when each of its boundaries has N(0, sigma) error:

| spell length | sigma=3 | sigma=6 | sigma=10 | sigma=15 |
|---|---|---|---|---|
| 25 f (1 s) | 0.98 | 0.81 | 0.56 | 0.37 |
| 50 f (2 s) | 1.00 | 0.98 | 0.88 | 0.71 |
| 75 f (3 s) | 1.00 | 1.00 | 0.97 | 0.88 |

q stays at or above 0.8 only while the length is at least ~4 sigma. Below ~2.5 sigma, q is about 0.5 or less.

## Q2: Short-spell economics

**Case A**: H(L1) A(s) H(L2), where the short A spell is real with probability p and is matched with probability q
if real. If it does not exist, the truth is a single H spell. c = credit of the larger piece vs the whole, which is
c(max(L1, L2)/T): 0.4 for an even split, 0.6 at 60/40, and 1.0 at 75/25 or beyond. The options are predict
(H A H), omit (H H H) or D-filler (H D H). Linearised expected values:
predict = p(2+q-1.5F) + (1-p)(c-1.5F); omit = p(c-F/2) + (1-p)(1-F/2); D-fill = p(2-F) + (1-p)(c-F).

| c (neighbour split) | F | q | omit if p < | D-filler band | predict if p > |
|---|---|---|---|---|---|
| 0.4 (even) | 0.45 | 1.0 | 0.33 | none | 0.33 |
| 0.4 | 0.45 | 0.4 | 0.38 | 0.38-0.56 | 0.56 |
| 0.4 | 0.60 | 0.7 | 0.41 | 0.41-0.43 | 0.43 |
| 0.4 | 0.60 | 0.4 | 0.41 | 0.41-0.75 | 0.75 |
| 1.0 (lopsided) | 0.45 | 1.0 | 0.23 | none | 0.23 |
| 1.0 | 0.45 | 0.7 | 0.23 | 0.23-0.32 | 0.32 |
| 1.0 | 0.60 | 0.7 | 0.30 | 0.30-0.43 | 0.43 |
| 1.0 | 0.60 | 0.4 | 0.30 | 0.30-0.75 | 0.75 |

Closed form: predict beats omit when p > p* = (1+F-c)/(3+q-2c). An exact check embedded in a 350-spell background
gave p* = 0.389 / 0.303 / 0.301 against linear predictions of 0.388 / 0.301 / 0.300.

**Case B**: merge two same-team spells across a short D gap (g = 1 s). p is P(gap is a real stoppage). Keeping the
split beats merging when p > (1+F/2-c)/(3-2c):

| L1/(L1+L2) | c | F=0.3 | F=0.45 | F=0.6 |
|---|---|---|---|---|
| 0.50 | 0.4 | 0.34 | 0.37 | 0.41 |
| 0.65 | 0.8 | 0.25 | 0.30 | 0.36 |
| >= 0.75 | 1.0 | 0.15 | 0.22 | 0.30 |

When one neighbour is much longer, the merged prediction still matches it, so a wrong split only costs the extra
spell (F/2). That is why lopsided gaps should be split already at p of about 0.2-0.3.

**Calibrated-posterior check (Q2 + Q5).** Synthetic templates had 120 possessions, uncertain short spells,
phantom blips, short D gaps, and per-boundary sigma. Truth was drawn from the posterior, 80 templates x 5 truths,
and differences are paired against MAP (include if p >= 0.5):

| decoder | easy (F~0.90) | hard (F~0.62, sigma x3) |
|---|---|---|
| include if p >= 0.2 | -0.031 +/- 0.003 | -0.053 +/- 0.008 |
| include if p >= 0.3 | -0.014 +/- 0.002 | -0.038 +/- 0.008 |
| include if p >= 0.4 | -0.005 +/- 0.001 | -0.006 +/- 0.005 |
| include if p >= 0.6 | -0.004 +/- 0.005 | -0.017 +/- 0.009 |
| D-fill p in [0.3, 0.5) | -0.003 +/- 0.003 | -0.032 +/- 0.008 |
| D-fill p in [0.4, 0.5) | +0.000 +/- 0.002 | -0.011 +/- 0.004 |

The optimum sits at a threshold of about 0.4-0.5, not at the naive F/2. Splitting the neighbour (the 1-c term)
and low q push p* up, and in the hard regime (q ~ 0.4-0.7) the formula gives 0.4-0.5 too. **A blanket D-filler
for uncertain spells hurts.** It only wins in the narrow, low-q band in the table above.

## Q3: D frames at boundaries

D does **not** trim the IoU denominator. A predicted spell shorter than the truth has union = L_gt, so IoU =
(L-x)/L, the same penalty as any undershoot. At an H|A boundary with an uncertain zone of width w (true boundary
uniform in the zone), filling the zone with D undershoots both spells, while a single boundary gives one overshoot
and one undershoot of half the error on average:

| La, Lb, w (frames) | boundary at zone middle | D-fill whole zone |
|---|---|---|
| 60, 60, 60 | 1.852 | 1.119 |
| 50, 50, 40 | 1.949 | 1.428 |
| 200, 50, 60 | 1.872 | 1.395 |
| 500, 40, 80 | 1.658 | 1.000 |

D-fill loses 0.5-1.1 TP-units per boundary on short spells. At a spell|true-D boundary the order is: median end
(1.000) > zone-to-spell (0.95-0.98) > zone-to-D (0.93-0.97) (L = 100-500, w = 50-200). Extending into D is cheaper
than retreating from it, consistent with Q1. **D is not a safe filler.** Its only spellF1 role is to separate
same-team possessions (case B). It also costs macro-F1 (each wrongly-D frame is an H/A recall miss plus a D
precision miss, roughly 0.3/n_class per frame), so D-fill has no upside on either metric.

Also note that X is dominated. On spellF1, X is identical to writing the previous label (forward fill), while on
macro-F1 X is always a miss. So write the previous label explicitly, and only use X if a rule forbids guessing.

## Q4: Minimum-Bayes-risk (expected-spellF1) decoding: feasibility

Timing at full size (F = 200k frames, ~290 spells):

| operation | cost |
|---|---|
| `match_tp` on spell arrays (exact) | 0.36 ms/eval |
| frames to spells (RLE) | 0.10 ms |
| package `SpellF1.f1` on a frame array (fill + RLE + match) | 4.15 ms/eval |
| naive-Python FFBS sampling, 3-state HMM, K=100 | 8.5 s (vectorised over K; numba would be well under 1 s) |

- **Sample-MBR** (choose among C candidates by mean spellF1 vs K samples): C x K evals. With C=K=200 that is 40k x
  0.46 ms, about 18 s per match. Use spell arrays, not `SpellF1.f1` (9x slower).
- **Local MBR** (coordinate ascent over include/omit/merge/boundary-shift decisions): spellF1 is 2TP/(n_gt+n_pred)
  with TP additive over connected components of the overlap graph, so a decision only changes a local window.
  Incremental evaluation is about 20 us per sample per option, which for ~400 decisions x 3 options x K=100 is
  about 2-3 s per match. A naive full re-evaluation is about 1 min.
- Toy outcome (calibrated posterior): sample-MBR over raw samples plus MAP picked MAP in 40/40 templates (gain
  +0.000). Raw samples are never consensus-optimal. Local MBR with K=30 came out at -0.007 +/- 0.002 (easy) and
  +0.04 +/- 0.03 (hard, n=12, inconclusive), because sample noise is overfitted at small K.
- **Verdict**: feasible in compute, but there is no demonstrated gain. It only pays off if the posterior samples
  are calibrated at the *segment* level. HMM/CRF FFBS samples are typically overconfident on segment existence,
  so they need temperature tuning on CV. The closed-form rules below are the analytic MBR solution for isolated
  decisions, and they are the cheaper first step.

## Q5: Exploitable properties / post-processing rules (toy-estimated)

| rule | toy effect | condition |
|---|---|---|
| Include an uncertain spell iff P(real) > p* = (1+F-c)/(3+q-2c), which is about 0.3-0.5 | wrong threshold costs 0.005-0.05 (tau 0.2-0.3 vs 0.5) | biggest lever; needs a calibrated P(spell real) |
| Split same-team spells across an uncertain short D iff P(stoppage) > (1+F/2-c)/(3-2c): about 0.2-0.3 lopsided, about 0.35-0.4 even | per-decision, +0.2 to +0.6 units when it flips | strongest for lopsided neighbours |
| H/A boundary nudge into the longer neighbour by about 0.25 sigma (up to 0.4 sigma when the short side is 2 s or less and 4x shorter) | +0.0017 +/- 0.0003 (hard), +0.0001 (easy); k=0.5 is neutral, k=1 costs 0.008-0.013 | only when sigma_boundary is 0.15 L_short or more |
| Extend spells into adjacent D by about 0.5 sigma | +0.0007 +/- 0.0006 (hard), -0.0009 (easy); k=1 costs 0.005-0.012 | only short spells (sigma >= 0.15 L); costs macro-F1 |
| Length-only min-duration filter | a proxy for rule 1: q < 0.6 when length < ~2.5 sigma | do not use a fixed seconds cut-off; condition on sigma and p |

Bottom line: on spellF1 the existence and merge/split decisions of 1-3 s spells dominate everything else. Boundary
nudges are second-order (at most +0.002). Calibrate a per-spell P(real) and per-boundary sigma, and apply the
closed-form thresholds with F set to the current CV spellF1 (about 0.45). Tune the thresholds on CV around 0.4-0.5
rather than trusting F/2.
