# Feed-adaptive touch detector (branch `fix-touch`, 2026-10-24)

Label-free. Aggregates only: the public ground-truth file is never read, sim match 9007 is never loaded, nothing is
registered. Base commit is `f65bd8c` (master + the a10 learned-stream code; master itself lacks the a10 sources, which
step 4 needs). Reproduce with `scripts/touch_{diagnose,validate,adversarial,candidates,sweep}.py` (outputs under the
gitignored `runs/touch/`).

## 1. Why `detect_touches` (mode abs) fires 15 times on the real match

Real = the Metrica tactical feed (94,214 visible graded frames = 62.8 min); sim = 9001-9006 seed 100 (12 halves).

| statistic (visible graded frames) | real p50 / p90 / p99 | sim p50 / p90 / p99 (median over halves) |
|---|---|---|
| ball speed (m/s) | 5.1 / 14.3 / 24.6 | 6.6 / 14.9 / 23.7 |
| abs(delta speed) per frame (m/s) | 0.17 / 0.66 / 1.32 | 0.71 / 6.1 / 13.0 |
| abs(delta direction) per frame (deg, speed > 2 m/s) | 0.6 / 3.6 / 10.4 | 8.4 / 155 / 178 |
| dv, k=3 velocity change (m/s) - the thresholded statistic | 0.68 / 2.18 / 4.43 | 1.41 / 6.9 / 16.1 |
| robust sigma of dv (1.4826 MAD) | 0.73 | 1.75 |
| share of frames with dv >= 5 (theta_dv) | 0.7 % | 19.8 % |
| nearest tracked player (m) | p50 1.9 | p50 2.4 |

Three separate reasons, in order of damage:

1. **The sim ball is jittery, the real ball is smooth.** The simulator's corruption leaves the ball direction essentially
   random frame to frame (p90 155 degrees); the real track is a smoothed trajectory (a kick is spread over 0.5-1 s, speed
   moves 0.17 m/s per frame at the median). theta_dv = 5 m/s is 2 robust sigma of the sim's dv but 6 sigma of the real
   one, so 20 % of sim frames exceed it against 0.7 % of real frames (2,268 raw sim peaks per half, 403 real peaks in the
   whole match).
2. **The real peaks that do exceed it are not touches.** 89 % of the 403 real peaks sit within 2.4 s of a ball gap
   (41 % of graded frames have no ball; tracking glitches at re-acquisition), the median speed at a peak is 18.6 m/s
   (balls in flight) and the median nearest player is 7.3 m away; only 15 of 375 (after the teleport gate) pass the
   1.5 m gate. In the sim (9001 H1) 95 % of strong peaks (k = 8, z >= 3) are within 2 m of a player (median 0.5 m); in the real feed strong peaks
   are *less* often near a player than a random frame (7-18 % vs a 42 % base rate for dv >= 6 at k = 3/5), at every scale and lag
   checked. So the absolute-threshold + at-the-peak-frame proximity rule selects nearly nothing.
3. **The real ball-player geometry itself is fine.** Share of visible frames with a player within 1 m: 28.5 % real vs
   33.9 % sim (9001 H1); attachment episodes 23 vs 32 per visible minute (9001 H1). The information about *who has the ball* is there;
   what is missing is a sharp kink at the kick. Blackouts matter less (windows are NaN near gaps, which only costs a few
   peaks), but gap-adjacent glitches are the dominant false peaks.

## 2. The fix: `TouchParams(mode="rel")` (default stays `"abs"`)

`mtah/evidence/touches.py`; `abs` is byte-identical to before (a5c and a10 reproduce their registered CSVs).

- scale k_rel = 8 (0.32 s each side, matches the real smoothing); dv_8 turned into a **per-half robust z** (median /
  1.4826 MAD over the half's visible frames; sigma floor 0.25 m/s; pooled fallback when a half has < 150 frames);
- trigger = z >= z_thr (2.5) **or** a direction change >= 45 degrees with both speeds >= 2 m/s (and z >= 1.5), plus a
  physical floor dv >= 2 m/s so a noise-only path never fires;
- local maximum over +-max(k, 0.15 s) (min gap 0.3 s); ignore +-10 frames around ball gaps and teleports (> 3 m/frame);
- **time-tolerant proximity gate**: a player within 1.5 m at some frame of +-8 around the peak (the closest
  (frame, player) gives the player and team; contested rule unchanged);
- **rate control per half**: everything >= z_thr, topped up with weaker peaks (down to z 1.5) to >= 6 touches per visible
  minute, never more than 24 per visible minute. Output column 3 is z * 1.75 so `theta_hi` (strong touch) keeps its
  meaning. All fields are new defaults of `TouchParams`; old config JSONs load unchanged.

Unit tests (`tests/test_touches.py`, 6 new): a pass gives one touch at the kick and one at the reception (right player
and team); noise-only path with a player on the ball gives none; the same pass at 4x the jitter gives the same two
touches; a noisy half where abs fires > 10 times gives fewer under rel; default mode is abs and identical to before;
unknown mode raises; the rate cap holds; teleport and gap guards. Full suite: 322 passed (the one failure,
`test_readlog::test_public_guard`, exists on the base commit: the public-read budget is 5 there).

## 3. Label-free validation on the real match (`runs/touch/validation.json`)

| | abs | rel | chance (same criterion) |
|---|---|---|---|
| touches per visible minute, real | 0.24 (15) | 7.96 (500) | |
| same detector on sims (mean, range) | 27.9 (20.6-36.7) | 19.1 (16.1-24.0) | sim DFL events ~17.7/min (9001 H1) |
| touches within 1.5 m / 2 m of a player at the touch frame | 100 % / 100 % | 51.6 % / 69.4 % | random visible frame: 42 % / 51.5 % |
| tag-implied hand-overs (45 evaluable of 51): touch of the gaining team within +-1 s | 0 % | 13.3 % | 9.4 % |
| ... any touch within +-1 s | 0 % | 24.4 % | 18.6 % |
| a5c H<->A boundaries (192 evaluable): touch within +-0.5 s | 0 % | 15.1 % | 10.0 % |
| a10 H<->A boundaries (200): touch within +-0.5 s | 0 % | 12.0 % | 10.4 % |
| a5c boundaries: median abs lag to the nearest touch | none | 3.0 s | 4.2 s |
| PathCRF-confident hand-overs (447 evaluable): any touch / gaining-team touch inside | 0 % / 0 % | 17.2 % / 8.3 % | 16.6 % / 8.3 % |

a5c lag histogram (0.5 s bins over -2..+2 s, rel): 4, 12, 7, 15, 14, 6, 10, 7 (61 % of boundaries have no touch within 2 s;
mild concentration at -0.5..+0.5 s). Sensitivity variants (k 5, z 2, z 3.5, r 2 m; `touch_sweep.py`, 72 cells) all
land at 7-10 touches/min with a boundary-alignment lift of 1.4-1.7 and tag-handover lift 0.5-1.3 (n = 45: noise): no
better region exists, so the defaults, fixed before the sweep, stay.

Reading: `rel` gives the real feed a plausible touch density (8/min vs the sim-event 17/min and an inflated 19-28/min sim
detector rate) and a physically sensible gate, but only a **weak** alignment signal (about 1.3-1.5x chance on a5c
boundaries and tag hand-overs, none on PathCRF hand-overs). The real ball track does not carry sharp kicks, so this is
close to what kink detection can give on this feed.

### Adversarial check (`runs/touch/adversarial.json`, same protocol as exp-robust, touch columns swapped on both sides)

| step features, sim vs real | before (abs; reproduces the cached 0.9970) | after (rel) |
|---|---|---|
| base-feature AUC | 0.9970 | **0.9648** |
| windowed AUC (Model A's input) | 0.9955 | 0.9891 |
| touch group alone | 0.988 | **0.614** |
| log_since_touch / log_until_touch single AUC | 0.917 / 0.915 | 0.637 / 0.635 |
| all features except the touch group | 0.956 | 0.956 |

The touch shift is closed (the touch group goes from near-perfectly separable to 0.61), but sim and real remain
separable at 0.956-0.965 through the other columns (the non-touch features alone give 0.956): touches were not the only
shift, only the biggest single one.

## 4. Effect on the current candidates (decoded with `rel`, configs and models unchanged, nothing registered)

| real match | a5c abs | a5c rel | a10 abs | a10 rel |
|---|---|---|---|---|
| visible-dead share (band 0.10-0.18) | 0.178 | 0.177 | 0.166 | 0.130 |
| spells per graded min | 3.10 | 3.06 | 3.52 | 3.40 |
| tag agreement (implied) | 0.729 | 0.755 | 0.766 | 0.770 |
| agreement with confident PathCRF (H/A frames) | 0.858 | 0.832 | 0.799 | 0.771 |
| D share inside replay-cue frames / outside | 0.451 / 0.063 | 0.451 / 0.062 | 0.396 / 0.063 | 0.339 / 0.041 |
| frame agreement rel vs abs (graded) | | 0.958 | | 0.907 |

a5c barely moves (its teamctrl and snapping use touches lightly). a10 moves more: its learned streams read the touch
columns heavily; with the real touches now present they lower the visible-dead share (0.166 to 0.130, near the lower
band edge) and its D share inside replay frames, and agree less with PathCRF. The models were trained on sim touches
at 19-28/min with an almost-perfect proximity relation; the real `rel` touches (8/min, weaker relation) are still
off-distribution, so this is a feed change without retraining. Not an improvement signal on its own.

## Concerns

- `rel` touches are weakly informative (lift ~1.3-1.5); do not expect retraining on them to be a big win by itself, and
  the validation hand-over sample is small (45).
- The sim rate (19/min under rel) is still 2.4x the real rate: the sim's injected jitter still creates near-player
  peaks. Matching the feeds properly means smoothing the simulator's ball (sim-side fix) or retraining on rel touches
  computed on a smoothed sim ball; the adversarial residual (0.956 without touches) needs its own look.
- a10 (and the merge model, teamctrl) were fit with abs touches; switching the default requires a re-registration, not a
  silent flip. Default left at `abs`.

## 5. Plan 7 T1: smoothed sim ball, `rel` default, rebuild (label-free; no sealed/public reads)

`CorruptParams.ball_smooth_mode/win/poly` (`mtah/sim/corrupt.py: smooth_ball`): Savitzky-Golay (poly 2) of the sim ball x, y per
run of visible frames, applied last in `build_half` (after dead-ball realism and blackouts), no random draws. Runs shorter
than the window are untouched. `scripts/calib_ball_smooth.py` (sims 9001-9006 seed 100, median over the 12 halves, vs the
real match, graded visible frames; `sigma` = 1.4826 MAD of the velocity change |dv|; the 0.73 reference is the k=3 value):

| window | sigma k=3 | sigma k=8 | speed p50 | speed p90 | sum of rel. errors (k=3, p50, p90) |
|---|---|---|---|---|---|
| real | 0.733 | 1.897 | 5.08 | 14.30 | |
| none | 1.741 | 1.619 | 5.00 | 13.90 | 1.418 |
| 5 / 7 / 9 | 1.563 / 1.372 / 1.174 | 1.474 / 1.430 / 1.394 | 4.38 / 3.93 / 3.67 | 13.81 / 13.72 / 13.66 | 1.305 / 1.139 / 0.924 |
| 11 / **13** | 1.035 / **0.913** | 1.371 / 1.347 | 3.51 / 3.42 | 13.60 / 13.55 | 0.770 / **0.626** |
| 17 / 25 (outside the grid, info) | 0.761 / 0.598 | 1.291 / 1.212 | 3.32 / 3.33 | 13.46 / 13.29 | 0.444 / 0.599 |

Window 13 (grid edge) is the closest of {5..13} and is the new default (dataclass and `configs/corrupt.json`). The k=3 sigma
is still 1.25x the real one, and the smoothing costs the median speed (5.0 to 3.4 m/s: the dead-ball hover random walk and
the jitter carried speed; the real median is 5.1) while the k=8 sigma moves away from the real value (1.62 to 1.35 vs 1.90).
One smoother cannot match all of them; the choice is the stated criterion, not a fit of the touches.

`TouchParams.mode` default is now `rel`; configs saved without a touch mode (a5c, a10 and older) load as `abs`
(`PipelineConfig.from_dict`), so registered candidates are unchanged.

Rebuild (`build_sim.py`, 5 seeds, plus `--extra-dir runs/dfl_extra --only 9007`): `pos`, `speed`, `team`, `half`, the
blackout masks and the truth variants are byte-identical to the previous build on all 60 halves of 9001-9006 (PathCRF cache
hit 0.03 s); the ball changes in all 60; the simulated tags are re-drawn (they read the ball), 99 vs 101 tags in 9001 H1.
`rel` touches per visible minute on the rebuilt sims: 17.7 (14.6-22.1) vs 19.1 before and 8.0 real: the near-player peaks that
survive the smoothing keep the sim at 2.2x the real rate.

Adversarial sim-vs-real AUC (`scripts/touch_adversarial_rebuild.py`: sim step features fully recomputed on the rebuilt sims
with fold models, real side as in section 3 with `rel` touches):

| step features | abs (cached) | rel, old build | rel, new build |
|---|---|---|---|
| base AUC | 0.9970 | 0.9648 | **0.9518** |
| windowed AUC | 0.9955 | 0.9891 | 0.9855 |
| touch group alone | 0.988 | 0.614 | 0.590 |
| all except the touch group | 0.956 | 0.956 | 0.945 |
