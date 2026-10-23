# Plan 5 pre-registration (written 2026-10-23, before any integration run of the filled ball)

The filled ball (`runs/ball/filled_ball.npz`, fine-tuned detector on the tactical video, 14.6 % of blackout frames) exists only for the real match, so its use cannot be selected on the simulator. Rule fixed before looking at any result:
1. Base = the current best candidate configuration (a10 if its robustness audit passes, else a5c), unchanged.
2. Two ways to use the fill, each a scalar weight: (a) `fill_dead_w`: a D log-likelihood term on frames where the filled ball is present and slow (< 1 m/s over 1 s): +w*log(LR) with LR estimated label-free from the base candidate's own D vs H/A on those frames; (b) `fill_as_ball`: substitute the filled ball into `match.ball` on filled frames before computing the ball/dead/team features (on/off).
3. Sweep (a) w in {0, 0.5, 1, 2} x (b) {off, on}; report only label-free aggregates: visible-dead share, spells per graded minute, implied tag agreement, agreement of D with the broadcast replay cue on filled frames, and D share on PROGRESSION mid-tag frames.
4. Admissible: visible-dead in [0.10, 0.18]; spells in [2.7, 4.3]; PROGRESSION false-dead <= base + 0.02; tag agreement >= base - 0.01. Choose the admissible setting with the highest replay-cue agreement on filled frames; ties -> smaller w / off. If none beats the base on that agreement by >= 0.01, the fill is NOT used.
5. The public window plays no part.
