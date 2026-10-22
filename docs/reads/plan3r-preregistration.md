# Plan 3r pre-registration (written 2026-10-22, before any sweep of the replay stream)

The broadcast replay/close-up cue exists only for the real match (the DFL simulator has no video), so its weight cannot be selected on the simulator. Rule fixed before looking at any sweep result:

1. Base = candidate a5c (all artefacts, all parameters unchanged). The new stream adds a D log-likelihood term from the smoothed replay cue: +w*log(0.64/0.13) when replay_like, +w*log(0.36/0.87) otherwise (likelihood ratios estimated label-free from a5c's own D vs H/A frames on visible-ball frames). On the simulator the stream is absent, so simulator scores are identical to a5c by construction.
2. Sweep w in {0.5, 1.0, 1.5, 2.0} x dead_offset in {-2.0, -2.5, -3.0} on the real match; report only label-free aggregates: visible-ball D share, spells per graded minute, implied-team tag agreement, D share in the 30 s before goal-kick/corner/kick-off tags (pre-restart coverage), D share on PROGRESSION mid-tag frames (live-play false-dead proxy).
3. Admissible settings: visible-ball D share in [0.10, 0.18]; spells per graded minute in [2.7, 4.3]; PROGRESSION mid-tag D share <= a5c's value + 0.02; tag agreement >= a5c's value - 0.01.
4. Among admissible settings choose the one with the highest pre-restart coverage; ties -> smaller w. If none is admissible, the variant is rejected and a5c stays final.
5. The public window is not used in this choice. If the chosen variant replaces a5c, one report-only public read (read 4) is taken on it; its result does not reverse the choice.
