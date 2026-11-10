# Kill tests: camera motion cue (Diag 1) and pre-restart dead time (Diag 2)
Label-free. No contest ground truth read; aggregates only. Scripts: scripts/diag_camera.py, scripts/diag_prerestart.py. Raw numbers: diag_camera.json, diag_prerestart.json.

## Diag 1 - camera motion from the homography
Footprint per frame: centre = pitch point under the image centre; width = pitch length of the mid-row span (10-90 % of image width); zoom = log width; pan speed = |d centre/dt|, zoom rate = |d log width/dt|, 1 s moving average, central difference over 1 s. Fits on every 5th frame of VISIBLE-ball graded frames. Ridge alpha 10; GBT depth 3, 150 iterations; 2 threads. Half-split in both directions.

Camera movement (graded frames)
| camera | centre x sd (m) | centre y sd | log-width sd | pan m/s p25 / p50 / p75 / p95 | zoom rate /s p50 / p95 |
|---|---|---|---|---|---|
| tactical (100 % valid) | 19.2 | 3.6 | 0.118 | 0.02 / 0.16 / 1.33 / 3.9 | 0.001 / 0.039 |
| broadcast (usable frames) | 25.8 | 8.7 | 0.193 | 1.4 / 3.0 / 4.8 / 8.1 | 0.035 / 0.108 |

The tactical camera is NOT static. It follows the ball along the pitch (corr(centre x, ball x) = 0.83 in both halves) but not across (corr(centre y, ball y) 0.04 / 0.14); its width varies by about +-12 %.
Broadcast is mapped with offsets 4014 / 26920; usable = homography valid, in sync, not replay-like: 59 % of graded frames, 78 % of visible-ball frames, only 31 % of blackout frames (the rest are replays, cutaways or invalid). Replay frames are excluded because a replay cue already exists.

### (a) Ball-position proxy, visible frames, median error in m (euclid; along / across)
| | tactical h1 | tactical h2 | broadcast h1 | broadcast h2 |
|---|---|---|---|---|
| camera centre as ball | 23.6 (12.5 / 16.3) | 23.6 (11.8 / 15.8) | 16.4 (8.9 / 10.5) | 15.5 (6.9 / 10.5) |
| mean of tracked players | 19.9 (11.1 / 11.5) | 20.7 (11.4 / 12.8) | 19.3 (10.5 / 11.5) | 19.7 (10.4 / 12.7) |

Ridge on the test half, median euclid error (p90):
| split | camera only | players only | players + camera | reduction median / mean / p90 | along / across median reduction |
|---|---|---|---|---|---|
| tactical train1 -> test2 | 24.5 | 11.7 (26.6) | 11.9 (24.4) | -1.8 % / +2.9 % / +8.2 % | -3.4 / +2.2 % |
| tactical train2 -> test1 | 22.8 | 10.6 (22.2) | 10.8 (22.4) | -2.3 % / +0.5 % / -0.8 % | -4.6 / -5.8 % |
| broadcast train1 -> test2 | 12.7 | 10.5 (20.0) | 8.8 (16.2) | 16.1 % / 18.9 % / 19.3 % | 19.0 / 8.1 % |
| broadcast train2 -> test1 | 12.5 | 10.1 (20.0) | 8.6 (17.1) | 14.4 % / 15.8 % / 14.1 % | 19.6 / 8.7 % |

Primary metric fixed as median euclid reduction. Tactical: no gain. Broadcast: 16.1 % and 14.4 %, so it misses 15 % on the second split by 0.6 pt (passes on mean euclid and on the along-pitch median, fails across).

### (b) Dead-ball proxy (pseudo-label = the candidate D on visible-ball frames)
D share on visible frames: tactical set 15.4 % (a5c) / 14.4 % (a14); broadcast-usable set 6.1 % / 6.3 %.
GBT AUC on the test half (increment = players+camera minus players):
| camera | label | train1 -> test2: players / +cam / incr | train2 -> test1: players / +cam / incr | camera only (both) |
|---|---|---|---|---|
| tactical | a5c | .903 / .905 / +.002 | .908 / .915 / +.007 | .71 / .64 |
| tactical | a14 | .907 / .906 / -.001 | .928 / .917 / -.011 | .62 / .59 |
| broadcast | a5c | .881 / .889 / +.009 | .895 / .922 / +.028 | .83 / .82 |
| broadcast | a14 | .868 / .863 / -.005 | .883 / .899 / +.016 | .80 / .87 |

With ball speed added to the base model the increments are +.010 / -.000 (tactical a5c), +.011 / +.018 (broadcast a5c), about 0 for a14. Single-feature AUC (dead = positive; below .5 means dead has the smaller value): pan speed tactical .33-.41 (a5c), broadcast .13-.18; zoom level (log width) .51-.67; zoom rate .29-.47. Broadcast pan speed alone separates dead strongly (about .84 reversed) but the players already carry that (.88-.90).

### (c) Inside natural blackouts: median pan m/s [p25-p75]
| camera | visible live | visible dead | blackout called live | blackout called dead |
|---|---|---|---|---|
| tactical, a5c | 0.60 [.04-1.96] | 0.08 [.02-.78] | 0.16 [.03-1.55] | 0.03 [.02-.43] |
| tactical, a14 | 0.58 | 0.06 | 0.14 | 0.03 |
| broadcast, a5c | 3.56 [2.2-5.2] | 1.18 [.57-2.2] | 2.51 [1.3-4.1] | 0.61 [.23-1.2] |
| broadcast, a14 | 3.57 | 0.97 | 2.53 | 0.61 |

Candidates call 77 % (a5c) / 74 % (a14) of tactical-usable blackout frames dead, 54 % of broadcast-usable ones. The camera agrees with the candidate split: where they call live the camera moves like visible live play, where they call dead it is nearly still. A camera-only classifier trained on visible frames of one half and applied to the other half blackouts separates called-dead from called-live with AUC .58 / .57 (tactical, a5c / a14) and .82 / .79 (broadcast). Onset profile of tactical pan at the start of 79 long blackouts (median over blackouts, m/s): -5..-2 s 0.27, -2..0 s 0.34, 0..2 s 0.11, 2..5 s 0.09. The camera decelerates when the ball is lost but is not frozen at once, so it is not simply driven by the ball track (it could still derive from the same tracker).

### Verdict (rule fixed in advance: >= 15 % error reduction in (a) OR >= 0.03 AUC in (b), on BOTH half-splits)
- Tactical camera: NO-GO. (a) -2 % / -2 %; (b) increments between -.011 and +.007.
- Broadcast camera: NO-GO, narrowly in (a): 16.1 % / 14.4 % on the primary metric; (b) +.009 / +.028 (a5c), -.005 / +.016 (a14). It is usable on only 31 % of blackout frames, so even a GO could not reach most blackouts.
- (c) is mildly encouraging for the broadcast camera as agreement evidence (it matches the candidate live/dead split in blackouts, AUC about .8), but this is information the replay and player streams already hold; no independent gain is demonstrated.

## Diag 2 - pre-restart dead time
SkillCorner definition: for each player_possession event with game_interruption_before = <type>_for or _against (goal = kick-off after a goal), dead time = (its frame_start minus frame_end of the immediately preceding player_possession event in the same period) / 10 fps. It includes the last ball flight to the line plus the stoppage, up to the restart taker reception. 20 matches.
| type | n | p25 | median | p75 |
|---|---|---|---|---|
| throw-in | 834 | 10.3 | 16.3 | 24.0 |
| free kick | 405 | 19.1 | 29.3 | 49.8 |
| goal kick | 261 | 23.7 | 33.3 | 42.4 |
| corner | 141 | 31.2 | 36.7 | 43.0 |
| kick-off after goal | 44 | 63.5 | 72.3 | 80.3 |

This reproduces the briefed medians except goal (72 s vs 74 s). One penalty restart ignored.

Contest: the Smart Tagging SET PIECES start frame is taken as the restart time. Per tag, find the candidate D->live edge within a window; run length = the contiguous D run ending at that edge. Finding: the candidate D->live edge falls a median 4.2-7.0 s AFTER the tag start for every type (the tag lies inside a D run for 49 of 58 tags), so the requested +-3 s search misses most edges. Two windows are reported: A = [-3, +3] s (as specified), B = [-3, +10] s. D-through = D still running past the window end (no edge); none = no D in the window. Run-length stats are over all tags of the type (0 when none; for D-through, D length up to the tag).

Window A, a5c (a14 similar):
| type | n | edge | D-through | none | run s p25-med-p75 | below SK p25 | blackout before (n): run s | ball visible before (n): run s |
|---|---|---|---|---|---|---|---|---|
| throw-in | 26 | .04 | .85 | .12 | 1.8-5.4-15.0 | 65 % | 17: 5.0-12.8-16.4 | 9: 0-0-3.8 |
| free kick | 5 | 0 | 1.0 | 0 | 15.3-20.3-27.0 | 40 % | 5: same | 0 |
| goal kick | 10 | .10 | .70 | .20 | 0.5-4.4-27.9 | 60 % | 5: 27.6-28.0-57.8 | 5: 0-0-4.1 |
| corner | 12 | .17 | .83 | 0 | 18.7-22.3-26.2 | 92 % | 12: same | 0 |
| kick-off after goal | 4 | 0 | 1.0 | 0 | 40.6-56.2-87.4 | 50 % | 4: same | 0 |

Window B (edge within [-3, +10] s), a5c / a14:
| type | n | edge share | run s p25-med-p75 a5c | run s p25-med-p75 a14 | SK p25 / median | share below SK p25 |
|---|---|---|---|---|---|---|
| throw-in | 26 | .77 / .85 | 5.3-11.3-18.9 | 6.2-10.1-18.4 | 10.3 / 16.3 | 46 % / 50 % |
| free kick | 5 | .80 / .60 | 21.0-24.2-36.2 | 21.4-24.2-28.3 | 19.1 / 29.3 | 20 % / 20 % |
| goal kick | 10 | .80 / .80 | 4.9-9.6-33.8 | 5.6-9.0-33.4 | 23.7 / 33.3 | 60 % / 60 % |
| corner | 12 | 1.0 / 1.0 | 24.1-26.6-29.1 | 24.0-28.4-31.5 | 31.2 / 36.7 | 83 % / 75 % |
| kick-off after goal | 4 | .75 / .75 | 43.6-59.9-91.0 | 16.2-44.6-76.7 | 63.5 / 72.3 | 50 % / 50 % |

Blackout split (window B, a5c): throw-ins after a blackout (n=17) median 17.0 s vs ball-visible before (n=9) 3.8 s; goal kicks blackout (5) 35.8 s vs visible (5) 4.4 s; all corners, free kicks and kick-offs are in blackouts. The shortfall is therefore concentrated where the ball IS visible before the restart (median 3.8-4.4 s vs SkillCorner 16-33 s), while in blackouts run lengths already match the SkillCorner medians (throw-in 17 vs 16, goal kick 36 vs 33). Corners are 3-10 s short of the norm even in blackouts.

Systematic shortfall? Yes for corners (83 % / 75 % below other-game p25, median 26.6-28.4 s vs 36.7), goal kicks (60 %, median 9 s vs 33 s) and visible-ball throw-ins; not for free kicks or blackout throw-ins; kick-offs after goals unclear (n = 4).
Frames that would change if every tagged restart were extended to the SkillCorner p25: a5c 7,250 frames (4.6 % of 158,857 graded) with window B, 10,685 (6.7 %) with window A; a14 8,307 (5.2 %) and 11,704 (7.4 %). To the SkillCorner median: 13,650 (8.6 %) / 14,674 (9.2 %) in window B. The candidates already call 40.6 % (a5c) / 38.5 % (a14) of graded frames D, so headroom exists only where those frames are currently called live.

## Caveats
- Pseudo-labels are candidate outputs, themselves built from players and ball; the players-only AUC of .87-.93 is partly circular and inflates the baseline that the camera must beat. The ball-position test (a) has no such circularity.
- Camera features exist only where the broadcast shows live play; for the main target (blackouts) coverage is 31 %.
- Tag times lead the candidate D->live edge by about 4-7 s: either tags mark the phase start before the actual restart, or the candidates hold D late. Without truth this cannot be told apart, so window-B lengths include up to about 10 s after the tag and may overstate the pre-restart run.
- SkillCorner dead time contains ball flight and uses its own possession events; the Metrica convention will differ by a few seconds. n is small for free kicks (5) and goals (4). Extending to p25 is right only if the true D really lasts that long; no truth was consulted.