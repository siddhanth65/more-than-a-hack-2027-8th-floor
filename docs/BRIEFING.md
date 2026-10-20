# More Than A Hack 2027 — Team Briefing

*Status as of 2026-10-20. Working folder `mth/possession/`, data in `HOME/mtah-data` (never in git, never in CloudDrive, never pasted into any cloud LLM).*

---

## 1. The challenge in one paragraph

Metrica Sports + FC Barcelona give us one LaLiga match (Barcelona v Athletic, 2024-25) as a 199,319-frame tactical video at 25 fps with **automatic, uncorrected** tracking (22 anonymous player slots with team, plus the ball), 638 automatic "Smart Tagging" phases, a per-frame homography, and an unsynchronised broadcast recording with its own tracking. For every frame we output one label: **H** (Barcelona has the ball, in play), **A** (Athletic), **D** (dead ball), **X** (abstain). The file is scored automatically against a private frame-level ground truth; the top 20 teams go to the Innovation Challenge. Deadline **17 Nov 2026, 23:59 CET**. Team registration (3–5 members, one team name) by **10 Nov**.

## 2. How we are scored — and what that implies

- **Ranking metric: spellF1.** A *possession* is a maximal run of H or of A (D breaks it). Our possessions are matched one-to-one to the true ones by temporal IoU, greedy by IoU, same team required; F1 is averaged over IoU thresholds 0.3–0.7. Each possession counts once — a 0.8 s one-touch possession is worth as much as a 42 s one.
- **Tie-break: macroF1** over H/A/D per frame.
- **X is forward-filled** by the scorer before spell matching (back-filled at the start of a half). So on the ranking metric X means "repeat my last answer": free inside a possession, expensive across a hand-over. Ungraded frames (pre-match 0–11184, half-time 90652–110807, post-match 190198–199318) cost nothing.
- **Graded frames:** 1st half 11185–90651, 2nd half 110808–190197 (158,857 frames). The 2nd half is graded from 110808 but kicks off at 113742 — those 2,934 frames are all D.
- **Ground truth semantics** (derived mechanically from Metrica's *corrected* tracking): D where the corrected ball is dead; otherwise the team that owns the ball. Loose/contested balls stay with the last owner; a change requires *controlled* possession (a touch, deflection or aerial duel without control is not a change). A dead ball is a stoppage, never a possession; `H H D D H H` is two possessions.
- **Implications:** (1) possession *count* and *boundary placement* dominate; per-frame accuracy is secondary; (2) every spurious D frame or wrong-team flicker splits a possession into two half-credit pieces plus a false positive; (3) every missed D between same-team spells merges two possessions; (4) differences below ~0.04 spellF1 are noise on one match.
- **Published floor:** `baseline_smooth50` = 0.2433 (nearest player to ball + 2 s majority vote). The public 5-minute window (17 possessions) is for pipeline checks only — tuning the smoothing window on it picks the wrong one on the full match.
- **We will never know the top-20 cut-off** — no leaderboard, no feedback. Repo stats (20 forks / 47 stars) suggest tens of entrants.

## 3. What we have

| asset | status |
|---|---|
| Challenge repo (scorer, validator, baselines, public GT) | cloned at `mth/repo`, scorer reproduces published numbers, our regenerated baselines are byte-identical |
| Tactical bundle (video 5.2 GB, tracking, tagging, homography) | complete and verified (199,319 frames/rows, ball on 56.9 %) |
| Broadcast bundle (video 14 GB, tracking, homography) | complete and verified (164,488 records, ball 50.5 %, homography valid 65.8 %) |
| Python env | `HOME/mtah-venv` (3.12, numpy/pandas/kloppy/opencv) |
| Prior work (football-synthesizer BTP) | surveyed by 10-agent research sweep; reuse map in the design spec |
| Proxy ground truth | 6 DFL open matches (12 halves, 570 min) on E: with per-frame ball-alive flag and possession team, already parsed to npz |
| Same-provider reference | Metrica sample games 1–2 (corrected tracking, 39–41 % ball missing in the same long-run pattern) |
| Compute | laptop CPU + RTX 3050 4 GB (cluster busy) — the plan needs no GPU |

## 4. What the data actually looks like (measured locally, aggregates only)

- **The ball is missing in blackouts, not flicker.** 150 gaps on graded frames; median 10.7 s, p90 34 s, max 3.7 min; **85 % of missing frames sit in 79 stretches > 10 s = 37 minutes.** Zero gaps between 1 and 3 s. Interpolation/smoothing cannot help.
- **Blackouts are a mix of dead and live.** Public window: ball-missing frames are 56 % D, 44 % live (A 386 frames, H 5). Match-wide: 52 % of blackouts start with the ball ≤ 5 m from a line and moving outward (80 %); re-acquisition is also near a line in 49 %. 36 of 79 contain a set-piece tag. The live portions are overwhelmingly Athletic possessions (long balls lost in the air — the ball is present 99.9 % of the time Barcelona has it, 85 % when Athletic has it).
- **The automatic ball is clean when present:** no static-ball lock-ons (0.0 %), 13 teleports in 113 k frames. Nearest-player logic is therefore trustworthy *on ball frames*.
- **Smart Tagging** covers 82 % of graded frames (78 % with a team), 77 % inside blackouts; set-piece windows are long (median 35 s). **All 26 throw-ins and all kick-offs carry no team.**
- **The ground truth has one-touch possessions** (0.8 s, 1.3 s in the public window) and long D runs (5–20 s); no short D runs in the window.
- **The public window is unusually ball-rich** (ball present 88 % vs 59 % match-wide): any method validated only there will fail in the 37 blind minutes.

## 5. What we are doing (approach A → B)

A single pipeline, pure numpy/pandas/scikit-learn, deterministic, reproducible with one command.

1. **Loaders & periods** — tracking to metres on a 105.3 × 68 pitch, per-half direction of play, graded masks, tag intervals.
2. **Evidence producers** (each gives per-frame log-scores for H/A/D and abstains where it has no opinion):
   - `ball_ctrl` — on ball frames: carrier/flight/contested forward-backward model (ported from the BTP), touch detection from ball-velocity kinks, loose-ball-stays-with-last-owner semantics.
   - `dead` — dead-ball classifier trained on DFL ball-status: ball-out geometry at blackout entry, time since blackout start, set-piece tags and tag gaps, restart shapes (throw-in taker on the touchline, corner cluster, goal-kick GK, kick-off halves), player-speed collapse.
   - `shape` — ball-free team-in-possession from team shape (BTP FP1 model, retrained on DFL with simulated blackouts); used mainly on live blackout frames.
   - `tags` — team prior from tagged phases; set-piece tags give a D prior just before and a taker-team prior just after (no team for throw-ins → taker inferred from the isolated touchline player).
   - `broadcast_ball` (B) — the broadcast's independent ball track, aligned to the tactical timeline by matching player formations in pitch coordinates, borrowed inside blackouts.
3. **Decoder** — one 3-state (H/A/D) Viterbi per half combining the evidence with switch costs tuned for spellF1, with evidence-scale correction so smoothed posteriors don't swamp the switch costs.
4. **Post-processing tuned for the metric** — A-B-A merges below a minimum duration unless a strong touch says otherwise, D-hole handling, per-transition-type boundary offsets. X emitted nowhere.
5. **Validation** — the official scorer wrapped unchanged; DFL proxy with a blackout simulator fitted to the real match's blackout statistics (79 / 41 % / 10.7 s median / 3.7 min max); leave-one-match-out selection; one DFL match sealed; the public window read ≤ 3 times, for conventions and plausibility, never for tuning; label-free drift checks on the real match.

## 6. What it takes to win

1. **Get the blind third right.** Decide D vs live inside blackouts, and the team on the live part. This is where the ranking is decided; everyone can do the ball frames.
2. **Don't fragment.** One stray D frame or team flicker halves the credit for a possession. Stickiness lives in the decoder, not in a smoothing window.
3. **Don't merge.** Missed D between same-team spells costs a whole possession. Dead-ball recall matters as much as team accuracy.
4. **Boundaries at the right frame.** Ball-velocity kinks when the ball exists; restart events (tags, shapes) when it doesn't.
5. **Validate on something that looks like the real match** (DFL + simulated blackouts), not on the 5 ball-rich public minutes.
6. **Ship a reproducible repo + a clear technical document.** The organisers may re-run the winner's repo; a non-reproducible file is not a result.
7. **Register the team by 10 Nov.**

## 7. SWOT

**Strengths**
- A year of directly relevant prior work: possession Viterbi, touch detection, team-shape possession model, dead-ball rules, noise simulators, tests — with measured results and documented failures.
- A real proxy ground truth (DFL, 570 min, per-frame ball status + possession) nobody is obliged to have.
- Same-provider reference (Metrica sample games) to lock conventions and blackout shapes.
- All data verified locally; pipeline runs on CPU; no dependency on the busy cluster.
- Tooling discipline already in place (sealed splits, look budgets, registration before reads).

**Weaknesses**
- Only one match to score on; spellF1 noise floor ~0.04 — small gains are invisible.
- DFL's possession/dead definitions may differ from Metrica's corrected-tracking semantics (the public window looks "stickier" than DFL windows); mitigated by multiple truth variants and mean-rank selection, not solved.
- Learned decoders previously lost to nearest-player on new domains; we must keep rule-based baselines as the bar.
- Team-shape transfer dropped from 0.93 (FOOTPASS) to 0.76 at events on DFL — may drop further on automatic tracking.
- Time: 4 weeks, one person coding with AI assistance; the broadcast alignment is a project in itself.

**Opportunities**
- Broadcast ball likely exists exactly where the tactical tracker lost it (the TV camera follows the ball) — a ball in the blind third without any GPU.
- Throw-in taker inference and restart shapes give team priors on the most frequent restarts, which the tags leave blank.
- Athletic's long-ball blackouts are a *pattern* (ball in the air → lost): a "long ball by A" prior from last ball velocity + Long Ball tags.
- The 3050 is free for a 1-hour go/no-go test of video ball re-detection; if positive and time allows, a cluster request is justified.

**Threats**
- Blackout content on the full match may differ from the public window's 56/44 split; the dead-ball model is trained on a simulator.
- The organisers' scorer semantics are fixed and known, but our *reading* of "controlled possession" is only as good as 17 public possessions.
- Over-tuning to DFL; drift on the real match goes unnoticed without label-free checks.
- Logistics: validation of the exact uploaded file, naming (`<team>.csv`/`<team>.pdf`), Google account, one response per team.

## 8. Timeline (hard stop 17 Nov)

| dates | milestone | exit criterion |
|---|---|---|
| Oct 21–23 | harness: loaders, periods, official-scorer wrapper, DFL proxy + blackout simulator, baselines B0–B4 | simulator passes realism gates; baselines scored LOMO; `make submission` writes a valid file |
| Oct 24–28 | approach A: ball_ctrl, dead, shape, tags, decoder, post-processing | beats best baseline by ≥ 0.05 on ≥ 5/6 DFL matches; count ratio 0.8–1.25; drift checks pass on the real match |
| Oct 29–Nov 3 | approach B: broadcast ball census → alignment → borrowed-ball evidence | census ≥ 30 % coverage of blackout frames or B shelved; A vs B differ by one evidence column |
| Nov 3 | **checkpoint**: decide on C (video re-detection) using the 3050 go/no-go test | — |
| Nov 4–12 | improvement loop on DFL-sim, ablations, error decomposition; sealed-match read | registered candidate |
| Nov 13–16 | freeze; public repo; technical document; `validate_submission.py` on the exact file; dry-run reproduction from a clean clone | byte-identical regeneration |
| Nov 17 | submit (`<team>.csv`, repo URL, `<team>.pdf`) | before 23:59 CET |

## 9. Open items for the humans

- Team name and 3–5 registered members by **10 Nov** (register early).
- Who else codes? Natural split: one person owns the broadcast alignment (B), one owns the technical document + reproduction testing, the core pipeline is mine.
- Google account for the submission form.
