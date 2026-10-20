# Research notes: Vidal-Codina et al. event detection and PathCRF

Method: web fetch only, 2026-10-20. Extraction was done by a summarising fetch tool, not by reading raw text, so quotes are tool-relayed. Re-check key numbers against the paper before citing.

## A. Vidal-Codina, Evans, El Fakir, Billingham (2022)
"Automatic event detection in football using tracking data", arXiv:2202.00804; Sports Engineering 2022.
Sources: https://arxiv.org/abs/2202.00804 and https://ar5iv.labs.arxiv.org/html/2202.00804. The raw PDF was not text-extractable, so the numbers below come from the ar5iv HTML.

IMPORTANT: the paper leaves most tolerances symbolic (epsilon parameters) and defers their values to "Online Resource 1" (the Sports Engineering supplement). That supplement was NOT fetched. [UNVERIFIED] every epsilon value.

### A1. Ball-control states (Sec 2.2, 2.3.1-2.3.3)
- Four states: "dead ball, no possession, possession and duel". Events occur only in possession or duel frames.
- Possession Zone (PZ), Sec 2.3.1: circle of radius R_pz around each player. Ball inside a player's PZ means that player is in possession. R_pz = 50 cm (provider A) and 1 m (providers B and C), per Sec 3.
- Duel Zone (DZ): circle of radius R_dz >= R_pz around the ball. If at least two opponents are inside, the frame is a duel. R_dz = 1 m for all providers. If both conditions fire, the duel wins.
- Control frame: any frame labelled possession or duel.
- Smoothing: Savitzky-Golay on ball x-y, 2nd-order polynomial, 7-frame window (about 0.25 s at 25 Hz), for noisy data.
- Direction and speed: incoming (d0, v0) from frame f-1, outgoing (d1, v1) from frame f+1.
- Possession loss (Sec 2.3.2), both must hold: (1) the ball is outside the player's PZ at f+1 and displacement ds > eps_s [value not given]; (2) the player is absent from the next control frame.
- Possession gain (Sec 2.3.3), over a control sequence with the same player, either: direction change d0.d1 < eps_theta [value not given], or speed change |v0 - v1| > eps_v on at least one frame [value not given]. Otherwise the frames are treated as false positives.
- No minimum-duration threshold for possession is given. A save/claim is checked for a loss within 1 s (Sec 2.4.2).

### A2. Set-piece rules (Sec 2.3.4, 2.4.1)
Trigger: the ball is dead (boolean flag, or missing ball data). All tolerances are symbolic [UNVERIFIED values].
- Kick-off: all players in their own halves (tol eps_k1) and at least one player within eps_k2 of the centre mark. IFAB Law 8.
- Penalty: one player on the goal line between the posts (eps_p1). One opponent in a square box from eps_p2/4 in front of to 3*eps_p2/4 behind the penalty mark. All others outside the penalty area and beyond 9.15 m from the mark (eps_p3). Law 14.
- Goal kick: at least one player in their own goal area (eps_c in the text, eps_g in the Fig 4 caption). Law 16.
- Corner: at least one player within eps_c of an active corner mark. Law 17.
- Throw-in: at least one player beyond the auxiliary sideline (sideline minus eps_t). Law 15.
- Free kick: no trigger of its own. It is the default when no pattern matches, with the player having the ball in PZ on the first in-play frame.
- Confirmation: the trigger must persist to the last dead-ball frame, and the taker must be within R_pz of the ball on the first in-play frame. Ties follow the hierarchy in Fig 5. Several candidate takers resolve to the one closest to the ball.

### A3. Missing ball and in-play flag
- Tracab supplies an explicit in-play/dead boolean.
- Track160 and Hawk-Eye: dead ball is inferred from missing ball data during dead play.
- The dead/in-play information is "critical for set piece detection" and is listed as a limitation. The algorithm needs ball tracking. No numeric limit on the length of a ball-data gap is given.

### A4. Reported accuracy (Sec 3.1, Fig 7)
- Abstract: "+90%" detection rate in most event categories, across tournaments and providers.
- Passes: >90% precision and recall.
- Shots: precision 78%, recall 53%. Of 254 shot misclassifications, about 70% were blocked shots labelled as passes. 569 crosses were labelled as passes.
- Corner, throw-in, goal kick: >90% precision and recall. Free kicks were the worst set piece. Kick-offs and penalties were "paramount", but no figures were given.
- Ball-to-player distance at pass time: under 1 m.
- Runtime: about 3 min per 90-min match on 1 CPU and 4 GB RAM.
- Caveat: the "supra-90%" wording is relayed by the fetch tool. [UNVERIFIED] against the PDF figures.

## B. PathCRF (Kim, Lee, Seo, Ko, Yoon, Park)
"PathCRF: Ball-Free Soccer Event Detection via Possession Path Inference from Player Trajectories", arXiv:2602.12080 (submitted 2026-02-12).
Sources: https://arxiv.org/abs/2602.12080, https://arxiv.org/html/2602.12080v2 (first 100k of 129.7k chars read; appendices unread), https://github.com/hyunsungkim-ds/pathcrf

- Inputs at inference: player tracking only, no ball. Team membership is used (per-team attention groups). Node feature composition (velocities etc.) [UNVERIFIED], not stated in what was read.
- Frame rate: raw 25 FPS, downsampled to 5 FPS. Windows are 10 s (50 steps, stride 5 frames), built within in-play episodes. Predictions are upsampled to 25 FPS. This means in-play episode segmentation is needed (a dead-ball flag).
- Graph: fully connected directed graph on 26 nodes (22 players plus 4 outside nodes: left, right, top, bottom). One edge per step out of 676.
  - Self-loop (u,u): u controls the ball.
  - Edge (u,v): a kick from u to v.
  - Edge (u,o) then (o,o): out of play.
- Model: 2 stacked socio-temporal encoder blocks. The social part combines PPE (ISAB per team plus outside) and FPE (ISAB over all nodes). The temporal part is positional encoding plus SABs (a Bi-LSTM variant also exists). An edge-embedding MLP feeds the emission and transition scores. There is also a coarse-level auxiliary head.
- CRF: linear-chain. Emission = MLP(edge embedding). Transition is dynamic, MLP over the pair of edge embeddings at t-1 and t. Allowed transitions are identity, kick, reception and out-of-play; all others get -1e4 (masked). Decoding is Viterbi. An event is flagged where the selected edge changes (control, kick, out-of-play).
- Training data: Sportec Open DFL dataset (Bassek et al. 2025, Figshare), 7 Bundesliga 1/2 matches. Train 5 / valid 1 / test 1 (364k / 80k / 85k frames).
- Training supervision needs the ball and events. Labels are built from event data synced with ELASTIC (uses ball and players) and RDP ball-touch candidates aligned by Needleman-Wunsch. Ball is NOT needed at inference.
- Accuracy (test match, Table 2, dynamic masked CRF): edge acc 69.64%, sender 80.44%, receiver 82.28%, violation 0%, event P/R/F1 = 73.18/78.37/75.69%. Baseline "Ball TP" F1 = 48.39%. Frame-level team possession accuracy 92.40%. Ball TP localisation error 2.74 m. Event recall about 80% at 1 s / 4 m tolerance.
- Frame-rate ablation (Table 4): 25 FPS F1 77.21% but 6,727 s/epoch vs 565 s/epoch at 5 FPS (RTX 4090).
- Runtime: about 23 s per match at 5 FPS on an RTX 4090. Latency per 10 s episode is 0.24 s. Parameter count, hidden sizes, optimizer and epochs were not reported in the text read [UNVERIFIED].
- Licence: code MPL-2.0 (per the GitHub LICENSE/sidebar); paper CC BY 4.0.
- Repo: pretrained checkpoints in saved/ (trials 100-140), train.py, inference.py, evaluate.py, tutorial.ipynb, dataset.py, models/, set_transformer/, datatools/preprocess.py. Dependencies are in requirements.txt (versions not inspected), plus kloppy and ELASTIC (hyunsungkim-ds/elastic). The README states neither a GPU requirement nor a frame rate.

### Feasibility verdict (our 199k frames at 25 fps)
- Inference on CPU is feasible. 199k frames is about 133 min of play, or about 40k steps at 5 FPS (about 800 windows of 50 steps). The model is a small attention network. The GPU time of about 23 s per match scales to a rough guess of a few to tens of minutes on CPU [UNVERIFIED, not benchmarked].
- The ball is not needed at inference or for the pretrained checkpoints. It IS needed to build training labels, so retraining on our data requires ball/event labels. Retraining on CPU is impractical (565 s/epoch on a 4090). Use the pretrained Sportec checkpoints, or fine-tune only if we have labels.
- Port risks: (1) a domain shift to Metrica needs the pitch coordinate convention matched (Sportec is metres, centred origin [UNVERIFIED]) and the 22-player slot ordering reproduced; (2) the model needs in-play episode segmentation (Vidal-Codina-style dead-ball detection can supply it); (3) the checkpoints may assume CUDA, so load with map_location=cpu [UNVERIFIED]; (4) it detects ball-control/kick events and the possession path, not set-piece types.
- Verdict: run the pretrained model on CPU after downsampling to 5 FPS, then interpolate the path back to 25 fps. Check the repo's requirements.txt and tutorial.ipynb first.
