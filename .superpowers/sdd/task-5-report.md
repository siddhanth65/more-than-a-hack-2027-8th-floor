# Task 5 report: PathCRF precompute + selection a2 (code part)

Status: DONE (code + tests + commit). No data runs. `runs/sim` and the real match were not read.
Commit: `993c1b9` on `plan2b`, "feat(select): grid2 (PathCRF weight, dead policy), per-policy dead models, real-match plausibility; PathCRF precompute script". Only the six files below were staged. The other agent's corrupt files were left alone.

## Implementation
- `mtah/select.py`
  - `GRID2` has 64 cells: tau_HA {10,30} x tau_D {30,100} x w_dead {1,2} x w_pathcrf {0,0.5,1,2} x dead_policy {full, no_ball_at_all}, with aba_s 1.0.
  - `GRIDS = {"grid1": GRID, "grid2": GRID2}`.
  - `REAL_D_SHARE_BAND = (0.10, 0.18)`.
  - `is_real_plausible(x)` returns False for NaN.
  - `choose_plausible(ranked, plaus) -> int` returns the highest-ranked cell flagged `real_plausible`. Cells missing from `plaus` are never chosen. If no checked cell is plausible, it prints a WARNING and returns rank 1.
- `scripts/select_lomo.py`
  - `base_for_grid(grid, threads)`: `use_pathcrf` is True iff some cell has `w_pathcrf > 0`.
  - `policy_cfgs(base, cells, lengths)` returns `{policy: apply_cell(base, first cell with that policy)}`.
  - `run_grid` trains one dead model per policy for each fold, builds one test-evidence set per policy, and scores each cell with the evidence for its own policy. `estimate_lengths` is still a single call (base, policy "full"). Fold resume is unchanged: the policy evidence is not persisted and is recomputed when needed.
  - `real_plausibility(...)` runs after `summary.csv` for the top 5 cells:
    - lengths and one dead model per needed policy are built on all non-sealed sims (the run's seeds);
    - real evidence is computed once per policy (`load_tactical`, `load_tags`, `cache_key="real_tactical"`);
    - each cell's config is predicted, and `drift_report` (with `code_table`) supplies non_blackout_D_share, tag_agreement_implied, pred_D_share and spells_per_HA_min;
    - output goes to `summary_real.csv` and to `real_plausibility` in `best.json` and `registration.json`.
  - The chosen cell comes from `choose_plausible`. The spec bar and the out-of-sample drift reference are then computed for that chosen cell.
  - `best.json` and `registration.json` gain `cell_index`/`best_cell_index`, `chosen_rank` and `real_plausibility`. `registration.json` also records `grid`, `use_pathcrf` and `pathcrf_threads`, and its `selection` string states the plausibility rule.
  - New flags: `--grid` (default grid1, old behaviour), `--no-real`, `--data` (default `MTAH_DATA_DIR`) and `--pathcrf-threads` (default 4).
  - Sealed path: it uses `train_dead(train, policy=best_cell["dead_policy"])` and `base_for_grid([best_cell])`, so PathCRF is used only if the chosen cell weights it. A new `check_bar_gate` follows the brief: the sealed read is refused unless `spec_bar.bar_pass` is true or `--force-sealed REASON` is given. `bar_pass` and the reason are written to `sealed_reads.jsonl`.
  - The out-of-sample drift reference and the pilot use the chosen cell's policy (for the pilot, grid[0]'s policy).
- `scripts/precompute_pathcrf.py` (new)
  - Runs `pathcrf_posterior` for every (key, seed) in the sim, with key `<key>_s<seed>`. With `--real` it also runs on the real match (`real_tactical`).
  - Per-half output:
    - agreement of `p_home > 0.5` with `raw_Dfull` on graded H/A frames, labelled "optimistic (PathCRF train/val)" for 9001-9006 and never computed for sealed 9007;
    - the nearest-player proxy: ball visible, nearest tracked player within 2 m, graded frames;
    - NaN share, mean imputed_frac, seconds, and whether the half was computed or came from the cache.
  - Pure functions: `nearest_player_proxy`, `truth_agreement`, `posterior_stats`.
  - Writes aggregates to `runs/cache/pathcrf/precompute_summary.json`. There is no per-frame output.

## TDD
- RED: 12 new tests failed. They were ImportError on GRID2/GRIDS/is_real_plausible/choose_plausible, the missing `real_data_dir`/`base_for_grid`, and the missing precompute script. The 9 existing tests passed.
- GREEN:
  - `tests/test_select.py`: 14 pass (GRID2 size and axes, apply_cell on a GRID2 cell, plausibility band, pick / fallback / unchecked-cell cases).
  - `tests/test_select_smoke.py`: 4 pass. The new run_selection test uses a 3-cell grid whose first cell is `no_ball_at_all`; the old code would have raised the policy-mismatch error on it. The test also covers the PathCRF stream (monkeypatched `pathcrf_posterior`), the faked real match, summary_real.csv, the best.json/registration.json fields, consistency with choose_plausible, base_for_grid, and the sealed bar gate.
  - `tests/test_precompute.py`: 4 pass (two-player toy proxy, untracked slot, empty input, truth agreement and stats).
- Full unit suite: `PY -m pytest tests -q -m "not dfl and not real"` gave 184 passed, 7 deselected. test_corrupt.py also passed at the time.

## Self-review / concerns
1. **PathCRF cache is keyed on threads.** Precompute and selection must use the same thread count (default 4 in both). If precompute uses `--threads 8`, selection needs `--pathcrf-threads 8`, or it will recompute every half (about 1 h). `make_candidate` always uses the default 4 because `config.json` does not serialise threads, so it would recompute `real_tactical` (about 5 min). Recommendation: keep the default of 4 everywhere.
2. The real step adds one full-train `estimate_lengths`, up to 2 dead-model fits and the real-match evidence, roughly a few minutes. It needs the `real_tactical` PathCRF cache, or it computes it (about 5 min).
3. Lengths, including L_dead, are estimated with the "full"-policy dead model for all cells, as the design specifies.
4. Precompute compares against `raw_Dfull`, as the controller specified. The brief says ctrl_Dfull.
5. Precompute also runs PathCRF on the sealed 9007 halves (tracking only; needed by the sealed path). For 9007 it prints the label-free proxy only, never truth agreement.
6. The `bar_pass` / `--force-sealed` gate is new behaviour, taken from the brief. Pass a reason if the bar fails.
7. Each fold computes evidence twice, once per policy (ball and tag are recomputed; PathCRF is read from the cache). This is small next to scoring 64 cells.

## Commands for the controller (PY = HOME/mtah-venv\Scripts\python.exe, cwd = repo)
1. Precompute (about 1 h, run in background):
   `PY scripts/precompute_pathcrf.py --sim runs/sim --seeds 100 101 --real`
2. Selection a2:
   `PY scripts/select_lomo.py --grid grid2 --sim runs/sim --seeds 100 101 --out runs/select/a2`
3. Sealed read of 9007, gated on bar_pass; add `--force-sealed "<reason>"` if the bar fails:
   `PY scripts/select_lomo.py --grid grid2 --sealed --best-from runs/select/a2 --out runs/select/a2`
4. Candidate:
   `PY scripts/make_candidate.py --select runs/select/a2 --name a2`

## Fix report
1. Threads pinned (Important): removed `--pathcrf-threads` from `scripts/select_lomo.py` and `--threads` from `scripts/precompute_pathcrf.py` (`PathCRFParams()` default 4 is the only value; `registration.json` still records `pathcrf_threads`). `run_infer.py` honours the passed value (only `torch.set_num_threads(a.threads)`, line 186, no other override), so it is unchanged. `third_party/pathcrf/SETUP.md` gains a "Thread count is pinned" section. `--sealed` now reads `registration.json` (from `--best-from` or `--out`) and SystemExits if it is missing or `seeds` differ from `args.seeds` (`main`, select_lomo.py). Test: `test_sealed_refuses_seed_mismatch`.
2. Resume grid check: `check_resume_grid` + `GRID_PARAM_COLS` in select_lomo.py (called on resume in `run_grid`); SystemExit "fold CSVs in <out> were produced with a different grid; use a fresh --out". Test: `test_resume_rejects_fold_csv_from_other_grid` (changed tau_D, changed dead_policy, missing cell).
3. `test_run_selection_two_policies_pathcrf_and_real` now spies on `compute_evidence` (maps evidence objects to policy) and `score_cell`, asserting each cell is scored only with evidence of its own `dead_policy` and both policies are scored.
4. `--grid` help string and module docstring state the real-match plausibility step runs for every grid unless `--no-real`.
5. `--data` defaults to `$MTAH_DATA_DIR` with no hard-coded fallback in select_lomo.py (`data_dir_or_exit`, SystemExit when needed and unset), precompute_pathcrf.py (needed only with `--real`) and make_candidate.py. Test: `test_data_dir_unset_exits`.
6. `nan_to_none` in precompute_pathcrf.py; summary written with `allow_nan=False`. Test: `test_nan_to_none_gives_strict_json`.
Test command: `mtah-venv\Scripts\python.exe -m pytest tests -q -m "not dfl and not real"` -> 188 passed, 7 deselected (test_corrupt.py included, also passing).
