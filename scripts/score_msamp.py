"""Score the registered candidates (trained on the six DFL folds only) and the organisers' baselines on the corrupted
Metrica-sample halves (runs/sim_msamp). Resumable: one JSON line per (half, seed, system) in OUT/scores.jsonl.
Never touches the public window or the sealed sim match; aggregates only. PathCRF is read from the cache only.

  OMP_NUM_THREADS=2 python scripts/score_msamp.py --sim runs/sim_msamp --out runs/msamp [--systems a5c a13r ...]
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "2")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "scripts"))

import numpy as np  # noqa: E402

import mtah.evidence.pathcrf as pcrf  # noqa: E402
from mtah import baselines  # noqa: E402
from mtah.eval.drift import blackout_mask_from_match, drift_report  # noqa: E402
from mtah.eval.scorer import score  # noqa: E402
from mtah.sim.dataset import SEALED_MATCH, VARIANTS, list_sim_keys, load_half_sim, match_id  # noqa: E402

CANDIDATES = {"a5c": "models/a5c", "a13r": "runs/select/a13r", "a10": "models/a10", "a14": "runs/select/a14",
              "a16": "models/a16"}
BASELINES = {"b0": baselines.b0_nearest_hold, "b4": baselines.b4_blackout_dead,
             "smooth50": lambda m: baselines.smooth_vote(m, 50)}
DRIFT_KEYS = ("spells_per_graded_min", "non_blackout_D_share", "pred_D_share", "spell_len_median_s", "tag_agreement", "n_spells")


def _no_run(*a, **k):
    raise RuntimeError("PathCRF cache miss: refusing to run PathCRF in the scoring process")


pcrf._run_half = _no_run


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sim", default=str(ROOT / "runs" / "sim_msamp"))
    ap.add_argument("--out", default=str(ROOT / "runs" / "msamp"))
    ap.add_argument("--seeds", type=int, nargs="+", default=[100, 101])
    ap.add_argument("--systems", nargs="+", default=["truth"] + list(CANDIDATES) + list(BASELINES))
    args = ap.parse_args()
    from make_candidate import candidate_labels, load_selection
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    log = out / "scores.jsonl"
    done = {(r["key"], r["seed"], r["system"]) for r in map(json.loads, log.read_text().splitlines())} if log.exists() else set()
    keys = [k for k in list_sim_keys(args.sim) if match_id(k) != SEALED_MATCH]
    if not all(9211 <= match_id(k) <= 9213 for k in keys):
        raise SystemExit(f"{args.sim}: expected only the Metrica sample halves 9211..9213, got {keys}")
    loaded = {}
    for key in keys:
        for seed in args.seeds:
            todo = [s for s in args.systems if (key, seed, s) not in done]
            if not todo:
                continue
            sim = load_half_sim(args.sim, key, seed)
            m, ck = sim.match, f"{key}_s{seed}"
            mask = blackout_mask_from_match(m)
            rows = []
            if ("truth" in args.systems) and (key, seed, "truth") not in done:
                rows.append(("truth", sim.truth["raw_Dfull"], None))
            for name in todo:
                t0 = time.time()
                if name in BASELINES:
                    lab, table = BASELINES[name](m), None
                elif name == "truth":
                    continue
                else:
                    if name not in loaded:
                        sel = ROOT / CANDIDATES[name]
                        cfg, dead, lengths, team, merge, switch, seq = load_selection(sel, with_seq=True)
                        cfg.pathcrf.cache_dir = str(ROOT / "runs" / "cache" / "pathcrf")
                        loaded[name] = (sel, cfg, dead, team, merge, switch, seq)
                    sel, cfg, dead, team, merge, switch, seq = loaded[name]
                    lab = candidate_labels(m, sim.tags, cfg, dead, team, merge, switch, cache_key=ck, seq=seq, sel_dir=sel)
                    table = cfg.tag.code_table
                rows.append((name, lab, table))
                print(f"{ck} {name} {time.time() - t0:.0f}s", flush=True)
            for name, lab, table in rows:
                rec = {"key": key, "seed": seed, "system": name, "game": match_id(key), "half": int(key[-1])}
                for v in VARIANTS:
                    r = score(sim.truth[v], lab)
                    rec[v] = {"spell_f1": r["spell_f1"], "macro_f1": r["macro_f1"], "n_gt": r["n_gt_spells"], "n_pred": r["n_pred_spells"]}
                d = drift_report(lab, m, sim.tags, mask, code_table=table)
                rec["drift"] = {k: (None if not np.isfinite(d[k]) else float(d[k])) for k in DRIFT_KEYS}
                with log.open("a") as f:
                    f.write(json.dumps(rec) + "\n")
            del sim


if __name__ == "__main__":
    main()
