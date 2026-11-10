"""Task K: run registered candidates' full pipelines on the SkillCorner natural-visibility hold-out halves and score them.

  python scripts/score_kh.py --sim runs/kh/sim --cache <pathcrf cache dir> --sel NAME=DIR [NAME=DIR ...] --out runs/kh/score
        [--part I --of N] [--keys ...] [--baseline]
Per (half, candidate): labels saved (uint8 0/1/2 = H/A/D) and spellF1 / macroF1 / spell counts on ctrl_Dfull and raw_Dfull truth
(X frames ungraded by the official metric). PathCRF is read from the cache (computed by pathcrf_keys.py); a miss is an error.
Learned-stream outputs are memoised per (model files, dead-evidence bytes) so a14f / H share one BiGRU pass.
SkillCorner data only; no contest data is read.
"""
import argparse, hashlib, json, os, sys, time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "scripts"))

import numpy as np
import pandas as pd

import mtah.evidence.pathcrf as pcrf
import mtah.evidence.seqstream as sq
from mtah import baselines
from mtah.eval.scorer import score
from mtah.sim.dataset import SEALED_MATCH, list_sim_keys, load_half_sim, match_id
from make_candidate import candidate_labels, load_selection

CODE = {"H": 0, "A": 1, "D": 2}
_orig_cs = sq.compute_streams
_memo = {}


def _cs(z, spec, model_dir):
    files = [Path(model_dir) / spec[k] for k in ("modelA", "modelB") if k in spec]
    key = (tuple(hashlib.md5(f.read_bytes()).hexdigest() for f in files), tuple(sorted(spec["weights"])),
           hashlib.md5(np.ascontiguousarray(z["ev_dead"]).tobytes()).hexdigest(), int(z["half"].sum()))
    if key not in _memo:
        _memo.clear() if len(_memo) > 6 else None
        _memo[key] = _orig_cs(z, spec, model_dir)
    return {k: v.copy() for k, v in _memo[key].items()}


sq.compute_streams = _cs


def _no_run(*a, **k):
    raise RuntimeError("PathCRF cache miss")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sim", required=True); ap.add_argument("--cache", required=True); ap.add_argument("--out", required=True)
    ap.add_argument("--sel", nargs="*", default=[]); ap.add_argument("--seed", type=int, default=100)
    ap.add_argument("--part", type=int, default=0); ap.add_argument("--of", type=int, default=1)
    ap.add_argument("--keys", nargs="*", default=None); ap.add_argument("--baseline", action="store_true")
    a = ap.parse_args()
    pcrf._run_half = _no_run
    out = Path(a.out); (out / "labels").mkdir(parents=True, exist_ok=True)
    cands = {}
    for s in a.sel:
        name, d = s.split("=", 1)
        cands[name] = (load_selection(d, with_seq=True), Path(d))
    keys = a.keys or list_sim_keys(a.sim)
    keys = [k for k in keys if match_id(k) != SEALED_MATCH][a.part::a.of]
    rows = []
    for k in keys:
        t0 = time.time()
        h = load_half_sim(a.sim, k, a.seed); m, tags = h.match, h.tags
        ck = f"{k}_s{a.seed}"
        todo = dict(cands)
        if a.baseline:
            todo["smooth50"] = None
        for name, v in todo.items():
            lp = out / "labels" / f"{name}__{k}.npy"
            if lp.exists():
                lab = np.array(["H", "A", "D"])[np.load(lp)]
            elif name == "smooth50":
                lab = baselines.smooth_vote(m, 50)
            else:
                (cfg, dead, lengths, team, merge, switch, seq), sd = v
                cfg.pathcrf.cache_dir = a.cache
                lab = candidate_labels(m, tags, cfg, dead, team, merge, switch, cache_key=ck, seq=seq, sel_dir=sd)
            np.save(lp, np.vectorize(CODE.get)(lab).astype(np.uint8))
            for var in ("ctrl_Dfull", "raw_Dfull"):
                r = score(h.truth[var], lab)
                g = h.truth[var] != "X"
                rows.append({"cand": name, "key": k, "variant": var, "spell_f1": r["spell_f1"], "macro_f1": r["macro_f1"],
                             "n_gt": r["n_gt_spells"], "n_pred": r["n_pred_spells"],
                             "pred_D": float((lab[g] == "D").mean()), "graded_min": float(g.sum() / 25 / 60)})
        print(k, f"{time.time() - t0:.0f}s", flush=True)
        pd.DataFrame(rows).to_csv(out / f"scores_part{a.part}.csv", index=False)
    print("DONE", a.part)


if __name__ == "__main__":
    main()
