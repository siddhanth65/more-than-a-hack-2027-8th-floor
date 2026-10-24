"""One-off held-out confirmation of the registered a14 pipeline on the sealed sim match 9007 (after registration;
not used to choose). Scores a14 and, for comparison on the same build, a13r; appends ONE record to
docs/reads/sealed.jsonl. Refuses a second a14 read. Aggregates only.
"""
import json, os, subprocess, sys, time
from pathlib import Path

os.environ["OMP_NUM_THREADS"] = "4"
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "scripts"))

import numpy as np

import mtah.evidence.pathcrf as pcrf
from make_candidate import candidate_labels, load_selection
from mtah.eval.scorer import score
from mtah.io import readlog
from mtah.sim.dataset import SEALED_MATCH, VARIANTS, list_sim_keys, load_half_sim, match_id

SIM = ROOT / "runs" / "sim"
SEEDS = [100, 101]


def _no_run(*a, **k):
    raise RuntimeError("PathCRF cache miss: refusing to run PathCRF")


pcrf._run_half = _no_run


def run(sel, sims):
    cfg, dead, lengths, team, merge, switch, seq = load_selection(sel, with_seq=True)
    cfg.pathcrf.cache_dir = str(ROOT / "runs" / "cache" / "pathcrf")
    acc = {v: [] for v in VARIANTS}
    for s in sims:
        lab = candidate_labels(s.match, s.tags, cfg, dead, team, merge, switch, cache_key=f"{s.key}_s{s.seed}",
                               seq=seq, sel_dir=sel)
        for v in VARIANTS:
            acc[v].append(score(s.truth[v], lab)["spell_f1"])
    return {v: float(np.mean(x)) for v, x in acc.items()}


def main():
    if any(e.get("candidate") == "a14" and e.get("match") == SEALED_MATCH for e in readlog.read_log("sealed")):
        raise SystemExit("a14 sealed read already logged; refusing a second read")
    keys = [k for k in list_sim_keys(SIM) if match_id(k) == SEALED_MATCH]
    sims = [load_half_sim(SIM, k, s) for k in keys for s in SEEDS]
    res = {"a14": run(ROOT / "runs" / "select" / "a14", sims), "a13r": run(ROOT / "runs" / "select" / "a13r", sims)}
    sha = subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT).decode().strip()
    rec = {"time": time.strftime("%Y-%m-%d %H:%M"), "git": sha, "match": SEALED_MATCH, "candidate": "a14",
           "select_dir": "runs/select/a14", "seeds": SEEDS, "halves": keys, "sealed": res["a14"],
           "comparison_a13r_same_build": res["a13r"],
           "note": "held-out confirmation after registration (not used to choose)"}
    readlog.append_read("sealed", rec)
    print("SEALED", json.dumps(res))


if __name__ == "__main__":
    main()
