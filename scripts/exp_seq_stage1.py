"""exp-seq stage 1 (fold 9001 only): stream models, per-half evidence caches, a5c baseline on the test halves.

Test 9001 halves: dead/team models trained on all fold-train matches 9002-9006 (as scripts/select_lomo.py) + the fold
merge model (select.train_merge_model over GRID8, inner split held = 9006). Train halves 9002-9006: CROSS-FITTED
evidence (dead/team models trained on the other four train matches) so the learned sequence model sees out-of-sample
streams as it will on the test match. Reads the main tree's runs/sim and runs/cache/pathcrf (read-only; a PathCRF
cache miss raises instead of running PathCRF). Writes only under <main>/runs/select/exp_seq/. Never touches 9007.
"""
import json, os, sys, time
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "4")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import joblib
import numpy as np
import pandas as pd

import mtah.evidence.pathcrf as pcrf
import mtah.select as sel
from mtah.eval.drift import blackout_mask_from_match
from mtah.eval.scorer import score
from mtah.evidence.dead import DeadModel
from mtah.evidence.team_ctrl import TeamCtrlModel
from mtah.pipeline import PipelineConfig, compute_evidence, compute_strong, predict
from mtah.sim.dataset import SEALED_MATCH, load_half_sim, list_sim_keys, match_id
from mtah.spells.merge import MergeModel

MAIN = Path(r"WORKDIR/possession")
SIM = MAIN / "runs" / "sim"
OUT = MAIN / "runs" / "select" / "exp_seq"
SEEDS = [100, 101]
HELD = 9001
TRAIN_M = [9002, 9003, 9004, 9005, 9006]
STREAMS = ("ball", "tag", "dead", "pathcrf", "teamctrl")
LAB = {"H": 0, "A": 1, "D": 2}


def _no_run(*a, **k):
    raise RuntimeError("PathCRF cache miss: refusing to run PathCRF (read-only cache)")


pcrf._run_half = _no_run


def base_cfg():
    base = PipelineConfig()
    base.use_pathcrf = True; base.use_teamctrl = True; base.use_switch = False
    base.pathcrf.cache_dir = str(MAIN / "runs" / "cache" / "pathcrf")
    return base


def a5c_cfg(base):
    cfg = sel.apply_cell(base, sel.GRID8[0])
    cfg.decoder.lengths = sel.estimate_lengths([], cfg, None)   # unit lengths (LENGTHS_MODE "unit")
    return cfg


def load(keys):
    out = []
    for k in keys:
        assert match_id(k) != SEALED_MATCH
        for s in SEEDS:
            out.append(load_half_sim(SIM, k, s))
    return out


def keys_of(ms):
    return [k for k in list_sim_keys(SIM) if match_id(k) in ms and match_id(k) != SEALED_MATCH]


def get_models(tag, sims, base, log):
    md = OUT / "models"; md.mkdir(parents=True, exist_ok=True)
    dp, tp = md / f"dead_{tag}.joblib", md / f"team_{tag}.joblib"
    t0 = time.time()
    if dp.exists():
        dead = DeadModel.load(dp)
    else:
        dead = sel.train_dead(sims, policy="long_rel"); dead.save(dp)
    if tp.exists():
        team = TeamCtrlModel.load(tp)
    else:
        team = sel.train_team_ctrl(sims, policy="full", touch_params=base.touch); team.save(tp)
    log(f"models {tag}: {time.time() - t0:.0f}s")
    return dead, team


def save_half(s, cfg, dead, team, path):
    ev = compute_evidence(s.match, s.tags, cfg, dead, cache_key=f"{s.key}_s{s.seed}", team_model=team)
    strong = compute_strong(s.match, cfg)
    r = pcrf.pathcrf_posterior(s.match, f"{s.key}_s{s.seed}", cfg.pathcrf)
    m = s.match
    tr = {v: np.array([LAB.get(c, 3) for c in s.truth[v]], np.int8) for v in ("ctrl_Dfull", "raw_Dfull")}
    np.savez_compressed(path, **{f"ev_{k}": ev[k].astype(np.float32) for k in STREAMS},
                        touches=strong[2], sH=strong[0], sA=strong[1], half=m.half, ball_present=m.ball_present,
                        graded=m.graded, blackout_inf=blackout_mask_from_match(m), ball=m.ball.astype(np.float32),
                        ball_speed=np.asarray(m.ball_speed, np.float32), pitch=np.asarray(m.pitch, np.float32),
                        fps=np.float32(m.fps), p_home=r["p_home"], imputed_frac=r["imputed_frac"],
                        truth_ctrl=tr["ctrl_Dfull"], truth_raw=tr["raw_Dfull"])
    return ev, strong


def main():
    OUT.mkdir(parents=True, exist_ok=True); (OUT / "halves").mkdir(exist_ok=True)
    logf = open(OUT / "stage1.log", "a")

    def log(msg):
        line = f"[{time.strftime('%H:%M:%S')}] {msg}"; print(line, flush=True); logf.write(line + "\n"); logf.flush()

    base = base_cfg(); cfg = a5c_cfg(base)
    T0 = time.time()
    # ---- fold models (all five train matches) + merge model ------------------------------------------------------
    train = load(keys_of(TRAIN_M))
    dead, team = get_models("fold9001", train, base, log)
    mp = OUT / "models" / "merge_fold9001.joblib"
    if mp.exists():
        merge = MergeModel.load(mp)
    else:
        t0 = time.time()
        merge = sel.train_merge_model(train, base, sel.GRID8, cfg.decoder.lengths)
        merge.save(mp); (OUT / "models" / "merge_fold9001.json").write_text(json.dumps(merge.info, indent=1))
        log(f"merge model {time.time() - t0:.0f}s {merge.info}")
    del train
    # ---- test halves: evidence + a5c baseline --------------------------------------------------------------------
    rows = []
    for s in load(keys_of([HELD])):
        p = OUT / "halves" / f"{s.key}_s{s.seed}.npz"
        t0 = time.time(); ev, strong = save_half(s, cfg, dead, team, p)
        lab = predict(s.match, s.tags, cfg, None, evidence=ev, strong=strong, merge_model=merge)
        for v in ("ctrl_Dfull", "raw_Dfull"):
            r = score(s.truth[v], lab)
            rows.append({"key": s.key, "seed": s.seed, "variant": v, "spell_f1": r["spell_f1"], "macro_f1": r["macro_f1"],
                         "n_gt": r["n_gt_spells"], "n_pred": r["n_pred_spells"]})
        log(f"test half {s.key}_s{s.seed} {time.time() - t0:.0f}s  " + json.dumps(rows[-2:]))
    df = pd.DataFrame(rows); df.to_csv(OUT / "baseline_a5c_halves.csv", index=False)
    log("BASELINE a5c fold 9001: " + df.groupby("variant")[["spell_f1", "macro_f1", "n_pred"]].mean().to_string())
    # ---- cross-fitted evidence for the train halves -------------------------------------------------------------
    for m in TRAIN_M:
        todo = [k for k in keys_of([m]) for s in SEEDS if not (OUT / "halves" / f"{k}_s{s}.npz").exists()]
        if not todo:
            continue
        others = [x for x in TRAIN_M if x != m]
        sims = load(keys_of(others))
        dead, team = get_models(f"cf{m}", sims, base, log)
        del sims
        for s in load(keys_of([m])):
            p = OUT / "halves" / f"{s.key}_s{s.seed}.npz"
            if p.exists():
                continue
            t0 = time.time(); save_half(s, cfg, dead, team, p); log(f"half {s.key}_s{s.seed} (cf{m}) {time.time() - t0:.0f}s")
    log(f"stage1 total {time.time() - T0:.0f}s")


if __name__ == "__main__":
    main()
