"""Step 3 (after the ball-smoothing rebuild): the sim-vs-real adversarial check of scripts/touch_adversarial.py, but with the
SIM step features recomputed from the rebuilt sims (the old caches hold the unsmoothed ball). Fold F (9001-9006) test halves
(seed 100) get a5c-cell evidence from dead (long_rel) / team-control (full) models trained on the other five matches
(seeds 100 101, the exp_seq `fold{F}` recipe), touches = the TouchParams default (rel), PathCRF from the unchanged cache.
Real side = runs/select/exp_robust/real_cache.npz (registered a5c evidence) with the `rel` touch columns swapped in, as in
touch_adversarial.py. Aggregates only; 9007 and the ground truth are never loaded.   -> runs/touch/adversarial_rebuild.json"""
import json, os, sys, time
os.environ["OMP_NUM_THREADS"] = "4"
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np

import touch_adversarial as ta
from touch_common import MATCHES, OUT, SEED, SIM, load_real
import mtah.select as sel
import mtah.evidence.seqstream as sq
from mtah.evidence.dead import dead_evidence, dead_table
from mtah.evidence.touches import TouchParams
from mtah.evidence import seq_feats as fx
from mtah.pipeline import PipelineConfig, compute_evidence, compute_strong
from mtah.sim.dataset import SEALED_MATCH, list_sim_keys, load_half_sim, match_id


def sims_of(ms, seeds):
    return [load_half_sim(SIM, k, s) for k in list_sim_keys(SIM) if match_id(k) in ms and match_id(k) != SEALED_MATCH
            for s in seeds]


def main():
    t0 = time.time()
    base = PipelineConfig(); base.use_pathcrf = True; base.use_teamctrl = True
    cfg = sel.apply_cell(base, sel.GRID8[0])
    assert cfg.touch.mode == "rel"
    sim = []
    for F in MATCHES:
        train = sims_of([m for m in MATCHES if m != F], (100, 101))
        dead = sel.train_dead(train, policy="long_rel"); team = sel.train_team_ctrl(train, policy="full", touch_params=cfg.touch)
        del train
        for s in sims_of([F], (SEED,)):
            ev = compute_evidence(s.match, s.tags, cfg, None, cache_key=f"{s.key}_s{s.seed}", team_model=team)
            ev["dead"] = dead_evidence(dead.predict_proba(dead_table(s.match, s.tags, dead.policy)), logit_offset=cfg.dead_offset)
            z = sq.feature_inputs(s.match, ev, compute_strong(s.match, cfg), cfg.pathcrf, f"{s.key}_s{s.seed}")
            X, names = fx.frame_features(z); Xs, _, seg = fx.to_steps(X, names, s.match.half)
            sim.append((f"{s.key}_s{s.seed}", Xs, seg))
        print(f"fold {F} features done ({time.time() - t0:.0f}s)", flush=True)
    real, _, rr = ta.load_real_steps(TouchParams())            # rel default
    X, y, blocks = ta.assemble(sim, real, names); folds = ta.fold_of(blocks, y)
    auc, fa = ta.cv_auc(X, y, folds); sa = ta.single_auc(X, y, names)
    ti = [names.index(n) for n in ta.TOUCH_NAMES]; oi = [i for i in range(len(names)) if i not in ti]
    res = {"auc_base": auc, "auc_base_folds": fa, "auc_touch_group_only": ta.cv_auc(X[:, ti], y, folds)[0],
           "auc_without_touch_group": ta.cv_auc(X[:, oi], y, folds)[0],
           "single_auc_touch_cols": {n: sa[n] for n in ta.TOUCH_NAMES},
           "top_single_auc_other": dict(sorted(((n, v) for n, v in sa.items() if n not in ta.TOUCH_NAMES), key=lambda kv: -kv[1])[:6]),
           "real_touch_rate_per_vis_min": rr}
    print(json.dumps(res, indent=1)); (OUT / "adversarial_rebuild.json").write_text(json.dumps(res, indent=1))
    Xw, yw, bw = ta.assemble(sim, real, names, windowed=True)
    res["auc_windowed"] = ta.cv_auc(Xw, yw, ta.fold_of(bw, yw), sub=6)[0]
    print("windowed", res["auc_windowed"], f"({time.time() - t0:.0f}s)")
    (OUT / "adversarial_rebuild.json").write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
