"""Retrain a5c's per-fold dead and team-control models on the OLD sim build (the saved exp_seq/lomo ones for folds
9002-9006 were pickled by an older scikit-learn and cannot be loaded): select.train_dead (policy long_rel) and
select.train_team_ctrl (policy full, a5c's abs touch parameters) on the fold's five train matches, seeds 100/101.
Fold 9001's saved models still load: the retrained ones are checked against them (max |dp|).
  python scripts/arb_prep_a5c_models.py FOLD
"""
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "scripts"))
import numpy as np
import mtah.select as sel
from mtah.evidence.dead import DeadModel, dead_table
from mtah.evidence.team_ctrl import TeamCtrlModel, team_features
from mtah.sim.dataset import SEALED_MATCH, list_sim_keys, load_half_sim, match_id
from make_candidate import load_selection


def main():
    F = int(sys.argv[1]); assert F != SEALED_MATCH
    cfg0 = load_selection(ROOT / "runs" / "select" / "a5c")[0]
    sim = ROOT / "runs" / "sim_prev_nosmooth"
    keys = [k for k in list_sim_keys(sim) if match_id(k) not in (F, SEALED_MATCH) and match_id(k) <= 9006]
    train = [load_half_sim(sim, k, s) for k in keys for s in (100, 101)]
    dead = sel.train_dead(train, policy=cfg0.dead_policy)
    team = sel.train_team_ctrl(train, policy=cfg0.team_policy, touch_params=cfg0.touch)
    out = ROOT / "runs" / "arb" / "models"; out.mkdir(parents=True, exist_ok=True)
    dead.save(out / f"a5c_dead_f{F}.joblib"); team.save(out / f"a5c_team_f{F}.joblib")
    if F == 9001:
        h = load_half_sim(sim, "game_9001_H1", 100)
        od = DeadModel.load(ROOT / "runs/select/exp_seq/lomo/models/dead_fold9001.joblib")
        ot = TeamCtrlModel.load(ROOT / "runs/select/exp_seq/lomo/models/team_fold9001.joblib")
        print("check dead max|dp|", float(np.abs(od.predict_proba(dead_table(h.match, h.tags, od.policy)) - dead.predict_proba(dead_table(h.match, h.tags, dead.policy))).max()))
        X = team_features(h.match, cfg0.touch)
        print("check team max|dp|", float(np.abs(ot.predict_proba(X) - team.predict_proba(X)).max()))
    print("done", F)


if __name__ == "__main__":
    main()
