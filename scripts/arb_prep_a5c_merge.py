"""Retrain a5c's per-fold merge model (the saved ones, runs/select/exp_seq/lomo/merge/f*_a5c.joblib, were pickled by an
older scikit-learn and cannot be loaded). Same procedure as select_lomo's fold merge model: select.train_merge_model on
the fold's five train matches of the OLD sim build (inner split: stream models on four matches, reference cells =
GRID8's merge cells decoded on the fifth), abs touches as in a5c. Output runs/arb/models/a5c_merge_f{F}.joblib.
  python scripts/arb_prep_a5c_merge.py FOLD
"""
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "scripts"))
import json
import mtah.select as sel
from mtah.pipeline import PipelineConfig
from mtah.sim.dataset import SEALED_MATCH, list_sim_keys, load_half_sim, match_id
import arb_models as am
from make_candidate import load_selection


def main():
    F = int(sys.argv[1]); assert F != SEALED_MATCH
    cfg0 = load_selection(ROOT / "runs" / "select" / "a5c")[0]
    base = PipelineConfig(); base.touch = cfg0.touch
    base.use_pathcrf = True; base.use_teamctrl = True; base.pathcrf.cache_dir = str(ROOT / "runs" / "cache" / "pathcrf")
    sim = ROOT / "runs" / "sim_prev_nosmooth"
    keys = [k for k in list_sim_keys(sim) if match_id(k) not in (F, SEALED_MATCH) and match_id(k) <= 9006]
    train = [load_half_sim(sim, k, s) for k in keys for s in (100, 101)]
    lengths = sel.estimate_lengths(train, base, sel.train_dead)
    mm = sel.train_merge_model(train, base, sel.GRID8, lengths)
    out = ROOT / "runs" / "arb" / "models" / f"a5c_merge_f{F}.joblib"
    out.parent.mkdir(parents=True, exist_ok=True); mm.save(out); print(F, mm.info)


if __name__ == "__main__":
    main()
