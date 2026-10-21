"""Label-free sensitivity of the dead model's `sp_type` feature for a selection directory.

Evidence/decoding is run normally and with the dead features' sp_type column forced to a constant (default 4) before
predict_proba. Reports sim spellF1 (ctrl_Dfull, raw_Dfull; non-sealed halves, given seeds) and real-match drift
(non_blackout_D_share, pred_D_share), the deltas, and PASS/FAIL against |dF1| < 0.005 and |dDshare| < 0.01.
Only aggregates are printed; writes <select dir>/sp_type_sensitivity.json."""
import argparse, json, os, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np

from mtah.eval.drift import blackout_mask_from_match, drift_report
from mtah.evidence.dead import DEAD_FEATURES
from mtah.io.metrica_loader import load_tactical
from mtah.io.tags import load_tags
from mtah.pipeline import compute_evidence, compute_strong, predict
from mtah.select import score_cell
from mtah.sim.dataset import SEALED_MATCH, list_sim_keys, match_id

import select_lomo as sl

SP_COL = DEAD_FEATURES.index("sp_type")
F1_TOL, SHARE_TOL = 0.005, 0.01
VARS = ("ctrl_Dfull", "raw_Dfull")


class ConstColumnDead:
    """Wraps a DeadModel: predict_proba sees column `col` of X overwritten with `value`."""
    def __init__(self, inner, col=SP_COL, value=4.0):
        self.inner, self.col, self.value = inner, int(col), float(value)
        self.policy = inner.policy

    def predict_proba(self, X):
        X = np.array(X, copy=True)
        X[:, self.col] = self.value
        return self.inner.predict_proba(X)


def sim_scores(sims, cfg, dead, strongs, cache=True):
    evs = [compute_evidence(s.match, s.tags, cfg, dead, cache_key=f"{s.key}_s{s.seed}") for s in sims]
    res, _ = score_cell(sims, evs, cfg, VARS, strongs=strongs)
    return {v: res[v]["spell_f1"] for v in VARS}


def real_drift(m, tags, mask, cfg, dead, strong):
    ev = compute_evidence(m, tags, cfg, dead, cache_key="real_tactical")
    lab = predict(m, tags, cfg, None, evidence=ev, strong=strong)
    d = drift_report(lab, m, tags, mask, code_table=cfg.tag.code_table)
    return {k: float(d[k]) for k in ("non_blackout_D_share", "pred_D_share")}


def verdict(sim_n, sim_c, real_n, real_c):
    dF = {v: sim_c[v] - sim_n[v] for v in sim_n}
    dS = {k: real_c[k] - real_n[k] for k in real_n}
    ok = all(abs(x) < F1_TOL for x in dF.values()) and all(abs(x) < SHARE_TOL for x in dS.values())
    return dF, dS, ("PASS" if ok else "FAIL")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--select", required=True)
    ap.add_argument("--sim", default=str(ROOT / "runs" / "sim"))
    ap.add_argument("--seeds", type=int, nargs="+", default=[100, 101])
    ap.add_argument("--data", default=os.environ.get("MTAH_DATA_DIR"))
    ap.add_argument("--const", type=float, default=4.0)
    a = ap.parse_args()
    sel_dir = Path(a.select)
    cfg, dead, _ = sl.load_candidate_artifacts(sel_dir)
    best = json.loads((sel_dir / "best.json").read_text()) if (sel_dir / "best.json").exists() else {}
    const = ConstColumnDead(dead, value=a.const)
    keys = [k for k in list_sim_keys(a.sim) if match_id(k) != SEALED_MATCH]
    sims = sl.load_sims(a.sim, a.seeds, keys)
    strongs = [compute_strong(s.match, cfg) for s in sims]
    sim_n = sim_scores(sims, cfg, dead, strongs); sim_c = sim_scores(sims, cfg, const, strongs)
    del sims
    m = load_tactical(sl.data_dir_or_exit(a.data)); tags = load_tags(a.data); mask = blackout_mask_from_match(m)
    strong = compute_strong(m, cfg)
    real_n = real_drift(m, tags, mask, cfg, dead, strong); real_c = real_drift(m, tags, mask, cfg, const, strong)
    dF, dS, v = verdict(sim_n, sim_c, real_n, real_c)
    out = {"select": str(sel_dir), "best_cell": best.get("cell"), "const": a.const, "n_halves": len(keys) * len(a.seeds),
           "sim_normal": sim_n, "sim_const": sim_c, "delta_f1": dF, "real_normal": real_n, "real_const": real_c,
           "delta_share": dS, "thresholds": {"abs_dF1": F1_TOL, "abs_dDshare": SHARE_TOL}, "verdict": v}
    (sel_dir / "sp_type_sensitivity.json").write_text(json.dumps(out, indent=1))
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
