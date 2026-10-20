"""Dead-model feature ablation: LOMO (2 folds) on sim + label-free aggregates on the real match."""
import json, os, sys, time
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
from mtah.eval.drift import blackout_mask_from_match, drift_report
from mtah.eval.official import possession_metrics as pm
from mtah.eval.scorer import score, codes
from mtah.evidence.dead import DEAD_FEATURES, DeadModel, dead_features
from mtah.io.metrica_loader import load_tactical
from mtah.io.tags import load_tags
from mtah.pipeline import PipelineConfig, compute_evidence, compute_strong, predict
from mtah.select import apply_cell, estimate_lengths
from mtah.sim.blackout_sim import runs_of
from mtah.sim.dataset import SEALED_MATCH, list_sim_keys, load_half_sim, lomo_folds, match_id

SEEDS = [100, 101]
SIM = ROOT / "runs" / "sim"
SEL = ROOT / "runs" / "select" / "a1"
CUES = ["ball_speed", "ball_line_dist", "ball_out", "ball_static_1s"]
SETS = {"FULL": [], "NO_BALL_CUES": CUES, "NO_BALL_AT_ALL": CUES + ["ball_present", "entry_line_dist", "entry_outward"]}


class MaskedDead:
    def __init__(self, drop):
        self.cols = [DEAD_FEATURES.index(n) for n in drop]; self.m = DeadModel()
    def _mask(self, X):
        X = np.array(X, np.float32, copy=True); X[:, self.cols] = 0.0; return X  # constant (all-NaN columns crash HGB binning)
    def fit(self, X, y):
        self.m.fit(self._mask(X), y); return self
    def predict_proba(self, X):
        return self.m.predict_proba(self._mask(X))


def trainer(drop):
    def fn(sims, variant="ctrl_Dfull"):
        X = np.concatenate([dead_features(s.match, s.tags) for s in sims])
        y = np.concatenate([(s.truth[variant] == "D").astype(int) for s in sims])
        return MaskedDead(drop).fit(X, y)
    return fn


def load(keys):
    return [load_half_sim(str(SIM), k, s) for k in keys for s in SEEDS]


def main():
    DATA = os.environ.get("MTAH_DATA_DIR")
    if not DATA:
        raise SystemExit("set MTAH_DATA_DIR to the real-match data dir")
    best = json.loads((SEL / "best.json").read_text())["cell"]
    keys = [k for k in list_sim_keys(str(SIM)) if match_id(k) != SEALED_MATCH]
    base = PipelineConfig()
    out = {"cell": best, "seeds": SEEDS, "sim": {}, "real": {}}
    t0 = time.time()

    # 1. sim LOMO
    folds = [f for f in lomo_folds(keys) if str(f[0]) in ("9001", "9004")]
    cache = {}
    for held, tr_keys, te_keys in folds:
        train, test = load(tr_keys), load(te_keys)
        strongs = None
        for name, drop in SETS.items():
            fn = trainer(drop)
            lengths = {"ball": 1.0, "tag": 1.0, "dead": 1.0}  # a1 selection used unit lengths
            dead = fn(train)
            cfg = apply_cell(base, best); cfg.decoder.lengths = lengths
            cm = {v: np.zeros((4, 4), np.int64) for v in ("ctrl_Dfull", "raw_Dfull")}
            f1 = {v: [] for v in cm}; npred = {v: [] for v in cm}; ngt = {v: [] for v in cm}
            for s in test:
                ev = compute_evidence(s.match, s.tags, cfg, dead)
                lab = predict(s.match, s.tags, cfg, None, evidence=ev)
                for v in cm:
                    r = score(s.truth[v], lab)
                    f1[v].append(r["spell_f1"]); npred[v].append(r["n_pred_spells"]); ngt[v].append(max(1, r["n_gt_spells"]))
                    cm[v] += pm.confusion(codes(s.truth[v]), codes(lab))
            cache.setdefault(name, []).append((f1, npred, ngt, cm))
            print(f"fold {held} {name} ctrl={np.mean(f1['ctrl_Dfull']):.3f} raw={np.mean(f1['raw_Dfull']):.3f} t={time.time()-t0:.0f}s", flush=True)
            out["sim"].setdefault(name, {})[str(held)] = {v: {"spell_f1": float(np.mean(f1[v])), "ratio": float(np.mean(np.array(npred[v]) / np.array(ngt[v])))} for v in cm}
    for name in SETS:
        agg = {}
        for v in ("ctrl_Dfull", "raw_Dfull"):
            f1 = np.concatenate([c[0][v] for c in cache[name]]); ratio = np.concatenate([np.array(c[1][v]) / np.array(c[2][v]) for c in cache[name]])
            cm = sum(c[3][v] for c in cache[name]); cm3 = pm.scored_confusion(cm)
            p, r, f = pm.per_class_prf(cm3, 2)
            agg[v] = {"spell_f1": float(f1.mean()), "ratio": float(ratio.mean()), "D_precision": p, "D_recall": r, "D_f1": f}
        out["sim"][name]["pooled"] = agg

    # 2. real match
    raw = json.loads((SEL / "config.json").read_text()); raw.pop("lengths")
    rbase = PipelineConfig.from_dict(raw)
    m = load_tactical(DATA); tags = load_tags(DATA)
    bm = blackout_mask_from_match(m); g = m.graded; fps = m.fps
    sp = np.array([int(r.start) for r in tags.itertuples(index=False) if r.code == "SET PIECES"])
    train = load(keys)
    for name, drop in SETS.items():
        fn = trainer(drop)
        lengths = {"ball": 1.0, "tag": 1.0, "dead": 1.0}
        dead = fn(train)
        cfg = apply_cell(rbase, best); cfg.decoder.lengths = lengths
        lab = predict(m, tags, cfg, dead)
        isD = lab == "D"
        runs = list(runs_of(isD))
        starts = np.array([s for s, e in runs])
        near = float(np.mean([np.min(np.abs(sp - s)) <= 2 * fps for s in starts])) if len(starts) and len(sp) else None
        rec = float(np.mean([any((s <= t + 5 * fps) and (e >= t - 5 * fps) for s, e in runs) for t in sp])) if len(sp) else None
        dr = drift_report(lab, m, tags, bm, code_table=cfg.tag.code_table)
        out["real"][name] = {
            "D_share_graded": float(isD[g].mean()), "D_share_nonblackout": float(isD[g & ~bm].mean()),
            "D_share_blackout": float(isD[g & bm].mean()) if (g & bm).any() else None,
            "n_D_runs": len(runs), "D_starts_within_2s_of_SP": near, "SP_recall_proxy_pm5s": rec,
            "n_SP_tags": int(len(sp)), "spells_per_HA_min": dr["spells_per_HA_min"], "lengths": lengths}
        print(name, json.dumps(out["real"][name]), f"t={time.time()-t0:.0f}s", flush=True)

    (ROOT / "runs" / "select" / "exp_dead_ablation.json").write_text(json.dumps(out, indent=1))
    print(f"{'set':15}{'sF1c':>6}{'sF1r':>6}{'Dprec':>6}{'Drec':>6}{'Rtot':>6}{'Rnb':>6}{'Rbo':>6}{'SPrec':>6}{'sp/min':>7}")
    for n in SETS:
        a = out["sim"][n]["pooled"]; r = out["real"][n]
        print(f"{n:15}{a['ctrl_Dfull']['spell_f1']:6.3f}{a['raw_Dfull']['spell_f1']:6.3f}{a['ctrl_Dfull']['D_precision']:6.3f}{a['ctrl_Dfull']['D_recall']:6.3f}"
              f"{r['D_share_graded']:6.3f}{r['D_share_nonblackout']:6.3f}{(r['D_share_blackout'] or 0):6.3f}{(r['SP_recall_proxy_pm5s'] or 0):6.3f}{r['spells_per_HA_min']:7.2f}")


if __name__ == "__main__":
    main()
