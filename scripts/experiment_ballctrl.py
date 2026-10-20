"""Ball-control parameter sweep on LOMO fold 9001 (ctrl_Dfull truth)."""
import copy, json, sys, time
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
from mtah.eval.scorer import score
from mtah.evidence.ball_ctrl import ball_ctrl_evidence
from mtah.pipeline import PipelineConfig, compute_evidence, compute_strong, predict
from mtah.select import apply_cell, estimate_lengths, train_dead
from mtah.sim.dataset import SEALED_MATCH, list_sim_keys, load_half_sim, lomo_folds, match_id

SIM = str(ROOT / "runs" / "sim"); SEEDS = [100, 101]; HELD = 9001
cell = json.loads((ROOT / "runs/select/a1/best.json").read_text())["cell"]
base = PipelineConfig.from_dict(json.loads((ROOT / "runs/select/a1/config.json").read_text()))
keys = [k for k in list_sim_keys(SIM) if match_id(k) != SEALED_MATCH]
for held, tr, te in lomo_folds(keys):
    if int(held) == HELD: break
train = [load_half_sim(SIM, k, s) for k in tr for s in SEEDS]
test = [load_half_sim(SIM, k, s) for k in te for s in SEEDS]
print("test halves", len(test), flush=True)
lengths = estimate_lengths(train, base, train_dead); dead = train_dead(train)
cfg0 = apply_cell(base, cell); cfg0.decoder.lengths = {k: 1.0 for k in cfg0.decoder.lengths}
ev0 = [compute_evidence(s.match, s.tags, cfg0, dead) for s in test]
strong = [compute_strong(s.match, cfg0) for s in test]
ball_cache = {}

def run(ballp=None, w_ball=None, tau_HA=None):
    ballp = ballp or {}
    cfg = copy.deepcopy(cfg0)
    for k, v in ballp.items(): setattr(cfg.ball, k, v)
    if w_ball is not None: cfg.decoder.weights["ball"] = w_ball
    if tau_HA is not None: cfg.decoder.tau_HA = tau_HA
    bk = json.dumps(ballp, sort_keys=True)
    if bk not in ball_cache:
        ball_cache[bk] = [ball_ctrl_evidence(s.match, cfg.ball) for s in test]
    f1 = []; rat = []; dec = {"false_splits": 0, "false_merges": 0, "boundary_only": 0}
    for i, s in enumerate(test):
        ev = dict(ev0[i]); ev["ball"] = ball_cache[bk][i]
        lab = predict(s.match, s.tags, cfg, dead, evidence=ev, strong=strong[i])
        r = score(s.truth["ctrl_Dfull"], lab)
        f1.append(r["spell_f1"]); rat.append(r["n_pred_spells"] / max(r["n_gt_spells"], 1))
        for k in dec: dec[k] += int(r["decomposition"].get(k, 0))
    return {"spell_f1": float(np.mean(f1)), "ratio": float(np.mean(rat)), **dec}

results = {}
def go(name, **kw):
    t = time.time(); r = run(**kw); r["setting"] = kw; results[name] = r
    print(f"{name:28s} {r['spell_f1']:.4f} {r['ratio']:.3f} {r['false_splits']:5d} {r['false_merges']:5d} {r['boundary_only']:5d}  ({time.time()-t:.0f}s)", flush=True)
    (ROOT / "runs/select/exp_ballctrl.json").write_text(json.dumps({"fold": HELD, "seeds": SEEDS, "results": results}, indent=1))
    return r

go("baseline")
sweeps = {"sigma_m": [0.5, 0.7, 1.0, 1.5, 2.0], "c_flight": [1.5, 2, 3, 5], "tau_switch": [2, 4, 6, 12],
          "alpha_loose": [0.1, 0.25, 0.5, 1.0], "w_vel": [0, 0.15, 0.3], "cand_radius_m": [3, 5, 8]}
best = {}
for p, vals in sweeps.items():
    sc = {}
    for v in vals:
        sc[v] = go(f"{p}={v}", ballp={p: v})["spell_f1"]
    bv = max(sc, key=sc.get)
    if sc[bv] > results["baseline"]["spell_f1"]: best[p] = bv
sc = {}
for w in [0.5, 1, 2]:
    sc[w] = go(f"w_ball={w}", w_ball=w)["spell_f1"]
bw = max(sc, key=sc.get); wb = bw if sc[bw] > results["baseline"]["spell_f1"] else None
print("best", best, wb, flush=True)
for t in [None, 3, 10, 30]:
    go(f"combo tau_HA={t or 10}", ballp=best, w_ball=wb, tau_HA=t)
