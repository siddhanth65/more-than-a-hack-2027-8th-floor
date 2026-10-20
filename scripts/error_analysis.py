"""Out-of-sample error decomposition of the selected cell on DFL-sim (LOMO test halves, ctrl_Dfull truth)."""
import argparse, json, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import numpy as np

from mtah.eval.drift import blackout_mask_from_match
from mtah.eval.official import possession_metrics as pm
from mtah.eval.scorer import codes, score
from mtah.pipeline import PipelineConfig, compute_evidence, compute_strong, predict
from mtah.select import apply_cell, estimate_lengths, train_dead
from mtah.sim.dataset import SEALED_MATCH, list_sim_keys, load_half_sim, lomo_folds, match_id

BUCKETS = ("lt1s", "1to3s", "3to10s", "gt10s")
D = 2


def bucket(frames, fps):
    s = frames / fps
    return BUCKETS[0 if s < 1 else 1 if s < 3 else 2 if s <= 10 else 3]


def greedy_pred_matched(ov, n_pred, thr):
    gt_done, pred_done = set(), set()
    for iou, i, j in ov:
        if iou < thr:
            break
        if i in gt_done or j in pred_done:
            continue
        gt_done.add(i); pred_done.add(j)
    flags = np.zeros(n_pred, bool); flags[list(pred_done)] = True
    return flags


def d_runs(codes_, runs):
    return [(a, b) for a, b, k in pm.segments(codes_, runs) if k == D]


def any_overlap(a, bs):
    return any(b0 < a[1] and a[0] < b1 for b0, b1 in bs)


def new_acc():
    return {"strata": {}, "pred_unmatched": {"start_in_blackout": 0, "start_outside": 0, "total_pred": 0},
            "d": {"true_D_runs": 0, "pred_D_runs": 0, "missed_stoppages": 0, "false_stoppages": 0},
            "decomposition": {}}


def add(d, k, v):
    d[k] = d.get(k, 0) + v


def analyse_half(s, lab, thr=0.5):
    fps = s.match.fps
    bo = blackout_mask_from_match(s.match)
    g = codes(s.truth["ctrl_Dfull"]); p = codes(lab)
    runs = pm.valid_runs(g); filled = pm.fill_abstentions(p, runs)
    gs = pm.possession_spells(g, runs); ps = pm.possession_spells(filled, runs)
    ov = pm.spell_overlaps(gs, ps)
    mg = pm.matched_gt_flags(ov, len(gs), thr); mp = greedy_pred_matched(ov, len(ps), thr)
    a = new_acc()
    a["decomposition"] = {k: int(v) for k, v in score(s.truth["ctrl_Dfull"], lab)["decomposition"].items()}
    for (st, en, _), m in zip(gs, mg):
        key = f"start_{'in' if bo[st] else 'out'}|end_{'in' if bo[en - 1] else 'out'}|{bucket(en - st, fps)}"
        c = a["strata"].setdefault(key, {"n_true": 0, "n_matched": 0})
        c["n_true"] += 1; c["n_matched"] += int(m)
    for (st, en, _), m in zip(ps, mp):
        a["pred_unmatched"]["total_pred"] += 1
        if not m:
            a["pred_unmatched"]["start_in_blackout" if bo[st] else "start_outside"] += 1
    td, pd_ = d_runs(g, runs), d_runs(filled, runs)
    a["d"] = {"true_D_runs": len(td), "pred_D_runs": len(pd_),
              "missed_stoppages": sum(not any_overlap(r, pd_) for r in td),
              "false_stoppages": sum(not any_overlap(r, td) for r in pd_)}
    return a


def merge(into, a):
    for k, c in a["strata"].items():
        t = into["strata"].setdefault(k, {"n_true": 0, "n_matched": 0})
        t["n_true"] += c["n_true"]; t["n_matched"] += c["n_matched"]
    for k, v in a["pred_unmatched"].items():
        add(into["pred_unmatched"], k, v)
    for k, v in a["d"].items():
        add(into["d"], k, v)
    for k, v in a["decomposition"].items():
        add(into["decomposition"], k, v)


def finish(acc):
    for c in acc["strata"].values():
        c["matched_share"] = c["n_matched"] / c["n_true"] if c["n_true"] else None
    tot = {}
    for k, c in acc["strata"].items():
        parts = k.split("|")
        for lbl in (parts[0], parts[1], parts[2], "ALL"):
            t = tot.setdefault(lbl, {"n_true": 0, "n_matched": 0})
            t["n_true"] += c["n_true"]; t["n_matched"] += c["n_matched"]
    for t in tot.values():
        t["matched_share"] = t["n_matched"] / t["n_true"] if t["n_true"] else None
    acc["marginals"] = tot
    return acc


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sim", default=str(ROOT / "runs" / "sim"))
    ap.add_argument("--dir", default=str(ROOT / "runs" / "select" / "a1"))
    ap.add_argument("--seeds", type=int, nargs="+", default=[100, 101])
    args = ap.parse_args()
    best = json.loads((Path(args.dir) / "best.json").read_text()); cell = best["cell"]
    keys = [k for k in list_sim_keys(args.sim) if match_id(k) != SEALED_MATCH]
    base = PipelineConfig(); pooled = new_acc(); folds = {}
    for held, train_keys, test_keys in lomo_folds(keys):
        train = [load_half_sim(args.sim, k, s) for k in train_keys for s in args.seeds]
        test = [load_half_sim(args.sim, k, s) for k in test_keys for s in args.seeds]
        lengths = estimate_lengths(train, base, train_dead); dead = train_dead(train)
        cfg = apply_cell(base, cell); cfg.decoder.lengths = lengths
        acc = new_acc()
        for s in test:
            ev = compute_evidence(s.match, s.tags, cfg, dead, cache_key=f"{s.key}_s{s.seed}")
            lab = predict(s.match, s.tags, cfg, dead, evidence=ev, strong=compute_strong(s.match, cfg))
            merge(acc, analyse_half(s, lab))
        folds[str(held)] = finish(acc); merge(pooled, acc)
        print(f"fold {held} done", flush=True)
    pooled = finish(pooled)
    out = {"cell": cell, "seeds": list(args.seeds), "truth": "ctrl_Dfull", "iou": 0.5, "per_fold": folds, "pooled": pooled}
    (Path(args.dir) / "decomposition.json").write_text(json.dumps(out, indent=1))
    print("cell", cell)
    print(f"{'stratum':34s}{'n_true':>8s}{'matched':>8s}{'share':>7s}")
    for k in sorted(pooled["strata"]):
        c = pooled["strata"][k]; print(f"{k:34s}{c['n_true']:8d}{c['n_matched']:8d}{c['matched_share']:7.3f}")
    for k, c in sorted(pooled["marginals"].items()):
        print(f"[marg] {k:27s}{c['n_true']:8d}{c['n_matched']:8d}{c['matched_share']:7.3f}")
    print("decomposition", pooled["decomposition"]); print("pred_unmatched", pooled["pred_unmatched"]); print("D", pooled["d"])


if __name__ == "__main__":
    main()
