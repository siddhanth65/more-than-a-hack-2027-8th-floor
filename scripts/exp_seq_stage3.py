"""exp-seq stage 3: decode learned per-step log-probs with the SAME a5c Viterbi + post-processing (+ fold merge model).

Modes: "replace" (the model's log-probs are the only cost stream, weight w) and "extra" (a5c's five streams plus the
model's log-probs with weight w). w is chosen on the inner validation match 9006 (cross-fitted evidence, inner-model
predictions, merge OFF there because the fold merge model was trained on 9006 spells); the test fold 9001 is decoded
once per model/mode with the chosen w, merge ON (a5c pipeline) and also reported with merge OFF.
Usage: exp_seq_stage3.py A|B|base
"""
import copy, json, os, sys, time
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "4")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "scripts"))

import numpy as np
import pandas as pd

import exp_seq_feats as fx
from exp_seq_stage1 import MAIN, OUT, SEEDS, SIM, STREAMS, a5c_cfg, base_cfg
from mtah.decode.viterbi import decode
from mtah.eval.scorer import score
from mtah.sim.dataset import SEALED_MATCH, load_half_sim
from mtah.spells.merge import MergeModel, apply_merge_model
from mtah.spells.postprocess import postprocess

W_GRID = {"replace": (0.5, 1.0, 2.0, 4.0, 8.0, 16.0), "extra": (0.5, 1.0, 2.0, 4.0, 8.0)}
VARS = ("ctrl_Dfull", "raw_Dfull")


def names_for(ms):
    return [(f"game_{m}_{h}", s) for m in ms for h in ("H1", "H2") for s in SEEDS]


def load_half(key, seed, model):
    assert int(key.split("_")[1]) != SEALED_MATCH
    s = load_half_sim(SIM, key, seed)
    z = np.load(OUT / "halves" / f"{key}_s{seed}.npz")
    ev = {k: z[f"ev_{k}"] for k in STREAMS}
    strong = (z["sH"], z["sA"], z["touches"])
    seq = None
    if model in ("A", "B"):
        p = np.load(OUT / f"pred_{model}" / f"{key}_s{seed}.npz")
        seq = fx.upsample(p["logp"], p["starts"], s.match.n_frames, s.match.half)
    return s, ev, strong, seq


def run_decode(s, ev, strong, seq, cfg, mode, w, theta):
    dcfg = copy.deepcopy(cfg.decoder)
    if mode == "base":
        ev_dec = ev
    elif mode == "replace":
        ev_dec = {"seq": seq}; dcfg.weights = {"seq": w}; dcfg.lengths = {"seq": 1.0}
    else:
        ev_dec = {**ev, "seq": seq}; dcfg.weights = {**dcfg.weights, "seq": w}; dcfg.lengths = {**dcfg.lengths, "seq": 1.0}
    lab = decode(ev_dec, s.match.half, dcfg)
    out = postprocess(lab, s.match.half, s.match.fps, cfg.post, strong_H=strong[0], strong_A=strong[1], touches=strong[2])
    if theta < 1.0:
        out = apply_merge_model(out, s.match, ev, strong, MERGE, theta)
    return out


def scores(s, lab):
    r = {}
    for v in VARS:
        x = score(s.truth[v], lab)
        r[f"{v}"] = x["spell_f1"]; r[f"{v}_macro"] = x["macro_f1"]; r[f"{v}_ngt"] = x["n_gt_spells"]
        r["n_pred"] = x["n_pred_spells"]
    return r


def main():
    global MERGE
    model = sys.argv[1]
    MERGE = MergeModel.load(OUT / "models" / "merge_fold9001.joblib")
    cfg = a5c_cfg(base_cfg())
    logf = open(OUT / "stage3.log", "a")

    def log(msg):
        line = f"[{time.strftime('%H:%M:%S')}] {msg}"; print(line, flush=True); logf.write(line + "\n"); logf.flush()

    tune_csv = OUT / f"stage3_tune_{model}.csv"; test_csv = OUT / f"stage3_test_{model}.csv"
    chosen = {}
    if model in ("A", "B"):
        rows = []
        halves = [load_half(k, s, model) for k, s in names_for([9006])]
        for s, ev, strong, seq in halves:   # a5c reference on the same inner halves (merge off)
            rows.append({"mode": "a5c", "w": np.nan, "key": s.key, "seed": s.seed,
                         **scores(s, run_decode(s, ev, strong, seq, cfg, "base", 0, 1.0))})
        for mode, grid in W_GRID.items():
            for w in grid:
                for s, ev, strong, seq in halves:
                    rows.append({"mode": mode, "w": w, "key": s.key, "seed": s.seed,
                                 **scores(s, run_decode(s, ev, strong, seq, cfg, mode, w, 1.0))})
                pd.DataFrame(rows).to_csv(tune_csv, index=False)
                d = pd.DataFrame(rows); d = d[(d["mode"] == mode) & (d["w"] == w)]
                log(f"tune {model} {mode} w={w}: ctrl {d['ctrl_Dfull'].mean():.4f} raw {d['raw_Dfull'].mean():.4f} "
                    f"n_pred {d['n_pred'].mean():.0f}")
        d = pd.DataFrame(rows)
        log(f"tune {model} a5c ref (9006, merge off): ctrl {d[d['mode'] == 'a5c']['ctrl_Dfull'].mean():.4f}")
        for mode in W_GRID:
            g = d[d["mode"] == mode].groupby("w")["ctrl_Dfull"].mean()
            chosen[mode] = float(g.idxmax())
        log(f"chosen {model}: {chosen}")
        del halves
    rows = []
    for k, sd in names_for([9001]):
        s, ev, strong, seq = load_half(k, sd, model)
        settings = [("base", 0.0)] if model == "base" else [(m, chosen[m]) for m in W_GRID]
        for mode, w in settings:
            for theta in (0.7, 1.0):
                t0 = time.time()
                lab = run_decode(s, ev, strong, seq, cfg, mode, w, theta)
                rows.append({"model": model, "mode": mode, "w": w, "merge_theta": theta, "key": s.key, "seed": s.seed,
                             "decode_s": round(time.time() - t0, 2), **scores(s, lab)})
        pd.DataFrame(rows).to_csv(test_csv, index=False)
        log(f"test {model} {k}_s{sd} done")
    d = pd.DataFrame(rows)
    summ = d.groupby(["model", "mode", "w", "merge_theta"])[["ctrl_Dfull", "raw_Dfull", "ctrl_Dfull_macro", "n_pred",
                                                            "ctrl_Dfull_ngt", "decode_s"]].mean()
    log("TEST 9001\n" + summ.to_string())
    (OUT / f"stage3_chosen_{model}.json").write_text(json.dumps(chosen))


if __name__ == "__main__":
    main()
