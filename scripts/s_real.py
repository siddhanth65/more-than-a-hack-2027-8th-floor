"""Candidate S on the contest match (Plan 9 amendment 3): real-match inputs and cross-half fine-tuning of B.

  prep  (main env; needs MTAH_DATA_DIR, MTAH_PATHCRF_PY, the smoothed PathCRF cache runs/cache/pathcrf_fx)
        a17's pipeline on the contest match -> step features of B, the a17 decode and its merged posterior; pseudo-labels
        (confident ball control; V3: plus the a17 decode where the merged posterior >= 0.9), the organisers' public window
        EXCLUDED from every pseudo-label set -> <out>/prep.npz. Prints aggregates only.
  ft    (torch env) for each target half h: fine-tune a17's B on the simulator training sequences plus the pseudo labels of
        the OTHER half, schedule of amendment 3b, weight of the chosen variant -> <out>/seq_modelB_h{1,2}.pt
No score is computed on anything; the window frames are never labelled or scored.
"""
import argparse, copy, json, os, sys, time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "scripts"))

import numpy as np

from mtah import selftrain as st

MATCHES = [9001, 9002, 9003, 9004, 9005, 9006]
SEEDS = [100, 101]
CHOSEN = "V3"
RULE = ROOT / "docs" / "arb" / "rule.json"


def stage_prep(a):
    import arb_common as ac
    import mtah.evidence.seq_feats as fx
    import mtah.evidence.seqstream as sq
    from make_candidate import load_selection
    from mtah.decode.viterbi import cost_matrix
    from mtah.io.metrica_loader import load_tactical
    from mtah.io.tags import load_tags
    from mtah.pipeline import compute_evidence, compute_strong, predict
    sel = Path(a.base)
    cfg, dead, lengths, team, merge, switch, seq = load_selection(sel, with_seq=True)
    cfg.pathcrf.smooth_s = 1.0; cfg.pathcrf.cache_dir = str(ROOT / "runs" / "cache" / "pathcrf_fx")
    m = load_tactical(a.data); tags = load_tags(a.data)
    ev = compute_evidence(m, tags, cfg, dead, cache_key="real_tactical", team_model=team, switch_model=switch)
    cfg_in = copy.deepcopy(cfg); cfg_in.dead_offset = float(seq["input_dead_offset"])
    ev_in = compute_evidence(m, tags, cfg_in, dead, cache_key="real_tactical", team_model=team, switch_model=switch)
    z = sq.feature_inputs(m, ev_in, compute_strong(m, cfg), cfg.pathcrf, "real_tactical")
    X, names = fx.frame_features(z)
    Xs, starts, seg = fx.to_steps(X, names, z["half"])
    cfg2, ev2 = sq.add_to_decoder(cfg, ev, sq.compute_streams(z, seq, sel), seq)
    lab = predict(m, tags, cfg2, dead, evidence=ev2, merge_model=merge)
    C = cost_matrix(ev2, cfg2.decoder)
    rule = ac.RuleParams(**json.loads(RULE.read_text())["rule"])
    conf = st.exclude_frames(ac.confident_control(m, rule), [st.PUBLIC_WINDOW])
    post = st.merged_posterior_labels(C, lab, st.MERGED_POSTERIOR_MIN)
    full = st.exclude_frames(st.combine_frame_labels(conf, post), [st.PUBLIC_WINDOW])
    y1 = st.to_step_labels(conf, starts); y3 = st.to_step_labels(full, starts)
    # guard: not one labelled step has its centre frame inside the public window
    centre = starts + st.STEP // 2
    inw = (centre >= st.PUBLIC_WINDOW[0]) & (centre <= st.PUBLIC_WINDOW[1])
    assert (y1[inw] == st.IGNORE).all() and (y3[inw] == st.IGNORE).all()
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out / "prep.npz", X=Xs, starts=starts, seg=seg, names=np.array(names), y_conf=y1, y_conf_a17=y3,
                        a17_hash=np.array(__import__("hashlib").md5(np.ascontiguousarray(lab.astype("S1")).tobytes()).hexdigest()))
    print(json.dumps({"steps": int(len(Xs)), "window_steps_excluded": int(inw.sum()),
                      "per_half": {int(h): {"steps": int((seg == h).sum()), "conf_steps": int(((seg == h) & (y1 != st.IGNORE)).sum()),
                                            "conf_a17_steps": int(((seg == h) & (y3 != st.IGNORE)).sum()),
                                            "D_in_a17_labels": int(((seg == h) & (y3 == 2)).sum())} for h in (1, 2)},
                      "a17_label_hash": str(__import__("hashlib").md5(np.ascontiguousarray(lab.astype("S1")).tobytes()).hexdigest())}))


def stage_ft(a):
    import torch
    import exp_seq_lomo_gru as gl
    import s_selftrain as S
    w = S.VARIANTS[CHOSEN]["w"]
    z = np.load(Path(a.out) / "prep.npz")
    X, seg, y = z["X"], z["seg"], z["y_conf_a17"] if CHOSEN == "V3" else z["y_conf"]
    ck = torch.load(Path(a.base) / "seq_modelB.pt", weights_only=False)
    ck["mu"] = ck["mu"].numpy(); ck["sd"] = ck["sd"].numpy()
    assert list(ck["names"]) == [str(n) for n in z["names"]]
    paths = [gl.fpath(f"fold{m}", k, s) for m in MATCHES for k, s in gl.halves_of(m)]
    sims = S.sim_train_seqs_paths(gl, paths, ck)
    for h in (1, 2):
        src = st.cross_half_source(h, [1, 2])
        st.leakage_check(h, src, [])
        i = np.flatnonzero(seg == src)
        pseudo = (((X[i] - ck["mu"]) / ck["sd"]).astype(np.float32), y[i].astype(np.int64))
        t0 = time.time()
        net = S.finetune(ck, sims, pseudo, w)
        torch.save({"state_dict": net.state_dict(), "mu": torch.tensor(ck["mu"]), "sd": torch.tensor(ck["sd"]),
                    "names": list(ck["names"]), "nf": ck["nf"], "hidden": ck["hidden"], "win": ck["win"], "stride": ck["stride"],
                    "target_half": h, "pseudo_source_half": src, "variant": CHOSEN, "pseudo_weight": w},
                   Path(a.out) / f"seq_modelB_h{h}.pt")
        print(f"half {h}: fine-tuned on pseudo labels of half {src} ({len(i)} steps, {int((y[i] != st.IGNORE).sum())} labelled) {time.time() - t0:.0f}s")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["prep", "ft"])
    ap.add_argument("--base", default=str(ROOT / "models" / "a17")); ap.add_argument("--out", default=str(ROOT / "runs" / "s" / "real"))
    ap.add_argument("--data", default=os.environ.get("MTAH_DATA_DIR"))
    a = ap.parse_args()
    {"prep": stage_prep, "ft": stage_ft}[a.stage](a)


if __name__ == "__main__":
    main()
