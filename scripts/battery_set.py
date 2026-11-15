"""Battery runner for one simulated/other-game data set (K-nat, K-sim, M, sealed 9007): a17, a5c_fx2, the organisers'
smoother and S (cross-half self-training of B, Amendment 3/3b recipe, applied per target half) on every half.

Stages (main env = env_cpu python; torch env = MTAH_PATHCRF_PY python):
  cands  main   a17, a5c_fx2 (registered select dirs) and smooth50 -> scores (four truth variants)
  sprep  main   a17's pipeline on each half: step features, a17 decode, merged posterior, V3 pseudo step labels
                (confident ball control + a17 decode at posterior >= 0.9; no public window exists outside the contest match)
  sft    torch  per target half: fine-tune a17's base B on the simulator sequences of the six DFL matches plus the pseudo labels
                of the OTHER half of the same match (same seed) only -> checkpoint
  sdec   main   S = models/s config (dead -4.0, seq_d +0.5, B weight 12) with the target half's own checkpoint -> scores
PathCRF is read from the cache only (a miss is an error). smooth_s = 0 on every set here: the 1.0 s input smoothing of the
registered candidates is a property of the contest feed (technical document section 6); these caches are the unsmoothed ones.
Aggregates only are ever printed.
"""
import argparse, copy, hashlib, json, os, sys, time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "scripts"))

import numpy as np

CODE = {"H": 0, "A": 1, "D": 2}
VARS = ("raw_Dfull", "ctrl_Dfull", "raw_Dnarrow", "ctrl_Dnarrow")
RULE = ROOT / "docs" / "arb" / "rule.json"
MATCHES = [9001, 9002, 9003, 9004, 9005, 9006]


def units(a):
    """[(key, seed)] of this part, pairs of halves kept together."""
    from mtah.sim.dataset import SEALED_MATCH, list_sim_keys, match_id
    keys = [k for k in list_sim_keys(a.sim) if a.allow_sealed or match_id(k) != SEALED_MATCH]
    if a.keys:
        keys = [k for k in keys if k in a.keys]
    u = [(k, s) for k in sorted(keys) for s in a.seeds]
    return u[a.part::a.of]


def pair_key(key):
    return key[:-1] + ("2" if key.endswith("1") else "1")


def setup_pathcrf():
    import mtah.evidence.pathcrf as pcrf

    def _no_run(*a, **k):
        raise RuntimeError("PathCRF cache miss: refusing to run PathCRF")
    pcrf._run_half = _no_run


def load_sel(path, cache):
    from make_candidate import load_selection
    cfg, dead, lengths, team, merge, switch, seq = load_selection(path, with_seq=True)
    cfg.pathcrf.cache_dir = str(cache); cfg.pathcrf.smooth_s = 0.0
    return cfg, dead, team, merge, switch, seq


def score_rows(tag, key, seed, sim, lab, cand):
    from mtah.eval.scorer import score
    rows = []
    for v in VARS:
        if v not in sim.truth:
            continue
        r = score(sim.truth[v], lab); g = sim.truth[v] != "X"
        rows.append({"set": tag, "cand": cand, "key": key, "seed": seed, "variant": v, "spell_f1": r["spell_f1"], "macro_f1": r["macro_f1"],
                     "n_gt": r["n_gt_spells"], "n_pred": r["n_pred_spells"], "pred_D": float((lab[g] == "D").mean())})
    return rows


def write_rows(out, name, part, rows):
    import pandas as pd
    out = Path(out); out.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out / f"{name}_part{part}.csv", index=False)


# ------------------------------------------------------------------------------------------------------------- cands
def stage_cands(a):
    from mtah import baselines
    from mtah.sim.dataset import load_half_sim
    from make_candidate import candidate_labels
    setup_pathcrf()
    cands = {n: (ROOT / "models" / n) for n in a.cands}
    sels = {n: (p, load_sel(p, a.cache)) for n, p in cands.items()}
    rows = []; out = Path(a.out); (out / "labels").mkdir(parents=True, exist_ok=True)
    for key, seed in units(a):
        t0 = time.time(); sim = load_half_sim(a.sim, key, seed); ck = f"{key}_s{seed}"
        for n, (p, (cfg, dead, team, merge, switch, seq)) in sels.items():
            lab = candidate_labels(sim.match, sim.tags, cfg, dead, team, merge, switch, cache_key=ck, seq=seq, sel_dir=p)
            np.save(out / "labels" / f"{n}__{ck}.npy", np.vectorize(CODE.get)(lab).astype(np.uint8))
            rows += score_rows(a.tag, key, seed, sim, lab, n)
        lab = baselines.smooth_vote(sim.match, 50)
        rows += score_rows(a.tag, key, seed, sim, lab, "smooth50")
        print(ck, f"{time.time() - t0:.0f}s", flush=True)
        write_rows(a.out, "cands", a.part, rows)
    print("DONE cands", a.part)


# ------------------------------------------------------------------------------------------------------------- sprep
def stage_sprep(a):
    import copy as cp
    import arb_common as ac
    import mtah.evidence.seq_feats as fx
    import mtah.evidence.seqstream as sq
    from mtah import selftrain as st
    from mtah.decode.viterbi import cost_matrix
    from mtah.pipeline import compute_evidence, compute_strong, predict
    from mtah.sim.dataset import load_half_sim
    setup_pathcrf()
    sel = ROOT / "models" / "a17"
    cfg, dead, team, merge, switch, seq = load_sel(sel, a.cache)
    rule = ac.RuleParams(**json.loads(RULE.read_text())["rule"])
    out = Path(a.out) / "prep"; out.mkdir(parents=True, exist_ok=True)
    for key, seed in units(a):
        t0 = time.time(); sim = load_half_sim(a.sim, key, seed); ck = f"{key}_s{seed}"
        m, tags = sim.match, sim.tags
        ev = compute_evidence(m, tags, cfg, dead, cache_key=ck, team_model=team, switch_model=switch)
        cfg_in = cp.deepcopy(cfg); cfg_in.dead_offset = float(seq["input_dead_offset"])
        ev_in = compute_evidence(m, tags, cfg_in, dead, cache_key=ck, team_model=team, switch_model=switch)
        z = sq.feature_inputs(m, ev_in, compute_strong(m, cfg), cfg.pathcrf, ck)
        X, names = fx.frame_features(z)
        Xs, starts, seg = fx.to_steps(X, names, z["half"])
        cfg2, ev2 = sq.add_to_decoder(cfg, ev, sq.compute_streams(z, seq, sel), seq)
        lab = predict(m, tags, cfg2, dead, evidence=ev2, merge_model=merge)
        C = cost_matrix(ev2, cfg2.decoder)
        conf = ac.confident_control(m, rule)
        post = st.merged_posterior_labels(C, lab, st.MERGED_POSTERIOR_MIN)
        y3 = st.to_step_labels(st.combine_frame_labels(conf, post), starts)
        np.savez_compressed(out / f"{ck}.npz", X=Xs, starts=starts, seg=seg, names=np.array(names), y=y3,
                            a17=np.vectorize(CODE.get)(lab).astype(np.uint8),
                            half_ids=np.array(sorted(int(h) for h in np.unique(z["half"]) if h > 0)))
        print(ck, "steps", len(Xs), "labelled", int((y3 != st.IGNORE).sum()), "D", int((y3 == 2).sum()), f"{time.time() - t0:.0f}s", flush=True)
    print("DONE sprep", a.part)


# ---------------------------------------------------------------------------------------------------------------- sft
def stage_sft(a):
    import torch
    import exp_seq_lomo_gru as gl
    import s_selftrain as S
    from mtah import selftrain as st
    ck = S.load_base(ROOT / "models" / "a17" / "seq_modelB.pt")
    paths = [gl.fpath(f"fold{m}", k, s) for m in MATCHES for k, s in gl.halves_of(m)]
    sims = S.sim_train_seqs_paths(gl, paths, ck)
    w = S.VARIANTS["V3"]["w"]
    out = Path(a.out) / "ckpt"; out.mkdir(parents=True, exist_ok=True)
    prepd = sorted((Path(a.out) / "prep").glob("*.npz"))
    for f in prepd[a.part::a.of]:
        key, seed = f.stem.rsplit("_s", 1); seed = int(seed)
        t0 = time.time(); k_src = pair_key(key)
        tgt = np.load(Path(a.out) / "prep" / f"{key}_s{seed}.npz")
        src = np.load(Path(a.out) / "prep" / f"{k_src}_s{seed}.npz")
        assert key != k_src and not np.array_equal(tgt["half_ids"], []) and list(src["names"]) == [str(n) for n in ck["names"]]
        st.leakage_check(int(key[-1]), int(k_src[-1]), [])
        pseudo = (((src["X"] - ck["mu"]) / ck["sd"]).astype(np.float32), src["y"].astype(np.int64))
        net = S.finetune(ck, sims, pseudo, w, log=lambda *x: None)
        torch.save({"state_dict": net.state_dict(), "mu": torch.tensor(ck["mu"]), "sd": torch.tensor(ck["sd"]), "names": list(ck["names"]),
                    "nf": ck["nf"], "hidden": ck["hidden"], "win": ck["win"], "stride": ck["stride"], "target": key,
                    "pseudo_source": k_src, "variant": "V3", "pseudo_weight": w}, out / f"{key}_s{seed}.pt")
        print(key, seed, "pseudo from", k_src, "labelled steps", int((src["y"] != st.IGNORE).sum()), f"{time.time() - t0:.0f}s", flush=True)
    print("DONE sft", a.part)


# ---------------------------------------------------------------------------------------------------------------- sdec
def stage_sdec(a):
    from mtah.sim.dataset import load_half_sim
    from make_candidate import candidate_labels
    setup_pathcrf()
    base = ROOT / "models" / "s"
    cfg, dead, team, merge, switch, seq = load_sel(base, a.cache)
    out = Path(a.out); (out / "labels").mkdir(parents=True, exist_ok=True); (out / "sel").mkdir(parents=True, exist_ok=True)
    rows = []
    for key, seed in units(a):
        t0 = time.time(); sim = load_half_sim(a.sim, key, seed); ck = f"{key}_s{seed}"
        prep = np.load(out / "prep" / f"{ck}.npz")
        d = out / "sel" / ck; d.mkdir(parents=True, exist_ok=True)
        for f in ("dead_model.joblib", "team_model.joblib"):
            if not (d / f).exists():
                (d / f).write_bytes((base / f).read_bytes())
        (d / "seq_modelB_t.pt").write_bytes((out / "ckpt" / f"{ck}.pt").read_bytes())
        sq = copy.deepcopy(seq); sq["modelB"] = {str(int(h)): "seq_modelB_t.pt" for h in prep["half_ids"]}
        raw = json.loads((base / "config.json").read_text()); raw["seq_stream"]["modelB"] = sq["modelB"]
        (d / "config.json").write_text(json.dumps(raw))
        lab = candidate_labels(sim.match, sim.tags, cfg, dead, team, merge, switch, cache_key=ck, seq=sq, sel_dir=d)
        np.save(out / "labels" / f"S__{ck}.npy", np.vectorize(CODE.get)(lab).astype(np.uint8))
        rows += score_rows(a.tag, key, seed, sim, lab, "S")
        print(ck, f"{time.time() - t0:.0f}s", flush=True)
        write_rows(a.out, "sdec", a.part, rows)
    print("DONE sdec", a.part)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["cands", "sprep", "sft", "sdec"])
    ap.add_argument("--sim", required=True); ap.add_argument("--cache", required=True); ap.add_argument("--out", required=True)
    ap.add_argument("--tag", required=True); ap.add_argument("--seeds", type=int, nargs="+", default=[100])
    ap.add_argument("--keys", nargs="*", default=None); ap.add_argument("--part", type=int, default=0); ap.add_argument("--of", type=int, default=1)
    ap.add_argument("--allow-sealed", action="store_true"); ap.add_argument("--cands", nargs="*", default=["a17", "a5c_fx2"])
    a = ap.parse_args()
    {"cands": stage_cands, "sprep": stage_sprep, "sft": stage_sft, "sdec": stage_sdec}[a.stage](a)


if __name__ == "__main__":
    main()
