"""Run the candidate pipelines on (pseudo-)masked data: the heavy part of both arbitration experiments.

Units: every selection-fold sim half (matches 9001-9006; sealed 9007 is refused) at seed 100, and the real match.
Per unit and mask seed (None = unmasked, 0..4 = pseudo-blackout masks, ~30 % of the naturally visible frames
hidden, windows drawn from the simulator's blackout length distribution and placed only where the ball is visible)
every family runs its FULL pipeline (evidence -> learned streams -> Viterbi -> post-processing -> merge), out of fold
on the sim. The unmasked run also decodes the family's ball-only pipeline (pathcrf / teamctrl / learned streams off,
merge off). Per-task output npz: mask, lab_<fam>, labball_<fam> (unmasked only), st_<fam>_<stream> float16 (F,3).

  python scripts/arb_run.py pool                      (build runs/arb/blackout_pool.json from the sim blackouts)
  python scripts/arb_run.py run --out runs/arb/out --workers 8 [--what sim real] [--seeds 0 1 2 3 4]
"""
import argparse, copy, json, os, sys, time
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "2"); os.environ.setdefault("OPENBLAS_NUM_THREADS", "2")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "scripts"))

import numpy as np

CODE = {"H": 0, "A": 1, "D": 2}
MASK_SEEDS = (0, 1, 2, 3, 4)
POOL = ROOT / "runs" / "arb" / "blackout_pool.json"
SIM_SEEDS_MASKED = (100,)
SIM_SEEDS_UNMASKED = (101,)      # extra unmasked halves: reproduction check of the registered 6-fold numbers


def _vec(lab):
    lab = np.asarray(lab).astype("<U1"); out = np.full(len(lab), 3, np.uint8)
    for k, v in CODE.items():
        out[lab == k] = v
    return out


def unit_list(what, with_unmasked_extra=True):
    from mtah.sim.dataset import SEALED_MATCH
    units = []
    if "real" in what:
        units.append(("real", None, None))
    if "sim" in what:
        for m in (9001, 9002, 9003, 9004, 9005, 9006):
            assert m != SEALED_MATCH
            for h in ("H1", "H2"):
                for s in SIM_SEEDS_MASKED:
                    units.append(("sim", f"game_{m}_{h}", s))
                if with_unmasked_extra:
                    for s in SIM_SEEDS_UNMASKED:
                        units.append(("sim", f"game_{m}_{h}", s))
    return units


def unit_name(u):
    return "real" if u[0] == "real" else f"{u[1]}_s{u[2]}"


def unit_id(u):
    if u[0] == "real":
        return 999
    from mtah.sim.dataset import half_index
    return half_index(u[1]) * 10 + (u[2] - 100)


def build_pool():
    import arb_common as ac
    sim = ROOT / "runs" / "sim"; ms = []
    for m in (9001, 9002, 9003, 9004, 9005, 9006):
        for h in ("H1", "H2"):
            for s in (100, 101, 102, 103, 104):
                p = sim / f"game_{m}_{h}_s{s}.npz"
                if p.exists():
                    with np.load(p) as z:
                        ms.append(z["blackout"].copy())
    pool = ac.length_pool_from_blackouts(ms)
    POOL.parent.mkdir(parents=True, exist_ok=True)
    POOL.write_text(json.dumps({"lengths_s": pool.tolist(), "n_halves": len(ms),
                                "quantiles_s": {str(q): float(np.percentile(pool, q)) for q in (10, 25, 50, 75, 90, 99)}}))
    print("pool", len(pool), "median", float(np.median(pool)), "n_halves", len(ms))


def _patch_pathcrf():
    import mtah.evidence.pathcrf as pcrf

    def _no_run(*a, **k):
        raise RuntimeError("PathCRF cache miss: refusing to run PathCRF (read-only cache)")
    pcrf._run_half = _no_run


def run_family(fam, mdl, m, tags, key, ball_only):
    from mtah.pipeline import compute_evidence, compute_strong, predict
    import mtah.evidence.seqstream as sq
    cfg = mdl["cfg"]; cfg.pathcrf.cache_dir = str(ROOT / "runs" / "cache" / "pathcrf")
    ev = compute_evidence(m, tags, cfg, mdl["dead"], cache_key=key, team_model=mdl["team"])
    strong = compute_strong(m, cfg)
    streams = {"pathcrf": ev["pathcrf"], "teamctrl": ev["teamctrl"]}
    cfg2, ev2 = cfg, ev
    seq = mdl["seq"]
    if seq is not None:
        ev_in = ev
        if seq.get("input_dead_offset") is not None:
            cfg_in = copy.deepcopy(cfg); cfg_in.dead_offset = float(seq["input_dead_offset"])
            ev_in = compute_evidence(m, tags, cfg_in, mdl["dead"], cache_key=key, team_model=mdl["team"])
        z = sq.feature_inputs(m, ev_in, strong, cfg.pathcrf, key)
        st = sq.compute_streams(z, seq, mdl["sel_dir"])
        streams.update(st)
        cfg2, ev2 = sq.add_to_decoder(cfg, ev, st, seq)
    lab = predict(m, tags, cfg2, mdl["dead"], evidence=ev2, strong=strong, merge_model=mdl["merge"])
    out = {f"lab_{fam}": _vec(lab)}
    for k, v in streams.items():
        out[f"st_{fam}_{k}"] = np.asarray(v, np.float16)
    if ball_only:
        cb = copy.deepcopy(cfg); cb.decoder.weights["pathcrf"] = 0.0; cb.decoder.weights["teamctrl"] = 0.0
        cb.post.merge_theta = 1.0
        out[f"labball_{fam}"] = _vec(predict(m, tags, cb, mdl["dead"], evidence=ev, strong=strong, merge_model=None))
    return out


def do_task(args):
    unit, mseed, outdir = args
    import arb_common as ac
    import arb_models as am
    outp = Path(outdir) / f"{unit_name(unit)}_m{'none' if mseed is None else mseed}.npz"
    if outp.exists():
        return str(outp), 0.0
    _patch_pathcrf()
    t0 = time.time()
    if unit[0] == "real":
        from mtah.io.metrica_loader import load_tactical
        from mtah.io.tags import load_tags
        d = os.environ["MTAH_DATA_DIR"]
        real = (load_tactical(d), load_tags(d))
        key = "real_tactical"; fams = am.FAMILIES; fold = None

        def get(f):
            return real
        base = real[0]
    else:
        from mtah.sim.dataset import load_half_sim
        key_, seed = unit[1], unit[2]; key = f"{key_}_s{seed}"; fams = am.SIM_FAMILIES
        fold = int(key_.split("_")[1])
        cache = {}

        def get(f):
            b = am.SPEC[f]["build"]
            if b not in cache:
                s = load_half_sim(ROOT / "runs" / b, key_, seed); cache[b] = (s.match, s.tags)
            return cache[b]
        base = get("a13r")[0]
    mask = None
    if mseed is not None:
        pool = np.asarray(json.loads(POOL.read_text())["lengths_s"])
        mask = ac.sample_mask(base, mseed, pool, unit_id=unit_id(unit))
    res = {"mask": mask if mask is not None else np.zeros(base.n_frames, bool)}
    for fam in fams:
        m, tags = get(fam)
        if mask is not None:
            assert np.array_equal(m.ball_present, base.ball_present)
            m = ac.apply_mask(m, mask)
        mdl = am.load_family(fam, fold)
        res.update(run_family(fam, mdl, m, tags, key, ball_only=mseed is None))
    outp.parent.mkdir(parents=True, exist_ok=True)
    tmp = outp.with_suffix(".tmp.npz"); np.savez(tmp, **res); os.replace(tmp, outp)
    return str(outp), time.time() - t0


def main():
    ap = argparse.ArgumentParser(); sp = ap.add_subparsers(dest="cmd", required=True)
    sp.add_parser("pool")
    r = sp.add_parser("run"); r.add_argument("--out", default="runs/arb/out"); r.add_argument("--workers", type=int, default=4)
    r.add_argument("--what", nargs="+", default=["real", "sim"]); r.add_argument("--seeds", type=int, nargs="*", default=list(MASK_SEEDS))
    r.add_argument("--unmasked-only", action="store_true"); r.add_argument("--limit", type=int, default=0)
    a = ap.parse_args()
    if a.cmd == "pool":
        build_pool(); return
    tasks = []
    for u in unit_list(a.what):
        masks = [None] + ([] if (a.unmasked_only or u[2] in SIM_SEEDS_UNMASKED) else list(a.seeds))
        tasks += [(u, ms, a.out) for ms in masks]
    tasks.sort(key=lambda t: t[0][0] != "real")   # the big real-match tasks first
    if a.limit:
        tasks = tasks[:a.limit]
    print(f"{len(tasks)} tasks, {a.workers} workers", flush=True)
    from concurrent.futures import ProcessPoolExecutor, as_completed
    t0 = time.time()
    with ProcessPoolExecutor(a.workers) as ex:
        futs = {ex.submit(do_task, t): t for t in tasks}
        for i, f in enumerate(as_completed(futs)):
            t = futs[f]
            try:
                p, dt = f.result(); print(f"[{i + 1}/{len(tasks)}] {Path(p).name} {dt:.0f}s  total {time.time() - t0:.0f}s", flush=True)
            except Exception as e:
                import traceback; print(f"FAILED {unit_name(t[0])} {t[1]}: {e!r}", flush=True); traceback.print_exc()


if __name__ == "__main__":
    main()
