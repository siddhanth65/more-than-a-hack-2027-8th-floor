"""Random search over BlackoutParams so DFL blackouts look like the real match's (aggregates only).

The search sees FIT_SEEDS; the best parameters are then re-evaluated on the disjoint EVAL_SEEDS (held out).
"""
import argparse, hashlib, json, subprocess
from pathlib import Path

import numpy as np
from dataclasses import replace as dc_replace

from mtah.io.dfl_proxy import list_halves, load_half
from mtah.sim import EVAL_SEEDS, FIT_SEEDS, half_rng
from mtah.sim.blackout_sim import BlackoutParams, blackout_mask, ks_statistic, pooled_blackout_stats

ROOT = Path(__file__).resolve().parents[1]
FIT_RNG_SEED = 2026
SCALARS = ("per_graded_min", "masked_share", "median_s", "p90_s", "max_s", "dead_share_inside", "long_share",
           "entry_near_line_share", "entry_outward_share")


def pooled_stats(halves, params, seeds):
    items = []
    for hi, (m, labels) in enumerate(halves):
        for seed in seeds:
            mask = blackout_mask(labels, m.ball, m.pitch, m.fps, params, half_rng(seed, hi))
            items.append((mask, labels, m.fps, None, m.ball, m.pitch))
    return pooled_blackout_stats(items)


def scalars(st):
    return {k: st[k] for k in SCALARS} | {"n_blackouts": st["n_blackouts"]}


def loss(st, T):
    tol = T["tolerance"]
    z = [(((st[k] - T[k]) / tol[k]) ** 2 if np.isfinite(st[k]) else 100.0) for k in tol]
    mx = st["max_s"] if np.isfinite(st["max_s"]) else 0.0
    z.append(max(0.0, (T["max_s_min"] - mx) / 30.0) ** 2)
    return float(sum(z))


def gates(st, T):
    g = {k: bool(np.isfinite(st[k]) and abs(st[k] - T[k]) <= T["tolerance"][k]) for k in T["tolerance"]}
    g["max_s"] = bool(np.isfinite(st["max_s"]) and st["max_s"] >= T["max_s_min"])
    return g


RANGES = {
    "p_mask_near": (0.3, 0.95), "p_mask_far": (0.0, 0.6),
    "c_lo": (0.1, 0.6), "c_hi": (0.6, 1.3),
    "pre_hi_s": (1.0, 8.0), "post_hi_s": (1.0, 12.0),
    "live_rate_per_min": (0.0, 1.5), "live_median_s": (5.0, 25.0),
    "live_sigma_log": (0.5, 1.6), "live_max_s": (60.0, 240.0),
    "merge_s": (1.0, 4.0), "short_rate_per_min": (0.1, 0.9),
}


def sample(rng):
    # Wide ranges on purpose: ~44 % of masked frames must be live and the longest blackout
    # must reach >= 150 s, which the dead-anchored defaults alone cannot produce.
    return BlackoutParams(**{k: rng.uniform(lo, hi) for k, (lo, hi) in RANGES.items()},
                          pre_lo_s=0.3, post_lo_s=0.3)


def perturb(params, rng, scale=0.15):
    d = {k: float(np.clip(getattr(params, k) + rng.normal(0, scale * (hi - lo)), lo, hi))
         for k, (lo, hi) in RANGES.items()}
    return dc_replace(params, **d)


def git_short_sha():
    try:
        return subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT, capture_output=True,
                              text=True, check=True).stdout.strip()
    except Exception:
        return "unknown"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dfl", default=r"E:\football-synthesizer\data\dfl_open\work\fp1d")
    ap.add_argument("--draws", type=int, default=150)
    ap.add_argument("--seeds", type=int, default=2, help="number of FIT_SEEDS used by the search (<= len(FIT_SEEDS))")
    ap.add_argument("--refine", type=int, default=150)
    args = ap.parse_args()
    fit_seeds = FIT_SEEDS[:args.seeds]
    targets_path = ROOT / "configs" / "blackout_targets.json"
    T = json.loads(targets_path.read_text())
    dfl = Path(args.dfl)
    halves = [load_half(p, dfl / "fp1d_census.json") for _, p in list_halves(dfl)]
    rng = np.random.default_rng(FIT_RNG_SEED)
    best = None
    for i in range(args.draws):
        params = sample(rng)
        st = pooled_stats(halves, params, fit_seeds)
        L = loss(st, T)
        if best is None or L < best[0]:
            best = (L, params, st)
            print(f"draw {i}: loss {L:.2f}  {json.dumps({k: round(v, 3) for k, v in scalars(st).items()})}", flush=True)
    for j in range(args.refine):
        cand = perturb(best[1], rng)
        st = pooled_stats(halves, cand, fit_seeds)
        L = loss(st, T)
        if L < best[0]:
            best = (L, cand, st)
            print(f"refine {j}: loss {L:.2f}  {json.dumps({k: round(v, 3) for k, v in scalars(st).items()})}", flush=True)
    L, params, st = best
    held = pooled_stats(halves, params, EVAL_SEEDS)
    real_path = ROOT / "runs" / "real_blackout_lengths.json"
    real = json.loads(real_path.read_text()) if real_path.is_file() else None
    ks = {"in_sample": ks_statistic(st["lengths_s"], real) if real else None,
          "held_out": ks_statistic(held["lengths_s"], real) if real else None}
    g_in, g_out = gates(st, T), gates(held, T)
    ok_in, ok_out = all(g_in.values()), all(g_out.values())
    in_stats, out_stats = scalars(st), scalars(held)
    (ROOT / "configs" / "blackout_sim.json").write_text(json.dumps(params.to_dict(), indent=1))
    (ROOT / "runs").mkdir(exist_ok=True)
    (ROOT / "runs" / "blackout_fit.json").write_text(json.dumps({
        "loss": L, "in_sample": {"seeds": list(fit_seeds), "stats": in_stats, "gates": g_in, "pass": ok_in},
        "held_out": {"seeds": list(EVAL_SEEDS), "stats": out_stats, "gates": g_out, "pass": ok_out},
        "ks_vs_real": ks, "targets": T}, indent=1))
    (ROOT / "configs" / "blackout_sim.meta.json").write_text(json.dumps({
        "fit_rng_seed": FIT_RNG_SEED, "fit_seeds": list(fit_seeds), "eval_seeds": list(EVAL_SEEDS),
        "draws": args.draws, "refine_steps": args.refine, "loss": L,
        "targets_md5": hashlib.md5(targets_path.read_bytes()).hexdigest(),
        "in_sample_stats": in_stats, "held_out_stats": out_stats, "ks_vs_real": ks,
        "git_sha": git_short_sha()}, indent=1))
    print("GATES", "PASS" if ok_in else "FAIL", "(in-sample)", json.dumps({k: round(v, 3) for k, v in in_stats.items()}), flush=True)
    print("GATES", "PASS" if ok_out else "FAIL", "(held-out)", json.dumps({k: round(v, 3) for k, v in out_stats.items()}), flush=True)
    print("failed held-out gates:", [k for k, v in g_out.items() if not v], " KS vs real:", json.dumps(ks), flush=True)


if __name__ == "__main__":
    main()
