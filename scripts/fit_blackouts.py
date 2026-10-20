"""Random search over BlackoutParams so DFL blackouts look like the real match's (aggregates only)."""
import argparse, json
from pathlib import Path

import numpy as np
from dataclasses import replace as dc_replace

from mtah.io.dfl_proxy import list_halves, load_half
from mtah.sim.blackout_sim import BlackoutParams, blackout_mask, blackout_stats

ROOT = Path(__file__).resolve().parents[1]


def pooled_stats(halves, params, seeds):
    acc = {k: [] for k in ("per_133min", "masked_share", "median_s", "p90_s", "max_s", "dead_share_inside", "long_share")}
    for m, labels in halves:
        for seed in seeds:
            mask = blackout_mask(labels, m.ball, m.pitch, m.fps, params, np.random.default_rng(seed))
            st = blackout_stats(mask, labels, m.fps)
            for k in acc:
                acc[k].append(st[k])
    return {k: float(np.nanmean(v)) for k, v in acc.items()} | {"max_s": float(np.nanmax(acc["max_s"]))}


def loss(st, T):
    tol = T["tolerance"]
    z = [((st[k] - T[k]) / tol[k]) ** 2 for k in tol]
    z.append(max(0.0, (T["max_s_min"] - st["max_s"]) / 30.0) ** 2)
    return float(sum(z))


RANGES = {
    "p_mask_near": (0.3, 0.95), "p_mask_far": (0.05, 0.6),
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dfl", default=r"E:\football-synthesizer\data\dfl_open\work\fp1d")
    ap.add_argument("--draws", type=int, default=150)
    ap.add_argument("--seeds", type=int, default=2)
    ap.add_argument("--refine", type=int, default=150)
    args = ap.parse_args()
    T = json.loads((ROOT / "configs" / "blackout_targets.json").read_text())
    dfl = Path(args.dfl)
    halves = [load_half(p, dfl / "fp1d_census.json") for _, p in list_halves(dfl)]
    rng = np.random.default_rng(2026)
    best = None
    for i in range(args.draws):
        params = sample(rng)
        st = pooled_stats(halves, params, range(args.seeds))
        L = loss(st, T)
        if best is None or L < best[0]:
            best = (L, params, st)
            print(f"draw {i}: loss {L:.2f}  {json.dumps({k: round(v, 3) for k, v in st.items()})}", flush=True)
    for j in range(args.refine):
        cand = perturb(best[1], rng)
        st = pooled_stats(halves, cand, range(args.seeds))
        L = loss(st, T)
        if L < best[0]:
            best = (L, cand, st)
            print(f"refine {j}: loss {L:.2f}  {json.dumps({k: round(v, 3) for k, v in st.items()})}", flush=True)
    L, params, st = best
    (ROOT / "configs" / "blackout_sim.json").write_text(json.dumps(params.to_dict(), indent=1))
    (ROOT / "runs").mkdir(exist_ok=True)
    (ROOT / "runs" / "blackout_fit.json").write_text(json.dumps({"loss": L, "stats": st, "targets": T}, indent=1))
    ok = all(abs(st[k] - T[k]) <= T["tolerance"][k] for k in T["tolerance"]) and st["max_s"] >= T["max_s_min"]
    print("GATES PASS" if ok else "GATES FAIL", json.dumps(st), flush=True)


if __name__ == "__main__":
    main()
