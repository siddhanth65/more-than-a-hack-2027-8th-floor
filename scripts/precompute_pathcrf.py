"""Pre-compute the PathCRF team posterior for every (half, seed) of the sim and, with --real, the real tactical match.

Fills runs/cache/pathcrf (keys `<key>_s<seed>` and `real_tactical`, as compute_evidence uses them) and prints per-half
aggregates only: agreement of p_home > 0.5 with the raw_Dfull truth on H/A frames (optimistic for 9001-9006, the
PathCRF train/val matches; never computed for the sealed match), a label-free proxy (agreement with the team of the
nearest player to the ball when the ball is visible and that player is within 2 m; comparable between sim and real),
NaN share, mean imputed_frac and seconds. Aggregates go to <cache_dir>/precompute_summary.json; no per-frame output.

--threads must equal the thread count the selection/candidate runs use (select_lomo --pathcrf-threads): the cache is
keyed on it.
"""
import argparse, json, os, sys, time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np

from mtah.evidence.pathcrf import PathCRFParams, pathcrf_posterior
from mtah.sim.dataset import SEALED_MATCH, list_sim_keys, load_half_sim, match_id

OPTIMISTIC = range(9001, 9007)   # PathCRF train (9001-9005) / val (9006)
PROXY_MAX_DIST = 2.0


def nearest_player_proxy(match, p_home, max_dist=PROXY_MAX_DIST):
    """(agreement, n): share of graded frames with a visible ball whose nearest tracked player is within max_dist
    metres and p_home is finite, on which (p_home > 0.5) equals (nearest player is home)."""
    p = np.asarray(p_home, np.float64)
    d = np.hypot(match.pos[..., 0] - match.ball[:, None, 0], match.pos[..., 1] - match.ball[:, None, 1])
    d = np.where(np.isfinite(d), d, np.inf)
    j = d.argmin(1)
    dmin = d[np.arange(len(j)), j]
    ok = match.graded & np.isfinite(p) & (dmin <= max_dist)
    n = int(ok.sum())
    if n == 0:
        return float("nan"), 0
    home = match.team[j[ok]] == 0
    return float(np.mean((p[ok] > 0.5) == home)), n


def truth_agreement(p_home, truth):
    """(agreement, n): on frames with truth in H/A and finite p_home, share where (p_home > 0.5) == (truth == "H")."""
    p = np.asarray(p_home, np.float64)
    ok = np.isin(truth, ("H", "A")) & np.isfinite(p)
    n = int(ok.sum())
    if n == 0:
        return float("nan"), 0
    return float(np.mean((p[ok] > 0.5) == (truth[ok] == "H"))), n


def posterior_stats(res, half):
    """NaN share of p_home and mean imputed_frac over graded frames (half > 0)."""
    g = np.asarray(half) > 0
    p = np.asarray(res["p_home"], np.float64)[g]
    imp = np.asarray(res["imputed_frac"], np.float64)[g]
    return {"nan_share": float(np.mean(~np.isfinite(p))) if p.size else float("nan"),
            "imputed_frac_mean": float(np.nanmean(imp)) if np.isfinite(imp).any() else float("nan")}


def _run(match, cache_key, params):
    cache = Path(params.cache_dir) / f"{cache_key}.npz"
    before = cache.stat().st_mtime if cache.is_file() else None
    t0 = time.time()
    r = pathcrf_posterior(match, cache_key, params)
    secs = time.time() - t0
    fresh = before is None or cache.stat().st_mtime != before
    rec = {"cache_key": cache_key, "seconds": round(secs, 1), "computed": bool(fresh), **posterior_stats(r, match.half)}
    rec["proxy_agreement"], rec["proxy_n"] = nearest_player_proxy(match, r["p_home"])
    return r, rec


def _fmt(x):
    return "  nan" if x is None or not np.isfinite(x) else f"{x:.3f}"


def _mean(recs, k):
    v = [r[k] for r in recs if r.get(k) is not None and np.isfinite(r[k])]
    return float(np.mean(v)) if v else float("nan")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sim", default=str(ROOT / "runs" / "sim"))
    ap.add_argument("--seeds", type=int, nargs="+", default=[100, 101])
    ap.add_argument("--real", action="store_true", help="also the real tactical match (cache key real_tactical)")
    ap.add_argument("--data", default=os.environ.get("MTAH_DATA_DIR", r"HOME/mtah-data"))
    ap.add_argument("--threads", type=int, default=PathCRFParams().threads,
                    help="PathCRF threads; must equal select_lomo --pathcrf-threads (cache key)")
    args = ap.parse_args()
    params = PathCRFParams(threads=args.threads)
    halves = []
    print(f"PathCRF precompute: ckpt {params.ckpt}, threads {params.threads}, cache {params.cache_dir}", flush=True)
    for key in list_sim_keys(args.sim):
        for seed in args.seeds:
            s = load_half_sim(args.sim, key, seed)
            mid = match_id(key)
            r, rec = _run(s.match, f"{key}_s{seed}", params)
            rec.update(key=key, seed=seed, match=mid)
            if mid == SEALED_MATCH:
                rec.update(truth_agreement=None, truth_n=0, label="sealed: truth agreement not computed")
            else:
                g = s.match.graded
                rec["truth_agreement"], rec["truth_n"] = truth_agreement(r["p_home"][g], s.truth["raw_Dfull"][g])
                rec["label"] = "optimistic (PathCRF train/val)" if mid in OPTIMISTIC else "clean"
            del s, r
            halves.append(rec)
            print(f"{key} s{seed}: truth_agree(raw_Dfull) {_fmt(rec['truth_agreement'])} [{rec['label']}]  "
                  f"proxy {_fmt(rec['proxy_agreement'])} (n={rec['proxy_n']})  nan {rec['nan_share']:.3f}  "
                  f"imputed {_fmt(rec['imputed_frac_mean'])}  {rec['seconds']:.0f}s{'' if rec['computed'] else ' (cached)'}",
                  flush=True)
    summary = {"ckpt": params.ckpt, "threads": params.threads, "seeds": args.seeds, "halves": halves,
               "sim_mean": {k: _mean(halves, k) for k in ("truth_agreement", "proxy_agreement", "nan_share", "imputed_frac_mean")},
               "sim_seconds_total": float(sum(r["seconds"] for r in halves)), "real": None}
    print("SIM MEAN", json.dumps(summary["sim_mean"]), f"total {summary['sim_seconds_total']:.0f}s", flush=True)
    if args.real:
        from mtah.io.metrica_loader import load_tactical
        m = load_tactical(args.data)
        _, rec = _run(m, "real_tactical", params)
        del m
        summary["real"] = rec
        print(f"real_tactical: proxy {_fmt(rec['proxy_agreement'])} (n={rec['proxy_n']})  nan {rec['nan_share']:.3f}  "
              f"imputed {_fmt(rec['imputed_frac_mean'])}  {rec['seconds']:.0f}s{'' if rec['computed'] else ' (cached)'}",
              flush=True)
    out = Path(params.cache_dir) / "precompute_summary.json"; out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(summary, indent=1))
    print("wrote", out)


if __name__ == "__main__":
    main()
