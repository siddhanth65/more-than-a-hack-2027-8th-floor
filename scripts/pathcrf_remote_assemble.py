"""Run PathCRF for a sim directory on a remote machine and merge the results into the normal posterior cache.

Two steps around the remote run (the wrapper `third_party/pathcrf/run_infer.py` itself is unchanged):

  prepare   python scripts/pathcrf_remote_assemble.py prepare --sim runs/sim_extra --seeds 100 --out runs/pathcrf_remote
            writes <out>/in/<cache_key>_h<i>.in.npz (one per graded half, the exact hand-off file pathcrf_posterior
            would write), <out>/manifest.json (input / wrapper sha per match) and <out>/run_all.sh (xargs, P jobs
            in parallel, one thread each by default; the thread count is part of the cache key, so use
            --threads 4 to reproduce the pinned PathCRFParams().threads, or edit it on both sides).
  (remote)  copy <out>/in and run_all.sh to the server, run it there with the PathCRF environment, copy <out>/out back.
  assemble  python scripts/pathcrf_remote_assemble.py assemble --out runs/pathcrf_remote --cache-dir runs/cache/pathcrf
            re-checks the manifest against the sims on disk, reads <out>/out/<cache_key>_h<i>.out.npz, applies the
            same MIN_OK_RATIO check and writes runs/cache/pathcrf/<cache_key>.npz with mtah.evidence.pathcrf.write_cache,
            i.e. byte-for-byte the format pathcrf_posterior writes (ckpt, threads, input_sha, wrapper_sha, p_home,
            p_none, imputed_frac). n_chunks / n_ok are not part of that format; they go to <out>/assemble_summary.json.

Only positions are shipped: for the SkillCorner sims (open data) that is fine; never use it for the challenge data.
"""
import argparse
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np

import mtah.evidence.pathcrf as pc
from mtah.io.match import half_runs
from mtah.sim.dataset import list_sim_keys, load_half_sim

SERVER_RUNNER = """#!/bin/bash
# usage: PY=<python with torch + torch_geometric> REPO=<pathcrf checkout> WRAPPER=<run_infer.py> bash run_all.sh [jobs]
set -e
cd "$(dirname "$0")"
mkdir -p out
export PY REPO WRAPPER TQDM_DISABLE=1 OMP_NUM_THREADS={threads} MKL_NUM_THREADS={threads}
ls in/*.in.npz | xargs -P "${{1:-8}}" -I{{}} bash -c '
  f="{{}}"; b=$(basename "$f" .in.npz)
  [ -f "out/$b.out.npz" ] && exit 0
  "$PY" "$WRAPPER" --in "$f" --out "out/$b.out.npz" --ckpt {ckpt} --repo "$REPO" --threads {threads} > "out/$b.log" 2>&1'
echo done
"""


def _match_sha(match):
    return pc._input_sha(match)


def prepare(sim_dir, seeds, out, params):
    out = Path(out); (out / "in").mkdir(parents=True, exist_ok=True)
    entries = []
    for key in list_sim_keys(sim_dir):
        for seed in seeds:
            if not (Path(sim_dir) / f"{key}_s{seed}.json").exists():
                continue
            m = load_half_sim(sim_dir, key, seed).match
            ck = f"{key}_s{seed}"
            runs = []
            for i, (h, a, b) in enumerate(half_runs(m.half)):
                pc.write_half_input(m, a, b, out / "in" / f"{ck}_h{i}.in.npz")
                runs.append([i, h, a, b])
            entries.append({"cache_key": ck, "n_frames": int(m.n_frames), "input_sha": _match_sha(m), "runs": runs})
    if not entries:
        raise SystemExit(f"no sim halves for seeds {seeds} in {sim_dir}")
    man = {"sim": str(sim_dir), "ckpt": params.ckpt, "threads": params.threads, "wrapper_sha": pc._wrapper_sha(),
           "entries": entries}
    (out / "manifest.json").write_text(json.dumps(man, indent=1))
    (out / "run_all.sh").write_text(SERVER_RUNNER.format(threads=params.threads, ckpt=params.ckpt), newline="\n")
    n_in = sum(len(e["runs"]) for e in entries)
    print(f"prepared {n_in} half inputs for {len(entries)} matches in {out}")
    return man


def assemble(out, cache_dir, sim_dir=None, only=None):
    out = Path(out)
    man = json.loads((out / "manifest.json").read_text())
    params = pc.PathCRFParams(cache_dir=str(cache_dir), ckpt=int(man["ckpt"]), threads=int(man["threads"]))
    sim_dir = sim_dir or man["sim"]
    if man["wrapper_sha"] != pc._wrapper_sha():
        raise SystemExit("the wrapper script changed since prepare: refusing to assemble")
    summary = []
    entries = [e for e in man["entries"] if only is None or e["cache_key"] in only]
    for e in entries:
        ck = e["cache_key"]; key, seed = ck.rsplit("_s", 1)
        m = load_half_sim(sim_dir, key, int(seed)).match
        if _match_sha(m) != e["input_sha"] or m.n_frames != e["n_frames"]:
            raise SystemExit(f"{ck}: the sim on disk differs from the prepared inputs; rerun prepare")
        res = {k: np.full(m.n_frames, np.nan, np.float32) for k in ("p_home", "p_none", "imputed_frac")}
        for i, h, a, b in e["runs"]:
            op = out / "out" / f"{ck}_h{i}.out.npz"
            if not op.is_file():
                raise SystemExit(f"missing remote output {op}")
            for k, v in pc.read_half_output(op, a).items():
                res[k][a:b] = v
            with np.load(op) as z:
                summary.append({"cache_key": ck, "run": i, "n_chunks": int(z["n_chunks"]), "n_ok": int(z["n_ok"])})
        pc.write_cache(Path(cache_dir) / f"{ck}.npz", params, dict(input_sha=e["input_sha"], wrapper_sha=man["wrapper_sha"]), res)
    (out / "assemble_summary.json").write_text(json.dumps(summary, indent=1))
    print(f"assembled {len(entries)} cache files into {cache_dir}")
    return summary


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("prepare")
    p.add_argument("--sim", required=True); p.add_argument("--seeds", type=int, nargs="+", default=[100])
    p.add_argument("--out", required=True)
    p.add_argument("--threads", type=int, default=pc.PathCRFParams().threads)
    a = sub.add_parser("assemble")
    a.add_argument("--out", required=True); a.add_argument("--cache-dir", default=pc.DEFAULT_CACHE)
    a.add_argument("--sim", default=None, help="override the sim dir recorded in the manifest")
    a.add_argument("--only", nargs="+", default=None, help="assemble only these cache keys (e.g. game_9101_H1_s100)")
    args = ap.parse_args()
    if args.cmd == "prepare":
        params = pc.PathCRFParams(); params.threads = args.threads
        prepare(args.sim, args.seeds, args.out, params)
    else:
        assemble(args.out, args.cache_dir, args.sim, args.only)


if __name__ == "__main__":
    main()
