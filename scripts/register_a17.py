"""Register a17 (= a14f configuration, PathCRF smoothing 1.0 s on the real feed) or a5c_fx2 (= a5c + smoothing) at the cell chosen
by calib_a17.py / calib_fx2.py. Writes runs/select/<name> (config.json, models) next to the base's.
  python scripts/register_a17.py --name a17 --base a14f --calib runs/kh/a17/calib_a17.json
  python scripts/register_a17.py --name a5c_fx2 --base-dir models/a5c --calib runs/kh/a5c_fx2/calib.json"""
import argparse, json, shutil
from pathlib import Path
MAIN = Path(__file__).resolve().parents[1]
ap = argparse.ArgumentParser()
ap.add_argument("--name", required=True); ap.add_argument("--base", default=None); ap.add_argument("--base-dir", default=None)
ap.add_argument("--calib", required=True); ap.add_argument("--smooth", type=float, default=1.0)
a = ap.parse_args()
src = Path(a.base_dir) if a.base_dir else MAIN / "runs" / "select" / a.base
ch = json.loads(Path(a.calib).read_text())["chosen"]
if ch is None:
    raise SystemExit("no admissible cell")
out = MAIN / "runs" / "select" / a.name; out.mkdir(parents=True, exist_ok=True)
for f in src.iterdir():
    if f.is_file() and f.name not in ("config.json", "registration.json") and not f.name.endswith((".drift.json", ".manifest.json", ".csv", "best.json")):
        shutil.copy2(f, out / f.name)
cfg = json.loads((src / "config.json").read_text())
cfg["dead_offset"] = float(ch["dead_offset"])
if "seq_stream" in cfg and cfg["seq_stream"]:
    cfg["seq_stream"]["d_offset"] = float(ch["seq_d_offset"]); cfg["seq_stream"]["input_dead_offset"] = -2.0
cfg["pathcrf"]["smooth_s"] = a.smooth
(out / "config.json").write_text(json.dumps(cfg, indent=1))
reg = {"candidate": a.name, "base": str(src), "pathcrf_smooth_s": a.smooth,
       "note": "simulator inputs unchanged (smooth_s 0 in the simulator caches); smoothing is a property of the real feed (runs/arb/FIX_NOTES.md)",
       "calibration": {"rule": "simulator-best (mean rank, 4 variants) admissible cell on the FIXED real evidence: visible-dead [0.116, 0.181], R/rho [2.28, 4.10]", "chosen": ch}}
(out / "registration.json").write_text(json.dumps(reg, indent=1, default=float))
print(json.dumps({"dead_offset": cfg["dead_offset"], "seq_d_offset": ch.get("seq_d_offset")}))
