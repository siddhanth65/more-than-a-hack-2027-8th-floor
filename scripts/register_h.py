"""Register H = a14 with the learned-stream weight halved (seqB 12 -> 6) at the dead offsets chosen by calib_h.py
(simulator-best admissible cell under the new bands). Writes runs/select/H (config.json etc.) next to a14's.
Inputs of the BiGRU stay at a13r's dead cell (input_dead_offset -2.0, the cell the model was trained and calibrated at).
  python scripts/register_h.py [--name H]   (then: scripts/make_candidate.py --select runs/select/H --name H)"""
import argparse, json, shutil
from pathlib import Path
MAIN = Path(r"WORKDIR/possession")
KH = Path(__file__).resolve().parents[1] / "runs" / "kh" / "H"
ap = argparse.ArgumentParser(); ap.add_argument("--name", default="H"); a = ap.parse_args()
ch = json.loads((KH / "calib_B6.json").read_text())["chosen"]
if ch is None:
    raise SystemExit("no admissible calibration cell")
src, out = MAIN / "runs" / "select" / "a14", MAIN / "runs" / "select" / a.name
out.mkdir(parents=True, exist_ok=True)
for f in ("dead_model.joblib", "team_model.joblib", "drift_reference.json", "seq_modelB.pt"):
    shutil.copy2(src / f, out / f)
cfg = json.loads((src / "config.json").read_text())
cfg["dead_offset"] = float(ch["dead_offset"])
s = cfg["seq_stream"]; s["weights"] = {"seqB": 6.0}; s["d_offset"] = float(ch["seq_d_offset"]); s["input_dead_offset"] = -2.0
(out / "config.json").write_text(json.dumps(cfg, indent=1))
reg = {"candidate": a.name, "base": "a14 (runs/select/a14: a13r base + BiGRU B, merge off), learned weight x0.5 (seqB 12 -> 6)",
       "calibration": {"rule": "simulator-best admissible (dead_offset, seq_d_offset): visible-dead in [0.116, 0.181] and "
                               "R / rho in [2.28, 4.10] (rho = ctrl_Dfull n_pred / n_gt, 6-fold LOMO, same cell)",
                       "chosen": ch}, "input_dead_offset": -2.0}
(out / "registration.json").write_text(json.dumps(reg, indent=1, default=float))
print(json.dumps({"dead_offset": cfg["dead_offset"], "seq_d_offset": s["d_offset"]}))
