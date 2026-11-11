"""Register S (a17 with the cross-half self-trained stream B) as models/s.

  python scripts/register_s.py --ckpts <dir holding seq_modelB_h1.pt, seq_modelB_h2.pt> --dest <selection dir>            # offsets as a17
  python scripts/register_s.py --ckpts ... --dest models/s --calib runs/s/calib/calib_s.json                            # offsets from calib_s.py
The dead / team models, the decoder and the drift reference are a17's (copied unchanged); only stream B differs: one
fine-tuned checkpoint per half ("modelB": {"1": ..., "2": ...}), each trained without its own half's pseudo-labels.
"""
import argparse, json, shutil
from pathlib import Path

MAIN = Path(__file__).resolve().parents[1]
ap = argparse.ArgumentParser()
ap.add_argument("--a17", default=str(MAIN / "models" / "a17")); ap.add_argument("--ckpts", required=True)
ap.add_argument("--dest", required=True); ap.add_argument("--calib", default=None)
ap.add_argument("--lomo-summary", default=str(MAIN / "runs" / "s" / "lomo" / "summary.json"))
a = ap.parse_args()
src, dst, ck = Path(a.a17), Path(a.dest), Path(a.ckpts)
dst.mkdir(parents=True, exist_ok=True)
for f in ("dead_model.joblib", "team_model.joblib", "drift_reference.json"):
    shutil.copy2(src / f, dst / f)
for h in (1, 2):
    shutil.copy2(ck / f"seq_modelB_h{h}.pt", dst / f"seq_modelB_h{h}.pt")
cfg = json.loads((src / "config.json").read_text())
cfg["seq_stream"]["modelB"] = {"1": "seq_modelB_h1.pt", "2": "seq_modelB_h2.pt"}
reg = {"candidate": "s", "base": "models/a17",
       "definition": "a17 with stream B fine-tuned by cross-half self-training (variant V3, weight 0.5): the checkpoint predicting half h is "
                     "fine-tuned on the simulator training data plus pseudo-labels of the OTHER half only; public-window frames excluded "
                     "from every pseudo-label set (docs/reads/plan9-preregistration.md, amendments 3 and 3b)"}
if a.calib:
    ch = json.loads(Path(a.calib).read_text())["chosen"]
    if ch is None:
        raise SystemExit("no admissible cell")
    cfg["dead_offset"] = float(ch["dead_offset"]); cfg["seq_stream"]["d_offset"] = float(ch["seq_d_offset"])
    reg["calibration"] = {"rule": "simulator-best (mean rank, 4 variants, S's V3 predictions) admissible cell on the real evidence: visible-dead [0.116, 0.181], R/rho [2.28, 4.10]",
                          "chosen": ch}
    if Path(a.lomo_summary).exists():
        reg["lomo_acceptance_run"] = {k: {"ctrl_Dfull": v["ctrl_Dfull"], "raw_Dfull": v["raw_Dfull"]} for k, v in json.loads(Path(a.lomo_summary).read_text()).items()}
(dst / "config.json").write_text(json.dumps(cfg, indent=1))
(dst / "registration.json").write_text(json.dumps(reg, indent=1, default=float))
print(json.dumps({"dead_offset": cfg["dead_offset"], "seq_d_offset": cfg["seq_stream"]["d_offset"]}))
