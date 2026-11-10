"""Register candidate a14 = a13r + learned sequence stream (exp-seq2 B12, merge off) at the calibrated dead cell.

runs/select/a14/: a13r's config.json with dead_offset = the chosen cell's, merge off (merge_theta 1.0) and a
"seq_stream" block (mtah.evidence.seqstream); a13r's dead/team models and drift reference; the final learned model
(seq_modelB.pt: BiGRU, run in the PathCRF venv) and registration.json.
Usage: python scripts/register_a14.py [--dead-offset X --seq-d-offset Y]  (default: lomo/calib_B12.json "chosen")
"""
import argparse, json, shutil, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MAIN = Path(__file__).resolve().parents[1]
BASE = MAIN / "runs" / "select" / "a13r"
EXP = MAIN / "runs" / "select" / "exp_seq2"
OUT = MAIN / "runs" / "select" / "a14"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dead-offset", type=float); ap.add_argument("--seq-d-offset", type=float)
    a = ap.parse_args()
    cal = json.loads((EXP / "lomo" / "calib_B12.json").read_text())
    ch = cal["chosen"]
    if a.dead_offset is not None:
        ch = next(r for r in cal["table"] if r["dead_offset"] == a.dead_offset and r["seq_d_offset"] == a.seq_d_offset)
    if ch is None or not ch["admissible"]:
        raise SystemExit("no admissible calibration cell chosen")
    fin = json.loads((EXP / "final" / "config.json").read_text())
    if fin["variant"] != "B12/off":
        raise SystemExit(f"final config is {fin['variant']}, expected B12/off")
    cfg = json.loads((BASE / "config.json").read_text())
    cfg["dead_offset"] = float(ch["dead_offset"]); cfg["post"]["merge_theta"] = 1.0
    cfg["seq_stream"] = {"weights": fin["stream_weights"], "lengths": fin["stream_lengths"],
                         "d_offset": float(ch["seq_d_offset"]), "modelB": "seq_modelB.pt",
                         "step_features": fin["feature_spec"]["step_features"]}
    OUT.mkdir(parents=True, exist_ok=True)
    for f in ("dead_model.joblib", "team_model.joblib", "drift_reference.json"):
        shutil.copy2(BASE / f, OUT / f)
    shutil.copy2(EXP / "final" / "modelB.pt", OUT / "seq_modelB.pt")
    (OUT / "config.json").write_text(json.dumps(cfg, indent=1))
    reg = {"candidate": "a14", "base": "a13r (runs/select/a13r: a13's cell refit under sklearn 1.9.1; config, dead + team models)",
           "learned_streams": {"variant": fin["variant"], "weights": fin["stream_weights"],
                               "feature_spec": fin["feature_spec"], "training": fin["training"],
                               "modelB": {k: v for k, v in json.loads((EXP / "final" / "modelB.json").read_text()).items()
                                          if k != "hist"},
                               "envs": {"modelB": "PathCRF venv via subprocess (mtah/evidence/seq_b_infer.py)"}},
           "merge": "off (merge_theta 1.0)",
           "adversarial": json.loads((EXP / "adversarial.json").read_text()),
           "calibration": {"rule": "dead_offset x seq_d_offset on 6-fold LOMO (decoding only) + real-match label-free "
                                   "bands (visible-dead 0.10-0.18, spells/graded min 2.7-4.3); best mean rank among "
                                   "admissible cells", "chosen": ch, "lomo_6fold_uncalibrated": fin["lomo_6fold"]}}
    (OUT / "registration.json").write_text(json.dumps(reg, indent=1, default=float))
    print(json.dumps({"dead_offset": cfg["dead_offset"], "seq_d_offset": cfg["seq_stream"]["d_offset"]}))


if __name__ == "__main__":
    main()
