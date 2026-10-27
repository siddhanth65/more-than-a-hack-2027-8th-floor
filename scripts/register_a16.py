"""Register candidate a16 = a13r + learned stream v1 (per-half-normalised BiGRU inputs, final model on all six matches),
merge off, at the calibrated cell chosen by scripts/calib_v1.py (runs/select/exp_seq3/calib_v1/chosen.json).
Writes <worktree>/runs/select/a16/ (config.json, a13r dead/team models + drift reference, seq_modelB.pt, registration.json)."""
import json, shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MAIN = Path(r"WORKDIR/possession")
BASE = MAIN / "runs" / "select" / "a13r"; E3 = MAIN / "runs" / "select" / "exp_seq3"; OUT = ROOT / "runs" / "select" / "a16"


def main():
    ch = json.loads((E3 / "calib_v1" / "chosen.json").read_text())["chosen"]
    assert ch["merge"] == "off"
    a14 = json.loads((MAIN / "runs" / "select" / "a14" / "config.json").read_text())
    cfg = json.loads((BASE / "config.json").read_text())
    cfg["dead_offset"] = float(ch["dead_offset"]); cfg["post"]["merge_theta"] = 1.0
    cfg["seq_stream"] = {**a14["seq_stream"], "d_offset": float(ch["seq_d_offset"]),
                        "input_dead_offset": -2.0}   # features as trained/calibrated (a13r cell); a14 lacks this (inputs at its -3.0)
    OUT.mkdir(parents=True, exist_ok=True)
    for f in ("dead_model.joblib", "team_model.joblib", "drift_reference.json"):
        shutil.copy2(BASE / f, OUT / f)
    shutil.copy2(E3 / "v1" / "model.pt", OUT / "seq_modelB.pt")
    (OUT / "config.json").write_text(json.dumps(cfg, indent=1))
    fin = json.loads((E3 / "v1" / "final.json").read_text()); adm = json.loads(Path(E3 / "calib_v1" / "chosen.json").read_text())
    reg = {"candidate": "a16", "base": "a13r", "learned_stream": {"variant": "exp_seq3 v1 (per-half robust input normalisation, "
           "mtah/evidence/seq_b_infer.py normalise_v1)", "final": fin, "weights": cfg["seq_stream"]["weights"]},
           "merge": "off (merge_theta 1.0); th0.7 was inadmissible on spells/min (2.4-2.5 < 2.7) in every cell",
           "calibration": {"script": "scripts/calib_v1.py", "grid": "dead {-3..-5 by .5} x seq_d {-1..-4} x merge {off, th0.7}",
                           "rule": "real visible-dead 0.10-0.18 AND spells/graded min 2.7-4.3; best 6-fold mean rank over the four spellF1 variants",
                           "chosen": ch}}
    (OUT / "registration.json").write_text(json.dumps(reg, indent=1, default=float))
    print(json.dumps({"dead_offset": cfg["dead_offset"], "d_offset": cfg["seq_stream"]["d_offset"]}))


if __name__ == "__main__":
    main()
