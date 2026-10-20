"""Run the selected pipeline on the real match; write the candidate CSV, drift report and manifest."""
import argparse, hashlib, json, os, subprocess, sys
from pathlib import Path

import numpy
import numpy as np
import pandas
import sklearn

from mtah import N_FRAMES
from mtah.eval.drift import blackout_mask_from_match, drift_report
from mtah.evidence.dead import DeadModel
from mtah.io.metrica_loader import STEMS, load_tactical
from mtah.io.tags import load_tags
from mtah.pipeline import PipelineConfig, compute_evidence, predict
from mtah.submission import validate, write_submission

ROOT = Path(__file__).resolve().parents[1]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--select", required=True); ap.add_argument("--name", required=True)
    ap.add_argument("--data", default=os.environ.get("MTAH_DATA_DIR", r"HOME/mtah-data"))
    args = ap.parse_args()
    sel = Path(args.select)
    raw = json.loads((sel / "config.json").read_text()); lengths = raw.pop("lengths")
    cfg = PipelineConfig.from_dict(raw); cfg.decoder.lengths = lengths
    dead = DeadModel.load(sel / "dead_model.joblib")
    d = Path(args.data) / "tactical"
    md5s = {n: hashlib.md5((d / f"{STEMS['tactical']}{suf}").read_bytes()).hexdigest()
            for n, suf in (("atd", "_atd.txt"), ("meta", "_atd.xml"), ("tags", "_smart_tagging.xml"), ("homography", "_homography.csv"))}
    m = load_tactical(args.data); tags = load_tags(args.data)
    ev = compute_evidence(m, tags, cfg, dead)
    labels = predict(m, tags, cfg, dead, evidence=ev)
    out = ROOT / "runs" / "candidates"; out.mkdir(parents=True, exist_ok=True)
    p = Path(write_submission(labels, out / f"{args.name}.csv")); reason = validate(p)
    if reason is not None:
        print(f"INVALID submission {p}: {reason}")
        sys.exit(1)
    mask = blackout_mask_from_match(m)
    drift = drift_report(labels, m, tags, mask)
    ref_path = sel / "drift_reference.json"
    ref = json.loads(ref_path.read_text()) if ref_path.exists() else None
    if ref is None:
        print(f"WARNING: no drift_reference.json in {sel}; drift not compared")
        oor = []
    else:
        # n_spells is an absolute count (full match vs single sim halves); only rate-style metrics are compared
        oor = [k for k, v in drift.items() if k != "n_spells" and k in ref and not np.isnan(v)
               and (v < ref[k]["min"] or v > ref[k]["max"])]
        print("out_of_range:", oor)
    drift = {"candidate": drift, "reference": ref, "out_of_range": oor}
    (out / f"{args.name}.drift.json").write_text(json.dumps(drift, indent=1))
    manifest = {"git": subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT).decode().strip(),
                "git_dirty": bool(subprocess.check_output(["git", "status", "--porcelain"], cwd=ROOT).strip()),
                "select_dir": str(sel), "config": {**cfg.to_dict(), "lengths": lengths}, "data_md5": md5s,
                "versions": {"numpy": numpy.__version__, "pandas": pandas.__version__, "sklearn": sklearn.__version__, "python": sys.version.split()[0]}}
    (out / f"{args.name}.manifest.json").write_text(json.dumps(manifest, indent=1))
    counts = {k: int((labels == k).sum()) for k in "HAD"}
    print(f"wrote {p}  validation: OK  counts {counts}")
    print(json.dumps(drift, indent=1))


if __name__ == "__main__":
    main()
