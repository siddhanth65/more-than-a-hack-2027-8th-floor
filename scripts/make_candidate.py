"""Run the selected pipeline on the real match; write the candidate CSV, drift report and manifest."""
import argparse, hashlib, json, os, subprocess, sys
from pathlib import Path

import numpy as np

from mtah import N_FRAMES
from mtah.eval.drift import drift_report
from mtah.evidence.dead import DeadModel
from mtah.io.metrica_loader import STEMS, load_tactical
from mtah.io.tags import load_tags
from mtah.pipeline import PipelineConfig, compute_evidence, predict
from mtah.sim.blackout_sim import runs_of
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
    m = load_tactical(args.data); tags = load_tags(args.data)
    ev = compute_evidence(m, tags, cfg, dead)
    labels = predict(m, tags, cfg, dead, evidence=ev)
    out = ROOT / "runs" / "candidates"; out.mkdir(parents=True, exist_ok=True)
    p = write_submission(labels, out / f"{args.name}.csv"); reason = validate(p)
    mask = np.zeros(N_FRAMES, bool)
    for s, e in runs_of(~m.ball_present & m.graded):
        if e - s > 10 * m.fps:
            mask[s:e] = True
    drift = drift_report(labels, m, tags, mask)
    (out / f"{args.name}.drift.json").write_text(json.dumps(drift, indent=1))
    d = Path(args.data) / "tactical"
    md5s = {n: hashlib.md5((d / f"{STEMS['tactical']}{suf}").read_bytes()).hexdigest()
            for n, suf in (("atd", "_atd.txt"), ("meta", "_atd.xml"), ("tags", "_smart_tagging.xml"))}
    import numpy, pandas, sklearn
    manifest = {"git": subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT).decode().strip(),
                "select_dir": str(sel), "config": {**cfg.to_dict(), "lengths": lengths}, "data_md5": md5s,
                "versions": {"numpy": numpy.__version__, "pandas": pandas.__version__, "sklearn": sklearn.__version__, "python": sys.version.split()[0]}}
    (out / f"{args.name}.manifest.json").write_text(json.dumps(manifest, indent=1))
    counts = {k: int((labels == k).sum()) for k in "HAD"}
    print(f"wrote {p}  validation: {'OK' if reason is None else reason}  counts {counts}")
    print(json.dumps(drift, indent=1))


if __name__ == "__main__":
    main()
