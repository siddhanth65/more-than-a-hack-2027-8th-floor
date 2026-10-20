"""Tracked look-budget logs: JSON lines in docs/reads/<kind>.jsonl (aggregates only)."""
import json
from pathlib import Path

READS_DIR = Path(__file__).resolve().parents[2] / "docs" / "reads"
KINDS = ("sealed", "public")


def _path(kind):
    if kind not in KINDS:
        raise ValueError(f"unknown read kind {kind!r}; expected one of {KINDS}")
    return Path(READS_DIR) / f"{kind}.jsonl"


def read_log(kind):
    p = _path(kind)
    if not p.exists():
        return []
    return [json.loads(l) for l in p.read_text(encoding="utf-8").splitlines() if l.strip()]


def append_read(kind, record):
    p = _path(kind); p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "a", encoding="utf-8", newline="\n") as f:
        f.write(json.dumps(record) + "\n")
    return p
