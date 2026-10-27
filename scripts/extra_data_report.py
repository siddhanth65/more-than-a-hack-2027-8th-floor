"""Aggregates (no positions) of the truth labels per source: DFL sims (9001-9006, never 9007), SkillCorner (91xx),
Metrica sample (92xx). Raw labels and the ctrl_Dfull truth variant (event-confirmed control).

Usage: OMP_NUM_THREADS=2 python -I scripts/extra_data_report.py --extra-dir runs/extra [--dfl DIR] [--out JSON]
"""
import site
import sys
from pathlib import Path

site.addsitedir(site.getusersitepackages())
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import argparse  # noqa: E402
import json  # noqa: E402

import numpy as np  # noqa: E402

from mtah.io.dfl_events import truth_variants  # noqa: E402
from mtah.io.dfl_proxy import list_halves, load_events, load_half  # noqa: E402
from mtah.io.extra_npz import aggregate  # noqa: E402
from mtah.sim.dataset import SEALED_MATCH, match_id  # noqa: E402

SOURCES = {"dfl": range(9001, 9100), "skillcorner": range(9101, 9200), "metrica_sample": range(9201, 9300)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--extra-dir", required=True)
    ap.add_argument("--dfl", default=r"E:\football-synthesizer\data\dfl_open\work\fp1d")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    items = []
    for d in (Path(args.dfl), Path(args.extra_dir)):
        if d.exists():
            items += [(k, p, d / "fp1d_census.json") for k, p in list_halves(d) if match_id(k) != SEALED_MATCH]
    acc = {s: {"raw": [], "ctrl_Dfull": []} for s in SOURCES}
    for key, p, census in items:
        src = next(s for s, r in SOURCES.items() if match_id(key) in r)
        m, labels = load_half(p, census)
        ev = load_events(p, census)
        with np.load(p) as z:
            vis = z["ball_visible"].astype(bool) if "ball_visible" in z.files else ~np.isnan(m.ball[:, 0])
        tv = truth_variants(labels, ev, m.ball, m.ball_speed, m.fps, m.pitch)
        acc[src]["raw"].append({"labels": labels, "ball_visible": vis})
        acc[src]["ctrl_Dfull"].append({"labels": tv["ctrl_Dfull"], "ball_visible": vis})
        print(key, "ok", flush=True)
    out = {s: {v: aggregate(h) for v, h in d.items()} for s, d in acc.items()}
    print(json.dumps(out, indent=1))
    if args.out:
        Path(args.out).write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
