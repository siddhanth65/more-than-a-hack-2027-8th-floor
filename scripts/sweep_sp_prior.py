"""Plan 2f T1: label-free real-match sweep of the type-specific, ramped pre-restart dead prior (sp_dead_bonus) on the
registered a5c pipeline, plus public-window SHARES (no scoring): truth visible-dead frames predicted D and the predicted
D share on non-blackout window frames. Writes <select>/sp_prior_sweep.json. Aggregates only."""
import argparse, copy, json, os, subprocess, sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from make_candidate import candidate_labels, load_selection   # noqa: E402

from mtah.eval.drift import blackout_mask_from_match, drift_report   # noqa: E402
from mtah.eval.official import possession_metrics as pm              # noqa: E402
from mtah.io.metrica_loader import load_tactical                     # noqa: E402
from mtah.io.tags import load_tags                                   # noqa: E402
from mtah.sim.blackout_sim import runs_of                            # noqa: E402

GT = Path(r"WORKDIR/repo\evaluation\gt\BAR-ATH_possession_gt_public.csv")
WINDOW = (11185, 18685)
KEYS = ("non_blackout_D_share", "spells_per_graded_min", "tag_agreement_implied", "pred_D_share", "blackout_D_share",
        "n_spells")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--select", default=str(ROOT / "runs" / "select" / "a5c"))
    ap.add_argument("--data", default=os.environ.get("MTAH_DATA_DIR") or str(Path.home() / "mtah-data"))
    ap.add_argument("--bonus", type=float, nargs="+", default=[0.0, 2.0, 4.0, 6.0])
    args = ap.parse_args()
    sel = Path(args.select)
    cfg0, dead, lengths, team, merge, switch = load_selection(sel)
    m = load_tactical(args.data); tags = load_tags(args.data)
    mask = blackout_mask_from_match(m)
    gt = pm.read_label_csv(GT)["label"].to_numpy().astype("<U1")
    win = np.isin(gt, ("H", "A", "D"))
    assert len(gt) == m.n_frames and win.sum() == WINDOW[1] - WINDOW[0] and win[WINDOW[0]] and win[WINDOW[1] - 1]
    vis_dead = win & (gt == "D") & m.ball_present
    nb = win & ~mask & m.graded
    runs = [(s, e) for s, e in runs_of(vis_dead)]
    rows = []
    for b in args.bonus:
        cfg = copy.deepcopy(cfg0); cfg.tag.sp_dead_bonus = float(b)
        labels = candidate_labels(m, tags, cfg, dead, team, merge, switch)
        d = drift_report(labels, m, tags, mask, code_table=cfg.tag.code_table)
        p = labels == "D"
        row = {"sp_dead_bonus": b, **{k: d[k] for k in KEYS},
               "window_truth_visible_dead_caught": float(p[vis_dead].mean()),
               "window_non_blackout_D_share": float(p[nb].mean()),
               "window_D_share": float(p[win].mean()),
               "window_visible_dead_runs": [{"len_s": round((e - s) / m.fps, 1), "caught": round(float(p[s:e].mean()), 2)}
                                            for s, e in runs]}
        rows.append(row)
        print(json.dumps({k: (round(v, 4) if isinstance(v, float) else v) for k, v in row.items()
                          if k != "window_visible_dead_runs"}), flush=True)
    out = {"git": subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT).decode().strip(),
           "select_dir": str(sel), "tag_params": {k: v for k, v in cfg0.tag.__dict__.items() if k != "code_table"},
           "window_truth": {"visible_dead_frames": int(vis_dead.sum()), "non_blackout_frames": int(nb.sum()),
                            "non_blackout_D_share": float((gt[nb] == "D").mean())},
           "note": "label-free real-match drift + public-window shares only (no scoring)", "rows": rows}
    (sel / "sp_prior_sweep.json").write_text(json.dumps(out, indent=1))
    print("wrote", sel / "sp_prior_sweep.json")


if __name__ == "__main__":
    main()
