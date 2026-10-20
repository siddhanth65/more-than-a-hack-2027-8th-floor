"""Baselines B0 / B4 / smooth50 on DFL-sim (fitted blackouts, 3 seeds) and on the public window."""
import json
from pathlib import Path

import numpy as np
import pandas as pd

from mtah import baselines
from mtah.eval.scorer import score, window_labels
from mtah.io.dfl_proxy import list_halves, load_half
from mtah.io.metrica_loader import load_tactical
from mtah.sim.blackout_sim import BlackoutParams, apply_blackouts, blackout_mask

ROOT = Path(__file__).resolve().parents[1]
DFL = Path(r"E:\football-synthesizer\data\dfl_open\work\fp1d")
DATA = Path(r"HOME/mtah-data")
GT = Path(r"WORKDIR/repo\evaluation\gt\BAR-ATH_possession_gt_public.csv")
BASELINES = {"b0": baselines.b0_nearest_hold, "b4": baselines.b4_blackout_dead,
             "smooth50": lambda m: baselines.smooth_vote(m, 50)}


def main():
    params = BlackoutParams.from_dict(json.loads((ROOT / "configs" / "blackout_sim.json").read_text()))
    rows = []
    for key, p in list_halves(DFL):
        m, labels = load_half(p, DFL / "fp1d_census.json")
        for seed in range(3):
            mask = blackout_mask(labels, m.ball, m.pitch, m.fps, params, np.random.default_rng(seed))
            sim = apply_blackouts(m, mask)
            for name, fn in BASELINES.items():
                r = score(labels, fn(sim))
                rows.append({"half": key, "seed": seed, "baseline": name, "spell_f1": r["spell_f1"],
                             "macro_f1": r["macro_f1"], "n_gt": r["n_gt_spells"], "n_pred": r["n_pred_spells"],
                             **{k: v for k, v in r["decomposition"].items() if k in ("false_splits", "false_merges")}})
        print(key, "done")
    (ROOT / "runs").mkdir(exist_ok=True)
    df = pd.DataFrame(rows); df.to_csv(ROOT / "runs" / "baselines_dfl.csv", index=False)
    print(df.groupby("baseline")[["spell_f1", "macro_f1", "n_gt", "n_pred", "false_splits", "false_merges"]].mean().round(3))

    m = load_tactical(DATA)
    gt = pd.read_csv(GT)["label"].to_numpy().astype("<U1")
    pub = {}
    for name, fn in BASELINES.items():
        r = score(gt, window_labels(fn(m), 11185, 18685))
        pub[name] = {k: r[k] for k in ("spell_f1", "macro_f1", "n_gt_spells", "n_pred_spells")}
    (ROOT / "runs" / "baselines_public.json").write_text(json.dumps(pub, indent=1))
    print(json.dumps(pub, indent=1))


if __name__ == "__main__":
    main()
