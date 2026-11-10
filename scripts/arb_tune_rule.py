"""Tune the confident-ball-control pseudo-label rule on the simulator selection folds 9001-9006 ONLY (true labels
exist there; sealed 9007 is never loaded). Target: precision >= 0.97 against the true team label; then maximise coverage.

Truth used for the criterion: raw_Dfull (the sim's ball-owner label, the analogue of the organisers' "corrected
tracking ball owner" ground truth). It is STRICT: a labelled frame whose true label is D counts as wrong (the rule
asserts live control). ctrl_Dfull (event-confirmed re-anchoring) disagrees with the nearest player on ~25 % of
near-player frames by construction, so 0.97 against it is unreachable; its precision is reported alongside.
Writes runs/arb/rule.json (frozen parameters + per-fold diagnostics) and runs/arb/rule_grid.csv.

  python scripts/arb_tune_rule.py --sim runs/sim --seeds 100 101 --out runs/arb
"""
import argparse, json, sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import arb_common as ac
from mtah.sim.dataset import SEALED_MATCH, list_sim_keys, load_half_sim, match_id


def stats(halves, p):
    t = {k: 0 for k in ("n", "ok_raw", "ok_ctrl", "D_raw", "nHA_ctrl", "vis", "H", "A")}; per = {}
    for k, s, m, tc, tr, geom in halves:
        lab = ac.confident_control(m, p, geom); i = lab >= 0; tl = np.where(lab[i] == 0, "H", "A")
        n = int(i.sum()); okr = int((tr[i] == tl).sum()); okc = int((tc[i] == tl).sum())
        for kk, v in (("n", n), ("ok_raw", okr), ("ok_ctrl", okc), ("D_raw", int((tr[i] == "D").sum())),
                      ("nHA_ctrl", int(np.isin(tc[i], ("H", "A")).sum())), ("vis", int((m.ball_present & m.graded).sum())),
                      ("H", int((tl == "H").sum())), ("A", int((tl == "A").sum()))):
            t[kk] += v
        d = per.setdefault(match_id(k), [0, 0]); d[0] += n; d[1] += okr
    return t, per


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sim", default="runs/sim"); ap.add_argument("--seeds", type=int, nargs="+", default=[100, 101])
    ap.add_argument("--out", default="runs/arb"); ap.add_argument("--target", type=float, default=0.97)
    a = ap.parse_args()
    halves = []
    for k in [k for k in list_sim_keys(a.sim) if match_id(k) != SEALED_MATCH]:
        for s in a.seeds:
            if (Path(a.sim) / f"{k}_s{s}.json").exists():
                h = load_half_sim(a.sim, k, s)
                halves.append((k, s, h.match, h.truth["ctrl_Dfull"], h.truth["raw_Dfull"], ac.frame_geometry(h.match)))
    print("halves", len(halves), flush=True)
    grid = [dict(r_near=rn, r_opp=ro, v_rel=v, v_ball_min=vb, min_run_s=mr)
            for rn in (1.0, 1.5, 2.0) for ro in (3.0, 4.0, 5.0, 6.0) for v in (1.0, 2.0, 3.0, 5.0)
            for vb in (0.0, 1.0, 2.0, 3.0) for mr in (1.0, 2.0)]
    rows = []
    for g in grid:
        t, per = stats(halves, ac.RuleParams(**g)); n = max(t["n"], 1)
        fp = [d[1] / d[0] for d in per.values() if d[0] >= 50]
        rows.append({**g, "n": t["n"], "prec_raw_strict": t["ok_raw"] / n, "D_share": t["D_raw"] / n,
                     "prec_ctrl_strict": t["ok_ctrl"] / n, "prec_ctrl_ignoringD": t["ok_ctrl"] / max(t["nHA_ctrl"], 1),
                     "cov_visible": t["n"] / t["vis"], "min_fold_prec_raw": float(min(fp)) if fp else np.nan})
    df = pd.DataFrame(rows)
    Path(a.out).mkdir(parents=True, exist_ok=True); df.to_csv(Path(a.out) / "rule_grid.csv", index=False)
    adm = df[(df["prec_raw_strict"] >= a.target) & (df["min_fold_prec_raw"] >= a.target - 0.02) & (df["n"] >= 2000)]
    if adm.empty:
        raise SystemExit("no admissible cell: loosen the target or the grid")
    best = adm.sort_values(["cov_visible", "prec_raw_strict"], ascending=False).iloc[0]
    p = ac.RuleParams(**{k: float(best[k]) for k in ("r_near", "r_opp", "v_rel", "v_ball_min", "min_run_s")})
    t, per = stats(halves, p)
    diag = {}
    for k, s, m, tc, tr, geom in halves:
        lab = ac.confident_control(m, p, geom); i = lab >= 0; tl = np.where(lab[i] == 0, "H", "A")
        d = diag.setdefault(str(match_id(k)), {"n": 0, "ok_raw": 0, "H": 0, "A": 0, "trueD_raw": 0, "graded_visible": 0, "ok_ctrl": 0})
        d["n"] += int(i.sum()); d["ok_raw"] += int((tr[i] == tl).sum()); d["ok_ctrl"] += int((tc[i] == tl).sum())
        d["H"] += int((tl == "H").sum()); d["A"] += int((tl == "A").sum()); d["trueD_raw"] += int((tr[i] == "D").sum())
        d["graded_visible"] += int((m.ball_present & m.graded).sum())
    out = {"rule": p.to_dict(), "criterion": "strict precision vs raw_Dfull >= target, pooled and each fold >= target-0.02; max coverage",
           "target": a.target, "seeds": a.seeds, "pooled": {k: (float(best[k]) if k != "n" else int(best[k])) for k in df.columns if k not in ("r_near", "r_opp", "v_rel", "v_ball_min", "min_run_s")},
           "class_balance_H_share": t["H"] / max(t["H"] + t["A"], 1), "per_fold": diag, "n_admissible": int(len(adm))}
    (Path(a.out) / "rule.json").write_text(json.dumps(out, indent=1))
    print(json.dumps({k: v for k, v in out.items() if k != "per_fold"}, indent=1))


if __name__ == "__main__":
    main()
