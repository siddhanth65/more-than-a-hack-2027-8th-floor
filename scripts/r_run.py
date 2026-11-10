"""Task R: type-specific pre-restart dead prior fitted on SkillCorner only (plan9 Amendment 1).

Synthetic set-piece tags on the K halves: SkillCorner typed restart events (runs/kh/R/restarts.csv) at the restart frame r,
thinned per type to the contest match's tag rate (runs/kh/R/contest_tag_stats.json), tag start = r - delta with delta drawn
from the contest's tag-start -> candidate D->live-edge offset distribution (pooled quantiles in that file), duration 35 s
(5 s for kick-offs), team = taking team for corner / free kick / goal kick, none for throw-in / kick-off, as in the contest
tags. They REPLACE the simulated SET PIECES rows of the half (other tag rows kept). Deterministic per half key.

R spec = {"name", "window": null | "p25" | "median", "bonus", "hard"}; windows are SkillCorner dead time before the
restart minus the median tag lead (so the window is measured from the tag start), per type, from the FIT matches
(9101-9110) only. hard = flat window with a bonus large enough to force D.

  python scripts/r_run.py --sim DIR --cache DIR --out DIR --sel NAME=DIR... --specs JSON --group fit|test --part I --of N
"""
import argparse, json, sys, zlib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "scripts"))

import numpy as np
import pandas as pd

import mtah.evidence.pathcrf as pcrf
from mtah.eval.scorer import score
from mtah.sim.dataset import list_sim_keys, load_half_sim, match_id
from make_candidate import candidate_labels, load_selection

R = ROOT / "runs" / "kh" / "R"
TAGNAME = {"throw_in": "Throw-in", "free_kick": "Free Kick", "goal_kick": "Goal Kick", "corner": "Corner Kick", "goal": "Kick Off (After Goal)"}
# prior key (tag_prior._sp_before_s: lower-case type prefix) per restart type
PRIORKEY = {"throw_in": "throw", "free_kick": "free", "goal_kick": "goal", "corner": "corner", "goal": "kick"}
SC_MIN = 1970.4   # SkillCorner minutes (40 halves, runs/extra/report.json)
FIT = range(9101, 9111); TEST = range(9111, 9121)
DELTA_Q = [(0.0, 2.5), (0.10, 3.32), (0.25, 4.02), (0.50, 4.5), (0.75, 5.98), (0.90, 7.34), (1.0, 9.0)]   # pooled contest offsets (s)
HARD_BONUS = 40.0


def lead_median():
    return 4.5


def windows(restarts, matches, q):
    """{prior key: seconds before the TAG start} from the fit matches: quantile q of dead time before the restart minus the lead."""
    d = restarts[restarts.key.map(match_id).isin(list(matches))]
    out = {}
    for ty, k in PRIORKEY.items():
        x = d[d.type == ty].dead_s.to_numpy()
        out[k] = float(max(np.quantile(x, q) - lead_median(), 2.0))
    return out


def synth_tags(tags, key, restarts, rates):
    rng = np.random.default_rng(zlib.crc32(key.encode()))
    keep = tags[tags.code != "SET PIECES"]
    rows = []
    p_u, p_d = [p for p, _ in DELTA_Q], [d for _, d in DELTA_Q]
    for r in restarts[restarts.key == key].itertuples():
        thin = min(1.0, rates["contest"][r.type] / rates["sc"][r.type])
        if rng.random() >= thin:
            continue
        delta = float(np.interp(rng.random(), p_u, p_d))
        s = int(r.frame - round(delta * 25)); dur = 5 if r.type == "goal" else 35
        team = r.team if r.type in ("free_kick", "goal_kick", "corner") else None
        rows.append({"start": s, "end": s + dur * 25, "code": "SET PIECES", "team": team, "type": TAGNAME[r.type], "half": None})
    syn = pd.DataFrame(rows, columns=list(tags.columns))
    for c in ("team", "type", "half"):
        syn[c] = syn[c].astype(object)
    out = pd.concat([keep, syn], ignore_index=True)
    return out.sort_values("start", kind="stable").reset_index(drop=True)


def apply_spec(cfg, spec, restarts):
    """cfg copy with the R prior set (spec window None = no prior)."""
    import copy
    cfg = copy.deepcopy(cfg)
    if spec.get("window") is None:
        cfg.tag.sp_dead_bonus = 0.0
        return cfg
    q = {"p25": 0.25, "median": 0.5}[spec["window"]]
    cfg.tag.sp_dead_before_by_type = windows(restarts, FIT, q)
    cfg.tag.sp_dead_bonus = HARD_BONUS if spec.get("hard") else float(spec["bonus"])
    cfg.tag.sp_dead_ramp = not spec.get("hard", False)
    return cfg


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sim", required=True); ap.add_argument("--cache", required=True); ap.add_argument("--out", required=True)
    ap.add_argument("--sel", nargs="+", required=True); ap.add_argument("--specs", required=True)
    ap.add_argument("--group", choices=("fit", "test", "all"), default="all"); ap.add_argument("--seed", type=int, default=100)
    ap.add_argument("--part", type=int, default=0); ap.add_argument("--of", type=int, default=1)
    ap.add_argument("--restarts", default=str(R / "restarts.csv")); ap.add_argument("--stats", default=str(R / "contest_tag_stats.json"))
    a = ap.parse_args()
    pcrf._run_half = lambda *x, **k: (_ for _ in ()).throw(RuntimeError("PathCRF cache miss"))
    restarts = pd.read_csv(a.restarts); stats = json.loads(Path(a.stats).read_text())
    rates = {"contest": {k: stats["tag_rate_per_graded_min"][k] for k in TAGNAME},
             "sc": {k: float((restarts.type == k).sum() / SC_MIN) for k in TAGNAME}}
    specs = json.loads(a.specs)
    cands = {}
    for s in a.sel:
        n, d = s.split("=", 1); cands[n] = (load_selection(d, with_seq=True), Path(d))
    grp = {"fit": FIT, "test": TEST, "all": range(9101, 9121)}[a.group]
    keys = [k for k in list_sim_keys(a.sim) if match_id(k) in grp][a.part::a.of]
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True); rows = []
    for k in keys:
        h = load_half_sim(a.sim, k, a.seed); tags = synth_tags(h.tags, k, restarts, rates)
        for name, ((cfg0, dead, lengths, team, merge, switch, seq), sd) in cands.items():
            for sp in specs:
                cfg = apply_spec(cfg0, sp, restarts); cfg.pathcrf.cache_dir = a.cache
                lab = candidate_labels(h.match, tags, cfg, dead, team, merge, switch, cache_key=f"{k}_s{a.seed}", seq=seq, sel_dir=sd)
                for var in ("ctrl_Dfull", "raw_Dfull"):
                    r = score(h.truth[var], lab); g = h.truth[var] != "X"
                    rows.append({"cand": name, "spec": sp["name"], "key": k, "variant": var, "spell_f1": r["spell_f1"], "macro_f1": r["macro_f1"],
                                 "n_gt": r["n_gt_spells"], "n_pred": r["n_pred_spells"], "pred_D": float((lab[g] == "D").mean())})
        print(k, flush=True)
        pd.DataFrame(rows).to_csv(out / f"r_part{a.part}.csv", index=False)
    print("DONE", a.part)


if __name__ == "__main__":
    main()
