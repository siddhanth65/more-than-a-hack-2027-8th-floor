"""Plan 3r: the pre-registered real-match sweep of the broadcast replay stream (docs/reads/plan3r-preregistration.md).

Base = candidate a5c (runs/select/a5c). For w in {0.5, 1.0, 1.5, 2.0} x dead_offset in {-2.0, -2.5, -3.0}, plus the
a5c baseline (w 0, offset -2.0), decode the real match and record label-free aggregates only:
  D_share          drift_report non_blackout_D_share (visible-ball D share)
  spell_rate       drift_report spells_per_graded_min
  tag_agree        drift_report tag_agreement_implied
  prerestart_D     share of graded visible-ball frames in the 30 s before Goal Kick / Corner Kick / Kick Off
                   SET PIECES tag starts that are labelled D
  progression_D    share of graded visible-ball frames between 25 % and 75 % of each PROGRESSION tag's span labelled D
Admissible (rule 3): D_share in [0.10, 0.18]; spell_rate in [2.7, 4.3]; progression_D <= a5c + 0.02;
tag_agree >= a5c - 0.01. Choice (rule 4): highest prerestart_D among admissible w > 0 settings; ties -> smaller w
(then the dead_offset closest to a5c's). None admissible -> REJECTED (a5c stays final).
The public window and no ground truth are read. Writes runs/select/a5c/replay_sweep.json (aggregates only)."""
import argparse, json, os, sys, time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from make_candidate import load_selection   # noqa: E402

from mtah.eval.drift import blackout_mask_from_match, drift_report   # noqa: E402
from mtah.evidence.dead import dead_evidence, dead_table   # noqa: E402
from mtah.evidence.replay import replay_cue, replay_evidence   # noqa: E402
from mtah.io.metrica_loader import load_tactical   # noqa: E402
from mtah.io.tags import load_tags   # noqa: E402
from mtah.pipeline import compute_evidence, compute_strong, predict   # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
WS = (0.5, 1.0, 1.5, 2.0); OFFS = (-2.0, -2.5, -3.0)
PRE_S = 30; PRE_TYPES = ("Goal Kick", "Corner Kick", "Kick Off")
D_BAND = (0.10, 0.18); RATE_BAND = (2.7, 4.3); PROG_TOL = 0.02; AGREE_TOL = 0.01


def masks(m, tags):
    F = m.n_frames; vis = m.ball_present & m.graded
    pre = np.zeros(F, bool)
    sp = tags[tags.code == "SET PIECES"]
    typ = sp["type"].fillna("")
    for s in sp[typ.str.startswith(PRE_TYPES)]["start"]:
        a = max(0, int(s) - int(round(PRE_S * m.fps))); pre[a:max(a, int(s))] = True
    prog = np.zeros(F, bool)
    for r in tags[tags.code == "PROGRESSION"].itertuples(index=False):
        q = (int(r.end) - int(r.start)) // 4; prog[int(r.start) + q:int(r.end) - q] = True
    return pre & vis, prog & vis


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--select", default=str(ROOT / "runs" / "select" / "a5c"))
    ap.add_argument("--data", default=os.environ.get("MTAH_DATA_DIR"))
    args = ap.parse_args()
    if not args.data:
        raise SystemExit("data dir needed: pass --data or set MTAH_DATA_DIR")
    T0 = time.time()
    cfg, dead, lengths, team, merge, switch = load_selection(args.select)
    if cfg.use_replay:
        raise SystemExit("base selection already uses the replay stream")
    m = load_tactical(args.data); tags = load_tags(args.data)
    cue = replay_cue(args.data, m.n_frames)
    if cue is None:
        raise SystemExit("no broadcast replay cue available")
    base_ev = compute_evidence(m, tags, cfg, dead, cache_key="real_tactical", team_model=team, switch_model=switch)
    p_dead = dead.predict_proba(dead_table(m, tags, dead.policy))
    strong = compute_strong(m, cfg)
    blk = blackout_mask_from_match(m); pre, prog = masks(m, tags)
    print(f"evidence ready {time.time() - T0:.0f}s; pre-restart frames {int(pre.sum())}, PROGRESSION mid frames {int(prog.sum())}",
          flush=True)

    rows = []
    for w, off in [(0.0, cfg.dead_offset)] + [(w, o) for w in WS for o in OFFS]:
        ev = dict(base_ev)
        ev["dead"] = dead_evidence(p_dead, logit_offset=off)
        if w > 0:
            ev["replay"] = replay_evidence(cue, w)
        lab = predict(m, tags, cfg, dead, evidence=ev, strong=strong, merge_model=merge)
        if w == 0 and (ROOT / "runs" / "candidates" / "a5c.csv").exists():   # baseline reproduces the a5c candidate
            import pandas as pd
            same = bool((pd.read_csv(ROOT / "runs" / "candidates" / "a5c.csv")["label"].to_numpy().astype("<U1") == lab).all())
            print(f"baseline labels identical to runs/candidates/a5c.csv: {same}", flush=True)
        dr = drift_report(lab, m, tags, blk, code_table=cfg.tag.code_table)
        rows.append({"w": w, "dead_offset": off, "D_share": dr["non_blackout_D_share"],
                     "spell_rate": dr["spells_per_graded_min"], "tag_agree": dr["tag_agreement_implied"],
                     "prerestart_D": float((lab[pre] == "D").mean()), "progression_D": float((lab[prog] == "D").mean()),
                     "pred_D_share": dr["pred_D_share"], "n_spells": dr["n_spells"]})
        print(f"w {w:.1f} off {off:+.1f} done {time.time() - T0:.0f}s", flush=True)

    base = rows[0]
    for r in rows:
        checks = {"D_share": D_BAND[0] <= r["D_share"] <= D_BAND[1],
                  "spell_rate": RATE_BAND[0] <= r["spell_rate"] <= RATE_BAND[1],
                  "progression_D": r["progression_D"] <= base["progression_D"] + PROG_TOL,
                  "tag_agree": r["tag_agree"] >= base["tag_agree"] - AGREE_TOL}
        r["fails"] = [k for k, ok in checks.items() if not ok]
        r["admissible"] = r["w"] > 0 and not r["fails"]
    adm = [r for r in rows if r["admissible"]]
    chosen = (sorted(adm, key=lambda r: (-r["prerestart_D"], r["w"], abs(r["dead_offset"] - cfg.dead_offset)))[0]
              if adm else None)
    if chosen is None:
        fails = sorted({k for r in rows[1:] for k in r["fails"]})
        decision = {"result": "REJECTED", "clause": "rule 4: no admissible setting (rule 3 failures seen: " + ", ".join(fails) + ")"}
    else:
        decision = {"result": "CHOSEN", "w": chosen["w"], "dead_offset": chosen["dead_offset"],
                    "clause": f"rule 4: highest pre-restart D coverage among {len(adm)} admissible settings",
                    "tied_with": [{"w": r["w"], "dead_offset": r["dead_offset"]} for r in adm
                                  if r is not chosen and r["prerestart_D"] == chosen["prerestart_D"]]}
        if decision["tied_with"]:
            decision["clause"] += " (exact tie broken by smaller w)"
    out = {"preregistration": "docs/reads/plan3r-preregistration.md", "select": str(args.select),
           "bounds": {"D_share": D_BAND, "spell_rate": RATE_BAND, "progression_D_max": base["progression_D"] + PROG_TOL,
                      "tag_agree_min": base["tag_agree"] - AGREE_TOL},
           "n_frames": {"prerestart_visible": int(pre.sum()), "progression_mid_visible": int(prog.sum())},
           "rows": rows, "decision": decision, "runtime_s": time.time() - T0}
    (Path(args.select) / "replay_sweep.json").write_text(json.dumps(out, indent=1))
    print(f"{'w':>4} {'off':>5} {'Dshare':>7} {'spell/min':>9} {'tagagr':>7} {'preD':>6} {'progD':>6}  adm fails")
    for r in rows:
        print(f"{r['w']:4.1f} {r['dead_offset']:+5.1f} {r['D_share']:7.4f} {r['spell_rate']:9.3f} {r['tag_agree']:7.4f} "
              f"{r['prerestart_D']:6.3f} {r['progression_D']:6.3f}  {'Y' if r['admissible'] else '-'} {','.join(r['fails'])}")
    print("decision:", json.dumps(decision))


if __name__ == "__main__":
    main()
