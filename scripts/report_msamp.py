"""Aggregate runs/msamp/scores.jsonl + truth aggregates + download manifest into runs/msamp/REPORT.md and results.json.
Aggregates only (no per-frame output)."""
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "runs" / "msamp"
SYS_ORDER = ["a5c", "a13r", "a10", "a14", "a16", "smooth50", "b0", "b4"]
GAME = {9211: "G1", 9212: "G2", 9213: "G3"}
REAL_DRIFT_FILES = {"a5c": "a5c", "a10": "a10", "a14": "a14", "a16": "a16", "a13r": "a13"}   # a13r = a13 refit; real drift of a13


def load_scores():
    return [json.loads(x) for x in (OUT / "scores.jsonl").read_text().splitlines()]


def cell_table(rows, variant, metric):
    """{system: {(game, seed): mean over the game's two halves}}"""
    t = {}
    for r in rows:
        t.setdefault(r["system"], {}).setdefault((r["game"], r["seed"]), []).append(r[variant][metric])
    return {s: {k: float(np.mean(v)) for k, v in d.items() if len(v) == 2} for s, d in t.items()}


def summarise(cells):
    v = np.array(list(cells.values()))
    games = sorted({g for g, _ in cells}); seeds = sorted({s for _, s in cells})
    per_game = {GAME[g]: float(np.mean([cells[(g, s)] for s in seeds if (g, s) in cells])) for g in games}
    per_seed = {int(s): float(np.mean([cells[(g, s)] for g in games if (g, s) in cells])) for s in seeds}
    return {"mean": float(v.mean()), "sd_cells": float(v.std(ddof=1)) if len(v) > 1 else float("nan"),
            "sd_seeds": float(np.std(list(per_seed.values()), ddof=1)) if len(per_seed) > 1 else float("nan"),
            "n_cells": int(len(v)), "per_game": per_game, "per_seed": per_seed,
            "cells": {f"{GAME[g]}_s{s}": float(x) for (g, s), x in sorted(cells.items())}}


def drift_means(rows):
    d = {}
    for r in rows:
        for k, v in r["drift"].items():
            if v is not None:
                d.setdefault(r["system"], {}).setdefault(k, []).append(v)
    return {s: {k: float(np.mean(v)) for k, v in kv.items()} for s, kv in d.items()}


def real_drift():
    out = {}
    for s, f in REAL_DRIFT_FILES.items():
        p = ROOT / "runs" / "candidates" / f"{f}.drift.json"
        if p.exists():
            c = json.loads(p.read_text())["candidate"]
            out[s] = {k: c.get(k) for k in ("non_blackout_D_share", "spells_per_graded_min", "tag_agreement")}
    return out


def fmt(x, n=3):
    return "n/a" if x is None or (isinstance(x, float) and not np.isfinite(x)) else f"{x:.{n}f}"


def main():
    rows = load_scores()
    res = {"scores": {}, "drift": drift_means(rows), "real_match_drift": real_drift()}
    for variant in ("raw_Dfull", "ctrl_Dfull", "raw_Dnarrow", "ctrl_Dnarrow"):
        res["scores"][variant] = {}
        for metric in ("spell_f1", "macro_f1"):
            res["scores"][variant][metric] = {s: summarise(c) for s, c in cell_table(rows, variant, metric).items() if s != "truth"}
    res["n_pred_vs_gt_raw_Dfull"] = {}
    for s in SYS_ORDER:
        rr = [r for r in rows if r["system"] == s]
        if rr:
            res["n_pred_vs_gt_raw_Dfull"][s] = {"n_pred_per_half": float(np.mean([r["raw_Dfull"]["n_pred"] for r in rr])),
                                                 "n_gt_per_half": float(np.mean([r["raw_Dfull"]["n_gt"] for r in rr]))}
    variants_equal = {}
    for s in SYS_ORDER:
        rr = [r for r in rows if r["system"] == s]
        if rr:
            variants_equal[s] = max(abs(r["raw_Dfull"]["spell_f1"] - r[v]["spell_f1"]) for r in rr
                                    for v in ("ctrl_Dfull", "raw_Dnarrow", "ctrl_Dnarrow"))
    res["max_abs_spellf1_diff_raw_Dfull_vs_other_variants"] = variants_equal
    for f, k in (("download_manifest.json", "download_manifest"), ("truth_aggregates.json", "truth_aggregates"),
                 ("skipped.json", "skipped")):
        if (OUT / f).exists():
            res[k] = json.loads((OUT / f).read_text())
    # per-half table (mean over seeds) and the a5c vs a14 decision statistics
    ph = {}
    for r in rows:
        ph.setdefault(r["system"], {}).setdefault(f"G{r['game'] - 9210}H{r['half']}", []).append(r["raw_Dfull"]["spell_f1"])
    ph = {sy: {h: float(np.mean(v)) for h, v in d.items()} for sy, d in ph.items()}
    res["per_half_spell_f1_raw_Dfull"] = ph
    halves = sorted(ph["a5c"])
    diff = {h: ph["a14"][h] - ph["a5c"][h] for h in halves}
    res["a14_vs_a5c"] = {"per_half_diff": diff, "mean_diff": float(np.mean(list(diff.values()))),
                         "sd_diff": float(np.std(list(diff.values()), ddof=1)),
                         "a14_wins": int(sum(d > 0 for d in diff.values())), "a5c_wins": int(sum(d < 0 for d in diff.values())),
                         "ties": int(sum(d == 0 for d in diff.values())), "n_halves": len(diff)}
    (OUT / "results.json").write_text(json.dumps(res, indent=1))

    L = ["# Metrica-sample other-game validation (games 1-3, Metrica convention)", ""]
    L += ["Scores: candidates registered on the six DFL folds (no sample-game data in any training set), halves of "
          "games 1-3 corrupted with the runs/sim settings (ball smoothing 13, rel touches), seeds 100/101, truth = "
          "`raw_Dfull` (event + blank-ball derived H/A/D). Cell = mean over the two halves of a game; mean/sd over the "
          "6 (game, seed) cells; sd_seeds = sd over the two seed means.", ""]
    for metric, title in (("spell_f1", "spellF1"), ("macro_f1", "macroF1")):
        L += [f"## {title} (raw_Dfull)", "", "| system | G1 | G2 | G3 | mean | sd cells | sd seeds |", "|---|---|---|---|---|---|---|"]
        for s in SYS_ORDER:
            m = res["scores"]["raw_Dfull"][metric].get(s)
            if m:
                L.append(f"| {s} | " + " | ".join(fmt(m["per_game"].get(g)) for g in ("G1", "G2", "G3")) +
                         f" | {fmt(m['mean'])} | {fmt(m['sd_cells'])} | {fmt(m['sd_seeds'])} |")
        L.append("")
    L += ["## Per game x seed spellF1 (raw_Dfull)", "", "| system | " + " | ".join(f"{g}_s{s}" for g in GAME.values() for s in (100, 101)) + " |",
          "|---|" + "---|" * 6]
    for s in SYS_ORDER:
        m = res["scores"]["raw_Dfull"]["spell_f1"].get(s)
        if m:
            L.append(f"| {s} | " + " | ".join(fmt(m["cells"].get(f"{g}_s{sd}")) for g in GAME.values() for sd in (100, 101)) + " |")
    L += ["", "## Per half spellF1 (raw_Dfull, mean over seeds)", "", "| system | " + " | ".join(halves) + " |", "|---|" + "---|" * len(halves)]
    for sy in SYS_ORDER:
        if sy in ph:
            L.append(f"| {sy} | " + " | ".join(fmt(ph[sy].get(h)) for h in halves) + " |")
    c = res["a14_vs_a5c"]
    L += ["", f"a14 - a5c over the {c['n_halves']} halves (seed-averaged): mean {c['mean_diff']:+.4f}, sd {c['sd_diff']:.4f}; "
          f"a14 wins {c['a14_wins']}, a5c wins {c['a5c_wins']}, ties {c['ties']}. Per half: " +
          ", ".join(f"{h} {d:+.3f}" for h, d in c["per_half_diff"].items()), ""]
    L += ["", "## Label-free drift (mean over the 12 half x seed runs) vs the real match", "",
          "| system | visible-dead share (sim) | spells/graded min (sim) | tag agr. (sim) | visible-dead (real) | spells/min (real) | tag agr. (real) |",
          "|---|---|---|---|---|---|---|"]
    for s in ["truth"] + SYS_ORDER:
        d = res["drift"].get(s)
        if d:
            rd = res["real_match_drift"].get(s, {})
            L.append(f"| {s} | {fmt(d.get('non_blackout_D_share'))} | {fmt(d.get('spells_per_graded_min'), 2)} | {fmt(d.get('tag_agreement'))} | "
                     f"{fmt(rd.get('non_blackout_D_share'))} | {fmt(rd.get('spells_per_graded_min'), 2)} | {fmt(rd.get('tag_agreement'))} |")
    L += ["", "Truth variants: max |spellF1(raw_Dfull) - spellF1(other variant)| per system = " +
          ", ".join(f"{s} {v:.4f}" for s, v in variants_equal.items()) + ".", ""]
    if "truth_aggregates" in res:
        L += ["## Truth-convention aggregates", "",
              "| game | dead share | spells/min | <1 s | <3 s | <5 s | median s | direct handover | blank-ball share | event-dead | blank-dead | dead with ball visible |",
              "|---|---|---|---|---|---|---|---|---|---|---|---|"]
        for g in res["truth_aggregates"]:
            t, c = g["truth"], g["dead_composition"]
            L.append(f"| G{g['game']} | {t['dead_share']} | {t['spells_per_min']} | {t['lt1s']} | {t['lt3s']} | {t['lt5s']} | {t['median_s']} | "
                     f"{t['direct_handover_share']} | {c['ball_blank']} | {c['event_dead']} | {c['blank_dead']} | {c['dead_ball_visible']} |")
        L += ["", "Contest anchors (sanity only): median spell ~7.6 s, share <3 s ~0.18, <1 s 0.06, <5 s 0.35, 12/16 direct handovers.", ""]
    if "download_manifest" in res:
        L += ["## Download manifest (raw.githubusercontent.com/metrica-sports/sample-data, master)", "", "| file | bytes | sha256 |", "|---|---|---|"]
        for f in res["download_manifest"]["files"]:
            L.append(f"| {f['path']} | {f['bytes']} | {f['sha256']} |")
        L.append("")
    if "skipped" in res:
        L += ["## Skipped / not done", ""] + [f"- {x}" for x in res["skipped"]] + [""]
    (OUT / "REPORT.md").write_text("\n".join(L) + "\n", encoding="utf-8")
    print("\n".join(L))


if __name__ == "__main__":
    main()
