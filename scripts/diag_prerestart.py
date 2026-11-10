"""Diagnostic 2: dead time before typed restarts, SkillCorner norms vs. candidate D runs before contest Smart Tagging
set pieces. Label-free (candidate outputs + tags + ball track only; no contest ground truth). Aggregates only.
Usage: OMP_NUM_THREADS=2 python -I scripts/diag_prerestart.py"""
import json, os, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd

from mtah.io.metrica_loader import load_tactical
from mtah.io.tags import load_tags

D = Path(os.environ.get("MTAH_DATA_DIR", os.path.join(os.path.expanduser("~"), "mtah-data")))
SC = Path(os.environ.get("MTAH_SKILLCORNER_DIR", os.path.join(os.path.expanduser("~"), "skillcorner-opendata", "data", "matches")))
OUT = Path(os.environ.get("DIAG_OUT", str(ROOT / "runs" / "diag")))
FPS = 25
TOL = 3 * FPS
TYPES = ["throw_in", "free_kick", "goal_kick", "corner", "goal"]
TAG2TYPE = {"Throw-in": "throw_in", "Free Kick": "free_kick", "Goal Kick": "goal_kick", "Corner Kick": "corner",
            "Kick Off (After Goal)": "goal", "Kick Off (Start)": "kickoff_start"}


def qs(x):
    x = np.asarray(x, float)
    if not len(x):
        return {"n": 0}
    return {"n": int(len(x)), "p25": float(np.percentile(x, 25)), "median": float(np.median(x)), "p75": float(np.percentile(x, 75)),
            "mean": float(x.mean())}


def skillcorner():
    """Dead time before a restart = (frame_start of the first player_possession after an interruption
    [game_interruption_before = <type>_for/_against; type 'goal' = kick-off after a goal] - frame_end of the immediately
    preceding player_possession event in the same period) / 10 fps. It therefore includes the last ball flight to the
    touchline plus the stoppage up to the restart reception (the restart taker's ball contact)."""
    rows = []
    for md in sorted(p for p in SC.iterdir() if p.is_dir()):
        ev = pd.read_csv(md / f"{md.name}_dynamic_events.csv", low_memory=False,
                         usecols=["event_type", "frame_start", "frame_end", "period", "game_interruption_before"])
        pp = ev[ev.event_type == "player_possession"].sort_values(["period", "frame_start"]).reset_index(drop=True)
        prev_end = pp.groupby("period").frame_end.shift(1)
        gib = pp.game_interruption_before.fillna("")
        typ = gib.str.replace(r"_(for|against)$", "", regex=True)
        gap = (pp.frame_start - prev_end) / 10.0
        m = (typ != "") & prev_end.notna() & (gap >= 0)
        for t_, g_, side in zip(typ[m], gap[m], gib[m].str.extract(r"_(for|against)$")[0]):
            rows.append((md.name, t_, g_, side))
    df = pd.DataFrame(rows, columns=["match", "type", "gap_s", "side"])
    out = {"definition": skillcorner.__doc__, "n_matches": int(df.match.nunique()), "by_type": {}}
    for t_ in TYPES:
        out["by_type"][t_] = qs(df[df["type"] == t_].gap_s.to_numpy())
    out["other_types_seen"] = df[~df["type"].isin(TYPES)]["type"].value_counts().to_dict()
    return out, df


def d_edge_run(lab, tag, ball_present, graded, after=TOL):
    """Locate the D->live edge nearest to the tag within +-TOL frames; return dict."""
    a, b = max(0, tag - TOL), min(len(lab) - 2, tag + after)
    isD = lab == "D"
    edges = [i for i in range(a, b) if isD[i] and not isD[i + 1]]    # last D frame i, live at i+1
    if edges:
        e = min(edges, key=lambda i: abs(i + 1 - tag))
        s = e
        while s > 0 and isD[s - 1]:
            s -= 1
        return {"kind": "edge", "len": (e + 1 - s) / FPS, "edge_offset_s": (e + 1 - tag) / FPS}
    if isD[a:b + 1].any():
        # D reaches past the window end (or the whole window is D) without a D->live edge: dead longer than the tag suggests
        s = tag
        while s > 0 and isD[s - 1]:
            s -= 1
        return {"kind": "d_through", "len": max(tag - s, 0) / FPS, "edge_offset_s": None}
    return {"kind": "none", "len": 0.0, "edge_offset_s": None}


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    sc, scdf = skillcorner()
    t = load_tactical(D)
    n = t.n_frames
    tags = load_tags(D)
    sp = tags[tags.code == "SET PIECES"].copy()
    sp["ty"] = sp["type"].map(TAG2TYPE)
    ball = t.ball_present; graded = t.graded
    res = {"skillcorner": sc, "contest": {}, "contest_after10s": {}}
    sc_p25 = {k: sc["by_type"][k]["p25"] for k in TYPES}
    sc_med = {k: sc["by_type"][k]["median"] for k in TYPES}
    for key, after in (("contest", TOL), ("contest_after10s", 10 * FPS)):
      for cand in os.environ.get("DIAG_CANDS", "a5c,a14").split(","):
          s = pd.read_csv(ROOT / "runs" / "candidates" / f"{cand}.csv")
          lab = np.full(n, "", object); lab[s.frame.to_numpy()] = s.label.to_numpy()
          recs = []
          for r in sp.itertuples():
              if r.ty is None or not graded[min(max(r.start, 0), n - 1)]:
                  continue
              o = d_edge_run(lab, int(r.start), ball, graded, after)
              pre = slice(max(0, r.start - 5 * FPS), r.start)
              o["ty"] = r.ty
              o["blackout_before"] = bool((~ball[pre]).mean() > 0.5)
              o["live_before_tag"] = bool(np.isin(lab[max(0, r.start - 2 * FPS):r.start], ["H", "A"]).mean() > 0.5)
              recs.append(o)
          df = pd.DataFrame(recs)
          cres = {}
          deficit_p25 = deficit_med = 0.0
          for ty in TYPES + ["kickoff_start"]:
              d = df[df.ty == ty]
              if not len(d):
                  continue
              def summ(dd):
                  if not len(dd):
                      return {"n": 0}
                  return {"n": int(len(dd)), "share_with_D_edge": float((dd.kind == "edge").mean()),
                          "share_D_through": float((dd.kind == "d_through").mean()),
                          "share_no_D": float((dd.kind == "none").mean()),
                          "run_len_s_all": qs(dd.len.to_numpy()),
                          "run_len_s_edge_only": qs(dd[dd.kind == "edge"].len.to_numpy()),
                          "share_pre_tag_live_labelled": float(dd.live_before_tag.mean())}
              c = summ(d)
              c["blackout_before"] = summ(d[d.blackout_before]); c["ball_visible_before"] = summ(d[~d.blackout_before])
              if ty in TYPES:
                  c["skillcorner_ref"] = sc["by_type"][ty]
                  dl25 = (sc_p25[ty] - d.len).clip(lower=0); dlm = (sc_med[ty] - d.len).clip(lower=0)
                  c["deficit_to_sc_p25_s_total"] = float(dl25.sum()); c["deficit_to_sc_median_s_total"] = float(dlm.sum())
                  c["share_below_sc_p25"] = float((d.len < sc_p25[ty]).mean())
                  c["share_below_sc_median"] = float((d.len < sc_med[ty]).mean())
                  deficit_p25 += dl25.sum(); deficit_med += dlm.sum()
              cres[ty] = c
          ng = int(graded.sum())
          cres["total"] = {"n_tags_used": int(len(df)), "deficit_to_sc_p25_frames": float(deficit_p25 * FPS), "deficit_to_sc_median_frames": float(deficit_med * FPS),
                           "graded_frames": ng, "deficit_p25_share_of_graded": float(deficit_p25 * FPS / ng),
                           "deficit_median_share_of_graded": float(deficit_med * FPS / ng),
                           "cand_D_share_graded": float((lab[graded] == "D").mean())}
          res[key][cand] = cres
    (OUT / "diag_prerestart.json").write_text(json.dumps(res, indent=1))
    print("done")


if __name__ == "__main__":
    main()
