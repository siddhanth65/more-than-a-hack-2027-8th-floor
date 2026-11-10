"""Regenerate the real-match outputs of a5c and a14 with the smoothed-PathCRF evidence as NEW files
(runs/candidates/<name>_fx.csv; registered files untouched) and report label-free drift old vs new (aggregates only).
  python scripts/arb_fx_candidates.py a5c a14 [--smooth 1.0]"""
import argparse, json, os, sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "scripts"))
import numpy as np
import arb_models as am
from make_candidate import candidate_labels
from mtah.eval.drift import blackout_mask_from_match, drift_report
from mtah.evidence.tag_prior import load_tag_codes
from mtah.io.match import half_runs
from mtah.io.metrica_loader import load_tactical
from mtah.io.tags import load_tags
from mtah.submission import validate, write_submission
import mtah.evidence.pathcrf as pcrf


def _no_run(*a, **k):
    raise RuntimeError("PathCRF cache miss")
pcrf._run_half = _no_run


def per_half(lab, m, tags, bo, code_table):
    out = {}
    for h, a, b in half_runs(m.half):
        L = lab[a:b]; bm = bo[a:b]; n = b - a
        alive = L != "D"
        vis = ~bm
        out[f"H{h}"] = {
            "D_share_nonblackout": float((L[vis] == "D").mean()),
            "D_share_blackout": float((L[bm] == "D").mean()) if bm.any() else float("nan"),
            "spells_per_graded_min": None,
            "team_changes_in_blackouts_per_min_blackout": None}
        seg = [(i, L[i]) for i in range(n) if i == 0 or L[i] != L[i - 1]]
        HA = [(i, l) for i, l in seg if l in "HA"]
        out[f"H{h}"]["spells_per_graded_min"] = len(HA) / (n / 25 / 60)
        chg = 0
        prev = None
        for i, l in seg:
            if l in "HA":
                if prev is not None and prev != l and bm[i]:
                    chg += 1
                prev = l
        out[f"H{h}"]["team_changes_in_blackouts"] = chg
        out[f"H{h}"]["blackout_minutes"] = float(bm.sum() / 25 / 60)
        # tag agreement implied, per half
        ag = []
        for r in tags.itertuples(index=False):
            if r.code != "SET PIECES" and r.team in ("H", "A") and int(r.start) >= a and int(r.end) < b and r.code in code_table:
                sign = int(code_table[r.code]["sign"])
                if sign == 0:
                    continue
                exp = r.team if sign > 0 else ("A" if r.team == "H" else "H")
                sel = lab[int(r.start):int(r.end) + 1]; al = sel != "D"
                if al.any():
                    ag.append((sel[al] == exp).mean())
        out[f"H{h}"]["tag_agreement_implied"] = float(np.mean(ag)) if ag else float("nan")
    return out


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("names", nargs="+"); ap.add_argument("--smooth", type=float, default=1.0)
    a = ap.parse_args(); d = os.environ["MTAH_DATA_DIR"]
    m = load_tactical(d); tags = load_tags(d); bo = blackout_mask_from_match(m); ct = load_tag_codes()
    res = {}
    for fam in a.names:
        mdl = am.load_family(fam); cfg = mdl["cfg"]
        cfg.pathcrf.cache_dir = str(ROOT / "runs/cache/pathcrf_fx"); cfg.pathcrf.smooth_s = a.smooth
        lab = candidate_labels(m, tags, cfg, mdl["dead"], mdl["team"], mdl["merge"], None, cache_key="real_tactical", seq=mdl["seq"], sel_dir=mdl["sel_dir"])
        out = ROOT / "runs/candidates"; out.mkdir(parents=True, exist_ok=True)
        p = write_submission(lab, out / f"{fam}_fx.csv"); assert validate(p) is None
        old = np.array([x.split(",")[1] for x in (ROOT / "runs/candidates" / f"{fam}.csv").read_text().split("\n")[1:] if x])
        new = np.asarray(lab)
        g = m.graded
        r = {"valid": True, "frames_changed_overall_graded": float((old[g] != new[g]).mean()),
             "frames_changed_in_blackouts": float((old[bo] != new[bo]).mean()),
             "frames_changed_outside_blackouts": float((old[g & ~bo] != new[g & ~bo]).mean())}
        for h, s, e in half_runs(m.half):
            r[f"frames_changed_H{h}"] = float((old[s:e] != new[s:e]).mean())
            r[f"frames_changed_H{h}_in_blackouts"] = float((old[s:e][bo[s:e]] != new[s:e][bo[s:e]]).mean())
        r["drift_old"] = {k: (float(v) if v == v else None) for k, v in drift_report(old, m, tags, bo, code_table=cfg.tag.code_table).items()}
        r["drift_new"] = {k: (float(v) if v == v else None) for k, v in drift_report(new, m, tags, bo, code_table=cfg.tag.code_table).items()}
        r["per_half_old"] = per_half(old, m, tags, bo, ct); r["per_half_new"] = per_half(new, m, tags, bo, ct)
        res[fam] = r
        print(fam, json.dumps({k: v for k, v in r.items() if k.startswith("frames")}), flush=True)
    (ROOT / "runs/arb/fx_candidates.json").write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
