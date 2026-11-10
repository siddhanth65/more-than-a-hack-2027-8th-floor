"""a5c_fx2 calibration: a5c + PathCRF smoothing 1.0 s on the real feed; dead_offset over a5c's grid8 cells {-2, -2.5, -3} with the stored
a5c 6-fold LOMO (runs/select/a5c/summary.csv: cell i <-> offset GRID8[i]); chosen = best mean rank among admissible cells
(visible-dead [0.116, 0.181], R / rho [2.28, 4.10], rho = stored ctrl_Dfull count ratio).
  python scripts/calib_fx2.py -> runs/kh/a5c_fx2/calib.csv / calib.json"""
import copy, json, os, sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "scripts"))
os.environ.setdefault("OMP_NUM_THREADS", "2")
import pandas as pd
import mtah.select as sel
from make_candidate import candidate_labels, load_selection
from mtah.eval.drift import blackout_mask_from_match, drift_report
from mtah.io.metrica_loader import load_tactical
from mtah.io.tags import load_tags

MAIN = Path(__file__).resolve().parents[1]
data = os.environ.get("MTAH_DATA_DIR", os.path.join(os.path.expanduser("~"), "mtah-data"))
base = ROOT / "models" / "a5c"; out = ROOT / "runs" / "kh" / "a5c_fx2"; out.mkdir(parents=True, exist_ok=True)
cfg0, dead, lengths, team, merge, switch = load_selection(base)
cfg0.pathcrf.smooth_s = 1.0; cfg0.pathcrf.cache_dir = str(MAIN / "runs" / "cache" / "pathcrf_fx")
m = load_tactical(data); tags = load_tags(data); mask = blackout_mask_from_match(m)
lo = pd.read_csv(MAIN / "runs" / "select" / "a5c" / "summary.csv"); rows = []
for i, o in enumerate(o["dead_offset"] for o in sel.GRID8):
    cfg = copy.deepcopy(cfg0); cfg.dead_offset = float(o)
    lab = candidate_labels(m, tags, cfg, dead, team, merge, switch)
    d = drift_report(lab, m, tags, mask, code_table=cfg.tag.code_table); r = lo[lo.cell == i].iloc[0]
    rows.append({"dead_offset": o, "ctrl_Dfull": r.ctrl_Dfull, "raw_Dfull": r.raw_Dfull, "ctrl_Dnarrow": r.ctrl_Dnarrow, "raw_Dnarrow": r.raw_Dnarrow,
                 "rho": r.ctrl_Dfull_ratio, "real_vis_dead": d["non_blackout_D_share"], "R": d["spells_per_graded_min"],
                 "R_over_rho": d["spells_per_graded_min"] / r.ctrl_Dfull_ratio, "mean_rank": r.mean_rank})
t = pd.DataFrame(rows)
t["admissible"] = [bool(sel.real_band_checks(v, R, rho=h)["real_plausible"]) for v, R, h in zip(t.real_vis_dead, t.R, t.rho)]
t = t.sort_values("mean_rank").reset_index(drop=True); t.to_csv(out / "calib.csv", index=False)
adm = t[t.admissible]
(out / "calib.json").write_text(json.dumps({"chosen": adm.iloc[0].to_dict() if len(adm) else None, "table": t.to_dict(orient="records")}, indent=1, default=float))
print(t.round(4).to_string(index=False))
