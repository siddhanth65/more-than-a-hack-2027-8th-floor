"""Step 4 (label-free; nothing registered, no reads): decode candidates a5c and a10 on the real match with `abs` (as
registered; must reproduce runs/candidates/<name>.csv) and `rel` touches (everything else in the configs unchanged:
the touch features reach teamctrl, the strong touches, boundary snapping, the merge model and a10's learned streams).
Reports drift_report aggregates, agreement with confident PathCRF, the replay-D cue and the abs labels.
The registered models (dead / team / merge / seq A, B) are NOT retrained, so `rel` is a pure input-feed change.
                                                                                    -> runs/touch/candidates.json"""
import json, os, sys, time
os.environ.setdefault("OMP_NUM_THREADS", "4")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd

import mtah.evidence.pathcrf as pcrf
from make_candidate import candidate_labels, load_selection
from touch_common import DATA, MAIN, OUT, load_real
from mtah.eval.drift import blackout_mask_from_match, drift_report
from mtah.evidence.touches import TouchParams
from mtah.io.tags import load_tags

CACHE = MAIN / "runs" / "select" / "exp_robust" / "real_cache.npz"


def _no_run(*a, **k):
    raise RuntimeError("PathCRF cache miss: refusing to run PathCRF (read-only cache)")


pcrf._run_half = _no_run


def agreement(lab, m, z):
    g = m.graded
    p = z["p_home"]; st = np.zeros(len(p), np.int8)
    st[np.nan_to_num(p, nan=0.5) >= 0.8] = 1; st[np.nan_to_num(p, nan=0.5) <= 0.2] = -1
    conf = g & (st != 0)
    ha = conf & np.isin(lab, ["H", "A"])
    agree = ((lab == "H") & (st == 1)) | ((lab == "A") & (st == -1))
    cue = z["cue"]; cv = g & np.isfinite(cue) & m.ball_present
    return {"pathcrf_confident_graded_frames": int(conf.sum()),
            "pathcrf_confident_HA_agree": float(agree[ha].mean()) if ha.any() else None,
            "pathcrf_confident_labelled_D": float((lab[conf] == "D").mean()),
            "replay_cue_visible_frames": int(cv.sum()),
            "D_share_when_replay_cue": float((lab[cv & (cue > 0.5)] == "D").mean()),
            "D_share_when_no_replay_cue": float((lab[cv & (cue <= 0.5)] == "D").mean())}


def main():
    T0 = time.time()
    m = load_real(); tags = load_tags(DATA); z = np.load(CACHE, allow_pickle=False)
    blk = blackout_mask_from_match(m)
    out = {}
    for name in ("a5c", "a10"):
        sel = MAIN / "runs" / "select" / name
        cfg, dead, lengths, team, merge, switch, seq = load_selection(sel, with_seq=True)
        cfg.pathcrf.cache_dir = str(MAIN / "runs" / "cache" / "pathcrf")
        labs = {}
        for mode in ("abs", "rel"):
            c = cfg
            c.touch = replace(cfg.touch, mode=mode)
            lab = candidate_labels(m, tags, c, dead, team=team, merge=merge, switch=switch, seq=seq, sel_dir=sel)
            labs[mode] = lab
            d = drift_report(lab, m, tags, blk, code_table=c.tag.code_table)
            out.setdefault(name, {})[mode] = {
                "non_blackout_D_share(visible-dead)": d["non_blackout_D_share"], "spells_per_graded_min": d["spells_per_graded_min"],
                "tag_agreement_implied": d["tag_agreement_implied"], "pred_D_share": d["pred_D_share"],
                "n_spells": d["n_spells"], "spell_len_median_s": d["spell_len_median_s"],
                "share_changes_in_blackout": d["share_changes_in_blackout"], "blackout_D_share": d["blackout_D_share"],
                "D_starts_near_sp_share": d["D_starts_near_sp_share"], **agreement(lab, m, z)}
            print(name, mode, json.dumps({k: (round(v, 4) if isinstance(v, float) else v) for k, v in out[name][mode].items()}),
                  f"{time.time() - T0:.0f}s", flush=True)
        reg = MAIN / "runs" / "candidates" / f"{name}.csv"
        if reg.exists():
            out[name]["abs_reproduces_registered_csv"] = bool((pd.read_csv(reg)["label"].to_numpy().astype("<U1") == labs["abs"]).all())
        g = m.graded
        out[name]["rel_vs_abs_frame_agreement_graded"] = float((labs["rel"][g] == labs["abs"][g]).mean())
        ha = g & np.isin(labs["rel"], ["H", "A"]) & np.isin(labs["abs"], ["H", "A"])
        out[name]["rel_vs_abs_HA_agreement_when_both_HA"] = float((labs["rel"][ha] == labs["abs"][ha]).mean())
        np.save(OUT / f"labels_{name}_rel.npy", labs["rel"].astype("<U1"))
        (OUT / "candidates.json").write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
