"""Plan 8 T2: label-free real-match drift of each exp-seq3 variant's FINAL model stream inside the a14 decode
(a13r base, B12, dead_offset -3.0, seq_d_offset -1.0, merge off) + the registered a14 stream as reference.
Aggregates only (no ground truth, no public/sealed read). Needs exp_seq3/<variant>/real_stream.npz (exp_seq3_gru.py --final).
  <mtah venv python> scripts/exp_seq3_robust_real.py base v1 touch v1touch   -> exp_seq3/robust_real.json"""
import json, os, sys
os.environ["OMP_NUM_THREADS"] = "4"
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import copy
import numpy as np

import exp_seq_lomo as e
import seq3_common as c3
import mtah.select as sel
from make_candidate import load_selection
from mtah.eval.drift import blackout_mask_from_match, drift_report
from mtah.evidence import seq_feats as fx
from mtah.evidence.dead import dead_evidence
from mtah.io.match import half_runs
from mtah.io.tags import load_tags
from mtah.pipeline import compute_evidence, compute_strong, predict
from mtah.evidence.dead import DeadModel  # noqa: F401
from touch_candidates import agreement
from touch_common import load_real

DEAD_OFF, SEQ_D, W = -3.0, -1.0, 12.0


def main():
    names = sys.argv[1:] or ["base", "v1", "touch", "v1touch"]
    from mtah.evidence.dead import dead_table
    data = os.environ.get("MTAH_DATA_DIR", os.path.join(os.path.expanduser("~"), "mtah-data"))
    cfg0, dead, lengths, team, _, _, _ = load_selection(e.MAIN / "runs" / "select" / "a13r", with_seq=True)
    cfg0.pathcrf.cache_dir = str(e.MAIN / "runs" / "cache" / "pathcrf"); cfg0.post.merge_theta = 1.0
    m = load_real(); tags = load_tags(data)
    zc = np.load(e.MAIN / "runs" / "select" / "exp_robust" / "real_cache.npz", allow_pickle=False)
    ev0 = compute_evidence(m, tags, cfg0, dead, cache_key="real_tactical", team_model=team)
    p_dead = dead.predict_proba(dead_table(m, tags, dead.policy))
    strong = compute_strong(m, cfg0); mask = blackout_mask_from_match(m)
    st = []
    for h, s, en in half_runs(m.half):
        n = (en - s) // fx.STEP * fx.STEP
        if n > 0:
            st.append(s + np.arange(0, n, fx.STEP))
    starts = np.concatenate(st)
    arr = np.load(e.FINAL / "real_stream.npy"); fc = json.loads((e.FINAL / "config.json").read_text())
    streams = {"a14": arr[list(fc["stream_weights"]).index("seqB")]}
    for v in names:
        f = c3.EXP3 / v / "real_stream.npz"
        if not f.exists():
            print(v, "no real_stream.npz, skipped"); continue
        z = np.load(f); assert len(z["logp"]) == len(starts), (len(z["logp"]), len(starts))
        streams[v] = fx.upsample(z["logp"], starts, m.n_frames, m.half)
    res = {}
    for k, base in streams.items():
        cfg = copy.deepcopy(cfg0); cfg.dead_offset = DEAD_OFF
        ev = dict(ev0); ev["dead"] = dead_evidence(p_dead, logit_offset=DEAD_OFF)
        v = base.copy(); v[:, 2] += np.float32(SEQ_D)
        ev["seqB"] = v; cfg.decoder.weights["seqB"] = W; cfg.decoder.lengths["seqB"] = 1.0
        lab = predict(m, tags, cfg, dead, evidence=ev, strong=strong)
        r = drift_report(lab, m, tags, mask, code_table=cfg.tag.code_table); ag = agreement(lab, m, zc)
        res[k] = {"vis_dead": r["non_blackout_D_share"], "spells_per_graded_min": r["spells_per_graded_min"],
                  "tag_implied": r["tag_agreement_implied"], "pathcrf_agree": ag["pathcrf_confident_HA_agree"],
                  "replay_D": ag["D_share_when_replay_cue"], "pred_D": r["pred_D_share"], "n_spells": r["n_spells"],
                  "admissible": bool(sel.real_band_checks(r["non_blackout_D_share"], r["spells_per_graded_min"])["real_plausible"])}
        print(k, json.dumps(res[k]), flush=True)
    (c3.EXP3 / "robust_real.json").write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
