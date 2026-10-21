"""Why does the dead model under-fire on real visible-ball frames? Label-free diagnostic (aggregates only).

Weak labels on the real match (never the hidden GT): frames inside SET PIECES tag intervals, the 2 s before a
set-piece start, and the central 50 % of BUILD UP / PROGRESSION tags. The same weak labels on the sim halves
(9001-9006, seed 100; 9007 is never loaded) are checked against the sim truth (ctrl_Dfull) for purity.

Writes runs/select/a2/visible_dead_diag.json. Prints only aggregates.
    python scripts/diagnose_visible_dead.py
"""
import argparse
import copy
import json
import os
import sys
from pathlib import Path

import numpy as np

os.environ.setdefault("OMP_NUM_THREADS", "2")

from mtah.decode.viterbi import cost_matrix  # noqa: E402
from mtah.eval.drift import blackout_mask_from_match  # noqa: E402
from mtah.eval.scorer import score  # noqa: E402
from mtah.evidence.ball_ctrl import ball_ctrl_evidence  # noqa: E402
from mtah.evidence.dead import DEAD_FEATURES, DeadModel, dead_evidence, dead_features  # noqa: E402
from mtah.evidence.pathcrf import PathCRFParams, _input_sha, pathcrf_evidence  # noqa: E402
from mtah.evidence.tag_prior import tag_prior_evidence  # noqa: E402
from mtah.io.metrica_loader import load_tactical  # noqa: E402
from mtah.io.tags import load_tags  # noqa: E402
from mtah.pipeline import PipelineConfig, compute_strong, predict  # noqa: E402
from mtah.sim.dataset import load_half_sim  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
SEL = ROOT / "runs" / "select" / "a2"
SIM_DIR = ROOT / "runs" / "sim"
SIM_KEYS = [f"game_{m}_H{h}" for m in range(9001, 9007) for h in (1, 2)]   # 9007 = sealed: never loaded
SEED = 100
PHASE_CODES = ("BUILD UP", "PROGRESSION")
OFFSETS = np.arange(-10, 31)          # seconds relative to the set-piece tag start


def r4(x):
    x = float(x)
    return round(x, 4) if np.isfinite(x) else None


def load_cfg():
    raw = json.loads((SEL / "config.json").read_text()); lengths = raw.pop("lengths")
    cfg = PipelineConfig.from_dict(raw); cfg.decoder.lengths = lengths
    return cfg


def pathcrf_cached(match, key):
    """PathCRF posterior from the cache file only (never launches PathCRF)."""
    p = Path(PathCRFParams().cache_dir) / f"{key}.npz"
    with np.load(p) as z:
        if len(z["p_home"]) != match.n_frames:
            raise SystemExit(f"pathcrf cache {key}: length mismatch")
        sha_ok = "input_sha" in z.files and str(z["input_sha"]) == _input_sha(match)
        return {k: np.array(z[k]) for k in ("p_home", "imputed_frac")}, bool(sha_ok)


def weak_labels(tags, F, fps):
    sp = np.zeros(F, bool); pre = np.zeros(F, bool); phase = np.zeros(F, bool)
    sp_starts = []
    for r in tags.itertuples(index=False):
        a = max(0, int(r.start)); b = min(F - 1, int(r.end))
        if b < a:
            continue
        if r.code == "SET PIECES":
            sp[a:b + 1] = True; sp_starts.append(a)
            pre[max(0, a - int(2 * fps)):a] = True
        elif r.code in PHASE_CODES:
            q = (b - a) // 4
            phase[a + q:b - q + 1] = True
    pre &= ~sp; phase &= ~sp & ~pre
    return {"sp_in": sp, "sp_pre2s": pre, "phase_mid": phase}, np.array(sorted(sp_starts), int)


def evidence_streams(match, tags, cfg, pD, pc):
    ev = {"ball": ball_ctrl_evidence(match, cfg.ball), "tag": tag_prior_evidence(tags, match.n_frames, cfg.tag),
          "dead": dead_evidence(pD)}
    if pc is not None:
        ev["pathcrf"] = pathcrf_evidence(pc["p_home"], pc["imputed_frac"], cfg.pathcrf.eps, cfg.pathcrf.conf_alpha)
    return ev


def stream_terms(ev, cfg):
    """Per-frame decoder terms: |H-A| margin per stream (NaN = no opinion), each stream's weighted advantage of
    the best alive state over D (0 if silent), the dead stream's weighted advantage of D, and net = C_alive_min - C_D."""
    out = {}
    for k, e in ev.items():
        if k == "dead":
            continue
        w = cfg.decoder.weights.get(k, 1.0) / max(1.0, cfg.decoder.lengths.get(k, 1.0))
        m = np.abs(e[:, 0].astype(float) - e[:, 1].astype(float))
        out[f"{k}_margin"] = m
        out[f"{k}_adv"] = np.where(np.isfinite(m), w * m / 2.0, 0.0)
    C = cost_matrix(ev, cfg.decoder)
    out["dead_adv"] = cost_matrix({"dead": ev["dead"]}, cfg.decoder)
    out["dead_adv"] = np.minimum(out["dead_adv"][:, 0], out["dead_adv"][:, 1]) - out["dead_adv"][:, 2]
    out["net"] = np.minimum(C[:, 0], C[:, 1]) - C[:, 2]
    return out


def qstats(v):
    v = v[np.isfinite(v)]
    if not v.size:
        return {"n": 0}
    return {"n": int(v.size), "mean": r4(v.mean()), "p10": r4(np.percentile(v, 10)), "p50": r4(np.median(v)),
            "p90": r4(np.percentile(v, 90))}


def group_pd(pd_, sel):
    if not sel.any():
        return {"n": 0}
    v = pd_[sel]
    return {"n": int(sel.sum()), "mean": r4(v.mean()), "gt05": r4((v > 0.5).mean())}


def collect(match, tags, cfg, models, pc, labels_pred, X, truth=None):
    """Per-frame arrays needed for the aggregates (kept in memory, never written out)."""
    F = match.n_frames; fps = match.fps
    nb = match.graded & ~blackout_mask_from_match(match)
    wl, sp_starts = weak_labels(tags, F, fps)
    pD = {name: mdl.predict_proba(X) for name, mdl in models.items()}
    terms = stream_terms(evidence_streams(match, tags, cfg, pD["a2_full"], pc), cfg)
    d = {"X": X, "nb": nb, "graded": match.graded, **wl, "pD": pD, "terms": terms, "pred": labels_pred,
         "sp_starts": sp_starts, "fps": fps, "present": match.ball_present, "speed": match.ball_speed}
    if truth is not None:
        d["truth_D"] = truth == "D"
    return d


def concat(ds):
    out = {}
    for k in ("X", "nb", "graded", "sp_in", "sp_pre2s", "phase_mid", "pred", "present", "speed"):
        out[k] = np.concatenate([d[k] for d in ds])
    out["pD"] = {n: np.concatenate([d["pD"][n] for d in ds]) for n in ds[0]["pD"]}
    out["terms"] = {n: np.concatenate([d["terms"][n] for d in ds]) for n in ds[0]["terms"]}
    if "truth_D" in ds[0]:
        out["truth_D"] = np.concatenate([d["truth_D"] for d in ds])
    return out


def sp_profile(ds):
    """Aggregate by whole-second offset from each set-piece tag start, all graded frames (blackout included)."""
    acc = {int(o): {"n": 0, "present": 0.0, "static": 0.0, "pD": 0.0, "predD": 0.0, "trueD": 0.0, "spd": []} for o in OFFSETS}
    for d in ds:
        F = len(d["nb"]); fps = d["fps"]; st = DEAD_FEATURES.index("ball_static_1s")
        for a in d["sp_starts"]:
            for o in OFFSETS:
                lo, hi = a + int(o * fps), a + int((o + 1) * fps)
                if lo < 0 or hi > F:
                    continue
                sel = np.arange(lo, hi)[d["graded"][lo:hi]]
                if not sel.size:
                    continue
                r = acc[int(o)]; r["n"] += sel.size
                r["present"] += d["present"][sel].sum()
                r["static"] += np.nan_to_num(d["X"][sel, st]).sum()
                r["pD"] += d["pD"]["a2_full"][sel].sum()
                r["predD"] += (d["pred"][sel] == "D").sum()
                if "truth_D" in d:
                    r["trueD"] += d["truth_D"][sel].sum()
                sp = d["speed"][sel]; r["spd"].extend(sp[np.isfinite(sp)].tolist())
    out = {}
    for o, r in acc.items():
        n = max(r["n"], 1)
        out[o] = {"n": r["n"], "ball_present": r4(r["present"] / n), "ball_static": r4(r["static"] / n),
                  "mean_pD_full": r4(r["pD"] / n), "pred_D": r4(r["predD"] / n),
                  "ball_speed_p50": r4(np.median(r["spd"])) if r["spd"] else None}
        if "truth_D" in ds[0]:
            out[o]["true_D"] = r4(r["trueD"] / n)
    return out


RUN_BUCKETS = ((0, 2), (2, 5), (5, 10), (10, 1e9))


def hi_pd_analysis(ds, model="a2_full"):
    """Where do the frames the dead model calls dead (P(D) > 0.5, non-blackout) get lost in the decoder?"""
    from mtah.sim.blackout_sim import runs_of
    n_nb = 0; acc = {"n_hi": 0, "hi_predD": 0, "hi_trueD": 0}
    buck = {f"{a}-{b if b < 1e9 else 'inf'}s": [0, 0] for a, b in RUN_BUCKETS}
    pres = {"ball_present": [0, 0], "ball_absent": [0, 0]}
    where = {"sp_in": [0, 0], "sp_pre2s": [0, 0], "phase_mid": [0, 0], "other": [0, 0]}
    terms = {"decoded_D": {}, "decoded_alive": {}}
    has_truth = "truth_D" in ds[0]
    for d in ds:
        nb = d["nb"]; hi = (d["pD"][model] > 0.5) & nb; isD = d["pred"] == "D"
        n_nb += nb.sum(); acc["n_hi"] += hi.sum(); acc["hi_predD"] += (hi & isD).sum()
        if has_truth:
            acc["hi_trueD"] += (hi & d["truth_D"]).sum()
        for s, e in runs_of(hi):
            L = (e - s) / d["fps"]
            for (a, b), k in zip(RUN_BUCKETS, buck):
                if a <= L < b:
                    buck[k][0] += e - s; buck[k][1] += isD[s:e].sum()
        for k, sel in (("ball_present", d["present"]), ("ball_absent", ~d["present"])):
            pres[k][0] += (hi & sel).sum(); pres[k][1] += (hi & sel & isD).sum()
        other = ~(d["sp_in"] | d["sp_pre2s"] | d["phase_mid"])
        for k, sel in (("sp_in", d["sp_in"]), ("sp_pre2s", d["sp_pre2s"]), ("phase_mid", d["phase_mid"]), ("other", other)):
            where[k][0] += (hi & sel).sum(); where[k][1] += (hi & sel & isD).sum()
        for nm, sel in (("decoded_D", hi & isD), ("decoded_alive", hi & ~isD)):
            for t in ("dead_adv", "ball_adv", "tag_adv", "pathcrf_adv", "net"):
                a = terms[nm].setdefault(t, [0.0, 0]); a[0] += d["terms"][t][sel].sum(); a[1] += sel.sum()
    nh = max(acc["n_hi"], 1)
    out = {"hi_share_of_nb": r4(acc["n_hi"] / max(n_nb, 1)), "hi_decoded_D": r4(acc["hi_predD"] / nh)}
    if has_truth:
        out["hi_true_D_purity"] = r4(acc["hi_trueD"] / nh)
    out["by_run_length"] = {k: {"frame_share": r4(v[0] / nh), "decoded_D": r4(v[1] / max(v[0], 1))} for k, v in buck.items()}
    out["by_ball"] = {k: {"frame_share": r4(v[0] / nh), "decoded_D": r4(v[1] / max(v[0], 1))} for k, v in pres.items()}
    out["by_weak_label"] = {k: {"frame_share": r4(v[0] / nh), "decoded_D": r4(v[1] / max(v[0], 1))} for k, v in where.items()}
    out["terms_mean"] = {nm: {t: r4(a[0] / max(a[1], 1)) for t, a in v.items()} for nm, v in terms.items()}
    return out


def tag_census(tags, fps):
    sp = tags[tags["code"] == "SET PIECES"]
    dur = ((sp["end"] - sp["start"]) / fps).to_numpy(float)
    types = sp["type"].fillna("None").value_counts().to_dict()
    return {"n_set_pieces": int(len(sp)), "types": {str(k): int(v) for k, v in types.items()},
            "sp_duration_s": qstats(dur),
            "codes": {str(k): int(v) for k, v in tags["code"].value_counts().to_dict().items()}}


def main():
    ap = argparse.ArgumentParser()
    args = ap.parse_args()
    data_dir = os.environ.get("MTAH_DATA_DIR")
    if not data_dir:
        raise SystemExit("set MTAH_DATA_DIR")
    cfg = load_cfg()
    a2 = DeadModel.load(SEL / "dead_model.joblib")
    assert a2.policy == "full"

    print("loading sims 9001-9006 seed", SEED, flush=True)
    sims = [load_half_sim(SIM_DIR, k, SEED) for k in SIM_KEYS]
    print("dead features on sims ...", flush=True)
    Xs = [dead_features(s.match, s.tags) for s in sims]
    ys = [(s.truth["ctrl_Dfull"] == "D").astype(int) for s in sims]

    def fit(idx, policy):   # == mtah.select.train_dead on sims[idx] (features computed once)
        return DeadModel(policy=policy).fit(np.concatenate([Xs[i] for i in idx]), np.concatenate([ys[i] for i in idx]))
    allk = list(range(len(sims))); trk = [i for i, s in enumerate(sims) if "9006" not in s.key]
    print("training no_ball_at_all / full dead models on sims (seed 100) ...", flush=True)
    models = {"a2_full": a2, "nba_s100": fit(allk, "no_ball_at_all"), "full_s100": fit(allk, "full")}
    print("held-out fold: train on 9001-9005, test 9006 ...", flush=True)
    ho = {"full": fit(trk, "full"), "no_ball_at_all": fit(trk, "no_ball_at_all")}

    # ---------- real ----------
    print("real match ...", flush=True)
    m = load_tactical(data_dir); tags = load_tags(data_dir)
    pc_real, sha_ok_real = pathcrf_cached(m, "real_tactical")
    strong_real = compute_strong(m, cfg)
    Xr_all = dead_features(m, tags)
    ev_real = evidence_streams(m, tags, cfg, a2.predict_proba(Xr_all), pc_real)
    lab_real = predict(m, tags, cfg, None, evidence=ev_real, strong=strong_real)
    R = collect(m, tags, cfg, models, pc_real, lab_real, Xr_all)

    # ---------- sims ----------
    S_list = []; sim_ev = []; sim_strong = []
    sha_ok_sim = []
    for i, s in enumerate(sims):
        print("  sim", s.key, flush=True)
        pc, ok = pathcrf_cached(s.match, f"{s.key}_s{s.seed}"); sha_ok_sim.append(ok)
        ev = evidence_streams(s.match, s.tags, cfg, a2.predict_proba(Xs[i]), pc)
        st = compute_strong(s.match, cfg)
        lab = predict(s.match, s.tags, cfg, None, evidence=ev, strong=st)
        d = collect(s.match, s.tags, cfg, models, pc, lab, Xs[i], truth=s.truth["ctrl_Dfull"])
        if "9006" in s.key:
            X = d["X"]
            d["pD"]["heldout_full"] = ho["full"].predict_proba(X); d["pD"]["heldout_nba"] = ho["no_ball_at_all"].predict_proba(X)
        S_list.append(d); sim_ev.append(ev); sim_strong.append(st)
    S = concat(S_list)
    H = concat([d for d, s in zip(S_list, sims) if "9006" in s.key])

    out = {"git_note": "label-free; real hidden GT never used; sims 9001-9006 seed 100 only",
           "pathcrf_cache_input_sha_ok": {"real": sha_ok_real, "sim_all": all(sha_ok_sim)},
           "tags": {"real": tag_census(tags, m.fps),
                    "sim": tag_census(__import__("pandas").concat([s.tags for s in sims]), sims[0].match.fps)}}

    # ---------- 1. weak-label P(D) ----------
    groups = ("sp_in", "sp_pre2s", "phase_mid")
    t1 = {"real": {}, "sim": {}, "sim_heldout_9006": {}}
    for g in groups:
        rs = R[g] & R["nb"]; ss = S[g] & S["nb"]; hs = H[g] & H["nb"]
        t1["real"][g] = {n: group_pd(v, rs) for n, v in R["pD"].items() if n in models}
        t1["real"][g]["pred_D_share"] = r4((R["pred"][rs] == "D").mean()) if rs.any() else None
        t1["sim"][g] = {n: group_pd(v, ss) for n, v in S["pD"].items() if n in models}
        t1["sim"][g]["pred_D_share"] = r4((S["pred"][ss] == "D").mean()) if ss.any() else None
        t1["sim"][g]["true_D_purity"] = r4(S["truth_D"][ss].mean()) if ss.any() else None
        t1["sim_heldout_9006"][g] = {n: group_pd(H["pD"][n], hs) for n in ("heldout_full", "heldout_nba")}
        t1["sim_heldout_9006"][g]["true_D_purity"] = r4(H["truth_D"][hs].mean()) if hs.any() else None
    # sim by truth (non-blackout)
    for nm, sel in (("true_D", S["truth_D"] & S["nb"]), ("true_alive", ~S["truth_D"] & S["nb"] & S["graded"])):
        t1["sim"][nm] = {n: group_pd(v, sel) for n, v in S["pD"].items() if n in models}
        t1["sim"][nm]["pred_D_share"] = r4((S["pred"][sel] == "D").mean())
    for nm, sel in (("true_D", H["truth_D"] & H["nb"]), ("true_alive", ~H["truth_D"] & H["nb"] & H["graded"])):
        t1["sim_heldout_9006"][nm] = {n: group_pd(H["pD"][n], sel) for n in ("heldout_full", "heldout_nba")}
    t1["real"]["all_nb"] = {n: group_pd(v, R["nb"]) for n, v in R["pD"].items()}
    t1["sim"]["all_nb"] = {n: group_pd(v, S["nb"]) for n, v in S["pD"].items() if n in models}
    t1["real"]["all_nb"]["pred_D_share"] = r4((R["pred"][R["nb"]] == "D").mean())
    t1["sim"]["all_nb"]["pred_D_share"] = r4((S["pred"][S["nb"]] == "D").mean())
    t1["sim"]["all_nb"]["true_D"] = r4(S["truth_D"][S["nb"]].mean())
    t1["share_of_nb_frames"] = {"real": {g: r4((R[g] & R["nb"]).sum() / R["nb"].sum()) for g in groups},
                                "sim": {g: r4((S[g] & S["nb"]).sum() / S["nb"].sum()) for g in groups}}
    out["weak_label_pD"] = t1

    # ---------- 2. feature shift ----------
    rng = np.random.default_rng(0)
    sims_D = S["truth_D"] & S["nb"]; sims_A = ~S["truth_D"] & S["nb"] & S["graded"]
    real_dead = (R["sp_in"] | R["sp_pre2s"]) & R["nb"]
    t2 = {}
    for j, f in enumerate(DEAD_FEATURES):
        row = {"real_sp": qstats(R["X"][R["sp_in"] & R["nb"], j]), "real_pre": qstats(R["X"][R["sp_pre2s"] & R["nb"], j]),
               "real_phase": qstats(R["X"][R["phase_mid"] & R["nb"], j]),
               "sim_sp": qstats(S["X"][S["sp_in"] & S["nb"], j]), "sim_pre": qstats(S["X"][S["sp_pre2s"] & S["nb"], j]),
               "sim_phase": qstats(S["X"][S["phase_mid"] & S["nb"], j]),
               "sim_trueD": qstats(S["X"][sims_D, j]), "sim_trueAlive": qstats(S["X"][sims_A, j])}
        mD, mA, mR = row["sim_trueD"].get("mean"), row["sim_trueAlive"].get("mean"), qstats(R["X"][real_dead, j]).get("mean")
        # position of the real weak-dead mean on the sim alive(0) -> dead(1) axis
        row["real_dead_on_simA2D_axis"] = r4((mR - mA) / (mD - mA)) if None not in (mD, mA, mR) and abs(mD - mA) > 1e-6 else None
        # model attribution: replace column j on real weak-dead frames with draws from sim true-D frames
        Xr = R["X"][real_dead].copy(); base = a2.predict_proba(Xr).mean()
        src = S["X"][sims_D, j]; Xr[:, j] = src[rng.integers(0, len(src), len(Xr))]
        row["swap_to_simD_dmeanPD_a2"] = r4(a2.predict_proba(Xr).mean() - base)
        Xr = R["X"][real_dead].copy(); base_n = models["nba_s100"].predict_proba(Xr).mean()
        Xr[:, j] = src[rng.integers(0, len(src), len(Xr))]
        row["swap_to_simD_dmeanPD_nba"] = r4(models["nba_s100"].predict_proba(Xr).mean() - base_n)
        t2[f] = row
    out["feature_shift"] = t2
    out["feature_shift_base_meanPD_real_weakdead"] = {"a2_full": r4(a2.predict_proba(R["X"][real_dead]).mean()),
                                                      "nba_s100": r4(models["nba_s100"].predict_proba(R["X"][real_dead]).mean())}

    # ---------- 3. decoder margins ----------
    t3 = {}
    for nm, D_, sel in (("real_sp_in", R, R["sp_in"] & R["nb"]), ("real_sp_pre2s", R, R["sp_pre2s"] & R["nb"]),
                        ("real_phase_mid", R, R["phase_mid"] & R["nb"]),
                        ("sim_trueD_nb", S, S["truth_D"] & S["nb"]), ("sim_sp_in_trueD", S, S["sp_in"] & S["truth_D"] & S["nb"]),
                        ("sim_trueAlive_nb", S, ~S["truth_D"] & S["nb"] & S["graded"])):
        T = D_["terms"]; row = {"n": int(sel.sum())}
        for k in ("ball", "tag", "pathcrf"):
            mg = T[f"{k}_margin"][sel]
            row[f"{k}_opinion_share"] = r4(np.isfinite(mg).mean())
            row[f"{k}_absHA_mean"] = r4(np.nanmean(mg)) if np.isfinite(mg).any() else None
            row[f"{k}_adv_vs_D_mean"] = r4(T[f"{k}_adv"][sel].mean())
        row["dead_adv_mean"] = r4(T["dead_adv"][sel].mean())
        row["net_mean"] = r4(T["net"][sel].mean()); row["net_pos_share"] = r4((T["net"][sel] > 0).mean())
        t3[nm] = row
    out["decoder_margins"] = t3
    out["hi_pD_frames"] = {"real_a2": hi_pd_analysis([R]), "sim_a2": hi_pd_analysis(S_list),
                           "real_nba": hi_pd_analysis([R], "nba_s100")}

    # ---------- 4. counterfactuals ----------
    def run_cf(cfg_cf, pD_real, pD_sims, extra=None):
        evr = dict(ev_real); evr["dead"] = dead_evidence(pD_real)
        if extra is not None:
            evr = extra(evr, R)
        if cfg_cf.tag is not cfg.tag:
            evr["tag"] = tag_prior_evidence(tags, m.n_frames, cfg_cf.tag)
        lab = predict(m, tags, cfg_cf, None, evidence=evr, strong=strong_real)
        res = {"real_nb_D_share": r4((lab[R["nb"]] == "D").mean()), "real_sp_in_D": r4((lab[R["sp_in"] & R["nb"]] == "D").mean()),
               "real_phase_mid_D": r4((lab[R["phase_mid"] & R["nb"]] == "D").mean()), "real_graded_D": r4((lab[m.graded] == "D").mean())}
        f1 = []; nbD = []
        for i, (s, ev) in enumerate(zip(sims, sim_ev)):
            evs = dict(ev); evs["dead"] = dead_evidence(pD_sims[i])
            if extra is not None:
                evs = extra(evs, S_list[i])
            if cfg_cf.tag is not cfg.tag:
                evs["tag"] = tag_prior_evidence(s.tags, s.match.n_frames, cfg_cf.tag)
            lb = predict(s.match, s.tags, cfg_cf, None, evidence=evs, strong=sim_strong[i])
            f1.append(score(s.truth["ctrl_Dfull"], lb)["spell_f1"]); nbD.append((lb[S_list[i]["nb"]] == "D").mean())
        res["sim_ctrl_Dfull_spellF1_insample"] = r4(np.mean(f1)); res["sim_nb_D_share"] = r4(np.mean(nbD))
        return res

    pr_a2 = R["pD"]["a2_full"]; ps_a2 = [d["pD"]["a2_full"] for d in S_list]
    pr_n = R["pD"]["nba_s100"]; ps_n = [d["pD"]["nba_s100"] for d in S_list]

    def with_tau(t):
        c = copy.deepcopy(cfg); c.decoder.tau_HD = c.decoder.tau_DH = t; return c

    def with_wdead(w):
        c = copy.deepcopy(cfg); c.decoder.weights["dead"] = w; return c

    def sp_neutral():
        c = copy.deepcopy(cfg); c.tag.code_table = dict(c.tag.code_table); c.tag.code_table["SET PIECES"] = {"sign": 0, "p": 0.5}; return c

    def sp_floor(bonus, pre_only=False):
        def f(ev, d):
            ev = dict(ev); add = np.full((len(d["nb"]), 3), np.nan, np.float32)
            sel = d["sp_pre2s"] if pre_only else (d["sp_pre2s"] | d["sp_in"])
            add[sel, 0] = add[sel, 1] = 0.0; add[sel, 2] = bonus
            ev["sp_dead_prior"] = add; return ev
        return f

    cfs = {"a2_as_is": (cfg, pr_a2, ps_a2, None),
           "tau_D_30": (with_tau(30.0), pr_a2, ps_a2, None), "tau_D_10": (with_tau(10.0), pr_a2, ps_a2, None),
           "w_dead_4": (with_wdead(4.0), pr_a2, ps_a2, None),
           "policy_nba_s100": (cfg, pr_n, ps_n, None),
           "sp_tag_neutral": (sp_neutral(), pr_a2, ps_a2, None),
           "sp_dead_prior_1.5": (cfg, pr_a2, ps_a2, sp_floor(1.5)),
           "sp_dead_prior_3": (cfg, pr_a2, ps_a2, sp_floor(3.0))}
    c = sp_neutral(); c.decoder.tau_HD = c.decoder.tau_DH = 30.0
    cfs["sp_neutral+tau_D_30"] = (c, pr_a2, ps_a2, None)

    def smooth(pv, w=50):   # centred 2 s mean of the logit
        lg = np.log(pv + 1e-3) - np.log(1 - pv + 1e-3)
        lg = np.convolve(lg, np.ones(w) / w, mode="same")
        return (1 / (1 + np.exp(-lg))).astype(np.float32)
    cfs["pD_logit_smooth_2s"] = (cfg, smooth(pr_a2), [smooth(v) for v in ps_a2], None)
    c = with_wdead(4.0); c.decoder.tau_HD = c.decoder.tau_DH = 30.0
    cfs["w_dead_4+tau_D_30"] = (c, pr_a2, ps_a2, None)
    cfs["w_dead_4+nba"] = (with_wdead(4.0), pr_n, ps_n, None)

    def no_pathcrf():
        c = copy.deepcopy(cfg); c.decoder.weights["pathcrf"] = 0.0; return c
    cfs["w_pathcrf_0"] = (no_pathcrf(), pr_a2, ps_a2, None)
    t4 = {}
    for nm, (c, pr, ps, ex) in cfs.items():
        print("  counterfactual", nm, flush=True)
        t4[nm] = run_cf(c, pr, ps, ex)
    out["counterfactuals"] = t4
    out["sp_profile"] = {"real": {str(k): v for k, v in sp_profile([R]).items()},
                         "sim": {str(k): v for k, v in sp_profile(S_list).items()}}

    dst = SEL / "visible_dead_diag.json"
    dst.write_text(json.dumps(out, indent=1))
    print("wrote", dst)
    # ---------- console summary (aggregates) ----------
    print(json.dumps({"weak_label_pD": t1, "decoder_margins": t3, "hi_pD_frames": out["hi_pD_frames"], "counterfactuals": t4}, indent=1))


if __name__ == "__main__":
    sys.exit(main())
