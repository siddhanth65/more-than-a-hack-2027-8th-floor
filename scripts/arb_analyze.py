"""Tables for Experiment 1 (stream transfer gap) and Experiment 2 (masked-ball proxy), from the arb_run.py outputs.
Aggregates only: no frame-level contest data is printed or stored.

  python scripts/arb_analyze.py exp1 --out runs/arb/out --res runs/arb/exp1.json
  python scripts/arb_analyze.py exp2 --out runs/arb/out --res runs/arb/exp2.json [--workers 16]
"""
import argparse, json, os, sys
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "scripts"))

import numpy as np

import arb_common as ac

BASE = ("a5c", "a13r", "a14", "a16")
P_TYPE = ("a5c", "a13r")                     # hand-built-stream family (rival: a14); everything else is learned (rival: a5c)
SIM_KEYS = [f"game_{m}_{h}_s100" for m in range(9001, 9007) for h in ("H1", "H2")]
SIM_ALL = SIM_KEYS + [k.replace("s100", "s101") for k in SIM_KEYS]
NEAR_SP = 125                               # +-5 s around SET PIECES tags
EDGE = 25                                   # 1 s around reference label changes (not confident)
CODES = np.array(["H", "A", "D", "X"])


def load_unit(name):
    """(match, tags, truth dict or None) of a unit: new-build sim half or the real match."""
    if name == "real":
        from mtah.io.metrica_loader import load_tactical
        from mtah.io.tags import load_tags
        d = os.environ["MTAH_DATA_DIR"]
        return load_tactical(d), load_tags(d), None
    from mtah.sim.dataset import load_half_sim
    key, seed = name.rsplit("_s", 1)
    s = load_half_sim(ROOT / "runs" / "sim", key, int(seed))
    return s.match, s.tags, s.truth


def family_of(cfgname):
    return cfgname.split("_")[0]


def is_ptype(cfgname):
    return family_of(cfgname) in P_TYPE


def sigm_margin(st):
    st = st.astype(np.float32)
    return st[:, 0] - st[:, 1]


# ---------------------------------------------------------------- Experiment 1
def cov_cells(match, idx, L=105.3):
    """(third, n_players_bin, speed_bin) per frame index: pitch third of the ball x (3), visible players (<=18, 19-20,
    21-22, >22), ball speed (<3, 3-6, >6 m/s)."""
    x = match.ball[idx, 0]
    third = np.digitize(x, [-L / 6, L / 6])
    npl = np.isfinite(match.pos[idx, :, 0]).sum(1)
    npb = np.digitize(npl, [19, 21, 23])
    sp = np.nan_to_num(match.ball_speed[idx], nan=0.0)
    spb = np.digitize(sp, [3.0, 6.0])
    return third * 12 + npb * 3 + spb


def exp1(a):
    rule = ac.RuleParams(**json.loads((ROOT / "runs" / "arb" / "rule.json").read_text())["rule"])
    out = Path(a.out)
    streams = []
    for fam in BASE:
        streams += [(fam, "pathcrf"), (fam, "teamctrl")] + ([(fam, "seqB")] if fam in ("a14", "a16") else [])
    rows = []      # per (unit, mask file): stream -> counts
    for name in SIM_KEYS + ["real"]:
        match, tags, truth = load_unit(name)
        conf = ac.confident_control(match, rule)
        for cm in ("c0", "c1"):
            f = out / f"{name}_m{cm}.npz"
            if not f.exists():
                continue
            z = np.load(f)
            mask, centres = z["mask"], z["centres"]
            near = np.zeros(match.n_frames, bool)
            for c in centres:
                near[max(0, c - 75):c + 76] = True            # +-3 s of a centre: >= 7 s of hidden ball on either side
            E = np.flatnonzero((conf >= 0) & near & mask)
            if len(E) == 0:
                continue
            y = conf[E]                                         # 0 = home
            cells = cov_cells(match, E)
            for fam, sn in streams:
                st = z[f"st_{fam}_{sn}"][E]
                m = sigm_margin(st); ok = np.isfinite(m)
                sgn = np.where(y == 0, 1.0, -1.0)
                ll = -np.logaddexp(0.0, -sgn * m)
                rows.append({"unit": name, "mask": cm, "fam": fam, "stream": sn, "half": (int(match.half[E[0]]) if name == "real" else 0),
                             "E": E, "ok": ok, "acc": (np.sign(m) == sgn)[ok], "ll": ll[ok], "cells": cells[ok],
                             "halfv": match.half[E][ok], "y": y[ok]})
    res = {"rule": rule.to_dict(), "tables": {}}
    # real mix of cells (confident frames in the hidden windows, per family-stream the same frames up to NaN)
    def pool(rs):
        return (np.concatenate([r["acc"] for r in rs]), np.concatenate([r["ll"] for r in rs]), np.concatenate([r["cells"] for r in rs]))
    for fam, sn in streams:
        sim_rs = [r for r in rows if r["fam"] == fam and r["stream"] == sn and r["unit"] != "real"]
        real_rs = [r for r in rows if r["fam"] == fam and r["stream"] == sn and r["unit"] == "real"]
        if not sim_rs or not real_rs:
            continue
        per_fold = {}
        for r in sim_rs:
            per_fold.setdefault(r["unit"].split("_")[1], []).append(r)
        fa = {k: float(np.concatenate([r["acc"] for r in v]).mean()) for k, v in per_fold.items()}
        fl = {k: float(np.concatenate([r["ll"] for r in v]).mean()) for k, v in per_fold.items()}
        acc_s, ll_s, cell_s = pool(sim_rs); acc_r, ll_r, cell_r = pool(real_rs)
        pr = np.bincount(cell_r, minlength=36) / len(cell_r); ps = np.bincount(cell_s, minlength=36) / len(cell_s)
        w = np.where(ps[cell_s] > 0, pr[cell_s] / np.maximum(ps[cell_s], 1e-12), 0.0)
        covered = float(pr[ps > 0].sum())
        wacc = float((w * acc_s).sum() / max(w.sum(), 1e-12))
        # reweighted per fold
        fwa = {}
        for k, v in per_fold.items():
            a_, _, c_ = pool(v); ww = np.where(ps[c_] > 0, pr[c_] / np.maximum(ps[c_], 1e-12), 0.0)
            fwa[k] = float((ww * a_).sum() / max(ww.sum(), 1e-12))
        halves = {}
        for h in (1, 2):
            hv = np.concatenate([r["acc"][r["halfv"] == h] for r in real_rs]); lv = np.concatenate([r["ll"][r["halfv"] == h] for r in real_rs])
            halves[str(h)] = {"acc": float(hv.mean()) if len(hv) else None, "ll": float(lv.mean()) if len(lv) else None, "n": int(len(hv))}
        res["tables"][f"{fam}/{sn}"] = {
            "sim_acc_folds_mean": float(np.mean(list(fa.values()))), "sim_acc_folds_sd": float(np.std(list(fa.values()), ddof=1)),
            "sim_acc_pooled": float(acc_s.mean()), "sim_acc_reweighted_pooled": wacc,
            "sim_acc_reweighted_folds_mean": float(np.mean(list(fwa.values()))), "reweight_real_mass_covered": covered,
            "sim_ll_folds_mean": float(np.mean(list(fl.values()))), "sim_n": int(len(acc_s)),
            "real_acc": float(acc_r.mean()), "real_ll": float(ll_r.mean()), "real_n": int(len(acc_r)), "real_halves": halves,
            "gap_raw": float(np.mean(list(fa.values())) - acc_r.mean()), "gap_reweighted": wacc - float(acc_r.mean()),
            "circular_ball_based": sn == "teamctrl"}
    # coverage / class balance of the pseudo-labels (unmasked, whole units)
    cov = {}
    for name in SIM_KEYS + ["real"]:
        match, _, _ = load_unit(name)
        c = ac.confident_control(match, rule)
        k = "real" if name == "real" else "sim"
        d = cov.setdefault(k, {"conf": 0, "H": 0, "A": 0, "visible": 0, "graded": 0})
        d["conf"] += int((c >= 0).sum()); d["H"] += int((c == 0).sum()); d["A"] += int((c == 1).sum())
        d["visible"] += int((match.ball_present & match.graded).sum()); d["graded"] += int(match.graded.sum())
    for k, d in cov.items():
        d["cov_visible"] = d["conf"] / d["visible"]; d["H_share"] = d["H"] / max(d["conf"], 1)
    res["coverage"] = cov
    # signal T
    t = res["tables"]
    sig = {}
    for fam in ("a14", "a16"):
        if f"{fam}/seqB" in t and "a14/pathcrf" in t:
            B = t[f"{fam}/seqB"]; Pc = t["a14/pathcrf"]
            sig[fam] = {"gap_B": B["gap_raw"], "gap_PathCRF": Pc["gap_raw"], "gapB_minus_gapPC": B["gap_raw"] - Pc["gap_raw"],
                        "gap_B_rw": B["gap_reweighted"], "gap_PC_rw": Pc["gap_reweighted"],
                        "real_B": B["real_acc"], "real_PC": Pc["real_acc"], "realB_minus_realPC": B["real_acc"] - Pc["real_acc"]}
    res["signal_T"] = sig
    Path(a.res).write_text(json.dumps(res, indent=1))
    print(json.dumps({"coverage": cov, "signal_T": sig}, indent=1))
    for k, v in t.items():
        print(f"{k:16s} sim {v['sim_acc_folds_mean']:.3f}+-{v['sim_acc_folds_sd']:.3f} (rw {v['sim_acc_reweighted_pooled']:.3f}) real {v['real_acc']:.3f} "
              f"[h1 {v['real_halves']['1']['acc']}, h2 {v['real_halves']['2']['acc']}] gap {v['gap_raw']:+.3f} (rw {v['gap_reweighted']:+.3f}) n_sim {v['sim_n']} n_real {v['real_n']}")


# ---------------------------------------------------------------- Experiment 2
def ref_filter(ref, half, tags, fps, tag_offset=0):
    """(F,) bool: frames where the reference is confident: not X, >= 1 s from a reference label change (within a half),
    and not within +-5 s of a SET PIECES tag."""
    ok = ref != 3
    ch = np.flatnonzero((np.diff(ref.astype(np.int16)) != 0) | (np.diff(half.astype(np.int16)) != 0)) + 1
    for c in ch:
        ok[max(0, c - EDGE):c + EDGE] = False
    sp = tags[tags["code"] == "SET PIECES"]["start"].to_numpy()
    for s in sp:
        s = int(s) + tag_offset
        ok[max(0, s - NEAR_SP):s + NEAR_SP + 1] = False
    return ok


def _str(codes):
    return CODES[np.asarray(codes)]


def score_codes(ref_codes, pred_codes, window):
    """spellF1, macroF1 of pred against ref restricted to the window (bool)."""
    from mtah.eval.scorer import score
    g = np.full(len(ref_codes), "X", dtype="<U1")
    w = window & (ref_codes != 3)
    if not w.any():
        return np.nan, np.nan
    g[w] = _str(ref_codes)[w]
    r = score(g, _str(np.minimum(pred_codes, 2)))
    return r["spell_f1"], r["macro_f1"]


def mask_windows(mask, half):
    from mtah.sim.blackout_sim import runs_of
    return runs_of(mask & (half > 0))


def exp2_unit(args):
    name, outdir = args
    outdir = Path(outdir)
    match, tags, truth = load_unit(name)
    none = np.load(outdir / f"{name}_mnone.npz")
    cfgs = sorted(k[4:] for k in none.files if k.startswith("lab_"))
    base = [c for c in BASE if c in cfgs] + (["a10"] if "a10" in cfgs else [])
    L = {c: none[f"lab_{c}"] for c in cfgs}
    cons_set = [c for c in BASE if c in cfgs]
    agree = np.ones(match.n_frames, bool)
    for c in cons_set[1:]:
        agree &= L[c] == L[cons_set[0]]
    cons = np.where(agree, L[cons_set[0]], 3).astype(np.uint8)
    refs = {}
    for c in cfgs:
        fam = family_of(c)
        refs[c] = {"own": L[c], "cons": cons, "ball": none[f"labball_{fam}"], "rival": L["a14"] if is_ptype(c) else L["a5c"]}
    filt = {}
    for r in ("own", "cons", "ball", "rival_a14", "rival_a5c"):
        pass
    filt_cache = {}

    def get_filter(refname, c):
        key = (refname, id(refs[c][refname]))
        if key not in filt_cache:
            filt_cache[key] = ref_filter(refs[c][refname], match.half, tags, match.fps)
        return filt_cache[key]
    rows = []
    half_arr = match.half
    halves = [0] if truth is not None else [1, 2]
    for ms in range(5):
        f = outdir / f"{name}_m{ms}.npz"
        if not f.exists():
            continue
        z = np.load(f); mask = z["mask"] & (half_arr > 0)
        wins = mask_windows(mask, half_arr)
        # strata: window is dead if the majority of the truth (sim) / of the a5c unmasked decode (real) is D
        strat_src = (truth["ctrl_Dfull"] == "D") if truth is not None else (L["a5c"] == 2)
        dead_w = np.zeros(match.n_frames, bool); live_w = np.zeros(match.n_frames, bool)
        for s, e in wins:
            (dead_w if strat_src[s:e].mean() > 0.5 else live_w)[s:e] = True
        for c in cfgs:
            pred = z[f"lab_{c}"]
            for h in halves:
                hm = (half_arr == h) if h else np.ones(match.n_frames, bool)
                if truth is not None:
                    tr_codes = np.searchsorted(np.array(["H", "A", "D"]), truth["ctrl_Dfull"]).astype(np.uint8)
                    tr_codes = np.where(np.isin(truth["ctrl_Dfull"], ["H", "A", "D"]), tr_codes, 3).astype(np.uint8)
                    for stratum, sw in (("all", mask), ("live", live_w), ("dead", dead_w)):
                        sf, mf = score_codes(tr_codes, pred, sw & hm)
                        rows.append((name, ms, c, h, "truth", stratum, True, sf, mf))
                for refname in ("own", "cons", "ball", "rival"):
                    ref = refs[c][refname]
                    for filt_on in (True, False):
                        ok = get_filter(refname, c) if filt_on else np.ones(match.n_frames, bool)
                        for stratum, sw in (("all", mask), ("live", live_w), ("dead", dead_w)):
                            if stratum != "all" and refname != "rival" and not filt_on:
                                continue
                            sf, mf = score_codes(ref, pred, sw & hm & ok)
                            rows.append((name, ms, c, h, refname, stratum, filt_on, sf, mf))
    return rows


def exp2(a):
    import pandas as pd
    from multiprocessing import Pool
    names = SIM_KEYS + ["real"]
    with Pool(a.workers) as p:
        allrows = sum(p.map(exp2_unit, [(n, a.out) for n in names]), [])
    df = pd.DataFrame(allrows, columns=["unit", "mask", "cfg", "half", "ref", "stratum", "filtered", "spell_f1", "macro_f1"])
    df.to_csv(Path(a.res).with_suffix(".csv"), index=False)
    print("rows", len(df))
    analyze_exp2(df, a)


def analyze_exp2(df, a):
    from scipy.stats import kendalltau, spearmanr
    res = {}
    sim = df[df["unit"] != "real"].copy(); sim["fold"] = sim["unit"].str.split("_").str[1].astype(int)
    real = df[df["unit"] == "real"]
    cfgs = sorted(sim["cfg"].unique())
    # truth and proxies, mean over halves x mask seeds per fold
    T = sim[(sim["ref"] == "truth") & (sim["stratum"] == "all")].groupby(["fold", "cfg"])["spell_f1"].mean().unstack()
    prox = {}
    for ref in ("rival", "cons", "ball", "own"):
        prox[ref] = sim[(sim["ref"] == ref) & (sim["stratum"] == "all") & (sim["filtered"])].groupby(["fold", "cfg"])["spell_f1"].mean().unstack()
    val = {}
    for ref, P in prox.items():
        taus, sps, best = {}, {}, {}
        ordered = {}
        for f in T.index:
            cc = [c for c in T.columns if np.isfinite(T.loc[f, c]) and np.isfinite(P.loc[f, c])]
            t = kendalltau(T.loc[f, cc], P.loc[f, cc])[0]; s = spearmanr(T.loc[f, cc], P.loc[f, cc])[0]
            taus[int(f)] = float(t); sps[int(f)] = float(s); best[int(f)] = bool(T.loc[f, cc].idxmax() == P.loc[f, cc].idxmax())
            ordered[int(f)] = bool(np.sign(T.loc[f, "a13r"] - T.loc[f, "a14"]) == np.sign(P.loc[f, "a13r"] - P.loc[f, "a14"]))
        dem = lambda X: (X.sub(X.mean(1), axis=0)).stack()
        pooled = float(spearmanr(dem(T), dem(P).reindex(dem(T).index))[0])
        val[ref] = {"kendall_per_fold": taus, "kendall_mean": float(np.mean(list(taus.values()))), "folds_tau_positive": int(sum(v > 0 for v in taus.values())),
                    "spearman_per_fold": sps, "spearman_mean": float(np.mean(list(sps.values()))), "pooled_spearman_fold_demeaned": pooled,
                    "same_best_config_per_fold": best, "n_same_best": int(sum(best.values())),
                    "a13r_lt_a14_as_truth_per_fold": ordered, "n_a13r_a14_ordered_as_truth": int(sum(ordered.values())),
                    "n_configs": len(cc)}
    res["proxy_validity_sim"] = val
    res["truth_by_fold"] = T.round(4).to_dict()
    res["sim_mean_truth"] = T.mean().round(4).to_dict()
    # strata (truth vs rival)
    st = {}
    for s in ("live", "dead"):
        Ts = sim[(sim["ref"] == "truth") & (sim["stratum"] == s)].groupby("cfg")["spell_f1"].mean()
        Ps = sim[(sim["ref"] == "rival") & (sim["stratum"] == s) & (sim["filtered"])].groupby("cfg")["spell_f1"].mean()
        c2 = Ts.dropna().index.intersection(Ps.dropna().index)
        st[s] = {"kendall_pooled_means": float(kendalltau(Ts[c2], Ps[c2])[0]), "truth_a5c_a14": [float(Ts.get("a5c", np.nan)), float(Ts.get("a14", np.nan))]}
    res["strata_sim"] = st
    # real table
    rr = real[real["stratum"] == "all"]
    tab = rr.groupby(["cfg", "ref", "filtered", "half"])[["spell_f1", "macro_f1"]].agg(["mean", "std"]).round(4)
    res["real_table"] = {f"{c}|{r}|filt={f}|half{h}": {"spell_mean": float(v[("spell_f1", "mean")]), "spell_sd": float(v[("spell_f1", "std")]),
                                                    "macro_mean": float(v[("macro_f1", "mean")]), "macro_sd": float(v[("macro_f1", "std")])}
                        for (c, r, f, h), v in tab.iterrows()}
    rs = real[(real["stratum"] != "all") & (real["ref"] == "rival") & (real["filtered"])]
    res["real_strata_rival"] = {f"{c}|{s}": float(v) for (c, s), v in rs.groupby(["cfg", "stratum"])["spell_f1"].mean().items()}
    Path(a.res).write_text(json.dumps(res, indent=1))
    for ref, v in val.items():
        print(f"proxy {ref:6s} tau {v['kendall_mean']:+.2f} (folds>0 {v['folds_tau_positive']}/6) spearman {v['spearman_mean']:+.2f} pooled {v['pooled_spearman_fold_demeaned']:+.2f} "
              f"best-match {v['n_same_best']}/6 a13r<a14 ordered {v['n_a13r_a14_ordered_as_truth']}/6")


# ---------------------------------------------------------------- real bootstrap
def bootstrap(a):
    from mtah.eval.scorer import score
    outdir = Path(a.out)
    match, tags, _ = load_unit("real")
    none = np.load(outdir / "real_mnone.npz"); R5, R14 = none["lab_a5c"], none["lab_a14"]
    f5 = ref_filter(R14, match.half, tags, match.fps)     # a5c is scored against a14's unmasked decode
    f14 = ref_filter(R5, match.half, tags, match.fps)
    wins = []                                            # (ref5, pred5, ref14, pred14) per window
    for ms in range(5):
        z = np.load(outdir / f"real_m{ms}.npz"); mask = z["mask"] & (match.half > 0)
        P5, P14 = z["lab_a5c"], z["lab_a14"]
        for s, e in mask_windows(mask, match.half):
            wins.append((np.where(f5[s:e], R14[s:e], 3), P5[s:e], np.where(f14[s:e], R5[s:e], 3), P14[s:e]))
    n = len(wins); rng = np.random.default_rng(20261110)

    def total(idx):
        g5 = []; p5 = []; g14 = []; p14 = []
        gap = np.full(1, 3, np.uint8)
        for i in idx:
            r5, q5, r14, q14 = wins[i]
            g5 += [r5, gap]; p5 += [q5, np.full(1, 0, np.uint8)]; g14 += [r14, gap]; p14 += [q14, np.full(1, 0, np.uint8)]
        out = []
        for g, p in ((g5, p5), (g14, p14)):
            gg = _str(np.concatenate(g)); pp = _str(np.minimum(np.concatenate(p), 2))
            out.append(score(gg, pp)["spell_f1"])
        return out
    base = total(range(n)); diffs = []
    for b in range(a.boot):
        s5, s14 = total(rng.integers(0, n, n)); diffs.append(s5 - s14)
    diffs = np.array(diffs)
    res = {"n_windows": n, "n_boot": a.boot, "score_a5c_vs_a14_ref": base[0], "score_a14_vs_a5c_ref": base[1],
           "P_a5c_beats_a14": float((diffs > 0).mean()), "P_a14_beats_a5c": float((diffs < 0).mean()),
           "diff_mean": float(diffs.mean()), "diff_ci95": [float(np.percentile(diffs, 2.5)), float(np.percentile(diffs, 97.5))]}
    Path(a.res).write_text(json.dumps(res, indent=1)); print(json.dumps(res, indent=1))


def repro(a):
    """Unmasked out-of-fold ctrl_Dfull / raw_Dfull spellF1 per base candidate (24 halves, seeds 100+101) vs the registered LOMO numbers."""
    from mtah.eval.scorer import score
    out = Path(a.out); acc = {}
    for name in SIM_ALL:
        _, _, truth = load_unit(name)
        z = np.load(out / f"{name}_mnone.npz")
        for c in [k[4:] for k in z.files if k.startswith("lab_")]:
            for v in ("ctrl_Dfull", "raw_Dfull"):
                acc.setdefault((c, v), []).append(score(truth[v], _str(np.minimum(z[f"lab_{c}"], 2)))["spell_f1"])
    res = {f"{c}|{v}": float(np.mean(x)) for (c, v), x in sorted(acc.items())}
    res["registered_ctrl_Dfull"] = {"a5c": 0.494, "a13": 0.477, "a14": 0.557, "a16": 0.519}
    Path(a.res).write_text(json.dumps(res, indent=1)); print(json.dumps(res, indent=1))


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("cmd", choices=["exp1", "exp2", "boot", "repro"])
    ap.add_argument("--out", default="runs/arb/out"); ap.add_argument("--res", default=None)
    ap.add_argument("--workers", type=int, default=8); ap.add_argument("--boot", type=int, default=200)
    a = ap.parse_args(); a.res = a.res or f"runs/arb/{a.cmd}.json"
    {"exp1": exp1, "exp2": exp2, "boot": bootstrap, "repro": repro}[a.cmd](a)


if __name__ == "__main__":
    main()
