"""Amendment 5 LOMO helpers (aggregates only).
  basepred --fold F --out O   (torch env) base B of fold F (O/base/fF.pt, unchanged) applied to F's held-out halves of the
                              CURRENT lomo tree (shifted-tag stand-in halves) -> O/pred/base/fF
  score --pred P --variant V --csv C [--lomo D]   (main env) decode every held-out half at S's registered cell
                              (dead -4.0, seq_d +0.5, B weight 12) and write per-half spellF1 rows for the four truth variants.
                              --lomo patches the lomo tree (reference run reads the original, unshifted tree)."""
import argparse, os, sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "scripts"))
import numpy as np

MATCHES = [9001, 9002, 9003, 9004, 9005, 9006]; SEEDS = [100, 101]


def basepred(a):
    import torch
    import exp_seq_lomo_gru as gl, exp_seq_gru as g
    import s_selftrain as S
    out = Path(a.out); F = a.fold
    ck = S.load_base(out / "base" / f"f{F}.pt")
    net = g.Net(ck["nf"], ck.get("hidden", 64)); net.load_state_dict(ck["state_dict"])
    d = out / "pred" / "base" / f"f{F}"; d.mkdir(parents=True, exist_ok=True)
    for k, s in gl.halves_of(F):
        X, starts, seg, y, names = gl.load(gl.fpath(gl.src_of(F, F), k, s))
        np.savez_compressed(d / f"{k}_s{s}.npz", logp=g.infer(net, X, ck["mu"], ck["sd"], names), starts=starts, y=y)
    print("basepred", F, flush=True)


def score(a):
    import pandas as pd
    import exp_seq_lomo as L
    import exp_seq_lomo_gru as gl
    import s_selftrain as S
    if a.lomo:
        L.L = Path(a.lomo); gl.L = Path(a.lomo)
    rows = []
    for F in MATCHES:
        for k in S.keys_of(F):
            for s in SEEDS:
                p = Path(a.pred) / a.variant / f"f{F}" / f"{k}_s{s}.npz"
                h = S.HalfB(Path(a.lomo or L.L), F, k, s, a.sim, p)
                lab, _ = S.decode_cell(h, dead_offset=-4.0, seq_d=0.5)
                rows += L.score_rows(h, lab, {"fold": F})
    df = pd.DataFrame(rows); df.to_csv(a.csv, index=False)
    for v in L.VARS:
        g = df[df.variant == v]
        print(a.csv, v, "n_rows", len(g), "mean spellF1 %.4f" % g.spell_f1.mean())


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("stage", choices=["basepred", "score"])
    ap.add_argument("--fold", type=int); ap.add_argument("--out"); ap.add_argument("--pred"); ap.add_argument("--variant", default="V3")
    ap.add_argument("--csv"); ap.add_argument("--lomo"); ap.add_argument("--sim", default=str(ROOT / "runs" / "sim"))
    a = ap.parse_args(); {"basepred": basepred, "score": score}[a.stage](a)


if __name__ == "__main__":
    main()
