"""Run PathCRF on one real-match half under an input transformation and score it on the confident-control frames
(label-free pseudo-labels, rule.json). Aggregates only.
  python scripts/arb_pcrf_variants.py HALF VARIANT       VARIANT in asis xflip yflip rot swap swap_xflip swap_rot
"""
import json, os, subprocess, sys, tempfile
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "scripts"))
import numpy as np
import arb_common as ac
from mtah.evidence.pathcrf import PathCRFParams, write_half_input, read_half_output, WRAPPER
from mtah.io.match import half_runs
from mtah.io.metrica_loader import load_tactical

TRANS = {"asis": (1, 1, 0), "xflip": (-1, 1, 0), "yflip": (1, -1, 0), "rot": (-1, -1, 0), "swap": (1, 1, 1),
         "swap_xflip": (-1, 1, 1), "swap_rot": (-1, -1, 1)}


def main():
    half, var = int(sys.argv[1]), sys.argv[2]
    sx, sy, sw = TRANS[var]
    m = load_tactical(os.environ["MTAH_DATA_DIR"])
    rule = ac.RuleParams(**json.load(open(ROOT / "runs/arb/rule.json"))["rule"]); conf = ac.confident_control(m, rule)
    p = PathCRFParams()
    (h, a, b), = [r for r in half_runs(m.half) if r[0] == half]
    pos = m.pos[a:b].copy(); pos[..., 0] *= sx; pos[..., 1] *= sy
    team = (1 - m.team) if sw else m.team
    with tempfile.TemporaryDirectory() as tmp:
        inp, out = f"{tmp}/i.npz", f"{tmp}/o.npz"
        np.savez(inp, pos=pos.astype(np.float32), team=team.astype(np.int8), fps=np.float32(m.fps), pitch=np.asarray(m.pitch, np.float32))
        subprocess.run([p.python, str(WRAPPER), "--in", inp, "--out", out, "--ckpt", str(p.ckpt), "--repo", p.repo, "--threads", "4"],
                       check=True, env=dict(os.environ, TQDM_DISABLE="1"), stdout=subprocess.DEVNULL)
        r = read_half_output(out, a)
    ph = r["p_home"]
    if sw:
        ph = 1 - ph
    c = conf[a:b]; i = np.flatnonzero((c >= 0) & np.isfinite(ph))
    acc = float(((ph[i] > 0.5) == (c[i] == 0)).mean())
    res = {"half": half, "variant": var, "n": int(len(i)), "acc": acc, "says_home_if_home": float((ph[i][c[i] == 0] > 0.5).mean()),
           "says_home_if_away": float((ph[i][c[i] == 1] > 0.5).mean())}
    Path(ROOT / "runs/arb/pcrf_variants").mkdir(parents=True, exist_ok=True)
    (ROOT / f"runs/arb/pcrf_variants/h{half}_{var}.json").write_text(json.dumps(res)); print(res)


if __name__ == "__main__":
    main()
