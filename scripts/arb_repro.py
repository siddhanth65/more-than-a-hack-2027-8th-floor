"""Regenerate the registered candidate CSVs (real match) in this checkout and compare md5s with the pinned ones.
  python scripts/arb_repro.py --out runs/arb/repro [--only a5c a16]
"""
import argparse, hashlib, json, os, sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "scripts"))
import arb_models as am
from make_candidate import candidate_labels
from mtah.io.metrica_loader import load_tactical
from mtah.io.tags import load_tags
from mtah.submission import validate, write_submission

EXPECT = {"a5c": "2a672be6795bdfb0c9699fea0f235730", "a10": "b8f428b35989587f4b0f78ad103f6b10",
          "a14": "60ab67b7a60e66a398898ce46410196c", "a16": "4a4ca7765dc3bafa51181ac853f77c65", "a13r": None}


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--out", default="runs/arb/repro"); ap.add_argument("--only", nargs="*")
    a = ap.parse_args(); d = os.environ["MTAH_DATA_DIR"]
    m = load_tactical(d); tags = load_tags(d); res = {}
    Path(a.out).mkdir(parents=True, exist_ok=True)
    for fam in (a.only or list(EXPECT)):
        mdl = am.load_family(fam)
        cfg = mdl["cfg"]; cfg.pathcrf.cache_dir = str(ROOT / "runs" / "cache" / "pathcrf")
        lab = candidate_labels(m, tags, cfg, mdl["dead"], mdl["team"], mdl["merge"], None, cache_key="real_tactical",
                               seq=mdl["seq"], sel_dir=mdl["sel_dir"])
        p = write_submission(lab, Path(a.out) / f"{fam}.csv"); md5 = hashlib.md5(Path(p).read_bytes()).hexdigest()
        res[fam] = {"md5": md5, "expected": EXPECT[fam], "match": (md5 == EXPECT[fam]) if EXPECT[fam] else None, "valid": validate(p) is None}
        print(fam, res[fam], flush=True)
    (Path(a.out) / "repro.json").write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
