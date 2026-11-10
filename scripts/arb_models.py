"""Candidate families for the arbitration experiments: registered models (real match) and leave-one-match-out fold
models (simulator folds, out of fold). Everything is read from existing artefacts under runs/; nothing is trained here
except wrapping the fold BiGRU checkpoints (see build_b_ckpts).

Fold-model sources (laptop paths relative to runs/select):
  a5c   old (pre-smoothing) sim build runs/sim_prev_nosmooth: exp_seq/lomo/models/{dead,team}_fold{F}, merge/f{F}_a5c
  a13r  new build runs/sim: exp_seq2/lomo/models/{dead,team}_fold{F}, merge/f{F}_a13 (a13r = a13 refit under sklearn 1.9.1)
  a14   new build: a13r dead/team fold models, merge OFF, Model B = exp_seq3/base/models/f{F}_final.pt
        (a recipe-identical retrain of a14's Model B; a14's own fold weights were not saved)
  a16   new build: same dead/team, Model B = exp_seq3/v1/models/f{F}_final.pt (exact a16 recipe, per-half norm v1)
  a10   real match only (its fold Model A was never saved): registered models
"""
import copy, json, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

FAMILIES = ("a5c", "a13r", "a14", "a16", "a10")
SIM_FAMILIES = ("a5c", "a13r", "a14", "a16")
FOLD_MATCHES = (9001, 9002, 9003, 9004, 9005, 9006)

SPEC = {
    "a5c": dict(build="sim_prev_nosmooth", dead="exp_seq/lomo/models/dead_fold{F}.joblib", team="exp_seq/lomo/models/team_fold{F}.joblib",
                merge="../arb/models/a5c_merge_f{F}.joblib", B=None),
    "a13r": dict(build="sim", dead="exp_seq2/lomo/models/dead_fold{F}.joblib", team="exp_seq2/lomo/models/team_fold{F}.joblib",
                 merge="exp_seq2/lomo/merge/f{F}_a13.joblib", B=None),
    "a14": dict(build="sim", dead="exp_seq2/lomo/models/dead_fold{F}.joblib", team="exp_seq2/lomo/models/team_fold{F}.joblib",
                merge=None, B="exp_seq3/base/models/f{F}_final.pt", norm=None),
    "a16": dict(build="sim", dead="exp_seq2/lomo/models/dead_fold{F}.joblib", team="exp_seq2/lomo/models/team_fold{F}.joblib",
                merge=None, B="exp_seq3/v1/models/f{F}_final.pt", norm="v1"),
}


def runs_dir():
    return ROOT / "runs"


def sim_dir(fam):
    return runs_dir() / SPEC[fam]["build"]


def fold_ckpt_dir(fam, F):
    return runs_dir() / "arb" / "models" / f"{fam}_f{F}"


def build_b_ckpts(torch):
    """Wrap each fold-final BiGRU state (keys state_dict/mu/sd) into the checkpoint format seq_b_infer reads
    (names, nf, hidden, win, stride, norm), under runs/arb/models/{fam}_f{F}/seq_modelB.pt. Needs torch."""
    for fam in ("a14", "a16"):
        sp = SPEC[fam]
        names = json.loads((runs_dir() / "select" / fam / "config.json").read_text())["seq_stream"]["step_features"]
        for F in FOLD_MATCHES:
            ck = torch.load(runs_dir() / "select" / sp["B"].format(F=F), weights_only=False)
            out = {"state_dict": ck["state_dict"], "mu": ck["mu"], "sd": ck["sd"], "names": list(names), "nf": len(names),
                   "hidden": 64, "win": 600, "stride": 300}
            if sp["norm"]:
                out["norm"] = sp["norm"]
            d = fold_ckpt_dir(fam, F); d.mkdir(parents=True, exist_ok=True)
            torch.save(out, d / "seq_modelB.pt")
    print("wrapped fold checkpoints for a14, a16")


def load_family(fam, fold=None):
    """dict(cfg, dead, team, merge, seq, sel_dir) for family `fam`: the registered models (fold None) or the
    out-of-fold models of held-out match `fold`."""
    from make_candidate import load_selection
    from mtah.evidence.dead import DeadModel
    from mtah.evidence.team_ctrl import TeamCtrlModel
    from mtah.spells.merge import MergeModel
    sel = runs_dir() / "select" / fam
    cfg, dead, lengths, team, merge, switch, seq = load_selection(sel, with_seq=True)
    sel_dir = sel
    if fold is not None:
        sp = SPEC[fam]; F = int(fold)
        S = runs_dir() / "select"
        dead = DeadModel.load(S / sp["dead"].format(F=F))
        team = TeamCtrlModel.load(S / sp["team"].format(F=F))
        merge = MergeModel.load(S / sp["merge"].format(F=F)) if sp["merge"] else None
        if seq is not None:
            seq = copy.deepcopy(seq); sel_dir = fold_ckpt_dir(fam, F)
            if "seqA" in seq["weights"]:
                raise RuntimeError(f"{fam}: no fold Model A")
    return dict(cfg=cfg, dead=dead, team=team, merge=merge, seq=seq, sel_dir=sel_dir, lengths=lengths)
