"""Label-free player-tracking quality: real match vs sim (seed 100), inside/outside ball blackouts.
Aggregates only. Writes runs/select/players_shift.json."""
import json, os, sys
from pathlib import Path
from dataclasses import asdict
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from mtah.sim.blackout_sim import runs_of
from mtah.sim.corrupt import CorruptParams
from mtah.sim.dataset import load_half_sim, list_sim_keys

THR = (2.0, 3.0, 5.0)


def gap_runs(nan):
    """nan (F,S) bool -> arrays slot, start, end (end exclusive) of NaN runs."""
    pad = np.zeros((nan.shape[0] + 2, nan.shape[1]), np.int8)
    pad[1:-1] = nan
    d = np.diff(pad, axis=0)
    ss, sc = np.nonzero(d.T == 1)
    es, ec = np.nonzero(d.T == -1)
    return ss, sc, ec  # slot, start, end (sorted consistently per slot)


def accumulate(pos, bo, fps, acc):
    """pos (F,S,2) one contiguous graded segment; bo (F,) bool."""
    F, S, _ = pos.shape
    nan = np.isnan(pos[..., 0])
    vis = ~nan
    for name, fm in (("in", bo), ("out", ~bo)):
        a = acc[name]
        n = vis[fm].sum(1)
        a["n_players"].append(n.astype(np.int16))
    # gaps
    slot, st, en = gap_runs(nan)
    ln = en - st
    cat_in = bo[st]
    for name, sel in (("in", cat_in), ("out", ~cat_in)):
        acc[name]["gap_len"].append(ln[sel])
        # reappearance jump: gap >= 1 s with visible both sides
        ok = sel & (ln >= fps) & (st > 0) & (en < F)
        p0 = pos[st[ok] - 1, slot[ok]]
        p1 = pos[en[ok], slot[ok]]
        acc[name]["reapp"].append(np.hypot(*(p1 - p0).T))
    # consecutive-frame jumps
    both = vis[1:] & vis[:-1]
    d = np.hypot(pos[1:, :, 0] - pos[:-1, :, 0], pos[1:, :, 1] - pos[:-1, :, 1])
    d = np.where(both, d, 0)
    for name, fm in (("in", bo[1:]), ("out", ~bo[1:])):
        a = acc[name]
        a["vis_frames"] += int(vis[1:][fm].sum())
        a["pair_frames"] += int(both[fm].sum())
        for t in THR:
            a[f"tp{t:g}"] += int((d[fm] > t).sum())
        a["fast"] += int((d[fm] * fps > 10).sum())


def finalize(acc, fps):
    out = {}
    for name in ("in", "out"):
        a = acc[name]
        n = np.concatenate(a["n_players"]) if a["n_players"] else np.zeros(0)
        gl = np.concatenate(a["gap_len"]) if a["gap_len"] else np.zeros(0)
        rj = np.concatenate(a["reapp"]) if a["reapp"] else np.zeros(0)
        pm = a["vis_frames"] / fps / 60
        q = lambda x, p: float(np.percentile(x, p)) if len(x) else None
        r = {"frames": int(len(n)),
             "n_players_p10": q(n, 10), "n_players_p50": q(n, 50), "n_players_mean": float(n.mean()) if len(n) else None,
             "share_ge20": float((n >= 20).mean()) if len(n) else None,
             "share_ge22": float((n >= 22).mean()) if len(n) else None,
             "gap_runs": int(len(gl)),
             "gap_p50_s": q(gl / fps, 50), "gap_p90_s": q(gl / fps, 90),
             "gap_frac_missing_in_runs_gt10s": float(gl[gl > 10 * fps].sum() / gl.sum()) if gl.sum() else None,
             "reapp_n": int(len(rj)), "reapp_jump_p50_m": q(rj, 50), "reapp_jump_p90_m": q(rj, 90),
             "fast_share_gt10ms": a["fast"] / max(a["pair_frames"], 1),
             "player_minutes": pm}
        for t in THR:
            r[f"teleport_per_player_min_{t:g}m"] = a[f"tp{t:g}"] / pm if pm else None
        out[name] = r
    return out


def new_acc():
    return {k: {"n_players": [], "gap_len": [], "reapp": [], "vis_frames": 0, "pair_frames": 0, "fast": 0,
                **{f"tp{t:g}": 0 for t in THR}} for k in ("in", "out")}


def run(pos, graded, bo, fps):
    acc = new_acc()
    for s, e in runs_of(graded):
        accumulate(pos[s:e], bo[s:e], fps, acc)
    return finalize(acc, fps)


def main():
    from mtah.io.metrica_loader import load_tactical
    from mtah.eval.drift import blackout_mask_from_match
    data_dir = os.environ.get("MTAH_DATA_DIR")
    if not data_dir:
        raise SystemExit("MTAH_DATA_DIR must point at the unpacked competition data bundle")
    m = load_tactical(data_dir)
    real = run(m.pos, np.asarray(m.graded, bool), blackout_mask_from_match(m), m.fps)
    d = ROOT / "runs" / "sim"
    per = {}
    for key in list_sim_keys(d):
        if not (d / f"{key}_s100.json").exists():
            continue
        h = load_half_sim(d, key, 100)
        per[key] = run(h.match.pos, np.asarray(h.match.half > 0), np.asarray(h.blackout, bool), h.match.fps)
    avg = {}
    for name in ("in", "out"):
        avg[name] = {k: float(np.mean([v[name][k] for v in per.values() if v[name][k] is not None]))
                     for k in per[next(iter(per))][name]}
    bp = json.loads((ROOT / "configs" / "blackout_sim.json").read_text())
    cp = CorruptParams()
    cj = ROOT / "configs" / "corrupt.json"
    if cj.exists():
        cp = CorruptParams.from_dict(json.loads(cj.read_text()))
    res = {"real": real, "sim": {"mean": avg, "per_key": per}, "params": {"corrupt": asdict(cp), "blackout": bp}}
    out = ROOT / "runs" / "select"; out.mkdir(parents=True, exist_ok=True)
    (out / "players_shift.json").write_text(json.dumps(res, indent=1))
    keys = [k for k in real["in"] if k != "frames"]
    print(f"{'metric':36s} {'real_in':>9s} {'sim_in':>9s} {'real_out':>9s} {'sim_out':>9s}")
    for k in keys:
        f = lambda v: "      n/a" if v is None else f"{v:9.3f}"
        print(f"{k:36s} {f(real['in'][k])} {f(avg['in'][k])} {f(real['out'][k])} {f(avg['out'][k])}")
    print("corrupt:", asdict(cp))


if __name__ == "__main__":
    main()
