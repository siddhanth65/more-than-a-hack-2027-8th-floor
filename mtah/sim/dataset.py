"""One simulated DFL half = corrupted tracking + blackouts + simulated tags + truth variants."""
import json
from dataclasses import dataclass
from dataclasses import replace as dc_replace
from pathlib import Path

import numpy as np
import pandas as pd

from mtah.io.dfl_events import truth_variants
from mtah.io.dfl_proxy import load_events, load_half
from mtah.io.match import Match, speeds_from_positions
from mtah.sim import half_rng
from mtah.sim.blackout_sim import apply_blackouts, blackout_mask
from mtah.sim.corrupt import PLAN2B_DEAD, corrupt, dead_ball_realism
from mtah.sim.tag_sim import simulate_tags

SEALED_MATCH = 9007
VARIANTS = ("raw_Dfull", "ctrl_Dfull", "raw_Dnarrow", "ctrl_Dnarrow")


@dataclass
class SimHalf:
    key: str
    seed: int
    match: Match
    tags: pd.DataFrame
    blackout: np.ndarray
    truth: dict
    events: pd.DataFrame


def match_id(key):
    return int(key.split("_")[1])


def half_index(key):
    return (match_id(key) - 9001) * 2 + (int(key[-1]) - 1)


def lomo_folds(keys):
    keys = sorted(keys)
    matches = sorted({match_id(k) for k in keys} - {SEALED_MATCH})
    return [(m, [k for k in keys if match_id(k) not in (m, SEALED_MATCH)], [k for k in keys if match_id(k) == m])
            for m in matches]


def build_half(key, npz, census, seed, bparams, cparams, tparams):
    m, labels = load_half(npz, census)
    events = load_events(npz, census)
    truth = truth_variants(labels, events, m.ball, m.ball_speed, m.fps, m.pitch)
    rng = half_rng(seed, half_index(key))
    mask = blackout_mask(labels, m.ball, m.pitch, m.fps, bparams, rng)
    sim = apply_blackouts(m, mask)
    dead_rng = rng.spawn(1)[0]            # spawning does not advance the half's stream
    dead_ball_realism(sim.ball, labels, m.pitch, m.fps, PLAN2B_DEAD, rng)   # replay: positions stay as in Plan 2b
    sim = dc_replace(sim, ball=dead_ball_realism(sim.ball, labels, m.pitch, m.fps, cparams, dead_rng))
    sim = dc_replace(sim, ball_speed=speeds_from_positions(sim.ball, m.fps))
    sim = corrupt(sim, mask, cparams, rng)
    tags = simulate_tags(sim, labels, events, mask, tparams, rng)
    return SimHalf(key=key, seed=seed, match=sim, tags=tags, blackout=mask, truth=truth, events=events)


def _stem(d, key, seed):
    return Path(d) / f"{key}_s{seed}"


def save_half(sim, d):
    Path(d).mkdir(parents=True, exist_ok=True)
    st = _stem(d, sim.key, sim.seed)
    m = sim.match
    np.savez_compressed(str(st) + ".npz", pos=m.pos, speed=m.speed, team=m.team, ball=m.ball,
                        ball_speed=m.ball_speed, half=m.half, blackout=sim.blackout,
                        **{f"truth_{k}": v for k, v in sim.truth.items()})
    sim.tags.to_parquet(str(st) + ".tags.parquet")
    sim.events.to_parquet(str(st) + ".events.parquet")
    (Path(str(st) + ".json")).write_text(json.dumps({"key": sim.key, "seed": sim.seed, "fps": m.fps, "pitch": list(m.pitch)}))


def load_half_sim(d, key, seed):
    st = _stem(d, key, seed)
    meta = json.loads(Path(str(st) + ".json").read_text())
    z = np.load(str(st) + ".npz", allow_pickle=False)
    m = Match(fps=meta["fps"], pos=z["pos"], speed=z["speed"], team=z["team"], ball=z["ball"],
              ball_speed=z["ball_speed"], half=z["half"], pitch=tuple(meta["pitch"]))
    truth = {k: z[f"truth_{k}"].astype("<U1") for k in VARIANTS}
    tags = pd.read_parquet(str(st) + ".tags.parquet")
    for c in ("team", "type", "half"):
        tags[c] = tags[c].astype(object).where(tags[c].notna(), None)
    events = pd.read_parquet(str(st) + ".events.parquet")
    return SimHalf(key=key, seed=seed, match=m, tags=tags, blackout=z["blackout"], truth=truth, events=events)


def list_sim_keys(d):
    return sorted({p.name.split("_s")[0] for p in Path(d).glob("game_*_s*.json")})
