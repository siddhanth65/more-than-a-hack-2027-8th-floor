"""Read Metrica's FIFA/EPTS export (metadata xml + raw txt) into a Match, in metres."""
import io
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
import pandas as pd

from mtah import N_FRAMES
from mtah.io.match import Match, speeds_from_positions
from mtah.io.periods import half_array

METRICA_PITCH = (105.3, 68.0)
STEMS = {"tactical": "BAR-ATH_tactical", "broadcast": "BAR-ATH_broadcast"}


def slot_teams(xml_path):
    """(S,) int8: 0 for slots of the first <Team> in the file (home), 1 otherwise."""
    root = ET.parse(xml_path).getroot()
    team_ids = [t.get("id") for t in root.iter("Team")]
    if len(team_ids) != 2:
        raise ValueError(f"{xml_path}: expected 2 teams, found {len(team_ids)}")
    slots = [p.get("teamId") for p in root.iter("Player")]
    if not slots:
        raise ValueError(f"{xml_path}: no Player entries")
    unknown = sorted({t for t in slots if t not in team_ids}, key=str)
    if unknown:
        raise ValueError(f"{xml_path}: Player teamId {unknown} is in neither team")
    return np.array([0 if t == team_ids[0] else 1 for t in slots], dtype=np.int8)


def parse_epts(txt_path, xml_path, pitch=METRICA_PITCH):
    team = slot_teams(xml_path)
    S = len(team)
    text = Path(txt_path).read_text().replace(":", ",").replace(";", ",")
    table = pd.read_csv(io.StringIO(text), header=None, na_values=["NaN"], dtype=np.float32).to_numpy()
    if table.shape[1] != 1 + 3 * S + 3:
        raise ValueError(f"{txt_path}: {table.shape[1]} columns, expected {1 + 3 * S + 3} for {S} slots")
    frame_ids = table[:, 0].astype(np.int64)
    F = int(frame_ids.max())
    if not np.array_equal(frame_ids, np.arange(1, F + 1)):
        raise ValueError(f"{txt_path}: frame ids are not 1..{F} in order")
    L, W = pitch
    players = table[:, 1:1 + 3 * S].reshape(F, S, 3)
    pos = np.empty((F, S, 2), np.float32)
    pos[..., 0] = (players[..., 0] - 0.5) * L
    pos[..., 1] = (players[..., 1] - 0.5) * W
    speed = players[..., 2].astype(np.float32)
    ballraw = table[:, 1 + 3 * S:]
    ball = np.empty((F, 2), np.float32)
    ball[:, 0] = (ballraw[:, 0] - 0.5) * L
    ball[:, 1] = (ballraw[:, 1] - 0.5) * W
    ball_speed = ballraw[:, 2].astype(np.float32)
    # provider speed is NaN on some tracked frames; fall back to central differences there
    own = speeds_from_positions(pos, 25.0)
    speed = np.where(np.isnan(speed), own, speed).astype(np.float32)
    ball_speed = np.where(np.isnan(ball_speed), speeds_from_positions(ball, 25.0), ball_speed).astype(np.float32)
    half = half_array(F) if F == N_FRAMES else np.ones(F, dtype=np.int8)
    return Match(fps=25.0, pos=pos, speed=speed, team=team, ball=ball,
                 ball_speed=ball_speed, half=half, pitch=pitch)


def _paths(data_dir, source):
    d = Path(data_dir) / source
    return d / f"{STEMS[source]}_atd.txt", d / f"{STEMS[source]}_atd.xml"


def load_tactical(data_dir):
    return parse_epts(*_paths(data_dir, "tactical"))


def load_broadcast(data_dir):
    m = parse_epts(*_paths(data_dir, "broadcast"))
    m.half[:] = 1  # its own timeline; no graded notion
    return m
