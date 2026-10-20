import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest

from mtah.io.match import Match, speeds_from_positions

ROOT = Path(__file__).resolve().parents[1]


def _load():
    spec = importlib.util.spec_from_file_location("precompute_pathcrf_script", ROOT / "scripts" / "precompute_pathcrf.py")
    mod = importlib.util.module_from_spec(spec); sys.modules["precompute_pathcrf_script"] = mod
    spec.loader.exec_module(mod)
    return mod


def _toy():
    """Two players (home slot 0 at x=-10, away slot 1 at x=+10); ball sits next to one of them or far from both."""
    F = 6
    pos = np.zeros((F, 2, 2), np.float32); pos[:, 0, 0] = -10; pos[:, 1, 0] = 10
    ball = np.array([[-9, 0], [9, 0], [9.5, 0], [0, 0], [np.nan, np.nan], [-10, 1]], np.float32)
    half = np.array([1, 1, 1, 1, 1, 0], np.int8)
    return Match(fps=25.0, pos=pos, speed=speeds_from_positions(pos, 25.0), team=np.array([0, 1], np.int8),
                 ball=ball, ball_speed=speeds_from_positions(ball, 25.0), half=half, pitch=(105.0, 68.0))


def test_nearest_player_proxy_two_players():
    mod = _load()
    m = _toy()
    # frames 0..2 qualify (ball within 2 m of a player); 3 is 10 m away; 4 has no ball; 5 is ungraded
    p = np.array([0.9, 0.2, 0.7, 0.9, 0.9, 0.1], np.float32)
    agree, n = mod.nearest_player_proxy(m, p)
    assert n == 3 and agree == pytest.approx(2 / 3)
    p2 = p.copy(); p2[2] = np.nan                       # no PathCRF opinion -> frame skipped
    agree, n = mod.nearest_player_proxy(m, p2)
    assert n == 2 and agree == 1.0


def test_nearest_player_proxy_ignores_untracked_slots():
    mod = _load()
    m = _toy(); m.pos[0, 0] = np.nan                    # home player missing at frame 0: away player is 19 m away
    agree, n = mod.nearest_player_proxy(m, np.array([0.9, 0.2, 0.3, 0.9, 0.9, 0.1], np.float32))
    assert n == 2 and agree == 1.0


def test_nearest_player_proxy_empty():
    mod = _load()
    agree, n = mod.nearest_player_proxy(_toy(), np.full(6, np.nan, np.float32))
    assert n == 0 and np.isnan(agree)


def test_truth_agreement_and_half_stats():
    mod = _load()
    truth = np.array(["H", "A", "D", "H", "A"])
    p = np.array([0.8, 0.6, 0.9, np.nan, 0.1], np.float32)
    agree, n = mod.truth_agreement(p, truth)
    assert n == 3 and agree == pytest.approx(2 / 3)
    st = mod.posterior_stats({"p_home": p, "imputed_frac": np.array([0, .5, 1, np.nan, 0], np.float32)},
                             np.array([1, 1, 1, 1, 0], np.int8))
    assert st["nan_share"] == pytest.approx(0.25) and st["imputed_frac_mean"] == pytest.approx(0.5)
