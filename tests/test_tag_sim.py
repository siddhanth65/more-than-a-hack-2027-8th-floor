import numpy as np
import pandas as pd

from mtah.sim.tag_sim import TagSimParams, _phase_runs, simulate_tags
from tests.test_baselines import two_player_match


def test_phases_and_setpiece():
    m = two_player_match(F=25 * 120)
    m.ball[:] = 0.0
    m.ball[:25 * 40, 0] = -8.0            # near home for 40 s
    m.ball[25 * 40:25 * 70, 0] = 8.0      # near away for 30 s
    m.ball[25 * 70:25 * 80, :] = np.nan   # blackout 10 s
    m.ball[25 * 80:, 0] = -8.0
    labels = np.full(m.n_frames, "H", dtype="<U1")
    labels[25 * 40:25 * 70] = "A"
    labels[25 * 70:25 * 80] = "D"
    m.ball[25 * 69:25 * 70, 1] = 34.0     # last ball frames before the D run (incl. 1749): on the touchline -> throw-in
    events = pd.DataFrame([(25 * 80, "H", 4, False)], columns=["frame", "team", "cls", "poss_only"])
    mask = np.zeros(m.n_frames, bool); mask[25 * 70:25 * 80] = True
    p = TagSimParams(keep_outside=1.0, keep_inside=1.0, jitter_s=0.0)
    df = simulate_tags(m, labels, events, mask, p, np.random.default_rng(0))
    phases = df[df["code"] == "PROGRESSION"]
    assert list(phases["team"]) == ["H", "A", "H"]
    assert int(phases.iloc[1]["end"]) == 25 * 80 - 1   # the A phase now persists through the 10 s blackout
    sp = df[df["code"] == "SET PIECES"]
    assert len(sp) == 1 and sp.iloc[0]["type"] == "Throw-in" and sp.iloc[0]["team"] is None
    assert sp.iloc[0]["start"] == 25 * 80 - 125 and sp.iloc[0]["end"] == 25 * 80 + 750


def test_deletion_rate():
    m = two_player_match(F=25 * 600); m.ball[:] = 0.0; m.ball[:, 0] = -8.0
    labels = np.full(m.n_frames, "H", dtype="<U1")
    events = pd.DataFrame(columns=["frame", "team", "cls", "poss_only"])
    p = TagSimParams(keep_outside=0.0, keep_inside=0.0)
    df = simulate_tags(m, labels, events, np.zeros(m.n_frames, bool), p, np.random.default_rng(0))
    assert len(df) == 0


def test_phase_runs_carry_through_gaps():
    fps = 25.0
    def codes(gap_s):
        c = np.full(25 * (40 + 40) + int(gap_s * 25), -1, np.int8)
        c[:25 * 40] = 1
        c[25 * 40 + int(gap_s * 25):] = 1
        return c
    r = _phase_runs(codes(20), fps, 1.0, 3.0, 60.0)
    assert r == [(0, len(codes(20)) - 1, 1)]          # one A phase, spanning the 20 s gap
    r = _phase_runs(codes(90), fps, 1.0, 3.0, 60.0)
    assert r[0] == (0, 25 * 40 - 1, 1)                # 90 s gap ends the run at the last ball frame
    assert r[1][0] == 25 * 40 + 90 * 25


def _restart_match():
    """Four 10 s dead runs whose last ball frame sits at a throw-in, corner, goal-kick and free-kick spot."""
    m = two_player_match(F=25 * 200)
    m.ball[:] = 0.0; m.ball[:, 0] = -8.0
    labels = np.full(m.n_frames, "H", dtype="<U1")
    spots = [(0.0, 34.0), (52.5, 31.5), (52.8, 0.0), (0.0, 10.0)]
    restarts = [(0.0, 33.8), (51.0, 33.0), (49.0, 0.0), (-8.0, 0.0)]
    ev = []
    for i, xy in enumerate(spots):
        s = 25 * (20 + 40 * i); e = s + 25 * 10
        labels[s:e] = "D"
        m.ball[s - 25:s] = xy
        m.ball[e:e + 25] = restarts[i]
        ev.append((e, "H", 4 if i == 0 else 8, False))
    events = pd.DataFrame(ev, columns=["frame", "team", "cls", "poss_only"])
    return m, labels, events


def _sp_types(keep, seed=0):
    m, labels, events = _restart_match()
    p = TagSimParams(keep_outside=0.0, keep_inside=0.0, jitter_s=0.0, keep_sp_by_type=keep)
    df = simulate_tags(m, labels, events, np.zeros(m.n_frames, bool), p, np.random.default_rng(seed))
    return list(df[df["code"] == "SET PIECES"]["type"])


def test_sim_sp_types_use_real_spellings():
    assert _sp_types({}) == ["Throw-in", "Corner", "Goal kick", "Free kick"]


def test_keep_sp_by_type_zero_removes_only_that_type():
    assert _sp_types({"Free kick": 0.0}) == ["Throw-in", "Corner", "Goal kick"]
    assert _sp_types({"free KICK": 0.0, "corner": 0.0}) == ["Throw-in", "Goal kick"]   # case-insensitive
    assert _sp_types({"Free kick": 1.0}) == ["Throw-in", "Corner", "Goal kick", "Free kick"]


def test_keep_sp_by_type_deterministic_and_partial():
    m, labels, events = _restart_match()
    p = TagSimParams(keep_outside=1.0, keep_inside=1.0, keep_sp_by_type={"Free kick": 0.5})
    a = simulate_tags(m, labels, events, np.zeros(m.n_frames, bool), p, np.random.default_rng(3))
    b = simulate_tags(m, labels, events, np.zeros(m.n_frames, bool), p, np.random.default_rng(3))
    pd.testing.assert_frame_equal(a, b)
    kept = sum("Free kick" in _sp_types({"Free kick": 0.5}, seed) for seed in range(200))
    assert 70 < kept < 130


def test_default_keeps_and_config_in_sync():
    import json
    from pathlib import Path
    assert TagSimParams().keep_sp_by_type == {"Free kick": 0.16, "Goal kick": 0.78, "Throw-in": 0.72}
    cfg = json.loads((Path(__file__).resolve().parents[1] / "configs" / "tag_sim.json").read_text())
    assert TagSimParams.from_dict(cfg) == TagSimParams()
    assert TagSimParams.from_dict(json.loads(json.dumps(TagSimParams().to_dict()))) == TagSimParams()


def test_sp_type_rules():
    from mtah.sim.tag_sim import _sp_type
    P = (105.0, 68.0)
    assert _sp_type(8, (0.2, -0.5), P) == "Kick-off"
    assert _sp_type(8, (51.5, 33.0), P) == "Corner"
    assert _sp_type(8, (-51.0, -32.0), P) == "Corner"
    assert _sp_type(8, (49.0, 2.0), P) == "Goal kick"
    assert _sp_type(8, (-30.0, 2.0), P) == "Free kick"
    assert _sp_type(8, (10.0, 33.8), P) == "Free kick"      # touchline geometry alone is not a throw-in
    assert _sp_type(4, (10.0, 5.0), P) == "Throw-in"        # THROW_IN class wins over geometry
    assert _sp_type(4, (51.5, 33.0), P) == "Throw-in"
    assert _sp_type(8, (np.nan, np.nan), P) == "Free kick"


def test_restart_typing_uses_raw_ball_not_blacked_out_ball():
    m, labels, events = _restart_match()
    raw = m.ball.copy()
    m.ball[:] = np.nan                                      # fully blacked-out sim ball
    p = TagSimParams(keep_outside=0.0, keep_inside=0.0, jitter_s=0.0, keep_sp_by_type={})
    mask = np.zeros(m.n_frames, bool)
    df = simulate_tags(m, labels, events, mask, p, np.random.default_rng(0), raw_ball=raw)
    assert list(df[df["code"] == "SET PIECES"]["type"]) == ["Throw-in", "Corner", "Goal kick", "Free kick"]


def test_restart_raw_ball_nan_at_restart_falls_back_to_last_visible():
    m, labels, events = _restart_match()
    raw = m.ball.copy()
    e = int(events["frame"].iloc[1])
    raw[e, :] = np.nan                                      # corner restart frame missing; frame e-1 is NaN-free (10 s dead, spot)
    raw[e - 1, :] = (51.0, 33.0)
    p = TagSimParams(keep_outside=0.0, keep_inside=0.0, jitter_s=0.0, keep_sp_by_type={})
    df = simulate_tags(m, labels, events, np.zeros(m.n_frames, bool), p, np.random.default_rng(0), raw_ball=raw)
    assert list(df[df["code"] == "SET PIECES"]["type"])[1] == "Corner"


def test_dead_run_at_frame_zero_without_event_is_kickoff_and_deterministic():
    m = two_player_match(F=25 * 60); m.ball[:] = 0.0; m.ball[:, 0] = -8.0
    labels = np.full(m.n_frames, "H", dtype="<U1"); labels[:25 * 5] = "D"
    events = pd.DataFrame(columns=["frame", "team", "cls", "poss_only"])
    p = TagSimParams(keep_outside=0.0, keep_inside=0.0, jitter_s=0.0, keep_sp_by_type={})
    a = simulate_tags(m, labels, events, np.zeros(m.n_frames, bool), p, np.random.default_rng(1))
    b = simulate_tags(m, labels, events, np.zeros(m.n_frames, bool), p, np.random.default_rng(1))
    pd.testing.assert_frame_equal(a, b)
    assert list(a["type"]) == ["Kick-off"]
