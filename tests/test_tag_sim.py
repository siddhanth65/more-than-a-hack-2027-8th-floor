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
    nan = np.nan
    # corner / goal kick need a goal-line crossing (max |x| > 52.0)
    assert _sp_type(8, (51.5, 33.0), P, 53.0, 30.0) == "Corner"
    assert _sp_type(8, (-49.8, -31.5), P, 53.0, 20.0) == "Corner"         # 3.7 m from the flag
    assert _sp_type(8, (-49.5, -31.0), P, 53.0, 20.0) == "Free kick"      # 4.2 m: too far; goal line crossed -> no throw-in
    assert _sp_type(8, (51.5, 33.0), P, 51.0, 30.0) == "Free kick"        # restart at the flag, no crossing
    assert _sp_type(8, (49.0, 2.0), P, 53.0, 10.0) == "Goal kick"
    assert _sp_type(8, (49.0, 2.0), P, 50.0, 10.0) == "Free kick"
    # kick-off: within 3 m of the centre spot AND (>= 15 s dead or half start)
    assert _sp_type(8, (2.0, -2.0), P, 53.0, 10.0, dead_s=40.0) == "Kick-off"
    assert _sp_type(8, (2.0, -2.0), P, 10.0, 10.0, dead_s=5.0, at_start=True) == "Kick-off"
    assert _sp_type(8, (2.0, -2.0), P, 10.0, 10.0, dead_s=10.0) == "Free kick"
    assert _sp_type(8, (3.5, 0.0), P, 53.0, 10.0, dead_s=40.0) == "Free kick"
    # throw-in: THROW_IN class wins; else touchline crossing without goal-line crossing
    assert _sp_type(4, (10.0, 5.0), P) == "Throw-in"
    assert _sp_type(4, (51.5, 33.0), P, 53.0, 34.5) == "Throw-in"
    assert _sp_type(8, (10.0, 33.0), P, 20.0, 34.2) == "Throw-in"
    assert _sp_type(8, (10.0, 33.8), P, 20.0, 33.0) == "Free kick"        # restart near the touchline, no crossing
    assert _sp_type(8, (50.0, 20.0), P, 53.0, 34.2) == "Free kick"        # both lines crossed, no corner/goal-kick spot
    assert _sp_type(8, (-30.0, 2.0), P, 30.0, 5.0) == "Free kick"
    assert _sp_type(8, (np.nan, np.nan), P) == "Free kick"
    assert _sp_type(8, (np.nan, np.nan), P, 20.0, 34.2) == "Throw-in"     # excursion alone still types a throw-in
    assert _sp_type(8, (0.0, 0.0), P, nan, nan, dead_s=20.0) == "Kick-off"


def _dead_run_match(path, restart, dead_s=10.0, start=25 * 30, ev_cls=8, F=25 * 120):
    """One dead run [start, start + dead_s) on a 105 x 68 pitch. `path`: list of (frame offset from start, x, y) raw-ball
    waypoints held until the next one (the ball before the run sits at (-8, 0)); restart ball + event at the run end."""
    m = two_player_match(F=F)
    m.pitch = (105.0, 68.0)
    m.ball[:] = 0.0; m.ball[:, 0] = -8.0
    labels = np.full(F, "H", dtype="<U1")
    e = start + int(dead_s * 25)
    labels[start:e] = "D"
    for i, (off, x, y) in enumerate(path):
        nxt = path[i + 1][0] if i + 1 < len(path) else e - start
        m.ball[start + off:start + nxt] = (x, y)
    m.ball[e:e + 50] = restart
    events = pd.DataFrame([(e, "H", ev_cls, False)], columns=["frame", "team", "cls", "poss_only"])
    return m, labels, events


def _one_type(m, labels, events, seed=0):
    p = TagSimParams(keep_outside=0.0, keep_inside=0.0, jitter_s=0.0, keep_sp_by_type={})
    df = simulate_tags(m, labels, events, np.zeros(m.n_frames, bool), p, np.random.default_rng(seed))
    return list(df[df["code"] == "SET PIECES"]["type"])


def test_corner_from_goal_line_excursion_then_walk_to_flag():
    # ball rolls over the goal line (x 53.5) just before the run is labelled dead, is carried back to the flag
    m, labels, events = _dead_run_match([(-25, 53.5, 20.0), (0, 54.0, 25.0), (100, 50.0, 30.0), (200, 52.0, 33.0)],
                                        restart=(52.0, 33.2))
    assert _one_type(m, labels, events) == ["Corner"]


def test_goal_kick_from_goal_line_excursion():
    m, labels, events = _dead_run_match([(0, -53.5, 5.0), (100, -50.0, 2.0)], restart=(-47.0, 3.0))
    assert _one_type(m, labels, events) == ["Goal kick"]


def test_kickoff_after_long_dead_run_and_not_after_short_one():
    # goal: ball in the net (x 54), 40 s celebration, restart on the centre spot
    m, labels, events = _dead_run_match([(0, 54.0, 1.0), (500, 20.0, 0.0), (900, 0.5, 0.3)], restart=(0.4, -0.3),
                                        dead_s=40.0)
    assert _one_type(m, labels, events) == ["Kick-off"]
    m, labels, events = _dead_run_match([(0, 2.0, 1.0)], restart=(1.0, 1.0), dead_s=8.0)
    assert _one_type(m, labels, events) == ["Free kick"]               # short stoppage near the centre: a free kick


def test_throw_in_from_touchline_excursion_without_event():
    m, labels, events = _dead_run_match([(-10, 10.0, 34.6), (50, 12.0, 33.5)], restart=(12.0, 33.7))
    assert _one_type(m, labels, events) == ["Throw-in"]
    m, labels, events = _dead_run_match([(0, 10.0, 20.0)], restart=(10.0, 20.0), ev_cls=4)
    assert _one_type(m, labels, events) == ["Throw-in"]                 # THROW_IN event, inside geometry


def test_free_kick_inside_the_pitch():
    m, labels, events = _dead_run_match([(0, 20.0, -10.0)], restart=(20.0, -10.0))
    assert _one_type(m, labels, events) == ["Free kick"]


def test_excursion_margin_is_two_seconds():
    # a goal-line crossing 3 s before the run is outside the margin -> the flag restart is not a corner
    m, labels, events = _dead_run_match([(0, 50.0, 30.0)], restart=(52.0, 33.2))
    m.ball[25 * 30 - 75:25 * 30 - 60] = (53.5, 30.0)
    assert _one_type(m, labels, events) == ["Free kick"]
    m.ball[25 * 30 - 40:25 * 30 - 30] = (53.5, 30.0)                    # 1.6 s before: inside
    assert _one_type(m, labels, events) == ["Corner"]


def test_restart_typing_deterministic():
    m, labels, events = _dead_run_match([(0, 54.0, 1.0), (500, 0.5, 0.3)], restart=(0.4, -0.3), dead_s=40.0)
    p = TagSimParams(keep_outside=0.5, keep_inside=0.5, keep_sp_by_type={"Kick-off": 0.5})
    a = simulate_tags(m, labels, events, np.zeros(m.n_frames, bool), p, np.random.default_rng(7))
    b = simulate_tags(m, labels, events, np.zeros(m.n_frames, bool), p, np.random.default_rng(7))
    pd.testing.assert_frame_equal(a, b)


def test_sp_rate_by_type():
    from mtah.sim.tag_sim import REAL_SP_RATES, REAL_SP_TOTAL, SP_TYPES, sp_rate_by_type
    assert abs(sum(REAL_SP_RATES.values()) - REAL_SP_TOTAL) < 2e-3   # per-type rates are rounded (sum 0.556)
    assert set(REAL_SP_RATES) == set(SP_TYPES)
    tags = pd.DataFrame({"start": [0, 1, 2, 3, 4], "end": [1, 2, 3, 4, 5],
                         "code": ["SET PIECES", "SET PIECES", "SET PIECES", "PROGRESSION", "SET PIECES"],
                         "team": [None, "H", "A", "H", None], "type": ["Throw-in", "Corner", "Corner", None, "Odd"],
                         "half": [1] * 5})
    r = sp_rate_by_type(tags, 10.0)
    assert r["Throw-in"] == 0.1 and r["Corner"] == 0.2 and r["Kick-off"] == 0.0 and r["Odd"] == 0.1
    assert r["total"] == 0.4
    empty = sp_rate_by_type(tags.iloc[:0], 10.0)
    assert empty["total"] == 0.0 and all(empty[t] == 0.0 for t in SP_TYPES)


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


def test_sp_rates_script_pools_halves_and_skips_sealed(tmp_path):
    import importlib.util
    import json
    from pathlib import Path
    spec = importlib.util.spec_from_file_location("sp_rates", Path(__file__).resolve().parents[1] / "scripts" / "sp_rates.py")
    mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
    for key, n_sp in (("game_9001_H1", 3), ("game_9007_H1", 50)):
        st = tmp_path / f"{key}_s100"
        np.savez(str(st) + ".npz", half=np.ones(25 * 60 * 10, np.int8))          # 10 graded minutes
        (Path(str(st) + ".json")).write_text(json.dumps({"key": key, "seed": 100, "fps": 25.0, "pitch": [105.0, 68.0]}))
        pd.DataFrame({"code": ["SET PIECES"] * n_sp + ["PROGRESSION"], "type": ["Corner"] * n_sp + [None]}
                     ).to_parquet(str(st) + ".tags.parquet")
    rates, minutes, n = mod.sim_rates(tmp_path, 100)
    assert n == 1 and minutes == 10.0 and abs(rates["Corner"] - 0.3) < 1e-12 and abs(rates["total"] - 0.3) < 1e-12
