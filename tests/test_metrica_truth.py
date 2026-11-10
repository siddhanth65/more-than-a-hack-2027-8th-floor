"""Truth derivation and coordinate handling for the Metrica sample games (synthetic snippets, no downloaded data)."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from mtah.io.dfl_proxy import DFL_PITCH, load_half  # noqa: E402
from mtah.io.extra_npz import to_dfl_metres  # noqa: E402
from mtah.io.metrica_truth import (blank_runs, derive_labels, event_dead, events_from_csv_rows, norm_to_metres,  # noqa: E402
                                   parse_subs, spell_stats, team_track)


def ev(typ, team, start, end=None, subs=(), period=1):
    return {"type": typ, "subs": tuple(subs), "team": team, "start": start, "end": start if end is None else end,
            "period": period, "player": None}


def test_parse_subs():
    assert parse_subs("HEAD-OFF TARGET-OUT") == ("HEAD", "OFF TARGET", "OUT")
    assert parse_subs(None) == () and parse_subs(float("nan")) == () and parse_subs("") == ()
    assert parse_subs("tackle-fault-won") == ("TACKLE", "FAULT", "WON")


def test_csv_rows_end_frame_and_team():
    rows = [{"Team": "Away", "Type": "SET PIECE", "Subtype": "KICK OFF", "Period": 1, "Start Frame": 5, "End Frame": 0, "From": "P1"},
            {"Team": "Home", "Type": "PASS", "Subtype": np.nan, "Period": 1, "Start Frame": 7, "End Frame": 9, "From": "P2"}]
    e = events_from_csv_rows(rows)
    assert [x["team"] for x in e] == ["A", "H"] and e[0]["end"] == 5 and e[1]["end"] == 9


def test_event_dead_head_ball_out_restart_tail():
    evs = [ev("SET PIECE", "H", 10, subs=("KICK OFF",)), ev("PASS", "H", 10, 20), ev("PASS", "H", 40, 60),
           ev("BALL OUT", "H", 60, 100), ev("SET PIECE", "A", 200, subs=("THROW IN",)), ev("PASS", "A", 200, 230)]
    d = event_dead(evs, 0, 299)
    assert d[:10].all() and not d[10:100].any()          # head up to the kick-off, play until the out
    assert d[100:200].all() and not d[200:231].any()      # [BALL OUT end, restart)
    assert d[231:].all()                                  # tail after the last event's end frame


def test_event_dead_goal_kick_is_a_restart_and_advantage_is_not():
    evs = [ev("SET PIECE", "H", 0, subs=("KICK OFF",)), ev("SHOT", "H", 20, 50, subs=("OFF TARGET", "OUT")),
           ev("RECOVERY", "A", 150), ev("PASS", "A", 150, 170, subs=("GOAL KICK",)),
           ev("CHALLENGE", "H", 200, subs=("GROUND", "FAULT", "LOST")), ev("PASS", "A", 210, 230),   # advantage played on
           ev("PASS", "A", 230, 260)]
    d = event_dead(evs, 0, 260)
    assert d[50:150].all() and not d[150:171].any()       # goal kick (RECOVERY + PASS GOAL KICK, same frame) closes the dead run
    assert not d[200:261].any()                           # a foul not followed by a restart is not dead


def test_team_track_does_not_flip_on_ball_lost_or_aerial_but_on_recovery_and_tackle():
    evs = [ev("PASS", "H", 0, 10), ev("BALL LOST", "H", 10, subs=("INTERCEPTION",)), ev("CHALLENGE", "A", 12, subs=("AERIAL", "WON")),
           ev("RECOVERY", "A", 30), ev("CHALLENGE", "H", 60, subs=("TACKLE", "WON")),
           ev("CHALLENGE", "A", 80, subs=("TACKLE", "FAULT", "WON")), ev("BALL OUT", "H", 90, 95)]
    t = team_track(evs, 0, 120)
    assert t[0] == "H" and t[29] == "H"       # BALL LOST and the aerial win do not hand over
    assert t[30] == "A" and t[59] == "A"      # the opponent's RECOVERY does
    assert t[60] == "H" and t[119] == "H"     # a won tackle does; a fouling "win" and a BALL OUT do not


def test_blank_runs_and_blank_dead_in_labels():
    vis = np.ones(300, bool); vis[50:60] = False; vis[100:200] = False
    br = blank_runs(vis, 50)
    assert br[100:200].all() and not br[50:60].any()
    evs = [ev("SET PIECE", "H", 0, subs=("KICK OFF",)), ev("PASS", "H", 0, 290)]
    r = derive_labels(evs, 0, 299, vis, 25.0, 2.0)
    assert (r["labels"][100:200] == "D").all() and (r["labels"][50:60] == "H").all()
    r0 = derive_labels(evs, 0, 299, vis, 25.0, None)
    assert not (r0["labels"][100:200] == "D").any()       # blank rule off


def test_spell_stats_handover_and_quantiles():
    lab = np.array(list("HH" * 25 + "AA" * 50 + "D" * 50 + "H" * 50))        # 50 H, 100 A, 50 D, 50 H (frames)
    s = spell_stats([lab], fps=25.0)
    assert s["n_spells"] == 3 and s["dead_share"] == 0.2
    assert s["direct_handover_share"] == 0.5               # H->A direct, A->D not
    assert s["median_s"] == 2.0 and s["lt3s"] == pytest.approx(2 / 3, abs=1e-3)


def test_norm_to_metres_corners_and_pitch():
    assert np.allclose(norm_to_metres([0, 0]), [-52.5, -34.0])
    assert np.allclose(norm_to_metres([1, 1]), [52.5, 34.0])
    assert np.allclose(norm_to_metres([0.5, 0.5]), [0, 0])
    assert np.isnan(norm_to_metres([np.nan, 0.3])[0])
    # another source pitch is rescaled onto the simulator's 105 x 68
    m = norm_to_metres([1.0, 1.0], (100.0, 64.0))
    assert np.allclose(m, [50, 32]) and np.allclose(to_dfl_metres(m, (100.0, 64.0)), [52.5, 34.0])


def _write_csv_game(tmp, g=1):
    """Tiny two-half game: Home plays left-to-right in half 1 (GK at x~0), 3 players per side."""
    d = Path(tmp) / f"metrica_sample_game{g}"; d.mkdir()
    n1, n2 = 400, 400
    frames = np.arange(1, n1 + n2 + 1)
    period = np.r_[np.ones(n1, int), 2 * np.ones(n2, int)]
    def side(team, base_x):
        head = [["", "", ""] + sum([[team, ""] for _ in range(3)], []) + [""] * 2,
                ["", "", ""] + sum([[str(i), ""] for i in range(3)], []) + [""] * 2,
                ["Period", "Frame", "Time [s]"] + sum([[f"Player{base_x[0] + i}", ""] for i in range(3)], []) + ["Ball", ""]]
        body = []
        for f, p in zip(frames, period):
            flip = p == 2
            xs = [0.02, 0.3, 0.45]                       # GK, defender, midfielder for the LEFT-side team in half 1
            xs = [1 - x for x in xs] if (flip ^ (team == "Away")) else xs
            ball = (np.nan, np.nan) if 150 <= f <= 249 or 650 <= f <= 749 else (0.5, 0.5)
            body.append([p, f, f / 25] + sum([[x, 0.4 + 0.1 * i] for i, x in enumerate(xs)], []) + list(ball))
        return head, body
    for team, nm, base in (("Home", "Home", (11, 1)), ("Away", "Away", (21, 1))):
        head, body = side(team, base)
        lines = [",".join(r) for r in head] + [",".join("NaN" if (isinstance(v, float) and np.isnan(v)) else str(v) for v in r) for r in body]
        (d / f"Sample_Game_{g}_RawTrackingData_{nm}_Team.csv").write_text("\n".join(lines) + "\n")
    ev = pd.DataFrame([
        ["Home", "SET PIECE", "KICK OFF", 1, 1, 1, 0, 0, "Player11", "", 0, 0, 0, 0],
        ["Home", "PASS", "", 1, 1, 0.04, 100, 4, "Player11", "Player12", 0, 0, 0, 0],
        ["Home", "BALL OUT", "", 1, 140, 5.6, 149, 6, "Player12", "", 0, 0, 0, 0],
        ["Away", "SET PIECE", "THROW IN", 1, 250, 10, 250, 10, "Player21", "", 0, 0, 0, 0],
        ["Away", "PASS", "", 1, 250, 10, 390, 15, "Player21", "Player22", 0, 0, 0, 0],
        ["Away", "SET PIECE", "KICK OFF", 2, 401, 16, 401, 16, "Player21", "", 0, 0, 0, 0],
        ["Away", "PASS", "", 2, 401, 16, 640, 25, "Player21", "Player22", 0, 0, 0, 0],
        ["Home", "RECOVERY", "", 2, 750, 30, 750, 30, "Player12", "", 0, 0, 0, 0],
        ["Home", "PASS", "", 2, 750, 30, 795, 31, "Player12", "Player11", 0, 0, 0, 0]],
        columns=["Team", "Type", "Subtype", "Period", "Start Frame", "Start Time [s]", "End Frame", "End Time [s]", "From", "To", "Start X", "Start Y", "End X", "End Y"])
    ev.to_csv(d / f"Sample_Game_{g}_RawEventsData.csv", index=False)


def test_convert_csv_game_end_to_end_coordinates_orientation_and_dead(tmp_path):
    import convert_metrica_msamp as cm
    dl = tmp_path / "dl"; dl.mkdir(); out = tmp_path / "out"; out.mkdir()
    _write_csv_game(dl)
    res = cm.convert_game(1, dl, out, blank_min_s=2.0)
    # H1 (h0 = 1, index = frame - 1): play to the BALL OUT end frame 149, dead [149, throw-in 250), play to the last event end 390
    h1 = np.load(out / "game_9211_H1.npz")
    st = h1["ball_status"]
    assert st[:148].all() and not st[148:249].any() and st[249:390].all() and not st[390:].any()
    assert (~h1["ball_visible"][149:249]).all() and h1["ball_visible"][:149].all()
    census = out / "fp1d_census.json"
    m1, lab1 = load_half(out / "game_9211_H1.npz", census)
    m2, lab2 = load_half(out / "game_9211_H2.npz", census)
    assert m1.pitch == DFL_PITCH
    # home GK (x_norm 0.02) in half 1: centred metres (0.02 - 0.5) * 105
    home_slots = np.flatnonzero(m1.team == 0)
    xs = np.nanmean(m1.pos[:, home_slots, 0], 0)  # unused slots are all-NaN (warning only)
    assert np.isclose(np.nanmin(xs), (0.02 - 0.5) * 105, atol=1e-3) and np.isclose(np.nanmax(xs), (0.45 - 0.5) * 105, atol=1e-3)
    assert np.isclose(np.nanmax(np.nanmean(m1.pos[:, np.flatnonzero(m1.team == 1), 0], 0)), (0.98 - 0.5) * 105, atol=1e-3)
    assert np.allclose(np.nanmax(np.abs(m1.ball[:, 0])), 0.0)                   # ball at (0.5, 0.5) -> centre
    assert np.isnan(m1.ball[200, 0])                                           # blanked ball stays NaN
    # orientation flips in half 2 (home GK on the right), labels: H until the throw-in team, A after
    assert np.nanmean(m2.pos[:, np.flatnonzero(m2.team == 0), 0]) > 0
    assert set(lab1[:148]) == {"H"} and set(lab1[249:390]) == {"A"} and set(lab1[148:249]) == {"D"}
    # H2: A from the kick-off to the blank (frames 650..749 dead), H after the RECOVERY; 2 + 2 spells overall
    assert set(lab2[:248]) == {"A"} and set(lab2[249:349]) == {"D"} and set(lab2[349:394]) == {"H"}
    assert res["truth"]["n_spells"] == 4
