import numpy as np
import pytest

from mtah.io.metrica_loader import load_tactical, parse_epts
from mtah.io.periods import HALF_BLOCKS, half_array

XML = """<?xml version="1.0"?><main><Metadata><Teams>
<Team id="T1"><Name>Home FC</Name></Team><Team id="T2"><Name>Away FC</Name></Team></Teams>
<Players>
<Player id="P1" teamId="T1"><Name>Track 1</Name></Player>
<Player id="P2" teamId="T2"><Name>Track 2</Name></Player>
<Player id="P3" teamId="T1"><Name>Track 3</Name></Player>
</Players></Metadata></main>"""

TXT = ("1:0.5,0.5,1.0;NaN,NaN,NaN;0.0,0.0,2.0:0.75,0.5,3.0\n"
       "2:0.5,0.5,1.0;0.25,0.25,1.5;0.0,0.0,2.0:NaN,NaN,NaN\n"
       "3:0.5,0.5,1.0;0.25,0.25,1.5;0.0,0.0,2.0:1.0,1.0,0.0\n")


def test_parse_epts_tiny(tmp_path):
    (tmp_path / "a.txt").write_text(TXT)
    (tmp_path / "a.xml").write_text(XML)
    m = parse_epts(tmp_path / "a.txt", tmp_path / "a.xml")
    assert m.n_frames == 3 and m.n_slots == 3
    assert m.team.tolist() == [0, 1, 0]
    assert m.pos[0, 0].tolist() == pytest.approx([0.0, 0.0])          # 0.5,0.5 is the centre
    assert np.isnan(m.pos[0, 1]).all()
    assert m.pos[1, 1].tolist() == pytest.approx([-0.25 * 105.3, -0.25 * 68.0])
    assert m.ball[0].tolist() == pytest.approx([0.25 * 105.3, 0.0])
    assert np.isnan(m.ball[1]).all()
    assert m.ball_present.tolist() == [True, False, True]
    assert m.speed[0, 2] == pytest.approx(2.0)                          # provider speed kept
    assert m.half.tolist() == [1, 1, 1]                                 # short file: all graded


def test_half_array():
    h = half_array(199319)
    assert h[11184] == 0 and h[11185] == 1 and h[90651] == 1 and h[90652] == 0
    assert h[110807] == 0 and h[110808] == 2 and h[190197] == 2 and h[190198] == 0
    assert (h > 0).sum() == 158857


@pytest.mark.real
def test_load_tactical_real(data_dir):
    m = load_tactical(data_dir)
    n_frames, n_slots = m.n_frames, m.n_slots
    share_all = float(m.ball_present.mean())
    share_graded = float(m.ball_present[m.graded].mean())
    n_team = int((m.team == 0).sum() + (m.team == 1).sum())
    assert n_frames == 199319 and n_slots == 46
    assert share_all == pytest.approx(0.569, abs=0.001)
    assert share_graded == pytest.approx(0.593, abs=0.002)
    assert n_team == 46


def test_slot_teams_rejects_unknown_team(tmp_path):
    from mtah.io.metrica_loader import slot_teams
    (tmp_path / "b.xml").write_text(XML.replace('teamId="T2"', 'teamId="T9"'))
    with pytest.raises(ValueError):
        slot_teams(tmp_path / "b.xml")
