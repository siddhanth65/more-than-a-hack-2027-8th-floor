import pytest

from mtah.io.tags import coverage, parse_sportscode

XML = """<?xml version="1.0"?><file><ALL_INSTANCES>
<instance><ID>1</ID><start>1.0</start><end>2.0</end><code>BUILD UP</code>
 <label><group>Team</group><text>Barcelona</text></label><label><group>Half</group><text>1st Half</text></label></instance>
<instance><ID>2</ID><start>3.04</start><end>4.0</end><code>SET PIECES</code>
 <label><group>Team</group><text>N/A</text></label><label><group>Type</group><text>Throw-in</text></label><label><group>Half</group><text>2nd Half</text></label></instance>
<instance><ID>3</ID><start>5.0</start><end>6.0</end><code>PROGRESSION</code>
 <label><group>Team</group><text>Atletic Club</text></label></instance>
</ALL_INSTANCES></file>"""


def test_parse(tmp_path):
    p = tmp_path / "t.xml"; p.write_text(XML)
    df = parse_sportscode(p)
    assert df["start"].tolist() == [25, 76, 125]
    assert df["end"].tolist() == [50, 100, 150]
    assert df["code"].tolist() == ["BUILD UP", "SET PIECES", "PROGRESSION"]
    assert df["team"].tolist() == ["H", None, "A"]
    assert df["type"].tolist() == [None, "Throw-in", None]
    assert df["half"].tolist() == [1, 2, None]


def test_coverage(tmp_path):
    p = tmp_path / "t.xml"; p.write_text(XML)
    df = parse_sportscode(p)
    cov = coverage(df, 200)
    assert cov[25] and cov[50] and not cov[51] and cov[76] and cov[125]
    assert coverage(df, 200, team_only=True)[76] is not True and coverage(df, 200, team_only=True)[76] == False
