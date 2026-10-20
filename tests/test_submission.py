import numpy as np

from mtah import N_FRAMES
from mtah.submission import validate, write_submission


def test_write_and_validate(tmp_path):
    labels = np.full(N_FRAMES, "D", dtype="<U1"); labels[11185:20000] = "H"
    p = write_submission(labels, tmp_path / "t.csv")
    text = p.read_text()
    assert text.startswith("frame,label\n0,D\n1,D\n") and text.endswith("199318,D\n")
    assert validate(p) is None


def test_validate_rejects_short(tmp_path):
    p = tmp_path / "s.csv"; p.write_text("frame,label\n0,H\n1,H\n")
    assert "row count" in validate(p)
