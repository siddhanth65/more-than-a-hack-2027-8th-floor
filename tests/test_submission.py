import numpy as np
import pytest

from mtah import N_FRAMES
from mtah.submission import validate, write_submission


def test_write_and_validate(tmp_path):
    labels = np.full(N_FRAMES, "D", dtype="<U1"); labels[11185:20000] = "H"
    p = write_submission(labels, tmp_path / "t.csv")
    data = p.read_bytes()
    assert data.startswith(b"frame,label\n0,D\n1,D\n") and data.endswith(b"199318,D\n")
    assert b"\r" not in data
    assert validate(p) is None


def test_validate_rejects_short(tmp_path):
    p = tmp_path / "s.csv"; p.write_text("frame,label\n0,H\n1,H\n")
    assert "row count" in validate(p)


def test_write_rejects_wrong_length(tmp_path):
    with pytest.raises(ValueError):
        write_submission(np.full(10, "D", dtype="<U1"), tmp_path / "x.csv")
