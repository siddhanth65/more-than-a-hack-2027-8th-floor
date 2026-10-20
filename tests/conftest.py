import os
from pathlib import Path

import pytest

DEFAULT_DATA = Path(r"HOME/mtah-data")
DEFAULT_DFL = Path(r"E:\football-synthesizer\data\dfl_open\work\fp1d")


@pytest.fixture(scope="session")
def data_dir():
    d = Path(os.environ.get("MTAH_DATA_DIR", DEFAULT_DATA))
    if not (d / "tactical" / "BAR-ATH_tactical_atd.txt").is_file():
        pytest.skip("competition data not present")
    return d


@pytest.fixture(scope="session")
def dfl_dir():
    d = Path(os.environ.get("MTAH_DFL_DIR", DEFAULT_DFL))
    if not (d / "game_9001_H1.npz").is_file():
        pytest.skip("DFL proxy data not present")
    return d
