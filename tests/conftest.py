import os
from pathlib import Path

import pytest

DEFAULT_DATA = Path(os.path.join(os.path.expanduser("~"), "mtah-data"))
DEFAULT_DFL = Path(os.environ.get("MTAH_DFL_DIR", os.path.join(os.path.expanduser("~"), "dfl-open", "work", "fp1d")))


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
