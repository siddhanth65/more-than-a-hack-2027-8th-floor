import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from sensitivity_sp_type import SP_COL, ConstColumnDead, verdict  # noqa: E402


class _Rec:
    policy = "full"
    def predict_proba(self, X):
        self.seen = X.copy(); return X[:, 0]


def test_wrapper_sets_column_without_mutating_input():
    X = np.arange(40, dtype=np.float32).reshape(4, 10)
    inner = _Rec(); w = ConstColumnDead(inner, col=3, value=4.0)
    w.predict_proba(X)
    assert (inner.seen[:, 3] == 4.0).all() and (inner.seen[:, [0, 1, 2, 4]] == X[:, [0, 1, 2, 4]]).all()
    assert X[0, 3] == 3.0 and w.policy == "full" and isinstance(SP_COL, int)


def test_verdict_thresholds():
    n = {"a": 0.5}; r = {"d": 0.3}
    assert verdict(n, {"a": 0.503}, r, {"d": 0.305})[2] == "PASS"
    assert verdict(n, {"a": 0.51}, r, r)[2] == "FAIL"
    assert verdict(n, n, r, {"d": 0.32})[2] == "FAIL"
