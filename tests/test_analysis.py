import pytest
import numpy as np
from analysis import gini_impurity


def test_gini_impurity():
    values = np.array([5, 5])
    gi = gini_impurity(values)
    assert gi == 0.5

    values = np.array([0, 10])
    gi = gini_impurity(values)
    assert gi == 0

    values = np.array([3, 7])
    gi = gini_impurity(values)
    assert gi == pytest.approx(0.42)

    values = np.array([0, 0])
    gi = gini_impurity(values)
    assert np.isnan(gi)
