from __future__ import annotations

import numpy as np
import pytest
from scientific_stats import zscore_features


def test_constant_feature_maps_to_finite_zeros() -> None:
    values = np.array([[2.0, 5.0], [2.0, 7.0], [2.0, 9.0]])
    transformed = zscore_features(values)
    assert np.isfinite(transformed).all()
    np.testing.assert_array_equal(transformed[:, 0], np.zeros(3))


def test_does_not_mutate_input() -> None:
    values = np.array([[1.0, 3.0], [2.0, 5.0], [3.0, 7.0]])
    original = values.copy()
    zscore_features(values)
    np.testing.assert_array_equal(values, original)


def test_rejects_non_matrix_input() -> None:
    with pytest.raises(ValueError, match="two-dimensional"):
        zscore_features(np.array([1.0, 2.0, 3.0]))

