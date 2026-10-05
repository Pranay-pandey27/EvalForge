from __future__ import annotations

import numpy as np
from scientific_stats import zscore_features


def test_standardizes_each_feature_column() -> None:
    values = np.array(
        [
            [1.0, 10.0],
            [2.0, 20.0],
            [3.0, 30.0],
            [4.0, 40.0],
        ]
    )

    transformed = zscore_features(values)

    np.testing.assert_allclose(transformed.mean(axis=0), np.zeros(2), atol=1e-12)
    np.testing.assert_allclose(transformed.std(axis=0), np.ones(2), atol=1e-12)


def test_preserves_shape() -> None:
    values = np.arange(15, dtype=float).reshape(5, 3)
    assert zscore_features(values).shape == values.shape

