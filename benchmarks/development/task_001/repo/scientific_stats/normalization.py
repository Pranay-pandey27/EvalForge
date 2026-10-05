"""Feature normalization utilities."""

from __future__ import annotations

import numpy as np


def zscore_features(values: np.ndarray) -> np.ndarray:
    """Return a column-wise z-score transform of a feature matrix.

    Constant-valued feature columns should map to finite zeros.
    """

    matrix = np.asarray(values, dtype=float)
    if matrix.ndim != 2:
        raise ValueError("values must be a two-dimensional feature matrix")

    means = matrix.mean(axis=1)
    standard_deviations = matrix.std(axis=1)
    return (matrix - means) / standard_deviations

