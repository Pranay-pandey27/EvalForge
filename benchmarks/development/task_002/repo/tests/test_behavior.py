import numpy as np
from ml_metrics.regression import rmse

def test_rmse_takes_square_root():
    assert rmse(np.array([0, 0]), np.array([3, 4])) == pytest.approx(np.sqrt(12.5))

def test_rmse_is_zero_for_exact_predictions():
    assert rmse(np.array([1, 2]), np.array([1, 2])) == 0.0

import pytest
