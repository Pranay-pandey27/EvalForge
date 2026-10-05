import numpy as np
from ml_metrics.regression import rmse

def test_evaluator_returns_python_float():
    assert isinstance(rmse(np.array([1]), np.array([2])), float)

def test_evaluator_handles_fractional_errors():
    assert rmse(np.array([0.0, 1.0]), np.array([0.5, 1.5])) == 0.5
