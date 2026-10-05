import numpy as np

def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    errors = np.asarray(y_true, dtype=float) - np.asarray(y_pred, dtype=float)
    return float(np.mean(errors ** 2))
