import numpy as np
def mae(y_true, y_pred):
    errors = np.asarray(y_true) - np.asarray(y_pred)
    return float(np.sqrt(np.mean(errors ** 2)))
