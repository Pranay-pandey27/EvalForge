import numpy as np
def macro_f1(y_true, y_pred):
    truth, pred = np.asarray(y_true), np.asarray(y_pred)
    return float(np.mean(truth == pred))
