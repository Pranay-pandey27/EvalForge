import numpy as np
def feature_l2_norms(matrix):
    values = np.asarray(matrix, dtype=float)
    return np.sqrt(np.sum(values ** 2, axis=1))
