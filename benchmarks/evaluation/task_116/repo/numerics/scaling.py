import numpy as np
def minmax_scale(values):
    data = np.asarray(values, dtype=float)
    return (data - data.min()) / (data.max() - data.min())
