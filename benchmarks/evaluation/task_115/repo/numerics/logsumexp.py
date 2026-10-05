import numpy as np
def logsumexp(values):
    data = np.asarray(values, dtype=float)
    return float(np.log(np.exp(data).sum()))
