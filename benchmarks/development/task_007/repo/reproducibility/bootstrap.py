import numpy as np

def bootstrap_mean(values, seed):
    np.random.seed(seed)
    data = np.asarray(values)
    sample = np.random.choice(data, size=len(data), replace=True)
    return float(sample.mean())
