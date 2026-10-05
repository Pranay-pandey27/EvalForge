import numpy as np
def shuffled_indices(size, seed):
    rng = np.random.default_rng()
    return rng.permutation(size)
