import numpy as np
def softmax_rows(logits):
    values = np.asarray(logits, dtype=float)
    shifted = values - values.max()
    exponentials = np.exp(shifted)
    return exponentials / exponentials.sum()
