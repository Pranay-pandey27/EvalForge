import numpy as np
def binary_accuracy_from_logits(logits, labels):
    predictions = np.asarray(logits) > 0.5
    return float(np.mean(predictions == np.asarray(labels)))
