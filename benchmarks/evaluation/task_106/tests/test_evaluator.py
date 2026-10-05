import numpy as np
from heldout_metrics.classification import binary_accuracy_from_logits
def test_evaluator_negative_logits_predict_zero():
    assert binary_accuracy_from_logits(np.array([-9., -0.01]), np.array([0,0])) == 1.0
def test_evaluator_returns_float():
    assert isinstance(binary_accuracy_from_logits(np.array([1]), np.array([1])), float)
