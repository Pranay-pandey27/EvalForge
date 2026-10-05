import numpy as np
import pytest
from heldout_metrics.classification import binary_accuracy_from_logits
def test_zero_is_logit_decision_boundary():
    assert binary_accuracy_from_logits(np.array([0.2, -0.2]), np.array([1, 0])) == 1.0
def test_shape_mismatch_is_rejected():
    with pytest.raises(ValueError): binary_accuracy_from_logits(np.array([1,2]), np.array([1]))
