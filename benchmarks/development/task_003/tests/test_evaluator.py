import numpy as np
from stable_losses.binary import binary_cross_entropy

def test_evaluator_does_not_mutate_probabilities():
    p = np.array([0.0, 0.8])
    before = p.copy()
    binary_cross_entropy(np.array([0, 1]), p)
    np.testing.assert_array_equal(p, before)

def test_evaluator_near_boundary_is_nonnegative():
    assert binary_cross_entropy(np.array([1]), np.array([1.0])) >= 0.0
