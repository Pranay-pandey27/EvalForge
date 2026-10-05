import numpy as np
from stable_losses.binary import binary_cross_entropy

def test_boundary_probabilities_are_finite():
    value = binary_cross_entropy(np.array([0, 1]), np.array([0.0, 1.0]))
    assert np.isfinite(value)

def test_typical_probabilities_match_formula():
    assert binary_cross_entropy(np.array([1]), np.array([0.5])) == np.log(2)
