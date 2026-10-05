import numpy as np
from reproducibility.bootstrap import bootstrap_mean

def test_does_not_reset_global_rng():
    np.random.seed(123)
    expected = np.random.random(2)
    np.random.seed(123)
    first = np.random.random()
    bootstrap_mean([1, 2, 3], 7)
    second = np.random.random()
    np.testing.assert_allclose([first, second], expected)

def test_same_seed_is_reproducible():
    assert bootstrap_mean([1, 2, 9], 4) == bootstrap_mean([1, 2, 9], 4)
