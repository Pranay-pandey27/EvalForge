import numpy as np
from reproducibility_eval.split import shuffled_indices
def test_evaluator_known_seed():
    np.testing.assert_array_equal(shuffled_indices(5,3),np.random.default_rng(3).permutation(5))
def test_evaluator_zero_size():
    assert shuffled_indices(0,9).size==0
