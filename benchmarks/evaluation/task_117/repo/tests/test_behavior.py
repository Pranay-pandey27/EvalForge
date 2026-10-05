import numpy as np
from reproducibility_eval.split import shuffled_indices
def test_same_seed_same_order():
    np.testing.assert_array_equal(shuffled_indices(20,7),shuffled_indices(20,7))
def test_result_is_a_permutation():
    assert sorted(shuffled_indices(10,1).tolist())==list(range(10))
