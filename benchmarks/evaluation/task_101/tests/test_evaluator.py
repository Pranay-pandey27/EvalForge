import numpy as np
from array_ops.norms import feature_l2_norms
def test_evaluator_single_row():
    np.testing.assert_allclose(feature_l2_norms([[-3, 4]]), [3, 4])
def test_evaluator_zero_column():
    np.testing.assert_allclose(feature_l2_norms([[0, 1], [0, 2]]), [0, np.sqrt(5)])
