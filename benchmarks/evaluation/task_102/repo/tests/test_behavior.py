import numpy as np
from array_ops.softmax import softmax_rows
def test_each_row_sums_to_one():
    result = softmax_rows([[1, 2], [3, 4]])
    np.testing.assert_allclose(result.sum(axis=1), [1, 1])
def test_large_logits_are_finite():
    assert np.isfinite(softmax_rows([[1000, 1001]])).all()
