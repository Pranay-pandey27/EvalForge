import numpy as np
from array_ops.softmax import softmax_rows
def test_evaluator_translation_invariance_per_row():
    base = np.array([[1., 2.], [4., 8.]])
    shifts = np.array([[100.], [-50.]])
    np.testing.assert_allclose(softmax_rows(base), softmax_rows(base + shifts))
def test_evaluator_shape_preserved():
    assert softmax_rows(np.zeros((4, 3))).shape == (4, 3)
