import numpy as np
from array_ops.norms import feature_l2_norms
def test_returns_one_norm_per_feature():
    result = feature_l2_norms([[3, 0], [4, 12]])
    np.testing.assert_allclose(result, [5, 12])
def test_output_shape_tracks_columns():
    assert feature_l2_norms(np.ones((2, 3))).shape == (3,)
