import numpy as np
from tensor_shapes.images import spatial_channel_mean
def test_evaluator_non_square_spatial_axes():
    data = np.arange(24).reshape(1, 2, 3, 4)
    np.testing.assert_allclose(spatial_channel_mean(data), data.mean(axis=(2, 3)))
def test_evaluator_dtype_is_floating():
    assert spatial_channel_mean(np.ones((1, 1, 1, 1), dtype=int)).dtype.kind == 'f'
