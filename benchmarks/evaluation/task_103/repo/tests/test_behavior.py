import numpy as np
from tensor_shapes.images import spatial_channel_mean
def test_output_is_batch_by_channel():
    assert spatial_channel_mean(np.ones((2, 3, 4, 5))).shape == (2, 3)
def test_channel_values_are_independent():
    image = np.stack([np.ones((2, 2)), np.full((2, 2), 7)])[None, ...]
    np.testing.assert_allclose(spatial_channel_mean(image), [[1, 7]])
