import numpy as np
from numerics.scaling import minmax_scale
def test_constant_values_become_finite_zeros():
    np.testing.assert_array_equal(minmax_scale([4,4,4]), [0,0,0])
def test_nonconstant_range_is_zero_to_one():
    np.testing.assert_allclose(minmax_scale([2,4,6]), [0,.5,1])
