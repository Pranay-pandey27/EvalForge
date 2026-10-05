import numpy as np
from pde_boundaries.laplacian import periodic_laplacian
def test_evaluator_dx_scaling():
    np.testing.assert_allclose(periodic_laplacian([1,0,0],.5),[-8,4,4])
def test_evaluator_sum_is_zero_periodic():
    assert abs(periodic_laplacian([1,3,-2,4]).sum()) < 1e-12
