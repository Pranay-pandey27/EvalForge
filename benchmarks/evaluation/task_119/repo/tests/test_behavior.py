import numpy as np
from pde_boundaries.laplacian import periodic_laplacian
def test_edges_use_wrapped_neighbors():
    np.testing.assert_allclose(periodic_laplacian([1,0,0]),[-2,1,1])
def test_constant_field_has_zero_laplacian():
    np.testing.assert_allclose(periodic_laplacian([4,4,4]),[0,0,0])
