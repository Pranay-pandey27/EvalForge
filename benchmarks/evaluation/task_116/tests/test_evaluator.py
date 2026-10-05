import numpy as np
from numerics.scaling import minmax_scale
def test_evaluator_matrix_shape_preserved():
    data=np.full((2,3),9); assert minmax_scale(data).shape==(2,3)
def test_evaluator_output_is_float():
    assert minmax_scale(np.array([1,1],dtype=int)).dtype.kind=='f'
