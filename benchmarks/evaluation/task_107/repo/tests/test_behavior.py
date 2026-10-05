import pytest
from heldout_metrics.regression import mae
def test_mae_uses_absolute_errors():
    assert mae([0, 0], [3, 4]) == 3.5
def test_shape_mismatch_rejected():
    with pytest.raises(ValueError): mae([1,2], [1])
