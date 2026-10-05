import pytest
from heldout_metrics.macro_f1 import macro_f1
def test_evaluator_includes_predicted_only_class():
    assert macro_f1([0,0], [0,2]) == pytest.approx((2/3 + 0)/2)
def test_evaluator_empty_inputs_return_zero():
    assert macro_f1([], []) == 0.0
