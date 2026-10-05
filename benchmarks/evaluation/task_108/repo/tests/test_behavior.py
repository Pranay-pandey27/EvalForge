import pytest
from heldout_metrics.macro_f1 import macro_f1
def test_macro_f1_differs_from_accuracy_for_imbalance():
    assert macro_f1([0,0,0,1], [0,0,0,0]) == pytest.approx(3/7)
def test_perfect_predictions_score_one():
    assert macro_f1([0,1,2], [0,1,2]) == 1.0
