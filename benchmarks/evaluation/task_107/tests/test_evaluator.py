from heldout_metrics.regression import mae
def test_evaluator_signed_errors_do_not_cancel():
    assert mae([0, 2], [1, 1]) == 1.0
def test_evaluator_exact_prediction():
    assert mae([1.5], [1.5]) == 0.0
