import pandas as pd
from tabular_preprocessing.centering import center_splits

def test_evaluator_test_values_do_not_change_fitted_mean():
    train = pd.DataFrame({'x': [2.0, 4.0]})
    _, _, mean_a = center_splits(train, pd.DataFrame({'x': [-999.0]}))
    _, _, mean_b = center_splits(train, pd.DataFrame({'x': [999.0]}))
    pd.testing.assert_series_equal(mean_a, mean_b)

def test_evaluator_applies_train_mean_to_test():
    train = pd.DataFrame({'x': [1.0, 3.0]})
    test = pd.DataFrame({'x': [5.0]})
    assert center_splits(train, test)[1].iloc[0, 0] == 3.0
