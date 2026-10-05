import pandas as pd
from tabular_preprocessing.centering import center_splits

def test_statistics_are_fit_on_train_only():
    train = pd.DataFrame({'x': [0.0, 2.0]})
    test = pd.DataFrame({'x': [100.0]})
    centered_train, _, mean = center_splits(train, test)
    assert mean['x'] == 1.0
    assert centered_train['x'].mean() == 0.0

def test_output_columns_are_preserved():
    train = pd.DataFrame({'a': [1], 'b': [2]})
    test = pd.DataFrame({'a': [3], 'b': [4]})
    assert list(center_splits(train, test)[0].columns) == ['a', 'b']
