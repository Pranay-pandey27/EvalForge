import pandas as pd
from tabular.features import split_features_target
def test_evaluator_target_can_be_first_column():
    x, _ = split_features_target(pd.DataFrame({'y':[1], 'x':[2]}), 'y')
    assert list(x.columns) == ['x']
def test_evaluator_preserves_feature_order():
    x, _ = split_features_target(pd.DataFrame({'b':[1], 'y':[0], 'a':[2]}), 'y')
    assert list(x.columns) == ['b', 'a']
