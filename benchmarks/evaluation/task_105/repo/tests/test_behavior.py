import pandas as pd
from tabular.features import split_features_target
def test_target_is_not_in_features():
    x, y = split_features_target(pd.DataFrame({'a':[1], 'label':[0]}), 'label')
    assert list(x.columns) == ['a'] and y.tolist() == [0]
def test_input_is_not_mutated():
    frame = pd.DataFrame({'a':[1], 'y':[2]}); before = frame.copy()
    split_features_target(frame, 'y'); pd.testing.assert_frame_equal(frame, before)
