import pandas as pd
from tabular.encoding import encode_splits
def test_test_only_category_does_not_create_column():
    train = pd.DataFrame({'color': ['red', 'blue']})
    test = pd.DataFrame({'color': ['green']})
    train_x, test_x = encode_splits(train, test, 'color')
    assert set(train_x.columns) == {'color_red', 'color_blue'}
    assert list(test_x.iloc[0]) == [0, 0]
def test_split_row_counts_are_preserved():
    a, b = encode_splits(pd.DataFrame({'x':['a','b']}), pd.DataFrame({'x':['a']}), 'x')
    assert (len(a), len(b)) == (2, 1)
