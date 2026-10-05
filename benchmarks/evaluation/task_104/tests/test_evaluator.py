import pandas as pd
from tabular.encoding import encode_splits
def test_evaluator_column_order_follows_training_observation():
    train_x, _ = encode_splits(pd.DataFrame({'x':['z','a']}), pd.DataFrame({'x':['a']}), 'x')
    assert list(train_x.columns) == ['x_z', 'x_a']
def test_evaluator_known_test_category_is_encoded():
    _, test_x = encode_splits(pd.DataFrame({'x':['a','b']}), pd.DataFrame({'x':['b']}), 'x')
    assert test_x['x_b'].iloc[0] == 1
