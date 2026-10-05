import pandas as pd
def encode_splits(train, test, column):
    combined = pd.concat([train[[column]], test[[column]]], ignore_index=True)
    encoded = pd.get_dummies(combined[column], prefix=column, dtype=int)
    return encoded.iloc[:len(train)].reset_index(drop=True), encoded.iloc[len(train):].reset_index(drop=True)
