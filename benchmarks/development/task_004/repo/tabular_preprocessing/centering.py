import pandas as pd

def center_splits(train: pd.DataFrame, test: pd.DataFrame):
    combined = pd.concat([train, test], ignore_index=True)
    mean = combined.mean()
    return train - mean, test - mean, mean
