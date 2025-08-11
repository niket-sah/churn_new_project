"""
utils.py

Helper functions for data loading, saving, and feature scaling.
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

def load_data(path: str) -> pd.DataFrame:
    """
    Load CSV data into a DataFrame.

    Args:
        path (str): File path to CSV.

    Returns:
        pd.DataFrame: Loaded data.
    """
    return pd.read_csv(path)

def save_data(df: pd.DataFrame, path: str) -> None:
    """
    Save a DataFrame to CSV.

    Args:
        df (pd.DataFrame): DataFrame to save.
        path (str): Destination CSV path.
    """
    df.to_csv(path, index=False)

def scale_features(train: pd.DataFrame,
                   val: pd.DataFrame,
                   test: pd.DataFrame,
                   feature_cols: list) -> tuple:
    """
    Fit a StandardScaler on training data and transform all splits.

    Args:
        train (pd.DataFrame): Training set.
        val (pd.DataFrame): Validation set.
        test (pd.DataFrame): Test set.
        feature_cols (list): List of numeric feature column names.

    Returns:
        tuple: Scaled (train, val, test) DataFrames.
    """
    scaler = StandardScaler()
    train_scaled = train.copy()
    val_scaled = val.copy()
    test_scaled = test.copy()

    scaler.fit(train[feature_cols])
    train_scaled[feature_cols] = scaler.transform(train[feature_cols])
    val_scaled[feature_cols]   = scaler.transform(val[feature_cols])
    test_scaled[feature_cols]  = scaler.transform(test[feature_cols])

    return train_scaled, val_scaled, test_scaled
