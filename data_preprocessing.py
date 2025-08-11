"""
data_preprocessing.py

Load raw data, compute features, and split into train/val/test sets.
"""

import ast
import argparse
import pandas as pd
from sklearn.model_selection import train_test_split
from utils import load_data, save_data

def compute_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute churn prediction features.

    Args:
        df (pd.DataFrame): Raw customer data with a 'call_duration' list per row.

    Returns:
        pd.DataFrame: DataFrame with new feature columns.
    """
    df['call_duration'] = df['call_duration'].apply(ast.literal_eval)
    df['call_total'] = df['call_duration'].apply(lambda x: sum(x))
    df['call_avg']   = df['call_duration'].apply(lambda x: sum(x) / len(x))
    df['call_max']   = df['call_duration'].apply(lambda x: max(x))
    # monthly_data_usage and contract_length remain unchanged
    return df

def split_and_save(df: pd.DataFrame, output_dir: str):
    """
    Split into train/val/test and save CSVs.

    Args:
        df (pd.DataFrame): Feature-engineered DataFrame.
        output_dir (str): Directory to save splits.
    """
    train, temp = train_test_split(
        df, test_size=0.3, random_state=42, stratify=df['churn']
    )
    val, test = train_test_split(
        temp, test_size=0.5, random_state=42, stratify=temp['churn']
    )

    save_data(train, f"{output_dir}/train.csv")
    save_data(val,   f"{output_dir}/val.csv")
    save_data(test,  f"{output_dir}/test.csv")

def main(raw_path: str, output_dir: str):
    """
    Entry point for preprocessing script.

    Args:
        raw_path (str): Path to raw_customers.csv.
        output_dir (str): Directory to save train/val/test CSVs.
    """
    df_raw = load_data(raw_path)
    df_feat = compute_features(df_raw)
    split_and_save(df_feat, output_dir)
    print("Data preprocessing complete. Splits saved in", output_dir)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Preprocess raw customer data."
    )
    parser.add_argument(
        "--raw_path", type=str, required=True,
        help="Path to raw_customers.csv"
    )
    parser.add_argument(
        "--output_dir", type=str, required=True,
        help="Directory to save train/val/test CSVs"
    )
    args = parser.parse_args()
    main(args.raw_path, args.output_dir)
