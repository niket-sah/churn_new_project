"""
generate_at_risk_list.py

Identify and export customers with high churn probability.
"""

import ast
import argparse
import tensorflow as tf
from utils import load_data

def main(raw_path: str, model_path: str, output_csv: str):
    """
    Entry point for at-risk list generation.

    Args:
        raw_path (str): Path to raw_customers.csv.
        model_path (str): Path to trained model.h5.
        output_csv (str): Path to save at-risk customers.
    """
    df = load_data(raw_path)

    df['call_duration'] = df['call_duration'].apply(ast.literal_eval)
    df['call_total'] = df['call_duration'].apply(lambda x: sum(x))
    df['call_avg']   = df['call_duration'].apply(lambda x: sum(x) / len(x))
    df['call_max']   = df['call_duration'].apply(lambda x: max(x))

    feature_cols = [
        "call_total", "call_avg", "call_max",
        "monthly_data_usage", "contract_length"
    ]
    X = df[feature_cols].values

    model = tf.keras.models.load_model(model_path)
    probs = model.predict(X).flatten()

    df['churn_prob'] = probs
    at_risk = df[df['churn_prob'] >= 0.25].sort_values(
        by="churn_prob", ascending=False
    )
    at_risk.to_csv(output_csv, index=False)
    print(f"At-risk list saved to {output_csv} ({len(at_risk)} records)")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate at-risk customer list."
    )
    parser.add_argument(
        "--raw_path", type=str, required=True,
        help="Path to raw_customers.csv"
    )
    parser.add_argument(
        "--model_path", type=str, required=True,
        help="Path to model.h5"
    )
    parser.add_argument(
        "--output_csv", type=str, required=True,
        help="Where to save at-risk list"
    )
    args = parser.parse_args()
    main(args.raw_path, args.model_path, args.output_csv)
