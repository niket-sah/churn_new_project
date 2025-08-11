"""
evaluate_model.py

Load a trained model and evaluate it on the test set.
"""

import argparse
import tensorflow as tf
from sklearn.metrics import accuracy_score, f1_score
from utils import load_data, scale_features

def main(model_path: str, test_path: str, metrics_path: str):
    """
    Entry point for evaluation script.

    Args:
        model_path (str): Path to saved model.h5.
        test_path (str): Path to test.csv.
        metrics_path (str): File to write accuracy and F1-score.
    """
    test = load_data(test_path)
    feature_cols = [
        "call_total", "call_avg", "call_max",
        "monthly_data_usage", "contract_length"
    ]
    label_col = "churn"

    _, _, test_scaled = scale_features(test, test, test, feature_cols)
    X_test = test_scaled[feature_cols].values
    y_test = test_scaled[label_col].values

    model = tf.keras.models.load_model(model_path)
    preds = model.predict(X_test).flatten()
    y_pred = (preds >= 0.5).astype(int)

    acc = accuracy_score(y_test, y_pred)
    f1  = f1_score(y_test, y_pred)

    with open(metrics_path, "w") as f:
        f.write(f"accuracy: {acc:.4f}\n")
        f.write(f"f1_score: {f1:.4f}\n")

    print("Evaluation complete. Metrics saved to", metrics_path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Evaluate churn model."
    )
    parser.add_argument(
        "--model_path", type=str, required=True,
        help="Path to model.h5"
    )
    parser.add_argument(
        "--test_path", type=str, required=True,
        help="Path to test.csv"
    )
    parser.add_argument(
        "--metrics_path", type=str, required=True,
        help="Where to save metrics.txt"
    )
    args = parser.parse_args()
    main(args.model_path, args.test_path, args.metrics_path)
