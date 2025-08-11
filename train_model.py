"""
train_model.py

Define and train an ANN binary classifier for churn prediction.
"""

import argparse
import tensorflow as tf
from utils import load_data, scale_features

def build_model(input_dim: int) -> tf.keras.Model:
    """
    Build a simple feedforward neural network.

    Args:
        input_dim (int): Number of input features.

    Returns:
        tf.keras.Model: Compiled Keras model.
    """
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(32, activation="relu", input_shape=(input_dim,)),
        tf.keras.layers.Dense(16, activation="relu"),
        tf.keras.layers.Dense(1, activation="sigmoid")
    ])
    model.compile(
        optimizer="adam",
        loss="binary_crossentropy",
        metrics=["accuracy"]
    )
    return model

def main(train_path: str, val_path: str, model_path: str):
    """
    Entry point for training script.

    Args:
        train_path (str): Path to train.csv.
        val_path (str): Path to val.csv.
        model_path (str): Path to save trained model.
    """
    train = load_data(train_path)
    val   = load_data(val_path)

    feature_cols = [
        "call_total", "call_avg", "call_max",
        "monthly_data_usage", "contract_length"
    ]
    label_col = "churn"

    train_scaled, val_scaled, _ = scale_features(
        train, val, train, feature_cols
    )

    X_train = train_scaled[feature_cols].values
    y_train = train_scaled[label_col].values
    X_val   = val_scaled[feature_cols].values
    y_val   = val_scaled[label_col].values

    model = build_model(len(feature_cols))
    model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=20,
        batch_size=32
    )
    model.save(model_path)
    print(f"Model trained and saved to {model_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train churn prediction model."
    )
    parser.add_argument(
        "--train_path", type=str, required=True,
        help="Path to train.csv"
    )
    parser.add_argument(
        "--val_path", type=str, required=True,
        help="Path to val.csv"
    )
    parser.add_argument(
        "--model_path", type=str, required=True,
        help="Where to save model.h5"
    )
    args = parser.parse_args()
    main(args.train_path, args.val_path, args.model_path)
