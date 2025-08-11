"""
generate_raw_data.py

Generate a synthetic raw_customers.csv for churn prediction.
"""

import pandas as pd
import numpy as np

def generate_data(n_customers=1000, n_calls=30, random_seed=42):
    """
    Create synthetic customer records.

    Args:
        n_customers (int): Number of customers to simulate.
        n_calls (int): Number of call durations per customer.
        random_seed (int): Seed for reproducibility.

    Returns:
        pd.DataFrame: Raw customer data.
    """
    np.random.seed(random_seed)
    customer_ids = [f"C{i:05d}" for i in range(n_customers)]
    call_durations = [
        np.random.randint(1, 60, size=n_calls).tolist()
        for _ in range(n_customers)
    ]
    data_usage = np.round(np.random.uniform(0.5, 50.0, size=n_customers), 2)
    contract_length = np.random.randint(1, 36, size=n_customers)
    churn_labels = np.random.choice([0, 1], size=n_customers, p=[0.85, 0.15])

    df = pd.DataFrame({
        "customer_id": customer_ids,
        "call_duration": call_durations,
        "monthly_data_usage": data_usage,
        "contract_length": contract_length,
        "churn": churn_labels
    })
    return df

if __name__ == "__main__":
    df_raw = generate_data()
    df_raw.to_csv("data/raw_customers.csv", index=False)
    print("Synthetic raw_customers.csv generated in data/")
