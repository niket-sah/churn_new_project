# Churn Prediction for ConnectSphere Telecom

This project builds a complete pipeline to predict customer churn using an ANN binary classifier.

## Directory Structure

```text
churn_project/
├── data/
│   ├── raw_customers.csv       # Generated synthetic data
│   └── generate_raw_data.py    # Script to create raw data
├── src/
│   ├── utils.py
│   ├── data_preprocessing.py
│   ├── train_model.py
│   ├── evaluate_model.py
│   └── generate_at_risk_list.py
├── models/
│   └── model.h5                # Trained model
├── results/
│   └── at_risk_customers.csv   # At-risk customers output
├── requirements.txt
└── README.md
