# data_pipeline.py
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import seaborn as sns
import matplotlib.pyplot as plt
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')

def main():
    logging.info("Starting Data Pipeline")

    # -----------------------------
    # 1. Data Ingestion
    # -----------------------------
    path = "farmer_suicide_large_realistic_train.csv"  # Replace with your CSV path
    data = pd.read_csv(path)
    logging.info(f"Data loaded with shape {data.shape}")

    # -----------------------------
    # 2. Fill missing numeric values
    # -----------------------------
    numeric_cols = data.select_dtypes(include=['float64', 'int64']).columns
    data[numeric_cols] = data[numeric_cols].fillna(data[numeric_cols].mean())
    logging.info(f"Filled missing values for numeric columns: {list(numeric_cols)}")

    # -----------------------------
    # 3. Encode categorical columns
    # -----------------------------
    categorical_cols = data.select_dtypes(include=['object']).columns
    data_encoded = pd.get_dummies(data, columns=categorical_cols, drop_first=True)
    logging.info(f"Encoded categorical columns: {list(categorical_cols)}")

    # -----------------------------
    # 4. Normalize numeric columns
    # -----------------------------
    numeric_cols_encoded = data_encoded.select_dtypes(include=['float64', 'int64']).columns
    scaler = MinMaxScaler()
    data_encoded[numeric_cols_encoded] = scaler.fit_transform(data_encoded[numeric_cols_encoded])
    logging.info("Normalized numeric columns using MinMaxScaler")

    # -----------------------------
    # 5. Feature Engineering
    # -----------------------------
    # Example: Debt-to-Income ratio
    if 'Loan_Amount' in data.columns and 'Farmer_Income' in data.columns:
        data_encoded['Debt_Income_Ratio'] = data['Loan_Amount'] / data['Farmer_Income']
        logging.info("Added Debt_Income_Ratio feature")

    # -----------------------------
    # 6. Exploratory Data Analysis (EDA)
    # -----------------------------
    corr = data_encoded.corr()
    plt.figure(figsize=(12,8))
    sns.heatmap(corr, cmap='coolwarm', annot=False)
    plt.title("Correlation Heatmap")
    plt.tight_layout()
    plt.savefig("correlation_heatmap.png")
    logging.info("Correlation heatmap saved as 'correlation_heatmap.png'")

    summary = data_encoded.describe()
    summary.to_csv("eda_summary_statistics.csv")
    logging.info("Summary statistics saved as 'eda_summary_statistics.csv'")

    # -----------------------------
    # 7. Save processed data for ML
    # -----------------------------
    data_encoded.to_csv("farmer_suicide_processed.csv", index=False)
    logging.info("Processed data saved as 'farmer_suicide_processed.csv'")

    logging.info("Data Pipeline completed successfully!")

if __name__ == "__main__":
    main()
