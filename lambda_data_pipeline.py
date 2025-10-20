# lambda_data_pipeline.py
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import seaborn as sns
import matplotlib.pyplot as plt
import boto3
from io import BytesIO
import logging

logging.basicConfig(level=logging.INFO)
s3 = boto3.client('s3')

# -----------------------------
# Configuration
# -----------------------------
BUCKET = "farmer-suicide-data"  # replace with your S3 bucket name
RAW_KEY = "raw-data/farmer_suicide_large_realistic_train.csv"
PROCESSED_KEY = "processed-data/farmer_suicide_processed.csv"
EDA_SUMMARY_KEY = "processed-data/eda_summary_statistics.csv"
HEATMAP_KEY = "processed-data/correlation_heatmap.png"

def lambda_handler(event, context):
    logging.info("Starting Lambda Data Pipeline")

    # -----------------------------
    # 1. Data Ingestion from S3
    # -----------------------------
    obj = s3.get_object(Bucket=BUCKET, Key=RAW_KEY)
    data = pd.read_csv(obj['Body'])
    logging.info(f"Data loaded from S3 with shape {data.shape}")

    # -----------------------------
    # 2. Fill missing numeric values
    # -----------------------------
    numeric_cols = data.select_dtypes(include=['float64', 'int64']).columns
    data[numeric_cols] = data[numeric_cols].fillna(data[numeric_cols].mean())
    logging.info(f"Filled missing numeric columns: {list(numeric_cols)}")

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
    if 'Loan_Amount' in data.columns and 'Farmer_Income' in data.columns:
        # Avoid division by zero
        data_encoded['Debt_Income_Ratio'] = data['Loan_Amount'] / data['Farmer_Income'].replace(0, np.nan)
        data_encoded['Debt_Income_Ratio'].fillna(0, inplace=True)
        logging.info("Added Debt_Income_Ratio feature")

    # -----------------------------
    # 6. EDA
    # -----------------------------
    corr = data_encoded.corr()
    plt.figure(figsize=(12,8))
    sns.heatmap(corr, cmap='coolwarm', annot=False)
    plt.title("Correlation Heatmap")
    plt.tight_layout()

    # Save heatmap to S3
    buf = BytesIO()
    plt.savefig(buf, format="png")
    buf.seek(0)
    s3.put_object(Bucket=BUCKET, Key=HEATMAP_KEY, Body=buf)
    plt.close()
    logging.info(f"Correlation heatmap saved to S3: {HEATMAP_KEY}")

    # Save summary statistics to S3
    summary = data_encoded.describe()
    csv_buffer = BytesIO()
    summary.to_csv(csv_buffer)
    csv_buffer.seek(0)
    s3.put_object(Bucket=BUCKET, Key=EDA_SUMMARY_KEY, Body=csv_buffer)
    logging.info(f"Summary statistics saved to S3: {EDA_SUMMARY_KEY}")

    # -----------------------------
    # 7. Save processed data for ML
    # -----------------------------
    csv_buffer2 = BytesIO()
    data_encoded.to_csv(csv_buffer2, index=False)
    csv_buffer2.seek(0)
    s3.put_object(Bucket=BUCKET, Key=PROCESSED_KEY, Body=csv_buffer2)
    logging.info(f"Processed data saved to S3: {PROCESSED_KEY}")

    logging.info("Lambda Data Pipeline completed successfully!")
    return {"status": "success", "records": len(data_encoded)}
