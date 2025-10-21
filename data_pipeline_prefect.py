# data_pipeline_prefect.py
from prefect import flow, task, get_run_logger
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import seaborn as sns
import matplotlib.pyplot as plt
import logging
import os

logging.basicConfig(level=logging.INFO)

@task
def ingest_data(path):
    logger = get_run_logger()
    logger.info("Loading data...")
    df = pd.read_csv(path)
    logger.info(f"Data loaded with shape: {df.shape}")
    return df

@task
def preprocess_data(data):
    logger = get_run_logger()
    logger.info("Preprocessing data...")
    
    numeric_cols = data.select_dtypes(include=['float64', 'int64']).columns
    data[numeric_cols] = data[numeric_cols].fillna(data[numeric_cols].mean())
    
    categorical_cols = data.select_dtypes(include=['object']).columns
    data_encoded = pd.get_dummies(data, columns=categorical_cols, drop_first=True)
    
    scaler = MinMaxScaler()
    data_encoded[numeric_cols] = scaler.fit_transform(data_encoded[numeric_cols])
    
    logger.info(f"Preprocessing completed. Processed data shape: {data_encoded.shape}")
    return data_encoded

@task
def perform_eda(data):
    logger = get_run_logger()
    logger.info("Performing EDA...")
    
    # Create outputs directory
    os.makedirs("eda_outputs", exist_ok=True)
    
    # Correlation heatmap
    corr = data.corr()
    heatmap_path = os.path.join("eda_outputs", "correlation_heatmap.png")
    plt.figure(figsize=(12,8))
    sns.heatmap(corr, cmap='coolwarm', annot=False)
    plt.savefig(heatmap_path)
    plt.close()
    logger.add_artifact(heatmap_path)
    
    # Summary statistics
    summary_path = os.path.join("eda_outputs", "eda_summary_statistics.csv")
    data.describe().to_csv(summary_path)
    logger.add_artifact(summary_path)
    
    logger.info("EDA completed. Artifacts saved.")
    return data

@task
def save_processed_data(data, path):
    logger = get_run_logger()
    os.makedirs(os.path.dirname(path), exist_ok=True)
    data.to_csv(path, index=False)
    logger.info(f"Processed data saved to {path}")
    # Log as Prefect artifact so it's visible in Prefect Cloud
    logger.add_artifact(path)

@flow(name="Farmer Suicide Data Pipeline")
def data_pipeline_flow():
    # 1. Load raw data
    data = ingest_data("farmer_suicide_large_realistic_train.csv")
    
    # 2. Preprocess
    data = preprocess_data(data)
    
    # 3. EDA
    data = perform_eda(data)
    
    # 4. Save processed CSV
    save_processed_data(data, "processed_outputs/farmer_suicide_processed.csv")

#if __name__ == "__main__":
#    data_pipeline_flow()
